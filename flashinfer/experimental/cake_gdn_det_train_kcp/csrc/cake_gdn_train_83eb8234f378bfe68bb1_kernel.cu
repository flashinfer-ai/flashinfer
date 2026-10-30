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
#define TMEM_ACC_DQS_OFFSET 0
#define TMEM_ACC_DKS_OFFSET 64
#define TMEM_ACC_DQ_OFFSET 128
#define TMEM_ACC_DK_OFFSET 192
#define TMEM_ACC_DW_OFFSET 256
#define TMEM_ACC_DA_OFFSET 320
#define TMEM_ACC_DAQ_OFFSET 384
#define TMEM_ACC_DVB_OFFSET 384
#define TMEM_ACC_DKGB_OFFSET 256
#define TMEM_ACC_DA1_OFFSET 256
#define TMEM_ACC_DA2_OFFSET 320
#define TMEM_ACC_DKSRC_OFFSET 256
#define TMEM_ACC_X_OFFSET 384
#define TMEM_ACC_GP_OFFSET 448
#define NUM_ONE_STAGES 1
#define NUM_TK_PIPE_STAGES 2
#define SMEM_SMEM_V0_OFF 1024
#define SMEM_SMEM_V0_STAGE_BYTES 8192
#define SMEM_SMEM_V0_STRIDE 8192
#define SMEM_SMEM_V0_0_OFF 17408
#define SMEM_SMEM_V0_0_STAGE_BYTES 8192
#define SMEM_SMEM_V0_0_STRIDE 8192
#define SMEM_SMEM_V0_1_OFF 148480
#define SMEM_SMEM_V0_1_STAGE_BYTES 8192
#define SMEM_SMEM_V0_1_STRIDE 8192
#define SMEM_SMEM_V0_2_OFF 99328
#define SMEM_SMEM_V0_2_STAGE_BYTES 8192
#define SMEM_SMEM_V0_2_STRIDE 8192
#define SMEM_SMEM_V0_3_OFF 115712
#define SMEM_SMEM_V0_3_STAGE_BYTES 8192
#define SMEM_SMEM_V0_3_STRIDE 8192
#define SMEM_SMEM_V0_4_OFF 132096
#define SMEM_SMEM_V0_4_STAGE_BYTES 8192
#define SMEM_SMEM_V0_4_STRIDE 8192
#define SMEM_SMEM_V6_OFF 1024
#define SMEM_SMEM_V6_STAGE_BYTES 16384
#define SMEM_SMEM_V6_STRIDE 16384
#define SMEM_SMEM_V6_0_OFF 17408
#define SMEM_SMEM_V6_0_STAGE_BYTES 16384
#define SMEM_SMEM_V6_0_STRIDE 16384
#define SMEM_SMEM_V6_1_OFF 148480
#define SMEM_SMEM_V6_1_STAGE_BYTES 16384
#define SMEM_SMEM_V6_1_STRIDE 16384
#define SMEM_SMEM_V9_OFF 1024
#define SMEM_SMEM_V9_STAGE_BYTES 16384
#define SMEM_SMEM_V9_STRIDE 16384
#define SMEM_SMEM_V9_0_OFF 17408
#define SMEM_SMEM_V9_0_STAGE_BYTES 16384
#define SMEM_SMEM_V9_0_STRIDE 16384
#define SMEM_SMEM_V6_2_OFF 99328
#define SMEM_SMEM_V6_2_STAGE_BYTES 16384
#define SMEM_SMEM_V6_2_STRIDE 16384
#define SMEM_SMEM_V6_3_OFF 115712
#define SMEM_SMEM_V6_3_STAGE_BYTES 16384
#define SMEM_SMEM_V6_3_STRIDE 16384
#define SMEM_SMEM_V6_4_OFF 132096
#define SMEM_SMEM_V6_4_STAGE_BYTES 16384
#define SMEM_SMEM_V6_4_STRIDE 16384
#define SMEM_SMEM_V9_1_OFF 132096
#define SMEM_SMEM_V9_1_STAGE_BYTES 16384
#define SMEM_SMEM_V9_1_STRIDE 16384
#define SMEM_H_P_OFF 33792
#define SMEM_H_P_STAGE_BYTES 16384
#define SMEM_H_P_STRIDE 16384
#define SMEM_DH_P_OFF 66560
#define SMEM_DH_P_STAGE_BYTES 16384
#define SMEM_DH_P_STRIDE 16384
#define SMEM_H_K_OFF 33792
#define SMEM_H_K_STAGE_BYTES 32768
#define SMEM_H_K_STRIDE 32768
#define SMEM_DH_K_OFF 66560
#define SMEM_DH_K_STAGE_BYTES 32768
#define SMEM_DH_K_STRIDE 32768
#define SMEM_AKK_S_OFF 164864
#define SMEM_AKK_S_STAGE_BYTES 8192
#define SMEM_AKK_S_STRIDE 8192
#define SMEM_SMEM_V20_OFF 164864
#define SMEM_SMEM_V20_STAGE_BYTES 8192
#define SMEM_SMEM_V20_STRIDE 8192
#define SMEM_SMEM_V21_OFF 164864
#define SMEM_SMEM_V21_STAGE_BYTES 8192
#define SMEM_SMEM_V21_STRIDE 8192
#define SMEM_DWBT_K_OFF 33792
#define SMEM_DWBT_K_STAGE_BYTES 16384
#define SMEM_DWBT_K_STRIDE 16384
#define SMEM_DWBT_MN_OFF 33792
#define SMEM_DWBT_MN_STAGE_BYTES 16384
#define SMEM_DWBT_MN_STRIDE 16384
#define SMEM_SMEM_V6_5_OFF 50176
#define SMEM_SMEM_V6_5_STAGE_BYTES 16384
#define SMEM_SMEM_V6_5_STRIDE 16384
#define SMEM_SMEM_V6_6_OFF 210944
#define SMEM_SMEM_V6_6_STAGE_BYTES 16384
#define SMEM_SMEM_V6_6_STRIDE 16384
#define SMEM_SMEM_V9_2_OFF 210944
#define SMEM_SMEM_V9_2_STAGE_BYTES 16384
#define SMEM_SMEM_V9_2_STRIDE 16384
#define SMEM_SMEM_V20_0_OFF 173056
#define SMEM_SMEM_V20_0_STAGE_BYTES 8192
#define SMEM_SMEM_V20_0_STRIDE 8192
#define SMEM_SMEM_V20_1_OFF 181248
#define SMEM_SMEM_V20_1_STAGE_BYTES 8192
#define SMEM_SMEM_V20_1_STRIDE 8192
#define SMEM_SMEM_V21_0_OFF 181248
#define SMEM_SMEM_V21_0_STAGE_BYTES 8192
#define SMEM_SMEM_V21_0_STRIDE 8192
#define SMEM_SMEM_V20_2_OFF 189440
#define SMEM_SMEM_V20_2_STAGE_BYTES 8192
#define SMEM_SMEM_V20_2_STRIDE 8192
#define SMEM_SMEM_V21_1_OFF 189440
#define SMEM_SMEM_V21_1_STAGE_BYTES 8192
#define SMEM_SMEM_V21_1_STRIDE 8192
#define SMEM_SMEM_V20_3_OFF 197632
#define SMEM_SMEM_V20_3_STAGE_BYTES 8192
#define SMEM_SMEM_V20_3_STRIDE 8192
#define SMEM_SMEM_V21_2_OFF 197632
#define SMEM_SMEM_V21_2_STAGE_BYTES 8192
#define SMEM_SMEM_V21_2_STRIDE 8192
#define SMEM_STAGE_DV_OFF 132096
#define SMEM_STAGE_DV_STAGE_BYTES 16384
#define SMEM_STAGE_DV_STRIDE 16384
#define SMEM_STAGE_DQ_OFF 173056
#define SMEM_STAGE_DQ_STAGE_BYTES 16384
#define SMEM_STAGE_DQ_STRIDE 16384
#define SMEM_STAGE_DK_OFF 189440
#define SMEM_STAGE_DK_STAGE_BYTES 16384
#define SMEM_STAGE_DK_STRIDE 16384
#define SMEM_BETA_S_OFF 205824
#define SMEM_BETA_S_STAGE_BYTES 256
#define SMEM_BETA_S_STRIDE 256
#define SMEM_EG_S_OFF 206080
#define SMEM_EG_S_STAGE_BYTES 256
#define SMEM_EG_S_STRIDE 256
#define SMEM_RS_S_OFF 206336
#define SMEM_RS_S_STAGE_BYTES 256
#define SMEM_RS_S_STRIDE 256
#define SMEM_GC_S_OFF 206592
#define SMEM_GC_S_STAGE_BYTES 256
#define SMEM_GC_S_STRIDE 256
#define SMEM_RED_DG_OFF 206848
#define SMEM_RED_DG_STAGE_BYTES 1024
#define SMEM_RED_DG_STRIDE 1024
#define SMEM_RED_DB_OFF 207872
#define SMEM_RED_DB_STAGE_BYTES 1024
#define SMEM_RED_DB_STRIDE 1024
#define SMEM_RED_HK_OFF 208896
#define SMEM_RED_HK_STAGE_BYTES 1024
#define SMEM_RED_HK_STRIDE 1024
#define SMEM_RED_HD_OFF 209920
#define SMEM_RED_HD_STAGE_BYTES 32
#define SMEM_RED_HD_STRIDE 32
#define SMEM_TK_S_OFF 209984
#define SMEM_TK_S_STAGE_BYTES 16
#define SMEM_TK_S_STRIDE 16
#define SMEM_TOTAL 227328
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



__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}



extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_83eb8234f378bfe68bb1(const __grid_constant__ CUtensorMap q_tma, const __grid_constant__ CUtensorMap k_tma, const __grid_constant__ CUtensorMap v_tma, const __grid_constant__ CUtensorMap do_tma, const __grid_constant__ CUtensorMap h_tma, const __grid_constant__ CUtensorMap dh_tma, const __grid_constant__ CUtensorMap vn_tma, const __grid_constant__ CUtensorMap dvn_tma, const __grid_constant__ CUtensorMap akk_tma, float* __restrict__ beta, float* __restrict__ g_cum, __nv_bfloat16* __restrict__ dq, __nv_bfloat16* __restrict__ dk, __nv_bfloat16* __restrict__ dv, float* __restrict__ dbeta, float* __restrict__ dg_part, float* __restrict__ gamma_dot, float* __restrict__ dq_part, float* __restrict__ dk_part, int num_qk_heads, int num_heads, float scale, int dqk_tstride, int dv_tstride, int group_reduce, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, unsigned int* __restrict__ ticket)
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
    #define qk_full_addr (mbar_base + 0)
    #define h_full_addr (mbar_base + 8)
    #define dh_full_addr (mbar_base + 16)
    #define dovn_full_addr (mbar_base + 24)
    #define dvn_full_addr (mbar_base + 32)
    #define v_full_addr (mbar_base + 40)
    #define akk_full_addr (mbar_base + 48)
    #define a_committed_addr (mbar_base + 56)
    #define b_committed_addr (mbar_base + 64)
    #define d_committed_addr (mbar_base + 72)
    #define e_committed_addr (mbar_base + 80)
    #define a_ready_addr (mbar_base + 88)
    #define b_ready_addr (mbar_base + 96)
    #define c_ready_addr (mbar_base + 104)
    #define d_ready_addr (mbar_base + 112)
    #define e_ready_addr (mbar_base + 120)
    #define a_drained_addr (mbar_base + 128)
    #define b_drained_addr (mbar_base + 136)
    #define c_drained_addr (mbar_base + 144)
    #define d_drained_addr (mbar_base + 152)
    #define e_drained_addr (mbar_base + 160)
    #define dv_stored_addr (mbar_base + 168)
    #define tk_full_addr (mbar_base + 176)
    #define tk_empty_addr (mbar_base + 192)
    #define e1_ready_addr (mbar_base + 208)
    #define b1_ready_addr (mbar_base + 216)
    #define daq_drained_addr (mbar_base + 224)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_v0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v0_addr = smem + 1024;
    __nv_bfloat16* smem_v0_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_v0_addr_0 = smem + 17408;
    __nv_bfloat16* smem_v0_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int smem_v0_addr_1 = smem + 148480;
    __nv_bfloat16* smem_v0_2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int smem_v0_addr_2 = smem + 99328;
    __nv_bfloat16* smem_v0_3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int smem_v0_addr_3 = smem + 115712;
    __nv_bfloat16* smem_v0_4 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int smem_v0_addr_4 = smem + 132096;
    __nv_bfloat16* smem_v6 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v6_addr = smem + 1024;
    __nv_bfloat16* smem_v6_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_v6_addr_0 = smem + 17408;
    __nv_bfloat16* smem_v6_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int smem_v6_addr_1 = smem + 148480;
    __nv_bfloat16* smem_v9 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v9_addr = smem + 1024;
    __nv_bfloat16* smem_v9_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_v9_addr_0 = smem + 17408;
    __nv_bfloat16* smem_v6_2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int smem_v6_addr_2 = smem + 99328;
    __nv_bfloat16* smem_v6_3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int smem_v6_addr_3 = smem + 115712;
    __nv_bfloat16* smem_v6_4 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int smem_v6_addr_4 = smem + 132096;
    __nv_bfloat16* smem_v9_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int smem_v9_addr_1 = smem + 132096;
    __nv_bfloat16* h_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int h_p_addr = smem + 33792;
    __nv_bfloat16* dh_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int dh_p_addr = smem + 66560;
    __nv_bfloat16* h_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int h_k_addr = smem + 33792;
    __nv_bfloat16* dh_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int dh_k_addr = smem + 66560;
    __nv_bfloat16* akk_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int akk_s_addr = smem + 164864;
    __nv_bfloat16* smem_v20 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int smem_v20_addr = smem + 164864;
    __nv_bfloat16* smem_v21 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int smem_v21_addr = smem + 164864;
    __nv_bfloat16* dwbt_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int dwbt_k_addr = smem + 33792;
    __nv_bfloat16* dwbt_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int dwbt_mn_addr = smem + 33792;
    __nv_bfloat16* smem_v6_5 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int smem_v6_addr_5 = smem + 50176;
    __nv_bfloat16* smem_v6_6 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 210944);
    const int smem_v6_addr_6 = smem + 210944;
    __nv_bfloat16* smem_v9_2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 210944);
    const int smem_v9_addr_2 = smem + 210944;
    __nv_bfloat16* smem_v20_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int smem_v20_addr_0 = smem + 173056;
    __nv_bfloat16* smem_v20_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 181248);
    const int smem_v20_addr_1 = smem + 181248;
    __nv_bfloat16* smem_v21_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 181248);
    const int smem_v21_addr_0 = smem + 181248;
    __nv_bfloat16* smem_v20_2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 189440);
    const int smem_v20_addr_2 = smem + 189440;
    __nv_bfloat16* smem_v21_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 189440);
    const int smem_v21_addr_1 = smem + 189440;
    __nv_bfloat16* smem_v20_3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int smem_v20_addr_3 = smem + 197632;
    __nv_bfloat16* smem_v21_2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int smem_v21_addr_2 = smem + 197632;
    __nv_bfloat16* stage_dv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int stage_dv_addr = smem + 132096;
    __nv_bfloat16* stage_dq = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int stage_dq_addr = smem + 173056;
    __nv_bfloat16* stage_dk = reinterpret_cast<__nv_bfloat16*>(smem_raw + 189440);
    const int stage_dk_addr = smem + 189440;
    float* beta_s = reinterpret_cast<float*>(smem_raw + 205824);
    const int beta_s_addr = smem + 205824;
    float* eg_s = reinterpret_cast<float*>(smem_raw + 206080);
    const int eg_s_addr = smem + 206080;
    float* rs_s = reinterpret_cast<float*>(smem_raw + 206336);
    const int rs_s_addr = smem + 206336;
    float* gc_s = reinterpret_cast<float*>(smem_raw + 206592);
    const int gc_s_addr = smem + 206592;
    float* red_dg = reinterpret_cast<float*>(smem_raw + 206848);
    const int red_dg_addr = smem + 206848;
    float* red_db = reinterpret_cast<float*>(smem_raw + 207872);
    const int red_db_addr = smem + 207872;
    float* red_hk = reinterpret_cast<float*>(smem_raw + 208896);
    const int red_hk_addr = smem + 208896;
    float* red_hd = reinterpret_cast<float*>(smem_raw + 209920);
    const int red_hd_addr = smem + 209920;
    unsigned int* tk_s = reinterpret_cast<unsigned int*>(smem_raw + 209984);
    const int tk_s_addr = smem + 209984;

    // Mbarrier init (27 pipeline groups, 0 ordered-sequence groups, 29 barriers)
    // Mbarriers at smem_raw[0..232)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'one' ---
            // qk_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // h_full: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // dh_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // dovn_full: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // dvn_full: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // v_full: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // akk_full: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // a_committed: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // b_committed: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // d_committed: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // e_committed: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // a_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // b_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // c_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // d_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // e_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // a_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 128, 256);
            // b_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 136, 256);
            // c_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 144, 256);
            // d_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 152, 256);
            // e_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            // dv_stored: 1 barriers, init_count=256
            mbarrier_init(smem + 168, 256);
            // --- pipeline 'tk_pipe' ---
            // tk_full: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // tk_empty: 2 barriers, init_count=9
            mbarrier_init(smem + 192, 9);
            mbarrier_init(smem + 200, 9);
            // --- pipeline 'one' ---
            // e1_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            // b1_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            // daq_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 224, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 232);
    if (warp == 0) {
        int _tmem_hold = smem + 232;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_dqs = taddr;
    const int tmem_acc_dks = taddr + 64;
    const int tmem_acc_dq = taddr + 128;
    const int tmem_acc_dk = taddr + 192;
    const int tmem_acc_dw = taddr + 256;
    const int tmem_acc_da = taddr + 320;
    const int tmem_acc_daq = taddr + 384;
    const int tmem_acc_dvb = taddr + 384;
    const int tmem_acc_dkgb = taddr + 256;
    const int tmem_acc_da1 = taddr + 256;
    const int tmem_acc_da2 = taddr + 320;
    const int tmem_acc_dksrc = taddr + 256;
    const int tmem_acc_x = taddr + 384;
    const int tmem_acc_gp = taddr + 448;
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }

    // ---- Role: compute ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // compute_main
            int wq = warp % 4;
            int ch = warp / 4;
            int lane_0 = lane;
            int tid_1 = warp * 32 + lane;
            int d_row = wq * 32 + lane_0;
            int row_base = wq * 32 << 16;
            int t0 = ch * 32;
            int trow16 = wq * 16 + (lane_0 & 15);
            int lh = lane_0 >> 4;
            int colh = ch * 32 + lh * 16;
            int te = tid_1 >> 2;
            int de0 = (tid_1 & 3) * 32;
            int r_c = num_heads / num_qk_heads;
            int ngroups_c = seq_chunk_start[num_seqs] * num_qk_heads;
            unsigned int kpk[16];
            unsigned int ck_stage = 0;
            unsigned int _phase_tk_full = 0;
            unsigned int _phase_h_full_0 = 0;
            unsigned int _phase_dh_full_0 = 0;
            unsigned int _phase_qk_full_0 = 0;
            unsigned int _phase_a_ready_0 = 0;
            unsigned int _phase_b1_ready_0 = 0;
            unsigned int _phase_v_full_0 = 0;
            unsigned int _phase_dvn_full_0 = 0;
            unsigned int _phase_b_ready_0 = 0;
            unsigned int _phase_c_ready_0 = 0;
            unsigned int _phase_d_ready_0 = 0;
            unsigned int _phase_e1_ready_0 = 0;
            unsigned int _phase_e_ready_0 = 0;
            #pragma unroll 1
            for (int _j = 0; _j < ngroups_c + 1; _j++) {
                mbarrier_wait(tk_full_addr + (ck_stage) * 8, _phase_tk_full);
                unsigned int tku[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tku[0])) : "r"(tk_s_addr + ck_stage * 4) : "memory");
                int u = (int)tku[0];
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(tk_empty_addr + (ck_stage) * 8);
                }
                ck_stage += 1;
                if (ck_stage == 2) { ck_stage = 0; _phase_tk_full ^= 1; }
                if (u >= ngroups_c) {
                    break;
                }
                unsigned int g = (unsigned int)u;
                int chunk = g / (unsigned int)num_qk_heads;
                int hq = g % (unsigned int)num_qk_heads;
                int row0 = chunk * 64;
                int trow = chunk_bos[chunk];
                int clen = chunk_len[chunk];
                #pragma unroll 1
                for (int hl = 0; hl < r_c; hl++) {
                    int hv = hq * r_c + hl;
                    long long hv64 = (long long)hv;
                    long long nh64 = (long long)num_heads;
                    if (tid_1 < 64) {
                        int _max_0 = ((clen) > (1) ? (clen) : (1));
                        int last = _max_0 - 1;
                        float bv = 0.0f;
                        float gv = 0.0f;
                        float egv = 0.0f;
                        float rsv = 0.0f;
                        if (tid_1 < clen) {
                            bv = beta[(long long)(trow + tid_1) * nh64 + hv64];
                            gv = g_cum[(long long)(row0 + tid_1) * nh64 + hv64];
                            float gl = g_cum[(long long)(row0 + last) * nh64 + hv64];
                            float _exp_1 = expf(gv);
                            egv = _exp_1;
                            float _exp_2 = expf(gl - gv);
                            rsv = _exp_2;
                        }
                        beta_s[tid_1] = bv;
                        gc_s[tid_1] = gv;
                        eg_s[tid_1] = egv;
                        rs_s[tid_1] = rsv;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    mbarrier_wait(h_full_addr, _phase_h_full_0);
                    _phase_h_full_0 ^= 1;
                    mbarrier_wait(dh_full_addr, _phase_dh_full_0);
                    _phase_dh_full_0 ^= 1;
                    float hd = 0.0f;
                    int krow = tid_1 >> 1 ^ (tid_1 & 1) << 2;
                    int vh0 = (tid_1 & 1) * 64;
                    #pragma unroll
                    for (int seg = 0; seg < 4; seg++) {
                        unsigned int hraw[8];
                        unsigned int dhraw[8];
                        #pragma unroll
                        for (int q = 0; q < 2; q++) {
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&hraw[q * 4])), "=r"(*reinterpret_cast<uint32_t*>(&hraw[(q * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&hraw[(q * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&hraw[(q * 4) + 3]))
                                : "r"((h_k_addr + (unsigned int)((vh0 + seg * 16 + q * 8) / 64 * 16384 + krow * 128 + (vh0 + seg * 16 + q * 8) % 64 * 2 ^ ((vh0 + seg * 16 + q * 8) / 64 * 16384 + krow * 128 + (vh0 + seg * 16 + q * 8) % 64 * 2 >> 7 & 7) << 4))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&dhraw[q * 4])), "=r"(*reinterpret_cast<uint32_t*>(&dhraw[(q * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dhraw[(q * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dhraw[(q * 4) + 3]))
                                : "r"((dh_k_addr + (unsigned int)((vh0 + seg * 16 + q * 8) / 64 * 16384 + krow * 128 + (vh0 + seg * 16 + q * 8) % 64 * 2 ^ ((vh0 + seg * 16 + q * 8) / 64 * 16384 + krow * 128 + (vh0 + seg * 16 + q * 8) % 64 * 2 >> 7 & 7) << 4))));
                        }
                        float _unpack_bf16_0[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hraw[_pair]) << 16);
                            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hraw[_pair]) & 0xffff0000u);
                        }
                        float _unpack_bf16_1[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dhraw[_pair]) << 16);
                            (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dhraw[_pair]) & 0xffff0000u);
                        }
                        float _hd_prod[16];
                        _hd_prod[0] = _unpack_bf16_0[0] * _unpack_bf16_1[0];
                        _hd_prod[1] = _unpack_bf16_0[1] * _unpack_bf16_1[1];
                        _hd_prod[2] = _unpack_bf16_0[2] * _unpack_bf16_1[2];
                        _hd_prod[3] = _unpack_bf16_0[3] * _unpack_bf16_1[3];
                        _hd_prod[4] = _unpack_bf16_0[4] * _unpack_bf16_1[4];
                        _hd_prod[5] = _unpack_bf16_0[5] * _unpack_bf16_1[5];
                        _hd_prod[6] = _unpack_bf16_0[6] * _unpack_bf16_1[6];
                        _hd_prod[7] = _unpack_bf16_0[7] * _unpack_bf16_1[7];
                        _hd_prod[8] = _unpack_bf16_0[8] * _unpack_bf16_1[8];
                        _hd_prod[9] = _unpack_bf16_0[9] * _unpack_bf16_1[9];
                        _hd_prod[10] = _unpack_bf16_0[10] * _unpack_bf16_1[10];
                        _hd_prod[11] = _unpack_bf16_0[11] * _unpack_bf16_1[11];
                        _hd_prod[12] = _unpack_bf16_0[12] * _unpack_bf16_1[12];
                        _hd_prod[13] = _unpack_bf16_0[13] * _unpack_bf16_1[13];
                        _hd_prod[14] = _unpack_bf16_0[14] * _unpack_bf16_1[14];
                        _hd_prod[15] = _unpack_bf16_0[15] * _unpack_bf16_1[15];
                        unsigned int _hd_prod_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_hd_prod[_lp*2 + 0], _hd_prod[_lp*2+1 + 0]));
                            _hd_prod_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float _hd_prod_r[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_hd_prod_r[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_hd_prod_bf16[_pair]) << 16);
                            (&_hd_prod_r[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_hd_prod_bf16[_pair]) & 0xffff0000u);
                        }
                        hd = hd + _hd_prod_r[0];
                        hd = hd + _hd_prod_r[1];
                        hd = hd + _hd_prod_r[2];
                        hd = hd + _hd_prod_r[3];
                        hd = hd + _hd_prod_r[4];
                        hd = hd + _hd_prod_r[5];
                        hd = hd + _hd_prod_r[6];
                        hd = hd + _hd_prod_r[7];
                        hd = hd + _hd_prod_r[8];
                        hd = hd + _hd_prod_r[9];
                        hd = hd + _hd_prod_r[10];
                        hd = hd + _hd_prod_r[11];
                        hd = hd + _hd_prod_r[12];
                        hd = hd + _hd_prod_r[13];
                        hd = hd + _hd_prod_r[14];
                        hd = hd + _hd_prod_r[15];
                    }
                    float hd_w = 0.0f;
                    float s = hd;
                    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, s, 16);
                    s = s + _shfl_xor_sync_3;
                    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, s, 8);
                    s = s + _shfl_xor_sync_4;
                    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, s, 4);
                    s = s + _shfl_xor_sync_5;
                    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, s, 2);
                    s = s + _shfl_xor_sync_6;
                    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, s, 1);
                    s = s + _shfl_xor_sync_7;
                    hd_w = s;
                    if (lane_0 == 0) {
                        red_hd[warp] = hd_w;
                    }
                    mbarrier_wait(qk_full_addr, _phase_qk_full_0);
                    _phase_qk_full_0 ^= 1;
                    if (hl == 0) {
                        unsigned int w0[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w0[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + t0 * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + t0 * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[0] = w0[0];
                        unsigned int w1[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w1[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 1) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 1) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[0] = kpk[0] | w1[0] << 16;
                        unsigned int w2[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w2[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 2) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 2) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[1] = w2[0];
                        unsigned int w3[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w3[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 3) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 3) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[1] = kpk[1] | w3[0] << 16;
                        unsigned int w4[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 4) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 4) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[2] = w4[0];
                        unsigned int w5[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w5[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 5) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 5) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[2] = kpk[2] | w5[0] << 16;
                        unsigned int w6[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w6[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 6) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 6) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[3] = w6[0];
                        unsigned int w7[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w7[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 7) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 7) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[3] = kpk[3] | w7[0] << 16;
                        unsigned int w8[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w8[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 8) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 8) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[4] = w8[0];
                        unsigned int w9[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w9[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 9) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 9) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[4] = kpk[4] | w9[0] << 16;
                        unsigned int w_r_r_r_0[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_r_r_0[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 10) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 10) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[5] = w_r_r_r_0[0];
                        unsigned int w_r_r_r_r_0[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_r_r_r_0[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 11) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 11) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[5] = kpk[5] | w_r_r_r_r_0[0] << 16;
                        unsigned int w_r_r_r_1[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_r_r_1[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 12) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 12) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[6] = w_r_r_r_1[0];
                        unsigned int w_r_r_r_r_1[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_r_r_r_1[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 13) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 13) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[6] = kpk[6] | w_r_r_r_r_1[0] << 16;
                        unsigned int w_r_r_r_2[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_r_r_2[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 14) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 14) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[7] = w_r_r_r_2[0];
                        unsigned int w_r_r_r_r_2[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_r_r_r_2[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 15) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 15) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[7] = kpk[7] | w_r_r_r_r_2[0] << 16;
                        unsigned int w16[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w16[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 16) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 16) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[8] = w16[0];
                        unsigned int w_r_0[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_0[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 17) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 17) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[8] = kpk[8] | w_r_0[0] << 16;
                        unsigned int w_r_r_0[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_r_0[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 18) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 18) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[9] = w_r_r_0[0];
                        unsigned int w_r_1[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_1[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 19) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 19) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[9] = kpk[9] | w_r_1[0] << 16;
                        unsigned int w_r_r_1[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_r_1[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 20) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 20) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[10] = w_r_r_1[0];
                        unsigned int w_r_2[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_2[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 21) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 21) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[10] = kpk[10] | w_r_2[0] << 16;
                        unsigned int w_r_r_2[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_r_2[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 22) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 22) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[11] = w_r_r_2[0];
                        unsigned int w_r_3[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_3[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 23) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 23) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[11] = kpk[11] | w_r_3[0] << 16;
                        unsigned int w_r_r_3[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_r_3[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 24) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 24) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[12] = w_r_r_3[0];
                        unsigned int w_r_4[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_4[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 25) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 25) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[12] = kpk[12] | w_r_4[0] << 16;
                        unsigned int w_r_r_4[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_r_4[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 26) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 26) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[13] = w_r_r_4[0];
                        unsigned int w_r_5[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_5[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 27) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 27) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[13] = kpk[13] | w_r_5[0] << 16;
                        unsigned int w_r_r_5[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_r_5[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 28) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 28) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[14] = w_r_r_5[0];
                        unsigned int w_r_6[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_6[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 29) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 29) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[14] = kpk[14] | w_r_6[0] << 16;
                        unsigned int w_r_r_6[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_r_r_6[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 30) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 30) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[15] = w_r_r_6[0];
                        unsigned int w31[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w31[0])) : "r"((smem_v6_addr_0 + (unsigned int)(d_row / 64 * 8192 + (t0 + 31) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 31) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                        kpk[15] = kpk[15] | w31[0] << 16;
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    mbarrier_wait(a_ready_addr, _phase_a_ready_0);
                    _phase_a_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float dg_acc = 0.0f;
                    float db_acc = 0.0f;
                    float hk_acc = 0.0f;
                    float tmp = 0.0f;
                    float _tmem_load_0[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                        : "r"(taddr + 384 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(daq_drained_addr);
                    float gt = gc_s[trow16];
                    float aqg[16];
                    float _exp_8 = expf(gt - gc_s[colh]);
                    aqg[0] = ((trow16 < clen) ? ((trow16 >= colh) ? _tmem_load_0[0] * scale * _exp_8 : 0.0f) : 0.0f);
                    float _exp_9 = expf(gt - gc_s[colh + 1]);
                    aqg[1] = ((trow16 < clen) ? ((trow16 >= colh + 1) ? _tmem_load_0[1] * scale * _exp_9 : 0.0f) : 0.0f);
                    float _exp_10 = expf(gt - gc_s[colh + 2]);
                    aqg[2] = ((trow16 < clen) ? ((trow16 >= colh + 2) ? _tmem_load_0[2] * scale * _exp_10 : 0.0f) : 0.0f);
                    float _exp_11 = expf(gt - gc_s[colh + 3]);
                    aqg[3] = ((trow16 < clen) ? ((trow16 >= colh + 3) ? _tmem_load_0[3] * scale * _exp_11 : 0.0f) : 0.0f);
                    float _exp_12 = expf(gt - gc_s[colh + 4]);
                    aqg[4] = ((trow16 < clen) ? ((trow16 >= colh + 4) ? _tmem_load_0[4] * scale * _exp_12 : 0.0f) : 0.0f);
                    float _exp_13 = expf(gt - gc_s[colh + 5]);
                    aqg[5] = ((trow16 < clen) ? ((trow16 >= colh + 5) ? _tmem_load_0[5] * scale * _exp_13 : 0.0f) : 0.0f);
                    float _exp_14 = expf(gt - gc_s[colh + 6]);
                    aqg[6] = ((trow16 < clen) ? ((trow16 >= colh + 6) ? _tmem_load_0[6] * scale * _exp_14 : 0.0f) : 0.0f);
                    float _exp_15 = expf(gt - gc_s[colh + 7]);
                    aqg[7] = ((trow16 < clen) ? ((trow16 >= colh + 7) ? _tmem_load_0[7] * scale * _exp_15 : 0.0f) : 0.0f);
                    float _exp_16 = expf(gt - gc_s[colh + 8]);
                    aqg[8] = ((trow16 < clen) ? ((trow16 >= colh + 8) ? _tmem_load_0[8] * scale * _exp_16 : 0.0f) : 0.0f);
                    float _exp_17 = expf(gt - gc_s[colh + 9]);
                    aqg[9] = ((trow16 < clen) ? ((trow16 >= colh + 9) ? _tmem_load_0[9] * scale * _exp_17 : 0.0f) : 0.0f);
                    float _exp_18 = expf(gt - gc_s[colh + 10]);
                    aqg[10] = ((trow16 < clen) ? ((trow16 >= colh + 10) ? _tmem_load_0[10] * scale * _exp_18 : 0.0f) : 0.0f);
                    float _exp_19 = expf(gt - gc_s[colh + 11]);
                    aqg[11] = ((trow16 < clen) ? ((trow16 >= colh + 11) ? _tmem_load_0[11] * scale * _exp_19 : 0.0f) : 0.0f);
                    float _exp_20 = expf(gt - gc_s[colh + 12]);
                    aqg[12] = ((trow16 < clen) ? ((trow16 >= colh + 12) ? _tmem_load_0[12] * scale * _exp_20 : 0.0f) : 0.0f);
                    float _exp_21 = expf(gt - gc_s[colh + 13]);
                    aqg[13] = ((trow16 < clen) ? ((trow16 >= colh + 13) ? _tmem_load_0[13] * scale * _exp_21 : 0.0f) : 0.0f);
                    float _exp_22 = expf(gt - gc_s[colh + 14]);
                    aqg[14] = ((trow16 < clen) ? ((trow16 >= colh + 14) ? _tmem_load_0[14] * scale * _exp_22 : 0.0f) : 0.0f);
                    float _exp_23 = expf(gt - gc_s[colh + 15]);
                    aqg[15] = ((trow16 < clen) ? ((trow16 >= colh + 15) ? _tmem_load_0[15] * scale * _exp_23 : 0.0f) : 0.0f);
                    unsigned int aqg_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aqg[_lp*2 + 0], aqg[_lp*2+1 + 0]));
                        aqg_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v20_addr_2 + (unsigned int)(colh / 64 * 8192 + trow16 * 128 + colh % 64 * 2 ^ (colh / 64 * 8192 + trow16 * 128 + colh % 64 * 2 >> 7 & 7) << 4))), "r"(aqg_bf16[0]), "r"(aqg_bf16[1]), "r"(aqg_bf16[2]), "r"(aqg_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v20_addr_2 + (unsigned int)((colh + 8) / 64 * 8192 + trow16 * 128 + (colh + 8) % 64 * 2 ^ ((colh + 8) / 64 * 8192 + trow16 * 128 + (colh + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(aqg_bf16[4]), "r"(aqg_bf16[5]), "r"(aqg_bf16[6]), "r"(aqg_bf16[7]) : "memory");
                    float _tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    _tmem_load_1[0] = _tmem_load_1[0] * eg_s[t0] * scale;
                    _tmem_load_1[1] = _tmem_load_1[1] * eg_s[t0 + 1] * scale;
                    _tmem_load_1[2] = _tmem_load_1[2] * eg_s[t0 + 2] * scale;
                    _tmem_load_1[3] = _tmem_load_1[3] * eg_s[t0 + 3] * scale;
                    _tmem_load_1[4] = _tmem_load_1[4] * eg_s[t0 + 4] * scale;
                    _tmem_load_1[5] = _tmem_load_1[5] * eg_s[t0 + 5] * scale;
                    _tmem_load_1[6] = _tmem_load_1[6] * eg_s[t0 + 6] * scale;
                    _tmem_load_1[7] = _tmem_load_1[7] * eg_s[t0 + 7] * scale;
                    _tmem_load_1[8] = _tmem_load_1[8] * eg_s[t0 + 8] * scale;
                    _tmem_load_1[9] = _tmem_load_1[9] * eg_s[t0 + 9] * scale;
                    _tmem_load_1[10] = _tmem_load_1[10] * eg_s[t0 + 10] * scale;
                    _tmem_load_1[11] = _tmem_load_1[11] * eg_s[t0 + 11] * scale;
                    _tmem_load_1[12] = _tmem_load_1[12] * eg_s[t0 + 12] * scale;
                    _tmem_load_1[13] = _tmem_load_1[13] * eg_s[t0 + 13] * scale;
                    _tmem_load_1[14] = _tmem_load_1[14] * eg_s[t0 + 14] * scale;
                    _tmem_load_1[15] = _tmem_load_1[15] * eg_s[t0 + 15] * scale;
                    _tmem_load_1[16] = _tmem_load_1[16] * eg_s[t0 + 16] * scale;
                    _tmem_load_1[17] = _tmem_load_1[17] * eg_s[t0 + 17] * scale;
                    _tmem_load_1[18] = _tmem_load_1[18] * eg_s[t0 + 18] * scale;
                    _tmem_load_1[19] = _tmem_load_1[19] * eg_s[t0 + 19] * scale;
                    _tmem_load_1[20] = _tmem_load_1[20] * eg_s[t0 + 20] * scale;
                    _tmem_load_1[21] = _tmem_load_1[21] * eg_s[t0 + 21] * scale;
                    _tmem_load_1[22] = _tmem_load_1[22] * eg_s[t0 + 22] * scale;
                    _tmem_load_1[23] = _tmem_load_1[23] * eg_s[t0 + 23] * scale;
                    _tmem_load_1[24] = _tmem_load_1[24] * eg_s[t0 + 24] * scale;
                    _tmem_load_1[25] = _tmem_load_1[25] * eg_s[t0 + 25] * scale;
                    _tmem_load_1[26] = _tmem_load_1[26] * eg_s[t0 + 26] * scale;
                    _tmem_load_1[27] = _tmem_load_1[27] * eg_s[t0 + 27] * scale;
                    _tmem_load_1[28] = _tmem_load_1[28] * eg_s[t0 + 28] * scale;
                    _tmem_load_1[29] = _tmem_load_1[29] * eg_s[t0 + 29] * scale;
                    _tmem_load_1[30] = _tmem_load_1[30] * eg_s[t0 + 30] * scale;
                    _tmem_load_1[31] = _tmem_load_1[31] * eg_s[t0 + 31] * scale;
                    tmem_st_x32_f32(taddr + 128 + (unsigned int)row_base + (unsigned int)(ch * 32), _tmem_load_1);
                    float _tmem_load_2[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                        : "r"(taddr + 192 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float hk[32];
                    _tmem_load_2[0] = _tmem_load_2[0] * rs_s[t0];
                    hk[0] = __uint_as_float(kpk[0] << 16) * _tmem_load_2[0];
                    _tmem_load_2[1] = _tmem_load_2[1] * rs_s[t0 + 1];
                    hk[1] = __uint_as_float(kpk[0] & 4294901760u) * _tmem_load_2[1];
                    _tmem_load_2[2] = _tmem_load_2[2] * rs_s[t0 + 2];
                    hk[2] = __uint_as_float(kpk[1] << 16) * _tmem_load_2[2];
                    _tmem_load_2[3] = _tmem_load_2[3] * rs_s[t0 + 3];
                    hk[3] = __uint_as_float(kpk[1] & 4294901760u) * _tmem_load_2[3];
                    _tmem_load_2[4] = _tmem_load_2[4] * rs_s[t0 + 4];
                    hk[4] = __uint_as_float(kpk[2] << 16) * _tmem_load_2[4];
                    _tmem_load_2[5] = _tmem_load_2[5] * rs_s[t0 + 5];
                    hk[5] = __uint_as_float(kpk[2] & 4294901760u) * _tmem_load_2[5];
                    _tmem_load_2[6] = _tmem_load_2[6] * rs_s[t0 + 6];
                    hk[6] = __uint_as_float(kpk[3] << 16) * _tmem_load_2[6];
                    _tmem_load_2[7] = _tmem_load_2[7] * rs_s[t0 + 7];
                    hk[7] = __uint_as_float(kpk[3] & 4294901760u) * _tmem_load_2[7];
                    _tmem_load_2[8] = _tmem_load_2[8] * rs_s[t0 + 8];
                    hk[8] = __uint_as_float(kpk[4] << 16) * _tmem_load_2[8];
                    _tmem_load_2[9] = _tmem_load_2[9] * rs_s[t0 + 9];
                    hk[9] = __uint_as_float(kpk[4] & 4294901760u) * _tmem_load_2[9];
                    _tmem_load_2[10] = _tmem_load_2[10] * rs_s[t0 + 10];
                    hk[10] = __uint_as_float(kpk[5] << 16) * _tmem_load_2[10];
                    _tmem_load_2[11] = _tmem_load_2[11] * rs_s[t0 + 11];
                    hk[11] = __uint_as_float(kpk[5] & 4294901760u) * _tmem_load_2[11];
                    _tmem_load_2[12] = _tmem_load_2[12] * rs_s[t0 + 12];
                    hk[12] = __uint_as_float(kpk[6] << 16) * _tmem_load_2[12];
                    _tmem_load_2[13] = _tmem_load_2[13] * rs_s[t0 + 13];
                    hk[13] = __uint_as_float(kpk[6] & 4294901760u) * _tmem_load_2[13];
                    _tmem_load_2[14] = _tmem_load_2[14] * rs_s[t0 + 14];
                    hk[14] = __uint_as_float(kpk[7] << 16) * _tmem_load_2[14];
                    _tmem_load_2[15] = _tmem_load_2[15] * rs_s[t0 + 15];
                    hk[15] = __uint_as_float(kpk[7] & 4294901760u) * _tmem_load_2[15];
                    _tmem_load_2[16] = _tmem_load_2[16] * rs_s[t0 + 16];
                    hk[16] = __uint_as_float(kpk[8] << 16) * _tmem_load_2[16];
                    _tmem_load_2[17] = _tmem_load_2[17] * rs_s[t0 + 17];
                    hk[17] = __uint_as_float(kpk[8] & 4294901760u) * _tmem_load_2[17];
                    _tmem_load_2[18] = _tmem_load_2[18] * rs_s[t0 + 18];
                    hk[18] = __uint_as_float(kpk[9] << 16) * _tmem_load_2[18];
                    _tmem_load_2[19] = _tmem_load_2[19] * rs_s[t0 + 19];
                    hk[19] = __uint_as_float(kpk[9] & 4294901760u) * _tmem_load_2[19];
                    _tmem_load_2[20] = _tmem_load_2[20] * rs_s[t0 + 20];
                    hk[20] = __uint_as_float(kpk[10] << 16) * _tmem_load_2[20];
                    _tmem_load_2[21] = _tmem_load_2[21] * rs_s[t0 + 21];
                    hk[21] = __uint_as_float(kpk[10] & 4294901760u) * _tmem_load_2[21];
                    _tmem_load_2[22] = _tmem_load_2[22] * rs_s[t0 + 22];
                    hk[22] = __uint_as_float(kpk[11] << 16) * _tmem_load_2[22];
                    _tmem_load_2[23] = _tmem_load_2[23] * rs_s[t0 + 23];
                    hk[23] = __uint_as_float(kpk[11] & 4294901760u) * _tmem_load_2[23];
                    _tmem_load_2[24] = _tmem_load_2[24] * rs_s[t0 + 24];
                    hk[24] = __uint_as_float(kpk[12] << 16) * _tmem_load_2[24];
                    _tmem_load_2[25] = _tmem_load_2[25] * rs_s[t0 + 25];
                    hk[25] = __uint_as_float(kpk[12] & 4294901760u) * _tmem_load_2[25];
                    _tmem_load_2[26] = _tmem_load_2[26] * rs_s[t0 + 26];
                    hk[26] = __uint_as_float(kpk[13] << 16) * _tmem_load_2[26];
                    _tmem_load_2[27] = _tmem_load_2[27] * rs_s[t0 + 27];
                    hk[27] = __uint_as_float(kpk[13] & 4294901760u) * _tmem_load_2[27];
                    _tmem_load_2[28] = _tmem_load_2[28] * rs_s[t0 + 28];
                    hk[28] = __uint_as_float(kpk[14] << 16) * _tmem_load_2[28];
                    _tmem_load_2[29] = _tmem_load_2[29] * rs_s[t0 + 29];
                    hk[29] = __uint_as_float(kpk[14] & 4294901760u) * _tmem_load_2[29];
                    _tmem_load_2[30] = _tmem_load_2[30] * rs_s[t0 + 30];
                    hk[30] = __uint_as_float(kpk[15] << 16) * _tmem_load_2[30];
                    _tmem_load_2[31] = _tmem_load_2[31] * rs_s[t0 + 31];
                    hk[31] = __uint_as_float(kpk[15] & 4294901760u) * _tmem_load_2[31];
                    tmem_st_x32_f32(taddr + 192 + (unsigned int)row_base + (unsigned int)(ch * 32), _tmem_load_2);
                    float hk_sum = 0.0f;
                    hk_sum = hk_sum + hk[0];
                    hk[0] = -hk[0];
                    hk_sum = hk_sum + hk[1];
                    hk[1] = -hk[1];
                    hk_sum = hk_sum + hk[2];
                    hk[2] = -hk[2];
                    hk_sum = hk_sum + hk[3];
                    hk[3] = -hk[3];
                    hk_sum = hk_sum + hk[4];
                    hk[4] = -hk[4];
                    hk_sum = hk_sum + hk[5];
                    hk[5] = -hk[5];
                    hk_sum = hk_sum + hk[6];
                    hk[6] = -hk[6];
                    hk_sum = hk_sum + hk[7];
                    hk[7] = -hk[7];
                    hk_sum = hk_sum + hk[8];
                    hk[8] = -hk[8];
                    hk_sum = hk_sum + hk[9];
                    hk[9] = -hk[9];
                    hk_sum = hk_sum + hk[10];
                    hk[10] = -hk[10];
                    hk_sum = hk_sum + hk[11];
                    hk[11] = -hk[11];
                    hk_sum = hk_sum + hk[12];
                    hk[12] = -hk[12];
                    hk_sum = hk_sum + hk[13];
                    hk[13] = -hk[13];
                    hk_sum = hk_sum + hk[14];
                    hk[14] = -hk[14];
                    hk_sum = hk_sum + hk[15];
                    hk[15] = -hk[15];
                    hk_sum = hk_sum + hk[16];
                    hk[16] = -hk[16];
                    hk_sum = hk_sum + hk[17];
                    hk[17] = -hk[17];
                    hk_sum = hk_sum + hk[18];
                    hk[18] = -hk[18];
                    hk_sum = hk_sum + hk[19];
                    hk[19] = -hk[19];
                    hk_sum = hk_sum + hk[20];
                    hk[20] = -hk[20];
                    hk_sum = hk_sum + hk[21];
                    hk[21] = -hk[21];
                    hk_sum = hk_sum + hk[22];
                    hk[22] = -hk[22];
                    hk_sum = hk_sum + hk[23];
                    hk[23] = -hk[23];
                    hk_sum = hk_sum + hk[24];
                    hk[24] = -hk[24];
                    hk_sum = hk_sum + hk[25];
                    hk[25] = -hk[25];
                    hk_sum = hk_sum + hk[26];
                    hk[26] = -hk[26];
                    hk_sum = hk_sum + hk[27];
                    hk[27] = -hk[27];
                    hk_sum = hk_sum + hk[28];
                    hk[28] = -hk[28];
                    hk_sum = hk_sum + hk[29];
                    hk[29] = -hk[29];
                    hk_sum = hk_sum + hk[30];
                    hk[30] = -hk[30];
                    hk_sum = hk_sum + hk[31];
                    hk[31] = -hk[31];
                    tmem_st_x32_f32(taddr + 448 + (unsigned int)row_base + (unsigned int)(ch * 32), hk);
                    float s_0 = hk_sum;
                    float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, s_0, 16);
                    s_0 = s_0 + _shfl_xor_sync_24;
                    float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, s_0, 8);
                    s_0 = s_0 + _shfl_xor_sync_25;
                    float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, s_0, 4);
                    s_0 = s_0 + _shfl_xor_sync_26;
                    float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, s_0, 2);
                    s_0 = s_0 + _shfl_xor_sync_27;
                    float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, s_0, 1);
                    s_0 = s_0 + _shfl_xor_sync_28;
                    hk_acc = s_0;
                    float _tmem_load_3[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                        : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    _tmem_load_3[0] = ((clen > t0) ? -_tmem_load_3[0] : 0.0f);
                    _tmem_load_3[1] = ((clen > t0 + 1) ? -_tmem_load_3[1] : 0.0f);
                    _tmem_load_3[2] = ((clen > t0 + 2) ? -_tmem_load_3[2] : 0.0f);
                    _tmem_load_3[3] = ((clen > t0 + 3) ? -_tmem_load_3[3] : 0.0f);
                    _tmem_load_3[4] = ((clen > t0 + 4) ? -_tmem_load_3[4] : 0.0f);
                    _tmem_load_3[5] = ((clen > t0 + 5) ? -_tmem_load_3[5] : 0.0f);
                    _tmem_load_3[6] = ((clen > t0 + 6) ? -_tmem_load_3[6] : 0.0f);
                    _tmem_load_3[7] = ((clen > t0 + 7) ? -_tmem_load_3[7] : 0.0f);
                    _tmem_load_3[8] = ((clen > t0 + 8) ? -_tmem_load_3[8] : 0.0f);
                    _tmem_load_3[9] = ((clen > t0 + 9) ? -_tmem_load_3[9] : 0.0f);
                    _tmem_load_3[10] = ((clen > t0 + 10) ? -_tmem_load_3[10] : 0.0f);
                    _tmem_load_3[11] = ((clen > t0 + 11) ? -_tmem_load_3[11] : 0.0f);
                    _tmem_load_3[12] = ((clen > t0 + 12) ? -_tmem_load_3[12] : 0.0f);
                    _tmem_load_3[13] = ((clen > t0 + 13) ? -_tmem_load_3[13] : 0.0f);
                    _tmem_load_3[14] = ((clen > t0 + 14) ? -_tmem_load_3[14] : 0.0f);
                    _tmem_load_3[15] = ((clen > t0 + 15) ? -_tmem_load_3[15] : 0.0f);
                    _tmem_load_3[16] = ((clen > t0 + 16) ? -_tmem_load_3[16] : 0.0f);
                    _tmem_load_3[17] = ((clen > t0 + 17) ? -_tmem_load_3[17] : 0.0f);
                    _tmem_load_3[18] = ((clen > t0 + 18) ? -_tmem_load_3[18] : 0.0f);
                    _tmem_load_3[19] = ((clen > t0 + 19) ? -_tmem_load_3[19] : 0.0f);
                    _tmem_load_3[20] = ((clen > t0 + 20) ? -_tmem_load_3[20] : 0.0f);
                    _tmem_load_3[21] = ((clen > t0 + 21) ? -_tmem_load_3[21] : 0.0f);
                    _tmem_load_3[22] = ((clen > t0 + 22) ? -_tmem_load_3[22] : 0.0f);
                    _tmem_load_3[23] = ((clen > t0 + 23) ? -_tmem_load_3[23] : 0.0f);
                    _tmem_load_3[24] = ((clen > t0 + 24) ? -_tmem_load_3[24] : 0.0f);
                    _tmem_load_3[25] = ((clen > t0 + 25) ? -_tmem_load_3[25] : 0.0f);
                    _tmem_load_3[26] = ((clen > t0 + 26) ? -_tmem_load_3[26] : 0.0f);
                    _tmem_load_3[27] = ((clen > t0 + 27) ? -_tmem_load_3[27] : 0.0f);
                    _tmem_load_3[28] = ((clen > t0 + 28) ? -_tmem_load_3[28] : 0.0f);
                    _tmem_load_3[29] = ((clen > t0 + 29) ? -_tmem_load_3[29] : 0.0f);
                    _tmem_load_3[30] = ((clen > t0 + 30) ? -_tmem_load_3[30] : 0.0f);
                    _tmem_load_3[31] = ((clen > t0 + 31) ? -_tmem_load_3[31] : 0.0f);
                    unsigned int _tmem_load_3_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                        _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dwbt_k_addr + (unsigned int)(t0 / 64 * 16384 + d_row * 128 + t0 % 64 * 2 ^ (t0 / 64 * 16384 + d_row * 128 + t0 % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[0]), "r"(_tmem_load_3_bf16[1]), "r"(_tmem_load_3_bf16[2]), "r"(_tmem_load_3_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dwbt_k_addr + (unsigned int)((t0 + 8) / 64 * 16384 + d_row * 128 + (t0 + 8) % 64 * 2 ^ ((t0 + 8) / 64 * 16384 + d_row * 128 + (t0 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[4]), "r"(_tmem_load_3_bf16[5]), "r"(_tmem_load_3_bf16[6]), "r"(_tmem_load_3_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dwbt_k_addr + (unsigned int)((t0 + 16) / 64 * 16384 + d_row * 128 + (t0 + 16) % 64 * 2 ^ ((t0 + 16) / 64 * 16384 + d_row * 128 + (t0 + 16) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[8]), "r"(_tmem_load_3_bf16[9]), "r"(_tmem_load_3_bf16[10]), "r"(_tmem_load_3_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dwbt_k_addr + (unsigned int)((t0 + 24) / 64 * 16384 + d_row * 128 + (t0 + 24) % 64 * 2 ^ ((t0 + 24) / 64 * 16384 + d_row * 128 + (t0 + 24) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[12]), "r"(_tmem_load_3_bf16[13]), "r"(_tmem_load_3_bf16[14]), "r"(_tmem_load_3_bf16[15]) : "memory");
                    float egt = eg_s[te];
                    float bte = beta_s[te];
                    unsigned int kraw0_1_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_0[(0) + 3]))
                        : "r"((smem_v6_addr_0 + (unsigned int)(de0 / 64 * 8192 + te * 128 + de0 % 64 * 2 ^ (de0 / 64 * 8192 + te * 128 + de0 % 64 * 2 >> 7 & 7) << 4))));
                    float _unpack_bf16_2[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kraw0_1_0[_pair]) << 16);
                        (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kraw0_1_0[_pair]) & 0xffff0000u);
                    }
                    float kg8_0[8];
                    float kb8_0[8];
                    kg8_0[0] = egt * _unpack_bf16_2[0];
                    kb8_0[0] = bte * _unpack_bf16_2[0];
                    kg8_0[1] = egt * _unpack_bf16_2[1];
                    kb8_0[1] = bte * _unpack_bf16_2[1];
                    kg8_0[2] = egt * _unpack_bf16_2[2];
                    kb8_0[2] = bte * _unpack_bf16_2[2];
                    kg8_0[3] = egt * _unpack_bf16_2[3];
                    kb8_0[3] = bte * _unpack_bf16_2[3];
                    kg8_0[4] = egt * _unpack_bf16_2[4];
                    kb8_0[4] = bte * _unpack_bf16_2[4];
                    kg8_0[5] = egt * _unpack_bf16_2[5];
                    kb8_0[5] = bte * _unpack_bf16_2[5];
                    kg8_0[6] = egt * _unpack_bf16_2[6];
                    kb8_0[6] = bte * _unpack_bf16_2[6];
                    kg8_0[7] = egt * _unpack_bf16_2[7];
                    kb8_0[7] = bte * _unpack_bf16_2[7];
                    unsigned int kg8_0_bf16_1_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg8_0[_lp*2 + 0], kg8_0[_lp*2+1 + 0]));
                        kg8_0_bf16_1_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int kb8_0_bf16_1_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb8_0[_lp*2 + 0], kb8_0[_lp*2+1 + 0]));
                        kb8_0_bf16_1_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v6_addr_5 + (unsigned int)(de0 / 64 * 8192 + te * 128 + de0 % 64 * 2 ^ (de0 / 64 * 8192 + te * 128 + de0 % 64 * 2 >> 7 & 7) << 4))), "r"(kg8_0_bf16_1_0[0]), "r"(kg8_0_bf16_1_0[1]), "r"(kg8_0_bf16_1_0[2]), "r"(kg8_0_bf16_1_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v6_addr_6 + (unsigned int)(de0 / 64 * 8192 + te * 128 + de0 % 64 * 2 ^ (de0 / 64 * 8192 + te * 128 + de0 % 64 * 2 >> 7 & 7) << 4))), "r"(kb8_0_bf16_1_0[0]), "r"(kb8_0_bf16_1_0[1]), "r"(kb8_0_bf16_1_0[2]), "r"(kb8_0_bf16_1_0[3]) : "memory");
                    unsigned int kraw0_1_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_1[(0) + 3]))
                        : "r"((smem_v6_addr_0 + (unsigned int)((de0 + 8) / 64 * 8192 + te * 128 + (de0 + 8) % 64 * 2 ^ ((de0 + 8) / 64 * 8192 + te * 128 + (de0 + 8) % 64 * 2 >> 7 & 7) << 4))));
                    float _unpack_bf16_3[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kraw0_1_1[_pair]) << 16);
                        (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kraw0_1_1[_pair]) & 0xffff0000u);
                    }
                    float kg8_1[8];
                    float kb8_1[8];
                    kg8_1[0] = egt * _unpack_bf16_3[0];
                    kb8_1[0] = bte * _unpack_bf16_3[0];
                    kg8_1[1] = egt * _unpack_bf16_3[1];
                    kb8_1[1] = bte * _unpack_bf16_3[1];
                    kg8_1[2] = egt * _unpack_bf16_3[2];
                    kb8_1[2] = bte * _unpack_bf16_3[2];
                    kg8_1[3] = egt * _unpack_bf16_3[3];
                    kb8_1[3] = bte * _unpack_bf16_3[3];
                    kg8_1[4] = egt * _unpack_bf16_3[4];
                    kb8_1[4] = bte * _unpack_bf16_3[4];
                    kg8_1[5] = egt * _unpack_bf16_3[5];
                    kb8_1[5] = bte * _unpack_bf16_3[5];
                    kg8_1[6] = egt * _unpack_bf16_3[6];
                    kb8_1[6] = bte * _unpack_bf16_3[6];
                    kg8_1[7] = egt * _unpack_bf16_3[7];
                    kb8_1[7] = bte * _unpack_bf16_3[7];
                    unsigned int kg8_0_bf16_1_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg8_1[_lp*2 + 0], kg8_1[_lp*2+1 + 0]));
                        kg8_0_bf16_1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int kb8_0_bf16_1_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb8_1[_lp*2 + 0], kb8_1[_lp*2+1 + 0]));
                        kb8_0_bf16_1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v6_addr_5 + (unsigned int)((de0 + 8) / 64 * 8192 + te * 128 + (de0 + 8) % 64 * 2 ^ ((de0 + 8) / 64 * 8192 + te * 128 + (de0 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(kg8_0_bf16_1_1[0]), "r"(kg8_0_bf16_1_1[1]), "r"(kg8_0_bf16_1_1[2]), "r"(kg8_0_bf16_1_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v6_addr_6 + (unsigned int)((de0 + 8) / 64 * 8192 + te * 128 + (de0 + 8) % 64 * 2 ^ ((de0 + 8) / 64 * 8192 + te * 128 + (de0 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(kb8_0_bf16_1_1[0]), "r"(kb8_0_bf16_1_1[1]), "r"(kb8_0_bf16_1_1[2]), "r"(kb8_0_bf16_1_1[3]) : "memory");
                    unsigned int kraw0_1_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_2[(0) + 3]))
                        : "r"((smem_v6_addr_0 + (unsigned int)((de0 + 16) / 64 * 8192 + te * 128 + (de0 + 16) % 64 * 2 ^ ((de0 + 16) / 64 * 8192 + te * 128 + (de0 + 16) % 64 * 2 >> 7 & 7) << 4))));
                    float _unpack_bf16_4[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kraw0_1_2[_pair]) << 16);
                        (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kraw0_1_2[_pair]) & 0xffff0000u);
                    }
                    float kg8_2[8];
                    float kb8_2[8];
                    kg8_2[0] = egt * _unpack_bf16_4[0];
                    kb8_2[0] = bte * _unpack_bf16_4[0];
                    kg8_2[1] = egt * _unpack_bf16_4[1];
                    kb8_2[1] = bte * _unpack_bf16_4[1];
                    kg8_2[2] = egt * _unpack_bf16_4[2];
                    kb8_2[2] = bte * _unpack_bf16_4[2];
                    kg8_2[3] = egt * _unpack_bf16_4[3];
                    kb8_2[3] = bte * _unpack_bf16_4[3];
                    kg8_2[4] = egt * _unpack_bf16_4[4];
                    kb8_2[4] = bte * _unpack_bf16_4[4];
                    kg8_2[5] = egt * _unpack_bf16_4[5];
                    kb8_2[5] = bte * _unpack_bf16_4[5];
                    kg8_2[6] = egt * _unpack_bf16_4[6];
                    kb8_2[6] = bte * _unpack_bf16_4[6];
                    kg8_2[7] = egt * _unpack_bf16_4[7];
                    kb8_2[7] = bte * _unpack_bf16_4[7];
                    unsigned int kg8_0_bf16_1_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg8_2[_lp*2 + 0], kg8_2[_lp*2+1 + 0]));
                        kg8_0_bf16_1_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int kb8_0_bf16_1_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb8_2[_lp*2 + 0], kb8_2[_lp*2+1 + 0]));
                        kb8_0_bf16_1_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v6_addr_5 + (unsigned int)((de0 + 16) / 64 * 8192 + te * 128 + (de0 + 16) % 64 * 2 ^ ((de0 + 16) / 64 * 8192 + te * 128 + (de0 + 16) % 64 * 2 >> 7 & 7) << 4))), "r"(kg8_0_bf16_1_2[0]), "r"(kg8_0_bf16_1_2[1]), "r"(kg8_0_bf16_1_2[2]), "r"(kg8_0_bf16_1_2[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v6_addr_6 + (unsigned int)((de0 + 16) / 64 * 8192 + te * 128 + (de0 + 16) % 64 * 2 ^ ((de0 + 16) / 64 * 8192 + te * 128 + (de0 + 16) % 64 * 2 >> 7 & 7) << 4))), "r"(kb8_0_bf16_1_2[0]), "r"(kb8_0_bf16_1_2[1]), "r"(kb8_0_bf16_1_2[2]), "r"(kb8_0_bf16_1_2[3]) : "memory");
                    unsigned int kraw0_1_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw0_1_3[(0) + 3]))
                        : "r"((smem_v6_addr_0 + (unsigned int)((de0 + 24) / 64 * 8192 + te * 128 + (de0 + 24) % 64 * 2 ^ ((de0 + 24) / 64 * 8192 + te * 128 + (de0 + 24) % 64 * 2 >> 7 & 7) << 4))));
                    float _unpack_bf16_5[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kraw0_1_3[_pair]) << 16);
                        (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kraw0_1_3[_pair]) & 0xffff0000u);
                    }
                    float kg8_3[8];
                    float kb8_3[8];
                    kg8_3[0] = egt * _unpack_bf16_5[0];
                    kb8_3[0] = bte * _unpack_bf16_5[0];
                    kg8_3[1] = egt * _unpack_bf16_5[1];
                    kb8_3[1] = bte * _unpack_bf16_5[1];
                    kg8_3[2] = egt * _unpack_bf16_5[2];
                    kb8_3[2] = bte * _unpack_bf16_5[2];
                    kg8_3[3] = egt * _unpack_bf16_5[3];
                    kb8_3[3] = bte * _unpack_bf16_5[3];
                    kg8_3[4] = egt * _unpack_bf16_5[4];
                    kb8_3[4] = bte * _unpack_bf16_5[4];
                    kg8_3[5] = egt * _unpack_bf16_5[5];
                    kb8_3[5] = bte * _unpack_bf16_5[5];
                    kg8_3[6] = egt * _unpack_bf16_5[6];
                    kb8_3[6] = bte * _unpack_bf16_5[6];
                    kg8_3[7] = egt * _unpack_bf16_5[7];
                    kb8_3[7] = bte * _unpack_bf16_5[7];
                    unsigned int kg8_0_bf16_1_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kg8_3[_lp*2 + 0], kg8_3[_lp*2+1 + 0]));
                        kg8_0_bf16_1_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int kb8_0_bf16_1_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kb8_3[_lp*2 + 0], kb8_3[_lp*2+1 + 0]));
                        kb8_0_bf16_1_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v6_addr_5 + (unsigned int)((de0 + 24) / 64 * 8192 + te * 128 + (de0 + 24) % 64 * 2 ^ ((de0 + 24) / 64 * 8192 + te * 128 + (de0 + 24) % 64 * 2 >> 7 & 7) << 4))), "r"(kg8_0_bf16_1_3[0]), "r"(kg8_0_bf16_1_3[1]), "r"(kg8_0_bf16_1_3[2]), "r"(kg8_0_bf16_1_3[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v6_addr_6 + (unsigned int)((de0 + 24) / 64 * 8192 + te * 128 + (de0 + 24) % 64 * 2 ^ ((de0 + 24) / 64 * 8192 + te * 128 + (de0 + 24) % 64 * 2 >> 7 & 7) << 4))), "r"(kb8_0_bf16_1_3[0]), "r"(kb8_0_bf16_1_3[1]), "r"(kb8_0_bf16_1_3[2]), "r"(kb8_0_bf16_1_3[3]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(a_drained_addr);
                    mbarrier_wait(b1_ready_addr, _phase_b1_ready_0);
                    _phase_b1_ready_0 ^= 1;
                    mbarrier_wait(v_full_addr, _phase_v_full_0);
                    _phase_v_full_0 ^= 1;
                    mbarrier_wait(dvn_full_addr, _phase_dvn_full_0);
                    _phase_dvn_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_4[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                        : "r"(taddr + 384 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float dvo[32];
                    float pv[32];
                    float vval = 0.0f;
                    dvo[0] = beta_s[t0] * _tmem_load_4[0];
                    unsigned int w[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + t0 * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + t0 * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w[0] << 16);
                    pv[0] = _tmem_load_4[0] * vval;
                    dvo[1] = beta_s[t0 + 1] * _tmem_load_4[1];
                    unsigned int w_0[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_0[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 1) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 1) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_0[0] << 16);
                    pv[1] = _tmem_load_4[1] * vval;
                    dvo[2] = beta_s[t0 + 2] * _tmem_load_4[2];
                    unsigned int w_1[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_1[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 2) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 2) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_1[0] << 16);
                    pv[2] = _tmem_load_4[2] * vval;
                    dvo[3] = beta_s[t0 + 3] * _tmem_load_4[3];
                    unsigned int w_2[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_2[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 3) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 3) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_2[0] << 16);
                    pv[3] = _tmem_load_4[3] * vval;
                    dvo[4] = beta_s[t0 + 4] * _tmem_load_4[4];
                    unsigned int w_3[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_3[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 4) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 4) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_3[0] << 16);
                    pv[4] = _tmem_load_4[4] * vval;
                    dvo[5] = beta_s[t0 + 5] * _tmem_load_4[5];
                    unsigned int w_4[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_4[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 5) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 5) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_4[0] << 16);
                    pv[5] = _tmem_load_4[5] * vval;
                    dvo[6] = beta_s[t0 + 6] * _tmem_load_4[6];
                    unsigned int w_5[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_5[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 6) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 6) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_5[0] << 16);
                    pv[6] = _tmem_load_4[6] * vval;
                    dvo[7] = beta_s[t0 + 7] * _tmem_load_4[7];
                    unsigned int w_6[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_6[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 7) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 7) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_6[0] << 16);
                    pv[7] = _tmem_load_4[7] * vval;
                    dvo[8] = beta_s[t0 + 8] * _tmem_load_4[8];
                    unsigned int w_7[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_7[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 8) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 8) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_7[0] << 16);
                    pv[8] = _tmem_load_4[8] * vval;
                    dvo[9] = beta_s[t0 + 9] * _tmem_load_4[9];
                    unsigned int w_8[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_8[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 9) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 9) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_8[0] << 16);
                    pv[9] = _tmem_load_4[9] * vval;
                    dvo[10] = beta_s[t0 + 10] * _tmem_load_4[10];
                    unsigned int w_9[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_9[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 10) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 10) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_9[0] << 16);
                    pv[10] = _tmem_load_4[10] * vval;
                    dvo[11] = beta_s[t0 + 11] * _tmem_load_4[11];
                    unsigned int w_10[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_10[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 11) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 11) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_10[0] << 16);
                    pv[11] = _tmem_load_4[11] * vval;
                    dvo[12] = beta_s[t0 + 12] * _tmem_load_4[12];
                    unsigned int w_11[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_11[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 12) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 12) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_11[0] << 16);
                    pv[12] = _tmem_load_4[12] * vval;
                    dvo[13] = beta_s[t0 + 13] * _tmem_load_4[13];
                    unsigned int w_12[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_12[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 13) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 13) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_12[0] << 16);
                    pv[13] = _tmem_load_4[13] * vval;
                    dvo[14] = beta_s[t0 + 14] * _tmem_load_4[14];
                    unsigned int w_13[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_13[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 14) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 14) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_13[0] << 16);
                    pv[14] = _tmem_load_4[14] * vval;
                    dvo[15] = beta_s[t0 + 15] * _tmem_load_4[15];
                    unsigned int w_14[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_14[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 15) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 15) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_14[0] << 16);
                    pv[15] = _tmem_load_4[15] * vval;
                    dvo[16] = beta_s[t0 + 16] * _tmem_load_4[16];
                    unsigned int w_15[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_15[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 16) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 16) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_15[0] << 16);
                    pv[16] = _tmem_load_4[16] * vval;
                    dvo[17] = beta_s[t0 + 17] * _tmem_load_4[17];
                    unsigned int w_16[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_16[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 17) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 17) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_16[0] << 16);
                    pv[17] = _tmem_load_4[17] * vval;
                    dvo[18] = beta_s[t0 + 18] * _tmem_load_4[18];
                    unsigned int w_17[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_17[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 18) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 18) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_17[0] << 16);
                    pv[18] = _tmem_load_4[18] * vval;
                    dvo[19] = beta_s[t0 + 19] * _tmem_load_4[19];
                    unsigned int w_18[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_18[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 19) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 19) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_18[0] << 16);
                    pv[19] = _tmem_load_4[19] * vval;
                    dvo[20] = beta_s[t0 + 20] * _tmem_load_4[20];
                    unsigned int w_19[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_19[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 20) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 20) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_19[0] << 16);
                    pv[20] = _tmem_load_4[20] * vval;
                    dvo[21] = beta_s[t0 + 21] * _tmem_load_4[21];
                    unsigned int w_20[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_20[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 21) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 21) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_20[0] << 16);
                    pv[21] = _tmem_load_4[21] * vval;
                    dvo[22] = beta_s[t0 + 22] * _tmem_load_4[22];
                    unsigned int w_21[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_21[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 22) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 22) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_21[0] << 16);
                    pv[22] = _tmem_load_4[22] * vval;
                    dvo[23] = beta_s[t0 + 23] * _tmem_load_4[23];
                    unsigned int w_22[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_22[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 23) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 23) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_22[0] << 16);
                    pv[23] = _tmem_load_4[23] * vval;
                    dvo[24] = beta_s[t0 + 24] * _tmem_load_4[24];
                    unsigned int w_23[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_23[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 24) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 24) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_23[0] << 16);
                    pv[24] = _tmem_load_4[24] * vval;
                    dvo[25] = beta_s[t0 + 25] * _tmem_load_4[25];
                    unsigned int w_24[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_24[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 25) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 25) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_24[0] << 16);
                    pv[25] = _tmem_load_4[25] * vval;
                    dvo[26] = beta_s[t0 + 26] * _tmem_load_4[26];
                    unsigned int w_25[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_25[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 26) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 26) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_25[0] << 16);
                    pv[26] = _tmem_load_4[26] * vval;
                    dvo[27] = beta_s[t0 + 27] * _tmem_load_4[27];
                    unsigned int w_26[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_26[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 27) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 27) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_26[0] << 16);
                    pv[27] = _tmem_load_4[27] * vval;
                    dvo[28] = beta_s[t0 + 28] * _tmem_load_4[28];
                    unsigned int w_27[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_27[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 28) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 28) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_27[0] << 16);
                    pv[28] = _tmem_load_4[28] * vval;
                    dvo[29] = beta_s[t0 + 29] * _tmem_load_4[29];
                    unsigned int w_28[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_28[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 29) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 29) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_28[0] << 16);
                    pv[29] = _tmem_load_4[29] * vval;
                    dvo[30] = beta_s[t0 + 30] * _tmem_load_4[30];
                    unsigned int w_29[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_29[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 30) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 30) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_29[0] << 16);
                    pv[30] = _tmem_load_4[30] * vval;
                    dvo[31] = beta_s[t0 + 31] * _tmem_load_4[31];
                    unsigned int w_30[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_30[0])) : "r"((smem_v6_addr_1 + (unsigned int)(d_row / 64 * 8192 + (t0 + 31) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 31) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    vval = __uint_as_float(w_30[0] << 16);
                    pv[31] = _tmem_load_4[31] * vval;
                    {
                        __nv_bfloat16 _bval_29 = __float2bfloat16_rn(dvo[0]);
                        uint16_t _bits_29 = *(uint16_t*)&_bval_29;
                        uint32_t _addr_29 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)(t0 * 256 + ((d_row >> 3 ^ t0 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_29), "h"(_bits_29) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_30 = __float2bfloat16_rn(dvo[1]);
                        uint16_t _bits_30 = *(uint16_t*)&_bval_30;
                        uint32_t _addr_30 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 1) * 256 + ((d_row >> 3 ^ t0 + 1 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_30), "h"(_bits_30) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_31 = __float2bfloat16_rn(dvo[2]);
                        uint16_t _bits_31 = *(uint16_t*)&_bval_31;
                        uint32_t _addr_31 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 2) * 256 + ((d_row >> 3 ^ t0 + 2 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_31), "h"(_bits_31) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_32 = __float2bfloat16_rn(dvo[3]);
                        uint16_t _bits_32 = *(uint16_t*)&_bval_32;
                        uint32_t _addr_32 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 3) * 256 + ((d_row >> 3 ^ t0 + 3 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_32), "h"(_bits_32) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_33 = __float2bfloat16_rn(dvo[4]);
                        uint16_t _bits_33 = *(uint16_t*)&_bval_33;
                        uint32_t _addr_33 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 4) * 256 + ((d_row >> 3 ^ t0 + 4 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_33), "h"(_bits_33) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_34 = __float2bfloat16_rn(dvo[5]);
                        uint16_t _bits_34 = *(uint16_t*)&_bval_34;
                        uint32_t _addr_34 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 5) * 256 + ((d_row >> 3 ^ t0 + 5 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_34), "h"(_bits_34) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_35 = __float2bfloat16_rn(dvo[6]);
                        uint16_t _bits_35 = *(uint16_t*)&_bval_35;
                        uint32_t _addr_35 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 6) * 256 + ((d_row >> 3 ^ t0 + 6 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_35), "h"(_bits_35) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_36 = __float2bfloat16_rn(dvo[7]);
                        uint16_t _bits_36 = *(uint16_t*)&_bval_36;
                        uint32_t _addr_36 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 7) * 256 + ((d_row >> 3 ^ t0 + 7 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_36), "h"(_bits_36) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_37 = __float2bfloat16_rn(dvo[8]);
                        uint16_t _bits_37 = *(uint16_t*)&_bval_37;
                        uint32_t _addr_37 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 8) * 256 + ((d_row >> 3 ^ t0 + 8 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_37), "h"(_bits_37) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_38 = __float2bfloat16_rn(dvo[9]);
                        uint16_t _bits_38 = *(uint16_t*)&_bval_38;
                        uint32_t _addr_38 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 9) * 256 + ((d_row >> 3 ^ t0 + 9 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_38), "h"(_bits_38) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_39 = __float2bfloat16_rn(dvo[10]);
                        uint16_t _bits_39 = *(uint16_t*)&_bval_39;
                        uint32_t _addr_39 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 10) * 256 + ((d_row >> 3 ^ t0 + 10 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_39), "h"(_bits_39) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_40 = __float2bfloat16_rn(dvo[11]);
                        uint16_t _bits_40 = *(uint16_t*)&_bval_40;
                        uint32_t _addr_40 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 11) * 256 + ((d_row >> 3 ^ t0 + 11 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_40), "h"(_bits_40) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_41 = __float2bfloat16_rn(dvo[12]);
                        uint16_t _bits_41 = *(uint16_t*)&_bval_41;
                        uint32_t _addr_41 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 12) * 256 + ((d_row >> 3 ^ t0 + 12 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_41), "h"(_bits_41) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_42 = __float2bfloat16_rn(dvo[13]);
                        uint16_t _bits_42 = *(uint16_t*)&_bval_42;
                        uint32_t _addr_42 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 13) * 256 + ((d_row >> 3 ^ t0 + 13 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_42), "h"(_bits_42) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_43 = __float2bfloat16_rn(dvo[14]);
                        uint16_t _bits_43 = *(uint16_t*)&_bval_43;
                        uint32_t _addr_43 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 14) * 256 + ((d_row >> 3 ^ t0 + 14 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_43), "h"(_bits_43) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_44 = __float2bfloat16_rn(dvo[15]);
                        uint16_t _bits_44 = *(uint16_t*)&_bval_44;
                        uint32_t _addr_44 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 15) * 256 + ((d_row >> 3 ^ t0 + 15 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_44), "h"(_bits_44) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_45 = __float2bfloat16_rn(dvo[16]);
                        uint16_t _bits_45 = *(uint16_t*)&_bval_45;
                        uint32_t _addr_45 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 16) * 256 + ((d_row >> 3 ^ t0 + 16 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_45), "h"(_bits_45) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_46 = __float2bfloat16_rn(dvo[17]);
                        uint16_t _bits_46 = *(uint16_t*)&_bval_46;
                        uint32_t _addr_46 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 17) * 256 + ((d_row >> 3 ^ t0 + 17 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_46), "h"(_bits_46) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_47 = __float2bfloat16_rn(dvo[18]);
                        uint16_t _bits_47 = *(uint16_t*)&_bval_47;
                        uint32_t _addr_47 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 18) * 256 + ((d_row >> 3 ^ t0 + 18 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_47), "h"(_bits_47) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_48 = __float2bfloat16_rn(dvo[19]);
                        uint16_t _bits_48 = *(uint16_t*)&_bval_48;
                        uint32_t _addr_48 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 19) * 256 + ((d_row >> 3 ^ t0 + 19 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_48), "h"(_bits_48) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_49 = __float2bfloat16_rn(dvo[20]);
                        uint16_t _bits_49 = *(uint16_t*)&_bval_49;
                        uint32_t _addr_49 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 20) * 256 + ((d_row >> 3 ^ t0 + 20 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_49), "h"(_bits_49) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_50 = __float2bfloat16_rn(dvo[21]);
                        uint16_t _bits_50 = *(uint16_t*)&_bval_50;
                        uint32_t _addr_50 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 21) * 256 + ((d_row >> 3 ^ t0 + 21 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_50), "h"(_bits_50) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_51 = __float2bfloat16_rn(dvo[22]);
                        uint16_t _bits_51 = *(uint16_t*)&_bval_51;
                        uint32_t _addr_51 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 22) * 256 + ((d_row >> 3 ^ t0 + 22 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_51), "h"(_bits_51) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_52 = __float2bfloat16_rn(dvo[23]);
                        uint16_t _bits_52 = *(uint16_t*)&_bval_52;
                        uint32_t _addr_52 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 23) * 256 + ((d_row >> 3 ^ t0 + 23 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_52), "h"(_bits_52) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_53 = __float2bfloat16_rn(dvo[24]);
                        uint16_t _bits_53 = *(uint16_t*)&_bval_53;
                        uint32_t _addr_53 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 24) * 256 + ((d_row >> 3 ^ t0 + 24 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_53), "h"(_bits_53) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_54 = __float2bfloat16_rn(dvo[25]);
                        uint16_t _bits_54 = *(uint16_t*)&_bval_54;
                        uint32_t _addr_54 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 25) * 256 + ((d_row >> 3 ^ t0 + 25 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_54), "h"(_bits_54) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_55 = __float2bfloat16_rn(dvo[26]);
                        uint16_t _bits_55 = *(uint16_t*)&_bval_55;
                        uint32_t _addr_55 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 26) * 256 + ((d_row >> 3 ^ t0 + 26 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_55), "h"(_bits_55) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_56 = __float2bfloat16_rn(dvo[27]);
                        uint16_t _bits_56 = *(uint16_t*)&_bval_56;
                        uint32_t _addr_56 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 27) * 256 + ((d_row >> 3 ^ t0 + 27 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_56), "h"(_bits_56) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_57 = __float2bfloat16_rn(dvo[28]);
                        uint16_t _bits_57 = *(uint16_t*)&_bval_57;
                        uint32_t _addr_57 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 28) * 256 + ((d_row >> 3 ^ t0 + 28 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_57), "h"(_bits_57) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_58 = __float2bfloat16_rn(dvo[29]);
                        uint16_t _bits_58 = *(uint16_t*)&_bval_58;
                        uint32_t _addr_58 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 29) * 256 + ((d_row >> 3 ^ t0 + 29 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_58), "h"(_bits_58) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_59 = __float2bfloat16_rn(dvo[30]);
                        uint16_t _bits_59 = *(uint16_t*)&_bval_59;
                        uint32_t _addr_59 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 30) * 256 + ((d_row >> 3 ^ t0 + 30 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_59), "h"(_bits_59) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_60 = __float2bfloat16_rn(dvo[31]);
                        uint16_t _bits_60 = *(uint16_t*)&_bval_60;
                        uint32_t _addr_60 = static_cast<uint32_t>(stage_dv_addr + (unsigned int)((t0 + 31) * 256 + ((d_row >> 3 ^ t0 + 31 & 7) << 4) + (d_row & 7) * 2));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_60), "h"(_bits_60) : "memory");
                    }
                    mbarrier_wait(b_ready_addr, _phase_b_ready_0);
                    _phase_b_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_5[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                        : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float pdg[32];
                    pv[0] = __fmaf_rn(_tmem_load_5[0] * __uint_as_float(kpk[0] << 16), eg_s[t0], pv[0]);
                    pdg[0] = _tmem_load_5[0] * (__uint_as_float(kpk[0] << 16) * (beta_s[t0] * eg_s[t0]));
                    _tmem_load_5[0] = _tmem_load_5[0] * (eg_s[t0] * beta_s[t0]);
                    pv[1] = __fmaf_rn(_tmem_load_5[1] * __uint_as_float(kpk[0] & 4294901760u), eg_s[t0 + 1], pv[1]);
                    pdg[1] = _tmem_load_5[1] * (__uint_as_float(kpk[0] & 4294901760u) * (beta_s[t0 + 1] * eg_s[t0 + 1]));
                    _tmem_load_5[1] = _tmem_load_5[1] * (eg_s[t0 + 1] * beta_s[t0 + 1]);
                    pv[2] = __fmaf_rn(_tmem_load_5[2] * __uint_as_float(kpk[1] << 16), eg_s[t0 + 2], pv[2]);
                    pdg[2] = _tmem_load_5[2] * (__uint_as_float(kpk[1] << 16) * (beta_s[t0 + 2] * eg_s[t0 + 2]));
                    _tmem_load_5[2] = _tmem_load_5[2] * (eg_s[t0 + 2] * beta_s[t0 + 2]);
                    pv[3] = __fmaf_rn(_tmem_load_5[3] * __uint_as_float(kpk[1] & 4294901760u), eg_s[t0 + 3], pv[3]);
                    pdg[3] = _tmem_load_5[3] * (__uint_as_float(kpk[1] & 4294901760u) * (beta_s[t0 + 3] * eg_s[t0 + 3]));
                    _tmem_load_5[3] = _tmem_load_5[3] * (eg_s[t0 + 3] * beta_s[t0 + 3]);
                    pv[4] = __fmaf_rn(_tmem_load_5[4] * __uint_as_float(kpk[2] << 16), eg_s[t0 + 4], pv[4]);
                    pdg[4] = _tmem_load_5[4] * (__uint_as_float(kpk[2] << 16) * (beta_s[t0 + 4] * eg_s[t0 + 4]));
                    _tmem_load_5[4] = _tmem_load_5[4] * (eg_s[t0 + 4] * beta_s[t0 + 4]);
                    pv[5] = __fmaf_rn(_tmem_load_5[5] * __uint_as_float(kpk[2] & 4294901760u), eg_s[t0 + 5], pv[5]);
                    pdg[5] = _tmem_load_5[5] * (__uint_as_float(kpk[2] & 4294901760u) * (beta_s[t0 + 5] * eg_s[t0 + 5]));
                    _tmem_load_5[5] = _tmem_load_5[5] * (eg_s[t0 + 5] * beta_s[t0 + 5]);
                    pv[6] = __fmaf_rn(_tmem_load_5[6] * __uint_as_float(kpk[3] << 16), eg_s[t0 + 6], pv[6]);
                    pdg[6] = _tmem_load_5[6] * (__uint_as_float(kpk[3] << 16) * (beta_s[t0 + 6] * eg_s[t0 + 6]));
                    _tmem_load_5[6] = _tmem_load_5[6] * (eg_s[t0 + 6] * beta_s[t0 + 6]);
                    pv[7] = __fmaf_rn(_tmem_load_5[7] * __uint_as_float(kpk[3] & 4294901760u), eg_s[t0 + 7], pv[7]);
                    pdg[7] = _tmem_load_5[7] * (__uint_as_float(kpk[3] & 4294901760u) * (beta_s[t0 + 7] * eg_s[t0 + 7]));
                    _tmem_load_5[7] = _tmem_load_5[7] * (eg_s[t0 + 7] * beta_s[t0 + 7]);
                    pv[8] = __fmaf_rn(_tmem_load_5[8] * __uint_as_float(kpk[4] << 16), eg_s[t0 + 8], pv[8]);
                    pdg[8] = _tmem_load_5[8] * (__uint_as_float(kpk[4] << 16) * (beta_s[t0 + 8] * eg_s[t0 + 8]));
                    _tmem_load_5[8] = _tmem_load_5[8] * (eg_s[t0 + 8] * beta_s[t0 + 8]);
                    pv[9] = __fmaf_rn(_tmem_load_5[9] * __uint_as_float(kpk[4] & 4294901760u), eg_s[t0 + 9], pv[9]);
                    pdg[9] = _tmem_load_5[9] * (__uint_as_float(kpk[4] & 4294901760u) * (beta_s[t0 + 9] * eg_s[t0 + 9]));
                    _tmem_load_5[9] = _tmem_load_5[9] * (eg_s[t0 + 9] * beta_s[t0 + 9]);
                    pv[10] = __fmaf_rn(_tmem_load_5[10] * __uint_as_float(kpk[5] << 16), eg_s[t0 + 10], pv[10]);
                    pdg[10] = _tmem_load_5[10] * (__uint_as_float(kpk[5] << 16) * (beta_s[t0 + 10] * eg_s[t0 + 10]));
                    _tmem_load_5[10] = _tmem_load_5[10] * (eg_s[t0 + 10] * beta_s[t0 + 10]);
                    pv[11] = __fmaf_rn(_tmem_load_5[11] * __uint_as_float(kpk[5] & 4294901760u), eg_s[t0 + 11], pv[11]);
                    pdg[11] = _tmem_load_5[11] * (__uint_as_float(kpk[5] & 4294901760u) * (beta_s[t0 + 11] * eg_s[t0 + 11]));
                    _tmem_load_5[11] = _tmem_load_5[11] * (eg_s[t0 + 11] * beta_s[t0 + 11]);
                    pv[12] = __fmaf_rn(_tmem_load_5[12] * __uint_as_float(kpk[6] << 16), eg_s[t0 + 12], pv[12]);
                    pdg[12] = _tmem_load_5[12] * (__uint_as_float(kpk[6] << 16) * (beta_s[t0 + 12] * eg_s[t0 + 12]));
                    _tmem_load_5[12] = _tmem_load_5[12] * (eg_s[t0 + 12] * beta_s[t0 + 12]);
                    pv[13] = __fmaf_rn(_tmem_load_5[13] * __uint_as_float(kpk[6] & 4294901760u), eg_s[t0 + 13], pv[13]);
                    pdg[13] = _tmem_load_5[13] * (__uint_as_float(kpk[6] & 4294901760u) * (beta_s[t0 + 13] * eg_s[t0 + 13]));
                    _tmem_load_5[13] = _tmem_load_5[13] * (eg_s[t0 + 13] * beta_s[t0 + 13]);
                    pv[14] = __fmaf_rn(_tmem_load_5[14] * __uint_as_float(kpk[7] << 16), eg_s[t0 + 14], pv[14]);
                    pdg[14] = _tmem_load_5[14] * (__uint_as_float(kpk[7] << 16) * (beta_s[t0 + 14] * eg_s[t0 + 14]));
                    _tmem_load_5[14] = _tmem_load_5[14] * (eg_s[t0 + 14] * beta_s[t0 + 14]);
                    pv[15] = __fmaf_rn(_tmem_load_5[15] * __uint_as_float(kpk[7] & 4294901760u), eg_s[t0 + 15], pv[15]);
                    pdg[15] = _tmem_load_5[15] * (__uint_as_float(kpk[7] & 4294901760u) * (beta_s[t0 + 15] * eg_s[t0 + 15]));
                    _tmem_load_5[15] = _tmem_load_5[15] * (eg_s[t0 + 15] * beta_s[t0 + 15]);
                    pv[16] = __fmaf_rn(_tmem_load_5[16] * __uint_as_float(kpk[8] << 16), eg_s[t0 + 16], pv[16]);
                    pdg[16] = _tmem_load_5[16] * (__uint_as_float(kpk[8] << 16) * (beta_s[t0 + 16] * eg_s[t0 + 16]));
                    _tmem_load_5[16] = _tmem_load_5[16] * (eg_s[t0 + 16] * beta_s[t0 + 16]);
                    pv[17] = __fmaf_rn(_tmem_load_5[17] * __uint_as_float(kpk[8] & 4294901760u), eg_s[t0 + 17], pv[17]);
                    pdg[17] = _tmem_load_5[17] * (__uint_as_float(kpk[8] & 4294901760u) * (beta_s[t0 + 17] * eg_s[t0 + 17]));
                    _tmem_load_5[17] = _tmem_load_5[17] * (eg_s[t0 + 17] * beta_s[t0 + 17]);
                    pv[18] = __fmaf_rn(_tmem_load_5[18] * __uint_as_float(kpk[9] << 16), eg_s[t0 + 18], pv[18]);
                    pdg[18] = _tmem_load_5[18] * (__uint_as_float(kpk[9] << 16) * (beta_s[t0 + 18] * eg_s[t0 + 18]));
                    _tmem_load_5[18] = _tmem_load_5[18] * (eg_s[t0 + 18] * beta_s[t0 + 18]);
                    pv[19] = __fmaf_rn(_tmem_load_5[19] * __uint_as_float(kpk[9] & 4294901760u), eg_s[t0 + 19], pv[19]);
                    pdg[19] = _tmem_load_5[19] * (__uint_as_float(kpk[9] & 4294901760u) * (beta_s[t0 + 19] * eg_s[t0 + 19]));
                    _tmem_load_5[19] = _tmem_load_5[19] * (eg_s[t0 + 19] * beta_s[t0 + 19]);
                    pv[20] = __fmaf_rn(_tmem_load_5[20] * __uint_as_float(kpk[10] << 16), eg_s[t0 + 20], pv[20]);
                    pdg[20] = _tmem_load_5[20] * (__uint_as_float(kpk[10] << 16) * (beta_s[t0 + 20] * eg_s[t0 + 20]));
                    _tmem_load_5[20] = _tmem_load_5[20] * (eg_s[t0 + 20] * beta_s[t0 + 20]);
                    pv[21] = __fmaf_rn(_tmem_load_5[21] * __uint_as_float(kpk[10] & 4294901760u), eg_s[t0 + 21], pv[21]);
                    pdg[21] = _tmem_load_5[21] * (__uint_as_float(kpk[10] & 4294901760u) * (beta_s[t0 + 21] * eg_s[t0 + 21]));
                    _tmem_load_5[21] = _tmem_load_5[21] * (eg_s[t0 + 21] * beta_s[t0 + 21]);
                    pv[22] = __fmaf_rn(_tmem_load_5[22] * __uint_as_float(kpk[11] << 16), eg_s[t0 + 22], pv[22]);
                    pdg[22] = _tmem_load_5[22] * (__uint_as_float(kpk[11] << 16) * (beta_s[t0 + 22] * eg_s[t0 + 22]));
                    _tmem_load_5[22] = _tmem_load_5[22] * (eg_s[t0 + 22] * beta_s[t0 + 22]);
                    pv[23] = __fmaf_rn(_tmem_load_5[23] * __uint_as_float(kpk[11] & 4294901760u), eg_s[t0 + 23], pv[23]);
                    pdg[23] = _tmem_load_5[23] * (__uint_as_float(kpk[11] & 4294901760u) * (beta_s[t0 + 23] * eg_s[t0 + 23]));
                    _tmem_load_5[23] = _tmem_load_5[23] * (eg_s[t0 + 23] * beta_s[t0 + 23]);
                    pv[24] = __fmaf_rn(_tmem_load_5[24] * __uint_as_float(kpk[12] << 16), eg_s[t0 + 24], pv[24]);
                    pdg[24] = _tmem_load_5[24] * (__uint_as_float(kpk[12] << 16) * (beta_s[t0 + 24] * eg_s[t0 + 24]));
                    _tmem_load_5[24] = _tmem_load_5[24] * (eg_s[t0 + 24] * beta_s[t0 + 24]);
                    pv[25] = __fmaf_rn(_tmem_load_5[25] * __uint_as_float(kpk[12] & 4294901760u), eg_s[t0 + 25], pv[25]);
                    pdg[25] = _tmem_load_5[25] * (__uint_as_float(kpk[12] & 4294901760u) * (beta_s[t0 + 25] * eg_s[t0 + 25]));
                    _tmem_load_5[25] = _tmem_load_5[25] * (eg_s[t0 + 25] * beta_s[t0 + 25]);
                    pv[26] = __fmaf_rn(_tmem_load_5[26] * __uint_as_float(kpk[13] << 16), eg_s[t0 + 26], pv[26]);
                    pdg[26] = _tmem_load_5[26] * (__uint_as_float(kpk[13] << 16) * (beta_s[t0 + 26] * eg_s[t0 + 26]));
                    _tmem_load_5[26] = _tmem_load_5[26] * (eg_s[t0 + 26] * beta_s[t0 + 26]);
                    pv[27] = __fmaf_rn(_tmem_load_5[27] * __uint_as_float(kpk[13] & 4294901760u), eg_s[t0 + 27], pv[27]);
                    pdg[27] = _tmem_load_5[27] * (__uint_as_float(kpk[13] & 4294901760u) * (beta_s[t0 + 27] * eg_s[t0 + 27]));
                    _tmem_load_5[27] = _tmem_load_5[27] * (eg_s[t0 + 27] * beta_s[t0 + 27]);
                    pv[28] = __fmaf_rn(_tmem_load_5[28] * __uint_as_float(kpk[14] << 16), eg_s[t0 + 28], pv[28]);
                    pdg[28] = _tmem_load_5[28] * (__uint_as_float(kpk[14] << 16) * (beta_s[t0 + 28] * eg_s[t0 + 28]));
                    _tmem_load_5[28] = _tmem_load_5[28] * (eg_s[t0 + 28] * beta_s[t0 + 28]);
                    pv[29] = __fmaf_rn(_tmem_load_5[29] * __uint_as_float(kpk[14] & 4294901760u), eg_s[t0 + 29], pv[29]);
                    pdg[29] = _tmem_load_5[29] * (__uint_as_float(kpk[14] & 4294901760u) * (beta_s[t0 + 29] * eg_s[t0 + 29]));
                    _tmem_load_5[29] = _tmem_load_5[29] * (eg_s[t0 + 29] * beta_s[t0 + 29]);
                    pv[30] = __fmaf_rn(_tmem_load_5[30] * __uint_as_float(kpk[15] << 16), eg_s[t0 + 30], pv[30]);
                    pdg[30] = _tmem_load_5[30] * (__uint_as_float(kpk[15] << 16) * (beta_s[t0 + 30] * eg_s[t0 + 30]));
                    _tmem_load_5[30] = _tmem_load_5[30] * (eg_s[t0 + 30] * beta_s[t0 + 30]);
                    pv[31] = __fmaf_rn(_tmem_load_5[31] * __uint_as_float(kpk[15] & 4294901760u), eg_s[t0 + 31], pv[31]);
                    pdg[31] = _tmem_load_5[31] * (__uint_as_float(kpk[15] & 4294901760u) * (beta_s[t0 + 31] * eg_s[t0 + 31]));
                    _tmem_load_5[31] = _tmem_load_5[31] * (eg_s[t0 + 31] * beta_s[t0 + 31]);
                    float _tmem_load_6[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                        : "r"(taddr + 192 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    _tmem_load_6[0] = _tmem_load_6[0] + _tmem_load_5[0];
                    _tmem_load_6[1] = _tmem_load_6[1] + _tmem_load_5[1];
                    _tmem_load_6[2] = _tmem_load_6[2] + _tmem_load_5[2];
                    _tmem_load_6[3] = _tmem_load_6[3] + _tmem_load_5[3];
                    _tmem_load_6[4] = _tmem_load_6[4] + _tmem_load_5[4];
                    _tmem_load_6[5] = _tmem_load_6[5] + _tmem_load_5[5];
                    _tmem_load_6[6] = _tmem_load_6[6] + _tmem_load_5[6];
                    _tmem_load_6[7] = _tmem_load_6[7] + _tmem_load_5[7];
                    _tmem_load_6[8] = _tmem_load_6[8] + _tmem_load_5[8];
                    _tmem_load_6[9] = _tmem_load_6[9] + _tmem_load_5[9];
                    _tmem_load_6[10] = _tmem_load_6[10] + _tmem_load_5[10];
                    _tmem_load_6[11] = _tmem_load_6[11] + _tmem_load_5[11];
                    _tmem_load_6[12] = _tmem_load_6[12] + _tmem_load_5[12];
                    _tmem_load_6[13] = _tmem_load_6[13] + _tmem_load_5[13];
                    _tmem_load_6[14] = _tmem_load_6[14] + _tmem_load_5[14];
                    _tmem_load_6[15] = _tmem_load_6[15] + _tmem_load_5[15];
                    _tmem_load_6[16] = _tmem_load_6[16] + _tmem_load_5[16];
                    _tmem_load_6[17] = _tmem_load_6[17] + _tmem_load_5[17];
                    _tmem_load_6[18] = _tmem_load_6[18] + _tmem_load_5[18];
                    _tmem_load_6[19] = _tmem_load_6[19] + _tmem_load_5[19];
                    _tmem_load_6[20] = _tmem_load_6[20] + _tmem_load_5[20];
                    _tmem_load_6[21] = _tmem_load_6[21] + _tmem_load_5[21];
                    _tmem_load_6[22] = _tmem_load_6[22] + _tmem_load_5[22];
                    _tmem_load_6[23] = _tmem_load_6[23] + _tmem_load_5[23];
                    _tmem_load_6[24] = _tmem_load_6[24] + _tmem_load_5[24];
                    _tmem_load_6[25] = _tmem_load_6[25] + _tmem_load_5[25];
                    _tmem_load_6[26] = _tmem_load_6[26] + _tmem_load_5[26];
                    _tmem_load_6[27] = _tmem_load_6[27] + _tmem_load_5[27];
                    _tmem_load_6[28] = _tmem_load_6[28] + _tmem_load_5[28];
                    _tmem_load_6[29] = _tmem_load_6[29] + _tmem_load_5[29];
                    _tmem_load_6[30] = _tmem_load_6[30] + _tmem_load_5[30];
                    _tmem_load_6[31] = _tmem_load_6[31] + _tmem_load_5[31];
                    tmem_st_x32_f32(taddr + 192 + (unsigned int)row_base + (unsigned int)(ch * 32), _tmem_load_6);
                    float w16_1[16];
                    float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[16] : pv[0]), 16);
                    w16_1[0] = (((lane_0 & 16) == 0) ? pv[0] : pv[16]) + _shfl_xor_sync_61;
                    float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[17] : pv[1]), 16);
                    w16_1[1] = (((lane_0 & 16) == 0) ? pv[1] : pv[17]) + _shfl_xor_sync_62;
                    float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[18] : pv[2]), 16);
                    w16_1[2] = (((lane_0 & 16) == 0) ? pv[2] : pv[18]) + _shfl_xor_sync_63;
                    float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[19] : pv[3]), 16);
                    w16_1[3] = (((lane_0 & 16) == 0) ? pv[3] : pv[19]) + _shfl_xor_sync_64;
                    float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[20] : pv[4]), 16);
                    w16_1[4] = (((lane_0 & 16) == 0) ? pv[4] : pv[20]) + _shfl_xor_sync_65;
                    float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[21] : pv[5]), 16);
                    w16_1[5] = (((lane_0 & 16) == 0) ? pv[5] : pv[21]) + _shfl_xor_sync_66;
                    float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[22] : pv[6]), 16);
                    w16_1[6] = (((lane_0 & 16) == 0) ? pv[6] : pv[22]) + _shfl_xor_sync_67;
                    float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[23] : pv[7]), 16);
                    w16_1[7] = (((lane_0 & 16) == 0) ? pv[7] : pv[23]) + _shfl_xor_sync_68;
                    float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[24] : pv[8]), 16);
                    w16_1[8] = (((lane_0 & 16) == 0) ? pv[8] : pv[24]) + _shfl_xor_sync_69;
                    float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[25] : pv[9]), 16);
                    w16_1[9] = (((lane_0 & 16) == 0) ? pv[9] : pv[25]) + _shfl_xor_sync_70;
                    float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[26] : pv[10]), 16);
                    w16_1[10] = (((lane_0 & 16) == 0) ? pv[10] : pv[26]) + _shfl_xor_sync_71;
                    float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[27] : pv[11]), 16);
                    w16_1[11] = (((lane_0 & 16) == 0) ? pv[11] : pv[27]) + _shfl_xor_sync_72;
                    float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[28] : pv[12]), 16);
                    w16_1[12] = (((lane_0 & 16) == 0) ? pv[12] : pv[28]) + _shfl_xor_sync_73;
                    float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[29] : pv[13]), 16);
                    w16_1[13] = (((lane_0 & 16) == 0) ? pv[13] : pv[29]) + _shfl_xor_sync_74;
                    float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[30] : pv[14]), 16);
                    w16_1[14] = (((lane_0 & 16) == 0) ? pv[14] : pv[30]) + _shfl_xor_sync_75;
                    float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pv[31] : pv[15]), 16);
                    w16_1[15] = (((lane_0 & 16) == 0) ? pv[15] : pv[31]) + _shfl_xor_sync_76;
                    float w8_1[8];
                    float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1[8] : w16_1[0]), 8);
                    w8_1[0] = (((lane_0 & 8) == 0) ? w16_1[0] : w16_1[8]) + _shfl_xor_sync_77;
                    float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1[9] : w16_1[1]), 8);
                    w8_1[1] = (((lane_0 & 8) == 0) ? w16_1[1] : w16_1[9]) + _shfl_xor_sync_78;
                    float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1[10] : w16_1[2]), 8);
                    w8_1[2] = (((lane_0 & 8) == 0) ? w16_1[2] : w16_1[10]) + _shfl_xor_sync_79;
                    float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1[11] : w16_1[3]), 8);
                    w8_1[3] = (((lane_0 & 8) == 0) ? w16_1[3] : w16_1[11]) + _shfl_xor_sync_80;
                    float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1[12] : w16_1[4]), 8);
                    w8_1[4] = (((lane_0 & 8) == 0) ? w16_1[4] : w16_1[12]) + _shfl_xor_sync_81;
                    float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1[13] : w16_1[5]), 8);
                    w8_1[5] = (((lane_0 & 8) == 0) ? w16_1[5] : w16_1[13]) + _shfl_xor_sync_82;
                    float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1[14] : w16_1[6]), 8);
                    w8_1[6] = (((lane_0 & 8) == 0) ? w16_1[6] : w16_1[14]) + _shfl_xor_sync_83;
                    float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1[15] : w16_1[7]), 8);
                    w8_1[7] = (((lane_0 & 8) == 0) ? w16_1[7] : w16_1[15]) + _shfl_xor_sync_84;
                    float w4_1[4];
                    float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_1[4] : w8_1[0]), 4);
                    w4_1[0] = (((lane_0 & 4) == 0) ? w8_1[0] : w8_1[4]) + _shfl_xor_sync_85;
                    float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_1[5] : w8_1[1]), 4);
                    w4_1[1] = (((lane_0 & 4) == 0) ? w8_1[1] : w8_1[5]) + _shfl_xor_sync_86;
                    float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_1[6] : w8_1[2]), 4);
                    w4_1[2] = (((lane_0 & 4) == 0) ? w8_1[2] : w8_1[6]) + _shfl_xor_sync_87;
                    float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_1[7] : w8_1[3]), 4);
                    w4_1[3] = (((lane_0 & 4) == 0) ? w8_1[3] : w8_1[7]) + _shfl_xor_sync_88;
                    float w2_1[2];
                    float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) == 0) ? w4_1[2] : w4_1[0]), 2);
                    w2_1[0] = (((lane_0 & 2) == 0) ? w4_1[0] : w4_1[2]) + _shfl_xor_sync_89;
                    float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) == 0) ? w4_1[3] : w4_1[1]), 2);
                    w2_1[1] = (((lane_0 & 2) == 0) ? w4_1[1] : w4_1[3]) + _shfl_xor_sync_90;
                    float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 1) == 0) ? w2_1[1] : w2_1[0]), 1);
                    tmp = (((lane_0 & 1) == 0) ? w2_1[0] : w2_1[1]) + _shfl_xor_sync_91;
                    db_acc = db_acc + tmp;
                    float w16_0[16];
                    float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[16] : pdg[0]), 16);
                    w16_0[0] = (((lane_0 & 16) == 0) ? pdg[0] : pdg[16]) + _shfl_xor_sync_92;
                    float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[17] : pdg[1]), 16);
                    w16_0[1] = (((lane_0 & 16) == 0) ? pdg[1] : pdg[17]) + _shfl_xor_sync_93;
                    float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[18] : pdg[2]), 16);
                    w16_0[2] = (((lane_0 & 16) == 0) ? pdg[2] : pdg[18]) + _shfl_xor_sync_94;
                    float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[19] : pdg[3]), 16);
                    w16_0[3] = (((lane_0 & 16) == 0) ? pdg[3] : pdg[19]) + _shfl_xor_sync_95;
                    float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[20] : pdg[4]), 16);
                    w16_0[4] = (((lane_0 & 16) == 0) ? pdg[4] : pdg[20]) + _shfl_xor_sync_96;
                    float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[21] : pdg[5]), 16);
                    w16_0[5] = (((lane_0 & 16) == 0) ? pdg[5] : pdg[21]) + _shfl_xor_sync_97;
                    float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[22] : pdg[6]), 16);
                    w16_0[6] = (((lane_0 & 16) == 0) ? pdg[6] : pdg[22]) + _shfl_xor_sync_98;
                    float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[23] : pdg[7]), 16);
                    w16_0[7] = (((lane_0 & 16) == 0) ? pdg[7] : pdg[23]) + _shfl_xor_sync_99;
                    float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[24] : pdg[8]), 16);
                    w16_0[8] = (((lane_0 & 16) == 0) ? pdg[8] : pdg[24]) + _shfl_xor_sync_100;
                    float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[25] : pdg[9]), 16);
                    w16_0[9] = (((lane_0 & 16) == 0) ? pdg[9] : pdg[25]) + _shfl_xor_sync_101;
                    float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[26] : pdg[10]), 16);
                    w16_0[10] = (((lane_0 & 16) == 0) ? pdg[10] : pdg[26]) + _shfl_xor_sync_102;
                    float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[27] : pdg[11]), 16);
                    w16_0[11] = (((lane_0 & 16) == 0) ? pdg[11] : pdg[27]) + _shfl_xor_sync_103;
                    float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[28] : pdg[12]), 16);
                    w16_0[12] = (((lane_0 & 16) == 0) ? pdg[12] : pdg[28]) + _shfl_xor_sync_104;
                    float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[29] : pdg[13]), 16);
                    w16_0[13] = (((lane_0 & 16) == 0) ? pdg[13] : pdg[29]) + _shfl_xor_sync_105;
                    float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[30] : pdg[14]), 16);
                    w16_0[14] = (((lane_0 & 16) == 0) ? pdg[14] : pdg[30]) + _shfl_xor_sync_106;
                    float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? pdg[31] : pdg[15]), 16);
                    w16_0[15] = (((lane_0 & 16) == 0) ? pdg[15] : pdg[31]) + _shfl_xor_sync_107;
                    float w8_0[8];
                    float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_0[8] : w16_0[0]), 8);
                    w8_0[0] = (((lane_0 & 8) == 0) ? w16_0[0] : w16_0[8]) + _shfl_xor_sync_108;
                    float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_0[9] : w16_0[1]), 8);
                    w8_0[1] = (((lane_0 & 8) == 0) ? w16_0[1] : w16_0[9]) + _shfl_xor_sync_109;
                    float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_0[10] : w16_0[2]), 8);
                    w8_0[2] = (((lane_0 & 8) == 0) ? w16_0[2] : w16_0[10]) + _shfl_xor_sync_110;
                    float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_0[11] : w16_0[3]), 8);
                    w8_0[3] = (((lane_0 & 8) == 0) ? w16_0[3] : w16_0[11]) + _shfl_xor_sync_111;
                    float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_0[12] : w16_0[4]), 8);
                    w8_0[4] = (((lane_0 & 8) == 0) ? w16_0[4] : w16_0[12]) + _shfl_xor_sync_112;
                    float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_0[13] : w16_0[5]), 8);
                    w8_0[5] = (((lane_0 & 8) == 0) ? w16_0[5] : w16_0[13]) + _shfl_xor_sync_113;
                    float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_0[14] : w16_0[6]), 8);
                    w8_0[6] = (((lane_0 & 8) == 0) ? w16_0[6] : w16_0[14]) + _shfl_xor_sync_114;
                    float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_0[15] : w16_0[7]), 8);
                    w8_0[7] = (((lane_0 & 8) == 0) ? w16_0[7] : w16_0[15]) + _shfl_xor_sync_115;
                    float w4_0[4];
                    float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_0[4] : w8_0[0]), 4);
                    w4_0[0] = (((lane_0 & 4) == 0) ? w8_0[0] : w8_0[4]) + _shfl_xor_sync_116;
                    float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_0[5] : w8_0[1]), 4);
                    w4_0[1] = (((lane_0 & 4) == 0) ? w8_0[1] : w8_0[5]) + _shfl_xor_sync_117;
                    float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_0[6] : w8_0[2]), 4);
                    w4_0[2] = (((lane_0 & 4) == 0) ? w8_0[2] : w8_0[6]) + _shfl_xor_sync_118;
                    float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_0[7] : w8_0[3]), 4);
                    w4_0[3] = (((lane_0 & 4) == 0) ? w8_0[3] : w8_0[7]) + _shfl_xor_sync_119;
                    float w2_0[2];
                    float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) == 0) ? w4_0[2] : w4_0[0]), 2);
                    w2_0[0] = (((lane_0 & 2) == 0) ? w4_0[0] : w4_0[2]) + _shfl_xor_sync_120;
                    float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) == 0) ? w4_0[3] : w4_0[1]), 2);
                    w2_0[1] = (((lane_0 & 2) == 0) ? w4_0[1] : w4_0[3]) + _shfl_xor_sync_121;
                    float _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 1) == 0) ? w2_0[1] : w2_0[0]), 1);
                    tmp = (((lane_0 & 1) == 0) ? w2_0[0] : w2_0[1]) + _shfl_xor_sync_122;
                    dg_acc = dg_acc + tmp;
                    float _tmem_load_7[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                        : "r"(taddr + 320 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float dab[16];
                    dab[0] = ((trow16 < clen) ? ((trow16 > colh) ? _tmem_load_7[0] * beta_s[colh] : 0.0f) : 0.0f);
                    dab[1] = ((trow16 < clen) ? ((trow16 > colh + 1) ? _tmem_load_7[1] * beta_s[colh + 1] : 0.0f) : 0.0f);
                    dab[2] = ((trow16 < clen) ? ((trow16 > colh + 2) ? _tmem_load_7[2] * beta_s[colh + 2] : 0.0f) : 0.0f);
                    dab[3] = ((trow16 < clen) ? ((trow16 > colh + 3) ? _tmem_load_7[3] * beta_s[colh + 3] : 0.0f) : 0.0f);
                    dab[4] = ((trow16 < clen) ? ((trow16 > colh + 4) ? _tmem_load_7[4] * beta_s[colh + 4] : 0.0f) : 0.0f);
                    dab[5] = ((trow16 < clen) ? ((trow16 > colh + 5) ? _tmem_load_7[5] * beta_s[colh + 5] : 0.0f) : 0.0f);
                    dab[6] = ((trow16 < clen) ? ((trow16 > colh + 6) ? _tmem_load_7[6] * beta_s[colh + 6] : 0.0f) : 0.0f);
                    dab[7] = ((trow16 < clen) ? ((trow16 > colh + 7) ? _tmem_load_7[7] * beta_s[colh + 7] : 0.0f) : 0.0f);
                    dab[8] = ((trow16 < clen) ? ((trow16 > colh + 8) ? _tmem_load_7[8] * beta_s[colh + 8] : 0.0f) : 0.0f);
                    dab[9] = ((trow16 < clen) ? ((trow16 > colh + 9) ? _tmem_load_7[9] * beta_s[colh + 9] : 0.0f) : 0.0f);
                    dab[10] = ((trow16 < clen) ? ((trow16 > colh + 10) ? _tmem_load_7[10] * beta_s[colh + 10] : 0.0f) : 0.0f);
                    dab[11] = ((trow16 < clen) ? ((trow16 > colh + 11) ? _tmem_load_7[11] * beta_s[colh + 11] : 0.0f) : 0.0f);
                    dab[12] = ((trow16 < clen) ? ((trow16 > colh + 12) ? _tmem_load_7[12] * beta_s[colh + 12] : 0.0f) : 0.0f);
                    dab[13] = ((trow16 < clen) ? ((trow16 > colh + 13) ? _tmem_load_7[13] * beta_s[colh + 13] : 0.0f) : 0.0f);
                    dab[14] = ((trow16 < clen) ? ((trow16 > colh + 14) ? _tmem_load_7[14] * beta_s[colh + 14] : 0.0f) : 0.0f);
                    dab[15] = ((trow16 < clen) ? ((trow16 > colh + 15) ? _tmem_load_7[15] * beta_s[colh + 15] : 0.0f) : 0.0f);
                    unsigned int dab_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dab[_lp*2 + 0], dab[_lp*2+1 + 0]));
                        dab_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v20_addr_0 + (unsigned int)(colh / 64 * 8192 + trow16 * 128 + colh % 64 * 2 ^ (colh / 64 * 8192 + trow16 * 128 + colh % 64 * 2 >> 7 & 7) << 4))), "r"(dab_bf16[0]), "r"(dab_bf16[1]), "r"(dab_bf16[2]), "r"(dab_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v20_addr_0 + (unsigned int)((colh + 8) / 64 * 8192 + trow16 * 128 + (colh + 8) % 64 * 2 ^ ((colh + 8) / 64 * 8192 + trow16 * 128 + (colh + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(dab_bf16[4]), "r"(dab_bf16[5]), "r"(dab_bf16[6]), "r"(dab_bf16[7]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(b_drained_addr);
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    int t = tid_1 >> 3;
                    int t1 = t + 32;
                    int c0 = tid_1 & 7;
                    if (t < clen) {
                        long long base = (long long)(trow + t) * (long long)dv_tstride + (long long)(hv * 128 + c0 * 8);
                        int chunk_0 = c0 ^ t & 7;
                        unsigned int q0_1_0[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q0_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_0[(0) + 3]))
                            : "r"(stage_dv_addr + (unsigned int)(t * 256 + (chunk_0 << 4))));
                        reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dv + base))[0] = reinterpret_cast<int4*>(q0_1_0)[0];
                        int chunk_1 = c0 + 8 ^ t & 7;
                        unsigned int q0_1_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q0_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_1[(0) + 3]))
                            : "r"(stage_dv_addr + (unsigned int)(t * 256 + (chunk_1 << 4))));
                        reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dv + (base + 64)))[0] = reinterpret_cast<int4*>(q0_1_1)[0];
                    }
                    if (t1 < clen) {
                        long long base1 = (long long)(trow + t1) * (long long)dv_tstride + (long long)(hv * 128 + c0 * 8);
                        int chunk1_0 = c0 ^ t1 & 7;
                        unsigned int q2_3_0[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q2_3_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_0[(0) + 3]))
                            : "r"(stage_dv_addr + (unsigned int)(t1 * 256 + (chunk1_0 << 4))));
                        reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dv + base1))[0] = reinterpret_cast<int4*>(q2_3_0)[0];
                        int chunk1_1 = c0 + 8 ^ t1 & 7;
                        unsigned int q2_3_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q2_3_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_1[(0) + 3]))
                            : "r"(stage_dv_addr + (unsigned int)(t1 * 256 + (chunk1_1 << 4))));
                        reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dv + (base1 + 64)))[0] = reinterpret_cast<int4*>(q2_3_1)[0];
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dv_stored_addr);
                    mbarrier_wait(c_ready_addr, _phase_c_ready_0);
                    _phase_c_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_8[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                        : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int _tmem_load_8_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                        _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v20_addr_1 + (unsigned int)(colh / 64 * 8192 + trow16 * 128 + colh % 64 * 2 ^ (colh / 64 * 8192 + trow16 * 128 + colh % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[0]), "r"(_tmem_load_8_bf16[1]), "r"(_tmem_load_8_bf16[2]), "r"(_tmem_load_8_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v20_addr_1 + (unsigned int)((colh + 8) / 64 * 8192 + trow16 * 128 + (colh + 8) % 64 * 2 ^ ((colh + 8) / 64 * 8192 + trow16 * 128 + (colh + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[4]), "r"(_tmem_load_8_bf16[5]), "r"(_tmem_load_8_bf16[6]), "r"(_tmem_load_8_bf16[7]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(c_drained_addr);
                    mbarrier_wait(d_ready_addr, _phase_d_ready_0);
                    _phase_d_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_9[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15]))
                        : "r"(taddr + 320 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float gs = gc_s[trow16];
                    float akg[16];
                    float _exp_123 = expf(gs - gc_s[colh]);
                    akg[0] = ((trow16 < clen) ? ((trow16 > colh) ? (-_tmem_load_9[0]) * _exp_123 : 0.0f) : 0.0f);
                    float _exp_124 = expf(gs - gc_s[colh + 1]);
                    akg[1] = ((trow16 < clen) ? ((trow16 > colh + 1) ? (-_tmem_load_9[1]) * _exp_124 : 0.0f) : 0.0f);
                    float _exp_125 = expf(gs - gc_s[colh + 2]);
                    akg[2] = ((trow16 < clen) ? ((trow16 > colh + 2) ? (-_tmem_load_9[2]) * _exp_125 : 0.0f) : 0.0f);
                    float _exp_126 = expf(gs - gc_s[colh + 3]);
                    akg[3] = ((trow16 < clen) ? ((trow16 > colh + 3) ? (-_tmem_load_9[3]) * _exp_126 : 0.0f) : 0.0f);
                    float _exp_127 = expf(gs - gc_s[colh + 4]);
                    akg[4] = ((trow16 < clen) ? ((trow16 > colh + 4) ? (-_tmem_load_9[4]) * _exp_127 : 0.0f) : 0.0f);
                    float _exp_128 = expf(gs - gc_s[colh + 5]);
                    akg[5] = ((trow16 < clen) ? ((trow16 > colh + 5) ? (-_tmem_load_9[5]) * _exp_128 : 0.0f) : 0.0f);
                    float _exp_129 = expf(gs - gc_s[colh + 6]);
                    akg[6] = ((trow16 < clen) ? ((trow16 > colh + 6) ? (-_tmem_load_9[6]) * _exp_129 : 0.0f) : 0.0f);
                    float _exp_130 = expf(gs - gc_s[colh + 7]);
                    akg[7] = ((trow16 < clen) ? ((trow16 > colh + 7) ? (-_tmem_load_9[7]) * _exp_130 : 0.0f) : 0.0f);
                    float _exp_131 = expf(gs - gc_s[colh + 8]);
                    akg[8] = ((trow16 < clen) ? ((trow16 > colh + 8) ? (-_tmem_load_9[8]) * _exp_131 : 0.0f) : 0.0f);
                    float _exp_132 = expf(gs - gc_s[colh + 9]);
                    akg[9] = ((trow16 < clen) ? ((trow16 > colh + 9) ? (-_tmem_load_9[9]) * _exp_132 : 0.0f) : 0.0f);
                    float _exp_133 = expf(gs - gc_s[colh + 10]);
                    akg[10] = ((trow16 < clen) ? ((trow16 > colh + 10) ? (-_tmem_load_9[10]) * _exp_133 : 0.0f) : 0.0f);
                    float _exp_134 = expf(gs - gc_s[colh + 11]);
                    akg[11] = ((trow16 < clen) ? ((trow16 > colh + 11) ? (-_tmem_load_9[11]) * _exp_134 : 0.0f) : 0.0f);
                    float _exp_135 = expf(gs - gc_s[colh + 12]);
                    akg[12] = ((trow16 < clen) ? ((trow16 > colh + 12) ? (-_tmem_load_9[12]) * _exp_135 : 0.0f) : 0.0f);
                    float _exp_136 = expf(gs - gc_s[colh + 13]);
                    akg[13] = ((trow16 < clen) ? ((trow16 > colh + 13) ? (-_tmem_load_9[13]) * _exp_136 : 0.0f) : 0.0f);
                    float _exp_137 = expf(gs - gc_s[colh + 14]);
                    akg[14] = ((trow16 < clen) ? ((trow16 > colh + 14) ? (-_tmem_load_9[14]) * _exp_137 : 0.0f) : 0.0f);
                    float _exp_138 = expf(gs - gc_s[colh + 15]);
                    akg[15] = ((trow16 < clen) ? ((trow16 > colh + 15) ? (-_tmem_load_9[15]) * _exp_138 : 0.0f) : 0.0f);
                    unsigned int akg_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(akg[_lp*2 + 0], akg[_lp*2+1 + 0]));
                        akg_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v20_addr_3 + (unsigned int)(colh / 64 * 8192 + trow16 * 128 + colh % 64 * 2 ^ (colh / 64 * 8192 + trow16 * 128 + colh % 64 * 2 >> 7 & 7) << 4))), "r"(akg_bf16[0]), "r"(akg_bf16[1]), "r"(akg_bf16[2]), "r"(akg_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v20_addr_3 + (unsigned int)((colh + 8) / 64 * 8192 + trow16 * 128 + (colh + 8) % 64 * 2 ^ ((colh + 8) / 64 * 8192 + trow16 * 128 + (colh + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(akg_bf16[4]), "r"(akg_bf16[5]), "r"(akg_bf16[6]), "r"(akg_bf16[7]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(d_drained_addr);
                    mbarrier_wait(e1_ready_addr, _phase_e1_ready_0);
                    _phase_e1_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_10[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_10[0]), "=f"(_tmem_load_10[1]), "=f"(_tmem_load_10[2]), "=f"(_tmem_load_10[3]), "=f"(_tmem_load_10[4]), "=f"(_tmem_load_10[5]), "=f"(_tmem_load_10[6]), "=f"(_tmem_load_10[7]), "=f"(_tmem_load_10[8]), "=f"(_tmem_load_10[9]), "=f"(_tmem_load_10[10]), "=f"(_tmem_load_10[11]), "=f"(_tmem_load_10[12]), "=f"(_tmem_load_10[13]), "=f"(_tmem_load_10[14]), "=f"(_tmem_load_10[15]), "=f"(_tmem_load_10[16]), "=f"(_tmem_load_10[17]), "=f"(_tmem_load_10[18]), "=f"(_tmem_load_10[19]), "=f"(_tmem_load_10[20]), "=f"(_tmem_load_10[21]), "=f"(_tmem_load_10[22]), "=f"(_tmem_load_10[23]), "=f"(_tmem_load_10[24]), "=f"(_tmem_load_10[25]), "=f"(_tmem_load_10[26]), "=f"(_tmem_load_10[27]), "=f"(_tmem_load_10[28]), "=f"(_tmem_load_10[29]), "=f"(_tmem_load_10[30]), "=f"(_tmem_load_10[31])
                        : "r"(taddr + 448 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_11[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_11[0]), "=f"(_tmem_load_11[1]), "=f"(_tmem_load_11[2]), "=f"(_tmem_load_11[3]), "=f"(_tmem_load_11[4]), "=f"(_tmem_load_11[5]), "=f"(_tmem_load_11[6]), "=f"(_tmem_load_11[7]), "=f"(_tmem_load_11[8]), "=f"(_tmem_load_11[9]), "=f"(_tmem_load_11[10]), "=f"(_tmem_load_11[11]), "=f"(_tmem_load_11[12]), "=f"(_tmem_load_11[13]), "=f"(_tmem_load_11[14]), "=f"(_tmem_load_11[15]), "=f"(_tmem_load_11[16]), "=f"(_tmem_load_11[17]), "=f"(_tmem_load_11[18]), "=f"(_tmem_load_11[19]), "=f"(_tmem_load_11[20]), "=f"(_tmem_load_11[21]), "=f"(_tmem_load_11[22]), "=f"(_tmem_load_11[23]), "=f"(_tmem_load_11[24]), "=f"(_tmem_load_11[25]), "=f"(_tmem_load_11[26]), "=f"(_tmem_load_11[27]), "=f"(_tmem_load_11[28]), "=f"(_tmem_load_11[29]), "=f"(_tmem_load_11[30]), "=f"(_tmem_load_11[31])
                        : "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float qval = 0.0f;
                    unsigned int w_31[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_31[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + t0 * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + t0 * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_31[0] << 16);
                    _tmem_load_10[0] = __fmaf_rn(qval, _tmem_load_11[0], _tmem_load_10[0]);
                    unsigned int w_32[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_32[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 1) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 1) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_32[0] << 16);
                    _tmem_load_10[1] = __fmaf_rn(qval, _tmem_load_11[1], _tmem_load_10[1]);
                    unsigned int w_33[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_33[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 2) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 2) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_33[0] << 16);
                    _tmem_load_10[2] = __fmaf_rn(qval, _tmem_load_11[2], _tmem_load_10[2]);
                    unsigned int w_34[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_34[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 3) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 3) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_34[0] << 16);
                    _tmem_load_10[3] = __fmaf_rn(qval, _tmem_load_11[3], _tmem_load_10[3]);
                    unsigned int w_35[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_35[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 4) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 4) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_35[0] << 16);
                    _tmem_load_10[4] = __fmaf_rn(qval, _tmem_load_11[4], _tmem_load_10[4]);
                    unsigned int w_36[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_36[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 5) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 5) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_36[0] << 16);
                    _tmem_load_10[5] = __fmaf_rn(qval, _tmem_load_11[5], _tmem_load_10[5]);
                    unsigned int w_37[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_37[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 6) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 6) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_37[0] << 16);
                    _tmem_load_10[6] = __fmaf_rn(qval, _tmem_load_11[6], _tmem_load_10[6]);
                    unsigned int w_38[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_38[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 7) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 7) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_38[0] << 16);
                    _tmem_load_10[7] = __fmaf_rn(qval, _tmem_load_11[7], _tmem_load_10[7]);
                    unsigned int w_39[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_39[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 8) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 8) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_39[0] << 16);
                    _tmem_load_10[8] = __fmaf_rn(qval, _tmem_load_11[8], _tmem_load_10[8]);
                    unsigned int w_40[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_40[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 9) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 9) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_40[0] << 16);
                    _tmem_load_10[9] = __fmaf_rn(qval, _tmem_load_11[9], _tmem_load_10[9]);
                    unsigned int w_41[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_41[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 10) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 10) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_41[0] << 16);
                    _tmem_load_10[10] = __fmaf_rn(qval, _tmem_load_11[10], _tmem_load_10[10]);
                    unsigned int w_42[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_42[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 11) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 11) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_42[0] << 16);
                    _tmem_load_10[11] = __fmaf_rn(qval, _tmem_load_11[11], _tmem_load_10[11]);
                    unsigned int w_43[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_43[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 12) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 12) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_43[0] << 16);
                    _tmem_load_10[12] = __fmaf_rn(qval, _tmem_load_11[12], _tmem_load_10[12]);
                    unsigned int w_44[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_44[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 13) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 13) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_44[0] << 16);
                    _tmem_load_10[13] = __fmaf_rn(qval, _tmem_load_11[13], _tmem_load_10[13]);
                    unsigned int w_45[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_45[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 14) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 14) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_45[0] << 16);
                    _tmem_load_10[14] = __fmaf_rn(qval, _tmem_load_11[14], _tmem_load_10[14]);
                    unsigned int w_46[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_46[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 15) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 15) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_46[0] << 16);
                    _tmem_load_10[15] = __fmaf_rn(qval, _tmem_load_11[15], _tmem_load_10[15]);
                    unsigned int w_47[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_47[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 16) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 16) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_47[0] << 16);
                    _tmem_load_10[16] = __fmaf_rn(qval, _tmem_load_11[16], _tmem_load_10[16]);
                    unsigned int w_48[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_48[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 17) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 17) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_48[0] << 16);
                    _tmem_load_10[17] = __fmaf_rn(qval, _tmem_load_11[17], _tmem_load_10[17]);
                    unsigned int w_49[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_49[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 18) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 18) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_49[0] << 16);
                    _tmem_load_10[18] = __fmaf_rn(qval, _tmem_load_11[18], _tmem_load_10[18]);
                    unsigned int w_50[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_50[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 19) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 19) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_50[0] << 16);
                    _tmem_load_10[19] = __fmaf_rn(qval, _tmem_load_11[19], _tmem_load_10[19]);
                    unsigned int w_51[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_51[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 20) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 20) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_51[0] << 16);
                    _tmem_load_10[20] = __fmaf_rn(qval, _tmem_load_11[20], _tmem_load_10[20]);
                    unsigned int w_52[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_52[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 21) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 21) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_52[0] << 16);
                    _tmem_load_10[21] = __fmaf_rn(qval, _tmem_load_11[21], _tmem_load_10[21]);
                    unsigned int w_53[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_53[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 22) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 22) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_53[0] << 16);
                    _tmem_load_10[22] = __fmaf_rn(qval, _tmem_load_11[22], _tmem_load_10[22]);
                    unsigned int w_54[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_54[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 23) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 23) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_54[0] << 16);
                    _tmem_load_10[23] = __fmaf_rn(qval, _tmem_load_11[23], _tmem_load_10[23]);
                    unsigned int w_55[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_55[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 24) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 24) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_55[0] << 16);
                    _tmem_load_10[24] = __fmaf_rn(qval, _tmem_load_11[24], _tmem_load_10[24]);
                    unsigned int w_56[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_56[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 25) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 25) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_56[0] << 16);
                    _tmem_load_10[25] = __fmaf_rn(qval, _tmem_load_11[25], _tmem_load_10[25]);
                    unsigned int w_57[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_57[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 26) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 26) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_57[0] << 16);
                    _tmem_load_10[26] = __fmaf_rn(qval, _tmem_load_11[26], _tmem_load_10[26]);
                    unsigned int w_58[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_58[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 27) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 27) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_58[0] << 16);
                    _tmem_load_10[27] = __fmaf_rn(qval, _tmem_load_11[27], _tmem_load_10[27]);
                    unsigned int w_59[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_59[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 28) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 28) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_59[0] << 16);
                    _tmem_load_10[28] = __fmaf_rn(qval, _tmem_load_11[28], _tmem_load_10[28]);
                    unsigned int w_60[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_60[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 29) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 29) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_60[0] << 16);
                    _tmem_load_10[29] = __fmaf_rn(qval, _tmem_load_11[29], _tmem_load_10[29]);
                    unsigned int w_61[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_61[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 30) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 30) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_61[0] << 16);
                    _tmem_load_10[30] = __fmaf_rn(qval, _tmem_load_11[30], _tmem_load_10[30]);
                    unsigned int w_62[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w_62[0])) : "r"((smem_v6_addr + (unsigned int)(d_row / 64 * 8192 + (t0 + 31) * 128 + d_row % 64 * 2 ^ (d_row / 64 * 8192 + (t0 + 31) * 128 + d_row % 64 * 2 >> 7 & 7) << 4))));
                    qval = __uint_as_float(w_62[0] << 16);
                    _tmem_load_10[31] = __fmaf_rn(qval, _tmem_load_11[31], _tmem_load_10[31]);
                    if (group_reduce != 0) {
                        if (hl == 0) {
                            tmem_st_x32_f32(taddr + (unsigned int)row_base + (unsigned int)(ch * 32), _tmem_load_11);
                        } else {
                            float _tmem_load_12[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_12[0]), "=f"(_tmem_load_12[1]), "=f"(_tmem_load_12[2]), "=f"(_tmem_load_12[3]), "=f"(_tmem_load_12[4]), "=f"(_tmem_load_12[5]), "=f"(_tmem_load_12[6]), "=f"(_tmem_load_12[7]), "=f"(_tmem_load_12[8]), "=f"(_tmem_load_12[9]), "=f"(_tmem_load_12[10]), "=f"(_tmem_load_12[11]), "=f"(_tmem_load_12[12]), "=f"(_tmem_load_12[13]), "=f"(_tmem_load_12[14]), "=f"(_tmem_load_12[15]), "=f"(_tmem_load_12[16]), "=f"(_tmem_load_12[17]), "=f"(_tmem_load_12[18]), "=f"(_tmem_load_12[19]), "=f"(_tmem_load_12[20]), "=f"(_tmem_load_12[21]), "=f"(_tmem_load_12[22]), "=f"(_tmem_load_12[23]), "=f"(_tmem_load_12[24]), "=f"(_tmem_load_12[25]), "=f"(_tmem_load_12[26]), "=f"(_tmem_load_12[27]), "=f"(_tmem_load_12[28]), "=f"(_tmem_load_12[29]), "=f"(_tmem_load_12[30]), "=f"(_tmem_load_12[31])
                                : "r"(taddr + (unsigned int)row_base + (unsigned int)(ch * 32)));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            _tmem_load_12[0] = _tmem_load_12[0] + _tmem_load_11[0];
                            _tmem_load_12[1] = _tmem_load_12[1] + _tmem_load_11[1];
                            _tmem_load_12[2] = _tmem_load_12[2] + _tmem_load_11[2];
                            _tmem_load_12[3] = _tmem_load_12[3] + _tmem_load_11[3];
                            _tmem_load_12[4] = _tmem_load_12[4] + _tmem_load_11[4];
                            _tmem_load_12[5] = _tmem_load_12[5] + _tmem_load_11[5];
                            _tmem_load_12[6] = _tmem_load_12[6] + _tmem_load_11[6];
                            _tmem_load_12[7] = _tmem_load_12[7] + _tmem_load_11[7];
                            _tmem_load_12[8] = _tmem_load_12[8] + _tmem_load_11[8];
                            _tmem_load_12[9] = _tmem_load_12[9] + _tmem_load_11[9];
                            _tmem_load_12[10] = _tmem_load_12[10] + _tmem_load_11[10];
                            _tmem_load_12[11] = _tmem_load_12[11] + _tmem_load_11[11];
                            _tmem_load_12[12] = _tmem_load_12[12] + _tmem_load_11[12];
                            _tmem_load_12[13] = _tmem_load_12[13] + _tmem_load_11[13];
                            _tmem_load_12[14] = _tmem_load_12[14] + _tmem_load_11[14];
                            _tmem_load_12[15] = _tmem_load_12[15] + _tmem_load_11[15];
                            _tmem_load_12[16] = _tmem_load_12[16] + _tmem_load_11[16];
                            _tmem_load_12[17] = _tmem_load_12[17] + _tmem_load_11[17];
                            _tmem_load_12[18] = _tmem_load_12[18] + _tmem_load_11[18];
                            _tmem_load_12[19] = _tmem_load_12[19] + _tmem_load_11[19];
                            _tmem_load_12[20] = _tmem_load_12[20] + _tmem_load_11[20];
                            _tmem_load_12[21] = _tmem_load_12[21] + _tmem_load_11[21];
                            _tmem_load_12[22] = _tmem_load_12[22] + _tmem_load_11[22];
                            _tmem_load_12[23] = _tmem_load_12[23] + _tmem_load_11[23];
                            _tmem_load_12[24] = _tmem_load_12[24] + _tmem_load_11[24];
                            _tmem_load_12[25] = _tmem_load_12[25] + _tmem_load_11[25];
                            _tmem_load_12[26] = _tmem_load_12[26] + _tmem_load_11[26];
                            _tmem_load_12[27] = _tmem_load_12[27] + _tmem_load_11[27];
                            _tmem_load_12[28] = _tmem_load_12[28] + _tmem_load_11[28];
                            _tmem_load_12[29] = _tmem_load_12[29] + _tmem_load_11[29];
                            _tmem_load_12[30] = _tmem_load_12[30] + _tmem_load_11[30];
                            _tmem_load_12[31] = _tmem_load_12[31] + _tmem_load_11[31];
                            tmem_st_x32_f32(taddr + (unsigned int)row_base + (unsigned int)(ch * 32), _tmem_load_12);
                        }
                    } else {
                        long long pbase = ((long long)trow * nh64 + hv64) * 128 + (long long)d_row;
                        if (clen > t0) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)(t0 * num_heads * 128))) + (0)) = _tmem_load_11[0];
                        }
                        if (clen > t0 + 1) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 1) * num_heads * 128))) + (0)) = _tmem_load_11[1];
                        }
                        if (clen > t0 + 2) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 2) * num_heads * 128))) + (0)) = _tmem_load_11[2];
                        }
                        if (clen > t0 + 3) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 3) * num_heads * 128))) + (0)) = _tmem_load_11[3];
                        }
                        if (clen > t0 + 4) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 4) * num_heads * 128))) + (0)) = _tmem_load_11[4];
                        }
                        if (clen > t0 + 5) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 5) * num_heads * 128))) + (0)) = _tmem_load_11[5];
                        }
                        if (clen > t0 + 6) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 6) * num_heads * 128))) + (0)) = _tmem_load_11[6];
                        }
                        if (clen > t0 + 7) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 7) * num_heads * 128))) + (0)) = _tmem_load_11[7];
                        }
                        if (clen > t0 + 8) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 8) * num_heads * 128))) + (0)) = _tmem_load_11[8];
                        }
                        if (clen > t0 + 9) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 9) * num_heads * 128))) + (0)) = _tmem_load_11[9];
                        }
                        if (clen > t0 + 10) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 10) * num_heads * 128))) + (0)) = _tmem_load_11[10];
                        }
                        if (clen > t0 + 11) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 11) * num_heads * 128))) + (0)) = _tmem_load_11[11];
                        }
                        if (clen > t0 + 12) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 12) * num_heads * 128))) + (0)) = _tmem_load_11[12];
                        }
                        if (clen > t0 + 13) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 13) * num_heads * 128))) + (0)) = _tmem_load_11[13];
                        }
                        if (clen > t0 + 14) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 14) * num_heads * 128))) + (0)) = _tmem_load_11[14];
                        }
                        if (clen > t0 + 15) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 15) * num_heads * 128))) + (0)) = _tmem_load_11[15];
                        }
                        if (clen > t0 + 16) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 16) * num_heads * 128))) + (0)) = _tmem_load_11[16];
                        }
                        if (clen > t0 + 17) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 17) * num_heads * 128))) + (0)) = _tmem_load_11[17];
                        }
                        if (clen > t0 + 18) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 18) * num_heads * 128))) + (0)) = _tmem_load_11[18];
                        }
                        if (clen > t0 + 19) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 19) * num_heads * 128))) + (0)) = _tmem_load_11[19];
                        }
                        if (clen > t0 + 20) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 20) * num_heads * 128))) + (0)) = _tmem_load_11[20];
                        }
                        if (clen > t0 + 21) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 21) * num_heads * 128))) + (0)) = _tmem_load_11[21];
                        }
                        if (clen > t0 + 22) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 22) * num_heads * 128))) + (0)) = _tmem_load_11[22];
                        }
                        if (clen > t0 + 23) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 23) * num_heads * 128))) + (0)) = _tmem_load_11[23];
                        }
                        if (clen > t0 + 24) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 24) * num_heads * 128))) + (0)) = _tmem_load_11[24];
                        }
                        if (clen > t0 + 25) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 25) * num_heads * 128))) + (0)) = _tmem_load_11[25];
                        }
                        if (clen > t0 + 26) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 26) * num_heads * 128))) + (0)) = _tmem_load_11[26];
                        }
                        if (clen > t0 + 27) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 27) * num_heads * 128))) + (0)) = _tmem_load_11[27];
                        }
                        if (clen > t0 + 28) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 28) * num_heads * 128))) + (0)) = _tmem_load_11[28];
                        }
                        if (clen > t0 + 29) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 29) * num_heads * 128))) + (0)) = _tmem_load_11[29];
                        }
                        if (clen > t0 + 30) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 30) * num_heads * 128))) + (0)) = _tmem_load_11[30];
                        }
                        if (clen > t0 + 31) {
                            *(reinterpret_cast<float*>(dq_part + (pbase + (long long)((t0 + 31) * num_heads * 128))) + (0)) = _tmem_load_11[31];
                        }
                    }
                    mbarrier_wait(e_ready_addr, _phase_e_ready_0);
                    _phase_e_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_13[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_13[0]), "=f"(_tmem_load_13[1]), "=f"(_tmem_load_13[2]), "=f"(_tmem_load_13[3]), "=f"(_tmem_load_13[4]), "=f"(_tmem_load_13[5]), "=f"(_tmem_load_13[6]), "=f"(_tmem_load_13[7]), "=f"(_tmem_load_13[8]), "=f"(_tmem_load_13[9]), "=f"(_tmem_load_13[10]), "=f"(_tmem_load_13[11]), "=f"(_tmem_load_13[12]), "=f"(_tmem_load_13[13]), "=f"(_tmem_load_13[14]), "=f"(_tmem_load_13[15]), "=f"(_tmem_load_13[16]), "=f"(_tmem_load_13[17]), "=f"(_tmem_load_13[18]), "=f"(_tmem_load_13[19]), "=f"(_tmem_load_13[20]), "=f"(_tmem_load_13[21]), "=f"(_tmem_load_13[22]), "=f"(_tmem_load_13[23]), "=f"(_tmem_load_13[24]), "=f"(_tmem_load_13[25]), "=f"(_tmem_load_13[26]), "=f"(_tmem_load_13[27]), "=f"(_tmem_load_13[28]), "=f"(_tmem_load_13[29]), "=f"(_tmem_load_13[30]), "=f"(_tmem_load_13[31])
                        : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    _tmem_load_10[0] = __fmaf_rn(-__uint_as_float(kpk[0] << 16), _tmem_load_13[0], _tmem_load_10[0]);
                    _tmem_load_10[1] = __fmaf_rn(-__uint_as_float(kpk[0] & 4294901760u), _tmem_load_13[1], _tmem_load_10[1]);
                    _tmem_load_10[2] = __fmaf_rn(-__uint_as_float(kpk[1] << 16), _tmem_load_13[2], _tmem_load_10[2]);
                    _tmem_load_10[3] = __fmaf_rn(-__uint_as_float(kpk[1] & 4294901760u), _tmem_load_13[3], _tmem_load_10[3]);
                    _tmem_load_10[4] = __fmaf_rn(-__uint_as_float(kpk[2] << 16), _tmem_load_13[4], _tmem_load_10[4]);
                    _tmem_load_10[5] = __fmaf_rn(-__uint_as_float(kpk[2] & 4294901760u), _tmem_load_13[5], _tmem_load_10[5]);
                    _tmem_load_10[6] = __fmaf_rn(-__uint_as_float(kpk[3] << 16), _tmem_load_13[6], _tmem_load_10[6]);
                    _tmem_load_10[7] = __fmaf_rn(-__uint_as_float(kpk[3] & 4294901760u), _tmem_load_13[7], _tmem_load_10[7]);
                    _tmem_load_10[8] = __fmaf_rn(-__uint_as_float(kpk[4] << 16), _tmem_load_13[8], _tmem_load_10[8]);
                    _tmem_load_10[9] = __fmaf_rn(-__uint_as_float(kpk[4] & 4294901760u), _tmem_load_13[9], _tmem_load_10[9]);
                    _tmem_load_10[10] = __fmaf_rn(-__uint_as_float(kpk[5] << 16), _tmem_load_13[10], _tmem_load_10[10]);
                    _tmem_load_10[11] = __fmaf_rn(-__uint_as_float(kpk[5] & 4294901760u), _tmem_load_13[11], _tmem_load_10[11]);
                    _tmem_load_10[12] = __fmaf_rn(-__uint_as_float(kpk[6] << 16), _tmem_load_13[12], _tmem_load_10[12]);
                    _tmem_load_10[13] = __fmaf_rn(-__uint_as_float(kpk[6] & 4294901760u), _tmem_load_13[13], _tmem_load_10[13]);
                    _tmem_load_10[14] = __fmaf_rn(-__uint_as_float(kpk[7] << 16), _tmem_load_13[14], _tmem_load_10[14]);
                    _tmem_load_10[15] = __fmaf_rn(-__uint_as_float(kpk[7] & 4294901760u), _tmem_load_13[15], _tmem_load_10[15]);
                    _tmem_load_10[16] = __fmaf_rn(-__uint_as_float(kpk[8] << 16), _tmem_load_13[16], _tmem_load_10[16]);
                    _tmem_load_10[17] = __fmaf_rn(-__uint_as_float(kpk[8] & 4294901760u), _tmem_load_13[17], _tmem_load_10[17]);
                    _tmem_load_10[18] = __fmaf_rn(-__uint_as_float(kpk[9] << 16), _tmem_load_13[18], _tmem_load_10[18]);
                    _tmem_load_10[19] = __fmaf_rn(-__uint_as_float(kpk[9] & 4294901760u), _tmem_load_13[19], _tmem_load_10[19]);
                    _tmem_load_10[20] = __fmaf_rn(-__uint_as_float(kpk[10] << 16), _tmem_load_13[20], _tmem_load_10[20]);
                    _tmem_load_10[21] = __fmaf_rn(-__uint_as_float(kpk[10] & 4294901760u), _tmem_load_13[21], _tmem_load_10[21]);
                    _tmem_load_10[22] = __fmaf_rn(-__uint_as_float(kpk[11] << 16), _tmem_load_13[22], _tmem_load_10[22]);
                    _tmem_load_10[23] = __fmaf_rn(-__uint_as_float(kpk[11] & 4294901760u), _tmem_load_13[23], _tmem_load_10[23]);
                    _tmem_load_10[24] = __fmaf_rn(-__uint_as_float(kpk[12] << 16), _tmem_load_13[24], _tmem_load_10[24]);
                    _tmem_load_10[25] = __fmaf_rn(-__uint_as_float(kpk[12] & 4294901760u), _tmem_load_13[25], _tmem_load_10[25]);
                    _tmem_load_10[26] = __fmaf_rn(-__uint_as_float(kpk[13] << 16), _tmem_load_13[26], _tmem_load_10[26]);
                    _tmem_load_10[27] = __fmaf_rn(-__uint_as_float(kpk[13] & 4294901760u), _tmem_load_13[27], _tmem_load_10[27]);
                    _tmem_load_10[28] = __fmaf_rn(-__uint_as_float(kpk[14] << 16), _tmem_load_13[28], _tmem_load_10[28]);
                    _tmem_load_10[29] = __fmaf_rn(-__uint_as_float(kpk[14] & 4294901760u), _tmem_load_13[29], _tmem_load_10[29]);
                    _tmem_load_10[30] = __fmaf_rn(-__uint_as_float(kpk[15] << 16), _tmem_load_13[30], _tmem_load_10[30]);
                    _tmem_load_10[31] = __fmaf_rn(-__uint_as_float(kpk[15] & 4294901760u), _tmem_load_13[31], _tmem_load_10[31]);
                    float w16_1_1[16];
                    float _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[16] : _tmem_load_10[0]), 16);
                    w16_1_1[0] = (((lane_0 & 16) == 0) ? _tmem_load_10[0] : _tmem_load_10[16]) + _shfl_xor_sync_139;
                    float _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[17] : _tmem_load_10[1]), 16);
                    w16_1_1[1] = (((lane_0 & 16) == 0) ? _tmem_load_10[1] : _tmem_load_10[17]) + _shfl_xor_sync_140;
                    float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[18] : _tmem_load_10[2]), 16);
                    w16_1_1[2] = (((lane_0 & 16) == 0) ? _tmem_load_10[2] : _tmem_load_10[18]) + _shfl_xor_sync_141;
                    float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[19] : _tmem_load_10[3]), 16);
                    w16_1_1[3] = (((lane_0 & 16) == 0) ? _tmem_load_10[3] : _tmem_load_10[19]) + _shfl_xor_sync_142;
                    float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[20] : _tmem_load_10[4]), 16);
                    w16_1_1[4] = (((lane_0 & 16) == 0) ? _tmem_load_10[4] : _tmem_load_10[20]) + _shfl_xor_sync_143;
                    float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[21] : _tmem_load_10[5]), 16);
                    w16_1_1[5] = (((lane_0 & 16) == 0) ? _tmem_load_10[5] : _tmem_load_10[21]) + _shfl_xor_sync_144;
                    float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[22] : _tmem_load_10[6]), 16);
                    w16_1_1[6] = (((lane_0 & 16) == 0) ? _tmem_load_10[6] : _tmem_load_10[22]) + _shfl_xor_sync_145;
                    float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[23] : _tmem_load_10[7]), 16);
                    w16_1_1[7] = (((lane_0 & 16) == 0) ? _tmem_load_10[7] : _tmem_load_10[23]) + _shfl_xor_sync_146;
                    float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[24] : _tmem_load_10[8]), 16);
                    w16_1_1[8] = (((lane_0 & 16) == 0) ? _tmem_load_10[8] : _tmem_load_10[24]) + _shfl_xor_sync_147;
                    float _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[25] : _tmem_load_10[9]), 16);
                    w16_1_1[9] = (((lane_0 & 16) == 0) ? _tmem_load_10[9] : _tmem_load_10[25]) + _shfl_xor_sync_148;
                    float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[26] : _tmem_load_10[10]), 16);
                    w16_1_1[10] = (((lane_0 & 16) == 0) ? _tmem_load_10[10] : _tmem_load_10[26]) + _shfl_xor_sync_149;
                    float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[27] : _tmem_load_10[11]), 16);
                    w16_1_1[11] = (((lane_0 & 16) == 0) ? _tmem_load_10[11] : _tmem_load_10[27]) + _shfl_xor_sync_150;
                    float _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[28] : _tmem_load_10[12]), 16);
                    w16_1_1[12] = (((lane_0 & 16) == 0) ? _tmem_load_10[12] : _tmem_load_10[28]) + _shfl_xor_sync_151;
                    float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[29] : _tmem_load_10[13]), 16);
                    w16_1_1[13] = (((lane_0 & 16) == 0) ? _tmem_load_10[13] : _tmem_load_10[29]) + _shfl_xor_sync_152;
                    float _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[30] : _tmem_load_10[14]), 16);
                    w16_1_1[14] = (((lane_0 & 16) == 0) ? _tmem_load_10[14] : _tmem_load_10[30]) + _shfl_xor_sync_153;
                    float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? _tmem_load_10[31] : _tmem_load_10[15]), 16);
                    w16_1_1[15] = (((lane_0 & 16) == 0) ? _tmem_load_10[15] : _tmem_load_10[31]) + _shfl_xor_sync_154;
                    float w8_1_1[8];
                    float _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1_1[8] : w16_1_1[0]), 8);
                    w8_1_1[0] = (((lane_0 & 8) == 0) ? w16_1_1[0] : w16_1_1[8]) + _shfl_xor_sync_155;
                    float _shfl_xor_sync_156 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1_1[9] : w16_1_1[1]), 8);
                    w8_1_1[1] = (((lane_0 & 8) == 0) ? w16_1_1[1] : w16_1_1[9]) + _shfl_xor_sync_156;
                    float _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1_1[10] : w16_1_1[2]), 8);
                    w8_1_1[2] = (((lane_0 & 8) == 0) ? w16_1_1[2] : w16_1_1[10]) + _shfl_xor_sync_157;
                    float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1_1[11] : w16_1_1[3]), 8);
                    w8_1_1[3] = (((lane_0 & 8) == 0) ? w16_1_1[3] : w16_1_1[11]) + _shfl_xor_sync_158;
                    float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1_1[12] : w16_1_1[4]), 8);
                    w8_1_1[4] = (((lane_0 & 8) == 0) ? w16_1_1[4] : w16_1_1[12]) + _shfl_xor_sync_159;
                    float _shfl_xor_sync_160 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1_1[13] : w16_1_1[5]), 8);
                    w8_1_1[5] = (((lane_0 & 8) == 0) ? w16_1_1[5] : w16_1_1[13]) + _shfl_xor_sync_160;
                    float _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1_1[14] : w16_1_1[6]), 8);
                    w8_1_1[6] = (((lane_0 & 8) == 0) ? w16_1_1[6] : w16_1_1[14]) + _shfl_xor_sync_161;
                    float _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_1_1[15] : w16_1_1[7]), 8);
                    w8_1_1[7] = (((lane_0 & 8) == 0) ? w16_1_1[7] : w16_1_1[15]) + _shfl_xor_sync_162;
                    float w4_1_1[4];
                    float _shfl_xor_sync_163 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_1_1[4] : w8_1_1[0]), 4);
                    w4_1_1[0] = (((lane_0 & 4) == 0) ? w8_1_1[0] : w8_1_1[4]) + _shfl_xor_sync_163;
                    float _shfl_xor_sync_164 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_1_1[5] : w8_1_1[1]), 4);
                    w4_1_1[1] = (((lane_0 & 4) == 0) ? w8_1_1[1] : w8_1_1[5]) + _shfl_xor_sync_164;
                    float _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_1_1[6] : w8_1_1[2]), 4);
                    w4_1_1[2] = (((lane_0 & 4) == 0) ? w8_1_1[2] : w8_1_1[6]) + _shfl_xor_sync_165;
                    float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_1_1[7] : w8_1_1[3]), 4);
                    w4_1_1[3] = (((lane_0 & 4) == 0) ? w8_1_1[3] : w8_1_1[7]) + _shfl_xor_sync_166;
                    float w2_1_1[2];
                    float _shfl_xor_sync_167 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) == 0) ? w4_1_1[2] : w4_1_1[0]), 2);
                    w2_1_1[0] = (((lane_0 & 2) == 0) ? w4_1_1[0] : w4_1_1[2]) + _shfl_xor_sync_167;
                    float _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) == 0) ? w4_1_1[3] : w4_1_1[1]), 2);
                    w2_1_1[1] = (((lane_0 & 2) == 0) ? w4_1_1[1] : w4_1_1[3]) + _shfl_xor_sync_168;
                    float _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 1) == 0) ? w2_1_1[1] : w2_1_1[0]), 1);
                    tmp = (((lane_0 & 1) == 0) ? w2_1_1[0] : w2_1_1[1]) + _shfl_xor_sync_169;
                    dg_acc = dg_acc + tmp;
                    float _tmem_load_14[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_14[0]), "=f"(_tmem_load_14[1]), "=f"(_tmem_load_14[2]), "=f"(_tmem_load_14[3]), "=f"(_tmem_load_14[4]), "=f"(_tmem_load_14[5]), "=f"(_tmem_load_14[6]), "=f"(_tmem_load_14[7]), "=f"(_tmem_load_14[8]), "=f"(_tmem_load_14[9]), "=f"(_tmem_load_14[10]), "=f"(_tmem_load_14[11]), "=f"(_tmem_load_14[12]), "=f"(_tmem_load_14[13]), "=f"(_tmem_load_14[14]), "=f"(_tmem_load_14[15]), "=f"(_tmem_load_14[16]), "=f"(_tmem_load_14[17]), "=f"(_tmem_load_14[18]), "=f"(_tmem_load_14[19]), "=f"(_tmem_load_14[20]), "=f"(_tmem_load_14[21]), "=f"(_tmem_load_14[22]), "=f"(_tmem_load_14[23]), "=f"(_tmem_load_14[24]), "=f"(_tmem_load_14[25]), "=f"(_tmem_load_14[26]), "=f"(_tmem_load_14[27]), "=f"(_tmem_load_14[28]), "=f"(_tmem_load_14[29]), "=f"(_tmem_load_14[30]), "=f"(_tmem_load_14[31])
                        : "r"(taddr + 384 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float px[32];
                    px[0] = _tmem_load_14[0] * __uint_as_float(kpk[0] << 16);
                    _tmem_load_14[0] = beta_s[t0] * _tmem_load_14[0];
                    px[1] = _tmem_load_14[1] * __uint_as_float(kpk[0] & 4294901760u);
                    _tmem_load_14[1] = beta_s[t0 + 1] * _tmem_load_14[1];
                    px[2] = _tmem_load_14[2] * __uint_as_float(kpk[1] << 16);
                    _tmem_load_14[2] = beta_s[t0 + 2] * _tmem_load_14[2];
                    px[3] = _tmem_load_14[3] * __uint_as_float(kpk[1] & 4294901760u);
                    _tmem_load_14[3] = beta_s[t0 + 3] * _tmem_load_14[3];
                    px[4] = _tmem_load_14[4] * __uint_as_float(kpk[2] << 16);
                    _tmem_load_14[4] = beta_s[t0 + 4] * _tmem_load_14[4];
                    px[5] = _tmem_load_14[5] * __uint_as_float(kpk[2] & 4294901760u);
                    _tmem_load_14[5] = beta_s[t0 + 5] * _tmem_load_14[5];
                    px[6] = _tmem_load_14[6] * __uint_as_float(kpk[3] << 16);
                    _tmem_load_14[6] = beta_s[t0 + 6] * _tmem_load_14[6];
                    px[7] = _tmem_load_14[7] * __uint_as_float(kpk[3] & 4294901760u);
                    _tmem_load_14[7] = beta_s[t0 + 7] * _tmem_load_14[7];
                    px[8] = _tmem_load_14[8] * __uint_as_float(kpk[4] << 16);
                    _tmem_load_14[8] = beta_s[t0 + 8] * _tmem_load_14[8];
                    px[9] = _tmem_load_14[9] * __uint_as_float(kpk[4] & 4294901760u);
                    _tmem_load_14[9] = beta_s[t0 + 9] * _tmem_load_14[9];
                    px[10] = _tmem_load_14[10] * __uint_as_float(kpk[5] << 16);
                    _tmem_load_14[10] = beta_s[t0 + 10] * _tmem_load_14[10];
                    px[11] = _tmem_load_14[11] * __uint_as_float(kpk[5] & 4294901760u);
                    _tmem_load_14[11] = beta_s[t0 + 11] * _tmem_load_14[11];
                    px[12] = _tmem_load_14[12] * __uint_as_float(kpk[6] << 16);
                    _tmem_load_14[12] = beta_s[t0 + 12] * _tmem_load_14[12];
                    px[13] = _tmem_load_14[13] * __uint_as_float(kpk[6] & 4294901760u);
                    _tmem_load_14[13] = beta_s[t0 + 13] * _tmem_load_14[13];
                    px[14] = _tmem_load_14[14] * __uint_as_float(kpk[7] << 16);
                    _tmem_load_14[14] = beta_s[t0 + 14] * _tmem_load_14[14];
                    px[15] = _tmem_load_14[15] * __uint_as_float(kpk[7] & 4294901760u);
                    _tmem_load_14[15] = beta_s[t0 + 15] * _tmem_load_14[15];
                    px[16] = _tmem_load_14[16] * __uint_as_float(kpk[8] << 16);
                    _tmem_load_14[16] = beta_s[t0 + 16] * _tmem_load_14[16];
                    px[17] = _tmem_load_14[17] * __uint_as_float(kpk[8] & 4294901760u);
                    _tmem_load_14[17] = beta_s[t0 + 17] * _tmem_load_14[17];
                    px[18] = _tmem_load_14[18] * __uint_as_float(kpk[9] << 16);
                    _tmem_load_14[18] = beta_s[t0 + 18] * _tmem_load_14[18];
                    px[19] = _tmem_load_14[19] * __uint_as_float(kpk[9] & 4294901760u);
                    _tmem_load_14[19] = beta_s[t0 + 19] * _tmem_load_14[19];
                    px[20] = _tmem_load_14[20] * __uint_as_float(kpk[10] << 16);
                    _tmem_load_14[20] = beta_s[t0 + 20] * _tmem_load_14[20];
                    px[21] = _tmem_load_14[21] * __uint_as_float(kpk[10] & 4294901760u);
                    _tmem_load_14[21] = beta_s[t0 + 21] * _tmem_load_14[21];
                    px[22] = _tmem_load_14[22] * __uint_as_float(kpk[11] << 16);
                    _tmem_load_14[22] = beta_s[t0 + 22] * _tmem_load_14[22];
                    px[23] = _tmem_load_14[23] * __uint_as_float(kpk[11] & 4294901760u);
                    _tmem_load_14[23] = beta_s[t0 + 23] * _tmem_load_14[23];
                    px[24] = _tmem_load_14[24] * __uint_as_float(kpk[12] << 16);
                    _tmem_load_14[24] = beta_s[t0 + 24] * _tmem_load_14[24];
                    px[25] = _tmem_load_14[25] * __uint_as_float(kpk[12] & 4294901760u);
                    _tmem_load_14[25] = beta_s[t0 + 25] * _tmem_load_14[25];
                    px[26] = _tmem_load_14[26] * __uint_as_float(kpk[13] << 16);
                    _tmem_load_14[26] = beta_s[t0 + 26] * _tmem_load_14[26];
                    px[27] = _tmem_load_14[27] * __uint_as_float(kpk[13] & 4294901760u);
                    _tmem_load_14[27] = beta_s[t0 + 27] * _tmem_load_14[27];
                    px[28] = _tmem_load_14[28] * __uint_as_float(kpk[14] << 16);
                    _tmem_load_14[28] = beta_s[t0 + 28] * _tmem_load_14[28];
                    px[29] = _tmem_load_14[29] * __uint_as_float(kpk[14] & 4294901760u);
                    _tmem_load_14[29] = beta_s[t0 + 29] * _tmem_load_14[29];
                    px[30] = _tmem_load_14[30] * __uint_as_float(kpk[15] << 16);
                    _tmem_load_14[30] = beta_s[t0 + 30] * _tmem_load_14[30];
                    px[31] = _tmem_load_14[31] * __uint_as_float(kpk[15] & 4294901760u);
                    _tmem_load_14[31] = beta_s[t0 + 31] * _tmem_load_14[31];
                    float w16_2[16];
                    float _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[16] : px[0]), 16);
                    w16_2[0] = (((lane_0 & 16) == 0) ? px[0] : px[16]) + _shfl_xor_sync_170;
                    float _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[17] : px[1]), 16);
                    w16_2[1] = (((lane_0 & 16) == 0) ? px[1] : px[17]) + _shfl_xor_sync_171;
                    float _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[18] : px[2]), 16);
                    w16_2[2] = (((lane_0 & 16) == 0) ? px[2] : px[18]) + _shfl_xor_sync_172;
                    float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[19] : px[3]), 16);
                    w16_2[3] = (((lane_0 & 16) == 0) ? px[3] : px[19]) + _shfl_xor_sync_173;
                    float _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[20] : px[4]), 16);
                    w16_2[4] = (((lane_0 & 16) == 0) ? px[4] : px[20]) + _shfl_xor_sync_174;
                    float _shfl_xor_sync_175 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[21] : px[5]), 16);
                    w16_2[5] = (((lane_0 & 16) == 0) ? px[5] : px[21]) + _shfl_xor_sync_175;
                    float _shfl_xor_sync_176 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[22] : px[6]), 16);
                    w16_2[6] = (((lane_0 & 16) == 0) ? px[6] : px[22]) + _shfl_xor_sync_176;
                    float _shfl_xor_sync_177 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[23] : px[7]), 16);
                    w16_2[7] = (((lane_0 & 16) == 0) ? px[7] : px[23]) + _shfl_xor_sync_177;
                    float _shfl_xor_sync_178 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[24] : px[8]), 16);
                    w16_2[8] = (((lane_0 & 16) == 0) ? px[8] : px[24]) + _shfl_xor_sync_178;
                    float _shfl_xor_sync_179 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[25] : px[9]), 16);
                    w16_2[9] = (((lane_0 & 16) == 0) ? px[9] : px[25]) + _shfl_xor_sync_179;
                    float _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[26] : px[10]), 16);
                    w16_2[10] = (((lane_0 & 16) == 0) ? px[10] : px[26]) + _shfl_xor_sync_180;
                    float _shfl_xor_sync_181 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[27] : px[11]), 16);
                    w16_2[11] = (((lane_0 & 16) == 0) ? px[11] : px[27]) + _shfl_xor_sync_181;
                    float _shfl_xor_sync_182 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[28] : px[12]), 16);
                    w16_2[12] = (((lane_0 & 16) == 0) ? px[12] : px[28]) + _shfl_xor_sync_182;
                    float _shfl_xor_sync_183 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[29] : px[13]), 16);
                    w16_2[13] = (((lane_0 & 16) == 0) ? px[13] : px[29]) + _shfl_xor_sync_183;
                    float _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[30] : px[14]), 16);
                    w16_2[14] = (((lane_0 & 16) == 0) ? px[14] : px[30]) + _shfl_xor_sync_184;
                    float _shfl_xor_sync_185 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 16) == 0) ? px[31] : px[15]), 16);
                    w16_2[15] = (((lane_0 & 16) == 0) ? px[15] : px[31]) + _shfl_xor_sync_185;
                    float w8_2[8];
                    float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_2[8] : w16_2[0]), 8);
                    w8_2[0] = (((lane_0 & 8) == 0) ? w16_2[0] : w16_2[8]) + _shfl_xor_sync_186;
                    float _shfl_xor_sync_187 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_2[9] : w16_2[1]), 8);
                    w8_2[1] = (((lane_0 & 8) == 0) ? w16_2[1] : w16_2[9]) + _shfl_xor_sync_187;
                    float _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_2[10] : w16_2[2]), 8);
                    w8_2[2] = (((lane_0 & 8) == 0) ? w16_2[2] : w16_2[10]) + _shfl_xor_sync_188;
                    float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_2[11] : w16_2[3]), 8);
                    w8_2[3] = (((lane_0 & 8) == 0) ? w16_2[3] : w16_2[11]) + _shfl_xor_sync_189;
                    float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_2[12] : w16_2[4]), 8);
                    w8_2[4] = (((lane_0 & 8) == 0) ? w16_2[4] : w16_2[12]) + _shfl_xor_sync_190;
                    float _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_2[13] : w16_2[5]), 8);
                    w8_2[5] = (((lane_0 & 8) == 0) ? w16_2[5] : w16_2[13]) + _shfl_xor_sync_191;
                    float _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_2[14] : w16_2[6]), 8);
                    w8_2[6] = (((lane_0 & 8) == 0) ? w16_2[6] : w16_2[14]) + _shfl_xor_sync_192;
                    float _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 8) == 0) ? w16_2[15] : w16_2[7]), 8);
                    w8_2[7] = (((lane_0 & 8) == 0) ? w16_2[7] : w16_2[15]) + _shfl_xor_sync_193;
                    float w4_2[4];
                    float _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_2[4] : w8_2[0]), 4);
                    w4_2[0] = (((lane_0 & 4) == 0) ? w8_2[0] : w8_2[4]) + _shfl_xor_sync_194;
                    float _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_2[5] : w8_2[1]), 4);
                    w4_2[1] = (((lane_0 & 4) == 0) ? w8_2[1] : w8_2[5]) + _shfl_xor_sync_195;
                    float _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_2[6] : w8_2[2]), 4);
                    w4_2[2] = (((lane_0 & 4) == 0) ? w8_2[2] : w8_2[6]) + _shfl_xor_sync_196;
                    float _shfl_xor_sync_197 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 4) == 0) ? w8_2[7] : w8_2[3]), 4);
                    w4_2[3] = (((lane_0 & 4) == 0) ? w8_2[3] : w8_2[7]) + _shfl_xor_sync_197;
                    float w2_2[2];
                    float _shfl_xor_sync_198 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) == 0) ? w4_2[2] : w4_2[0]), 2);
                    w2_2[0] = (((lane_0 & 2) == 0) ? w4_2[0] : w4_2[2]) + _shfl_xor_sync_198;
                    float _shfl_xor_sync_199 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 2) == 0) ? w4_2[3] : w4_2[1]), 2);
                    w2_2[1] = (((lane_0 & 2) == 0) ? w4_2[1] : w4_2[3]) + _shfl_xor_sync_199;
                    float _shfl_xor_sync_200 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0 & 1) == 0) ? w2_2[1] : w2_2[0]), 1);
                    tmp = (((lane_0 & 1) == 0) ? w2_2[0] : w2_2[1]) + _shfl_xor_sync_200;
                    db_acc = db_acc + tmp;
                    dg_acc = __fmaf_rn(beta_s[t0 + lane_0], tmp, dg_acc);
                    float _tmem_load_15[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_15[0]), "=f"(_tmem_load_15[1]), "=f"(_tmem_load_15[2]), "=f"(_tmem_load_15[3]), "=f"(_tmem_load_15[4]), "=f"(_tmem_load_15[5]), "=f"(_tmem_load_15[6]), "=f"(_tmem_load_15[7]), "=f"(_tmem_load_15[8]), "=f"(_tmem_load_15[9]), "=f"(_tmem_load_15[10]), "=f"(_tmem_load_15[11]), "=f"(_tmem_load_15[12]), "=f"(_tmem_load_15[13]), "=f"(_tmem_load_15[14]), "=f"(_tmem_load_15[15]), "=f"(_tmem_load_15[16]), "=f"(_tmem_load_15[17]), "=f"(_tmem_load_15[18]), "=f"(_tmem_load_15[19]), "=f"(_tmem_load_15[20]), "=f"(_tmem_load_15[21]), "=f"(_tmem_load_15[22]), "=f"(_tmem_load_15[23]), "=f"(_tmem_load_15[24]), "=f"(_tmem_load_15[25]), "=f"(_tmem_load_15[26]), "=f"(_tmem_load_15[27]), "=f"(_tmem_load_15[28]), "=f"(_tmem_load_15[29]), "=f"(_tmem_load_15[30]), "=f"(_tmem_load_15[31])
                        : "r"(taddr + 192 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    _tmem_load_15[0] = _tmem_load_15[0] + _tmem_load_13[0] + _tmem_load_14[0];
                    _tmem_load_15[1] = _tmem_load_15[1] + _tmem_load_13[1] + _tmem_load_14[1];
                    _tmem_load_15[2] = _tmem_load_15[2] + _tmem_load_13[2] + _tmem_load_14[2];
                    _tmem_load_15[3] = _tmem_load_15[3] + _tmem_load_13[3] + _tmem_load_14[3];
                    _tmem_load_15[4] = _tmem_load_15[4] + _tmem_load_13[4] + _tmem_load_14[4];
                    _tmem_load_15[5] = _tmem_load_15[5] + _tmem_load_13[5] + _tmem_load_14[5];
                    _tmem_load_15[6] = _tmem_load_15[6] + _tmem_load_13[6] + _tmem_load_14[6];
                    _tmem_load_15[7] = _tmem_load_15[7] + _tmem_load_13[7] + _tmem_load_14[7];
                    _tmem_load_15[8] = _tmem_load_15[8] + _tmem_load_13[8] + _tmem_load_14[8];
                    _tmem_load_15[9] = _tmem_load_15[9] + _tmem_load_13[9] + _tmem_load_14[9];
                    _tmem_load_15[10] = _tmem_load_15[10] + _tmem_load_13[10] + _tmem_load_14[10];
                    _tmem_load_15[11] = _tmem_load_15[11] + _tmem_load_13[11] + _tmem_load_14[11];
                    _tmem_load_15[12] = _tmem_load_15[12] + _tmem_load_13[12] + _tmem_load_14[12];
                    _tmem_load_15[13] = _tmem_load_15[13] + _tmem_load_13[13] + _tmem_load_14[13];
                    _tmem_load_15[14] = _tmem_load_15[14] + _tmem_load_13[14] + _tmem_load_14[14];
                    _tmem_load_15[15] = _tmem_load_15[15] + _tmem_load_13[15] + _tmem_load_14[15];
                    _tmem_load_15[16] = _tmem_load_15[16] + _tmem_load_13[16] + _tmem_load_14[16];
                    _tmem_load_15[17] = _tmem_load_15[17] + _tmem_load_13[17] + _tmem_load_14[17];
                    _tmem_load_15[18] = _tmem_load_15[18] + _tmem_load_13[18] + _tmem_load_14[18];
                    _tmem_load_15[19] = _tmem_load_15[19] + _tmem_load_13[19] + _tmem_load_14[19];
                    _tmem_load_15[20] = _tmem_load_15[20] + _tmem_load_13[20] + _tmem_load_14[20];
                    _tmem_load_15[21] = _tmem_load_15[21] + _tmem_load_13[21] + _tmem_load_14[21];
                    _tmem_load_15[22] = _tmem_load_15[22] + _tmem_load_13[22] + _tmem_load_14[22];
                    _tmem_load_15[23] = _tmem_load_15[23] + _tmem_load_13[23] + _tmem_load_14[23];
                    _tmem_load_15[24] = _tmem_load_15[24] + _tmem_load_13[24] + _tmem_load_14[24];
                    _tmem_load_15[25] = _tmem_load_15[25] + _tmem_load_13[25] + _tmem_load_14[25];
                    _tmem_load_15[26] = _tmem_load_15[26] + _tmem_load_13[26] + _tmem_load_14[26];
                    _tmem_load_15[27] = _tmem_load_15[27] + _tmem_load_13[27] + _tmem_load_14[27];
                    _tmem_load_15[28] = _tmem_load_15[28] + _tmem_load_13[28] + _tmem_load_14[28];
                    _tmem_load_15[29] = _tmem_load_15[29] + _tmem_load_13[29] + _tmem_load_14[29];
                    _tmem_load_15[30] = _tmem_load_15[30] + _tmem_load_13[30] + _tmem_load_14[30];
                    _tmem_load_15[31] = _tmem_load_15[31] + _tmem_load_13[31] + _tmem_load_14[31];
                    if (group_reduce != 0) {
                        if (hl == 0) {
                            tmem_st_x32_f32(taddr + 64 + (unsigned int)row_base + (unsigned int)(ch * 32), _tmem_load_15);
                        } else {
                            float _tmem_load_16[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_16[0]), "=f"(_tmem_load_16[1]), "=f"(_tmem_load_16[2]), "=f"(_tmem_load_16[3]), "=f"(_tmem_load_16[4]), "=f"(_tmem_load_16[5]), "=f"(_tmem_load_16[6]), "=f"(_tmem_load_16[7]), "=f"(_tmem_load_16[8]), "=f"(_tmem_load_16[9]), "=f"(_tmem_load_16[10]), "=f"(_tmem_load_16[11]), "=f"(_tmem_load_16[12]), "=f"(_tmem_load_16[13]), "=f"(_tmem_load_16[14]), "=f"(_tmem_load_16[15]), "=f"(_tmem_load_16[16]), "=f"(_tmem_load_16[17]), "=f"(_tmem_load_16[18]), "=f"(_tmem_load_16[19]), "=f"(_tmem_load_16[20]), "=f"(_tmem_load_16[21]), "=f"(_tmem_load_16[22]), "=f"(_tmem_load_16[23]), "=f"(_tmem_load_16[24]), "=f"(_tmem_load_16[25]), "=f"(_tmem_load_16[26]), "=f"(_tmem_load_16[27]), "=f"(_tmem_load_16[28]), "=f"(_tmem_load_16[29]), "=f"(_tmem_load_16[30]), "=f"(_tmem_load_16[31])
                                : "r"(taddr + 64 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            _tmem_load_16[0] = _tmem_load_16[0] + _tmem_load_15[0];
                            _tmem_load_16[1] = _tmem_load_16[1] + _tmem_load_15[1];
                            _tmem_load_16[2] = _tmem_load_16[2] + _tmem_load_15[2];
                            _tmem_load_16[3] = _tmem_load_16[3] + _tmem_load_15[3];
                            _tmem_load_16[4] = _tmem_load_16[4] + _tmem_load_15[4];
                            _tmem_load_16[5] = _tmem_load_16[5] + _tmem_load_15[5];
                            _tmem_load_16[6] = _tmem_load_16[6] + _tmem_load_15[6];
                            _tmem_load_16[7] = _tmem_load_16[7] + _tmem_load_15[7];
                            _tmem_load_16[8] = _tmem_load_16[8] + _tmem_load_15[8];
                            _tmem_load_16[9] = _tmem_load_16[9] + _tmem_load_15[9];
                            _tmem_load_16[10] = _tmem_load_16[10] + _tmem_load_15[10];
                            _tmem_load_16[11] = _tmem_load_16[11] + _tmem_load_15[11];
                            _tmem_load_16[12] = _tmem_load_16[12] + _tmem_load_15[12];
                            _tmem_load_16[13] = _tmem_load_16[13] + _tmem_load_15[13];
                            _tmem_load_16[14] = _tmem_load_16[14] + _tmem_load_15[14];
                            _tmem_load_16[15] = _tmem_load_16[15] + _tmem_load_15[15];
                            _tmem_load_16[16] = _tmem_load_16[16] + _tmem_load_15[16];
                            _tmem_load_16[17] = _tmem_load_16[17] + _tmem_load_15[17];
                            _tmem_load_16[18] = _tmem_load_16[18] + _tmem_load_15[18];
                            _tmem_load_16[19] = _tmem_load_16[19] + _tmem_load_15[19];
                            _tmem_load_16[20] = _tmem_load_16[20] + _tmem_load_15[20];
                            _tmem_load_16[21] = _tmem_load_16[21] + _tmem_load_15[21];
                            _tmem_load_16[22] = _tmem_load_16[22] + _tmem_load_15[22];
                            _tmem_load_16[23] = _tmem_load_16[23] + _tmem_load_15[23];
                            _tmem_load_16[24] = _tmem_load_16[24] + _tmem_load_15[24];
                            _tmem_load_16[25] = _tmem_load_16[25] + _tmem_load_15[25];
                            _tmem_load_16[26] = _tmem_load_16[26] + _tmem_load_15[26];
                            _tmem_load_16[27] = _tmem_load_16[27] + _tmem_load_15[27];
                            _tmem_load_16[28] = _tmem_load_16[28] + _tmem_load_15[28];
                            _tmem_load_16[29] = _tmem_load_16[29] + _tmem_load_15[29];
                            _tmem_load_16[30] = _tmem_load_16[30] + _tmem_load_15[30];
                            _tmem_load_16[31] = _tmem_load_16[31] + _tmem_load_15[31];
                            tmem_st_x32_f32(taddr + 64 + (unsigned int)row_base + (unsigned int)(ch * 32), _tmem_load_16);
                        }
                    } else {
                        long long kbase = ((long long)trow * nh64 + hv64) * 128 + (long long)d_row;
                        if (clen > t0) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)(t0 * num_heads * 128))) + (0)) = _tmem_load_15[0];
                        }
                        if (clen > t0 + 1) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 1) * num_heads * 128))) + (0)) = _tmem_load_15[1];
                        }
                        if (clen > t0 + 2) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 2) * num_heads * 128))) + (0)) = _tmem_load_15[2];
                        }
                        if (clen > t0 + 3) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 3) * num_heads * 128))) + (0)) = _tmem_load_15[3];
                        }
                        if (clen > t0 + 4) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 4) * num_heads * 128))) + (0)) = _tmem_load_15[4];
                        }
                        if (clen > t0 + 5) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 5) * num_heads * 128))) + (0)) = _tmem_load_15[5];
                        }
                        if (clen > t0 + 6) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 6) * num_heads * 128))) + (0)) = _tmem_load_15[6];
                        }
                        if (clen > t0 + 7) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 7) * num_heads * 128))) + (0)) = _tmem_load_15[7];
                        }
                        if (clen > t0 + 8) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 8) * num_heads * 128))) + (0)) = _tmem_load_15[8];
                        }
                        if (clen > t0 + 9) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 9) * num_heads * 128))) + (0)) = _tmem_load_15[9];
                        }
                        if (clen > t0 + 10) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 10) * num_heads * 128))) + (0)) = _tmem_load_15[10];
                        }
                        if (clen > t0 + 11) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 11) * num_heads * 128))) + (0)) = _tmem_load_15[11];
                        }
                        if (clen > t0 + 12) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 12) * num_heads * 128))) + (0)) = _tmem_load_15[12];
                        }
                        if (clen > t0 + 13) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 13) * num_heads * 128))) + (0)) = _tmem_load_15[13];
                        }
                        if (clen > t0 + 14) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 14) * num_heads * 128))) + (0)) = _tmem_load_15[14];
                        }
                        if (clen > t0 + 15) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 15) * num_heads * 128))) + (0)) = _tmem_load_15[15];
                        }
                        if (clen > t0 + 16) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 16) * num_heads * 128))) + (0)) = _tmem_load_15[16];
                        }
                        if (clen > t0 + 17) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 17) * num_heads * 128))) + (0)) = _tmem_load_15[17];
                        }
                        if (clen > t0 + 18) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 18) * num_heads * 128))) + (0)) = _tmem_load_15[18];
                        }
                        if (clen > t0 + 19) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 19) * num_heads * 128))) + (0)) = _tmem_load_15[19];
                        }
                        if (clen > t0 + 20) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 20) * num_heads * 128))) + (0)) = _tmem_load_15[20];
                        }
                        if (clen > t0 + 21) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 21) * num_heads * 128))) + (0)) = _tmem_load_15[21];
                        }
                        if (clen > t0 + 22) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 22) * num_heads * 128))) + (0)) = _tmem_load_15[22];
                        }
                        if (clen > t0 + 23) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 23) * num_heads * 128))) + (0)) = _tmem_load_15[23];
                        }
                        if (clen > t0 + 24) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 24) * num_heads * 128))) + (0)) = _tmem_load_15[24];
                        }
                        if (clen > t0 + 25) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 25) * num_heads * 128))) + (0)) = _tmem_load_15[25];
                        }
                        if (clen > t0 + 26) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 26) * num_heads * 128))) + (0)) = _tmem_load_15[26];
                        }
                        if (clen > t0 + 27) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 27) * num_heads * 128))) + (0)) = _tmem_load_15[27];
                        }
                        if (clen > t0 + 28) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 28) * num_heads * 128))) + (0)) = _tmem_load_15[28];
                        }
                        if (clen > t0 + 29) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 29) * num_heads * 128))) + (0)) = _tmem_load_15[29];
                        }
                        if (clen > t0 + 30) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 30) * num_heads * 128))) + (0)) = _tmem_load_15[30];
                        }
                        if (clen > t0 + 31) {
                            *(reinterpret_cast<float*>(dk_part + (kbase + (long long)((t0 + 31) * num_heads * 128))) + (0)) = _tmem_load_15[31];
                        }
                    }
                    red_dg[wq * 64 + t0 + lane_0] = dg_acc;
                    red_db[wq * 64 + t0 + lane_0] = db_acc;
                    if (lane_0 == 0) {
                        red_hk[warp] = hk_acc;
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    float egl = 0.0f;
                    if (tid_1 < 64) {
                        float dgs = red_dg[tid_1];
                        dgs = dgs + red_dg[64 + tid_1];
                        dgs = dgs + red_dg[128 + tid_1];
                        dgs = dgs + red_dg[192 + tid_1];
                        float dbs = red_db[tid_1];
                        dbs = dbs + red_db[64 + tid_1];
                        dbs = dbs + red_db[128 + tid_1];
                        dbs = dbs + red_db[192 + tid_1];
                        dg_part[(long long)(row0 + tid_1) * nh64 + hv64] = ((tid_1 < clen) ? dgs : 0.0f);
                        if (tid_1 < clen) {
                            dbeta[(long long)(trow + tid_1) * nh64 + hv64] = dbs;
                        }
                        if (clen > 0) {
                            egl = eg_s[clen - 1];
                        }
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (warp == 0) {
                        if (lane_0 == 0) {
                            float hdt = red_hd[0];
                            float hkt = red_hk[0];
                            hdt = hdt + red_hd[1];
                            hkt = hkt + red_hk[1];
                            hdt = hdt + red_hd[2];
                            hkt = hkt + red_hk[2];
                            hdt = hdt + red_hd[3];
                            hkt = hkt + red_hk[3];
                            hdt = hdt + red_hd[4];
                            hkt = hkt + red_hk[4];
                            hdt = hdt + red_hd[5];
                            hkt = hkt + red_hk[5];
                            hdt = hdt + red_hd[6];
                            hkt = hkt + red_hk[6];
                            hdt = hdt + red_hd[7];
                            hkt = hkt + red_hk[7];
                            gamma_dot[(long long)chunk * nh64 + hv64] = __fmaf_rn(egl, hdt, hkt);
                        }
                    }
                    if (group_reduce != 0) {
                        if (hl == r_c - 1) {
                            float _tmem_load_17[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_17[0]), "=f"(_tmem_load_17[1]), "=f"(_tmem_load_17[2]), "=f"(_tmem_load_17[3]), "=f"(_tmem_load_17[4]), "=f"(_tmem_load_17[5]), "=f"(_tmem_load_17[6]), "=f"(_tmem_load_17[7]), "=f"(_tmem_load_17[8]), "=f"(_tmem_load_17[9]), "=f"(_tmem_load_17[10]), "=f"(_tmem_load_17[11]), "=f"(_tmem_load_17[12]), "=f"(_tmem_load_17[13]), "=f"(_tmem_load_17[14]), "=f"(_tmem_load_17[15]), "=f"(_tmem_load_17[16]), "=f"(_tmem_load_17[17]), "=f"(_tmem_load_17[18]), "=f"(_tmem_load_17[19]), "=f"(_tmem_load_17[20]), "=f"(_tmem_load_17[21]), "=f"(_tmem_load_17[22]), "=f"(_tmem_load_17[23]), "=f"(_tmem_load_17[24]), "=f"(_tmem_load_17[25]), "=f"(_tmem_load_17[26]), "=f"(_tmem_load_17[27]), "=f"(_tmem_load_17[28]), "=f"(_tmem_load_17[29]), "=f"(_tmem_load_17[30]), "=f"(_tmem_load_17[31])
                                : "r"(taddr + (unsigned int)row_base + (unsigned int)(ch * 32)));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float _tmem_load_18[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_18[0]), "=f"(_tmem_load_18[1]), "=f"(_tmem_load_18[2]), "=f"(_tmem_load_18[3]), "=f"(_tmem_load_18[4]), "=f"(_tmem_load_18[5]), "=f"(_tmem_load_18[6]), "=f"(_tmem_load_18[7]), "=f"(_tmem_load_18[8]), "=f"(_tmem_load_18[9]), "=f"(_tmem_load_18[10]), "=f"(_tmem_load_18[11]), "=f"(_tmem_load_18[12]), "=f"(_tmem_load_18[13]), "=f"(_tmem_load_18[14]), "=f"(_tmem_load_18[15]), "=f"(_tmem_load_18[16]), "=f"(_tmem_load_18[17]), "=f"(_tmem_load_18[18]), "=f"(_tmem_load_18[19]), "=f"(_tmem_load_18[20]), "=f"(_tmem_load_18[21]), "=f"(_tmem_load_18[22]), "=f"(_tmem_load_18[23]), "=f"(_tmem_load_18[24]), "=f"(_tmem_load_18[25]), "=f"(_tmem_load_18[26]), "=f"(_tmem_load_18[27]), "=f"(_tmem_load_18[28]), "=f"(_tmem_load_18[29]), "=f"(_tmem_load_18[30]), "=f"(_tmem_load_18[31])
                                : "r"(taddr + 64 + (unsigned int)row_base + (unsigned int)(ch * 32)));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            {
                                __nv_bfloat16 _bval_201 = __float2bfloat16_rn(_tmem_load_17[0]);
                                uint16_t _bits_201 = *(uint16_t*)&_bval_201;
                                uint32_t _addr_201 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)(t0 * 256 + ((d_row >> 3 ^ t0 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_201), "h"(_bits_201) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_202 = __float2bfloat16_rn(_tmem_load_17[1]);
                                uint16_t _bits_202 = *(uint16_t*)&_bval_202;
                                uint32_t _addr_202 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 1) * 256 + ((d_row >> 3 ^ t0 + 1 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_202), "h"(_bits_202) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_203 = __float2bfloat16_rn(_tmem_load_17[2]);
                                uint16_t _bits_203 = *(uint16_t*)&_bval_203;
                                uint32_t _addr_203 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 2) * 256 + ((d_row >> 3 ^ t0 + 2 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_203), "h"(_bits_203) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_204 = __float2bfloat16_rn(_tmem_load_17[3]);
                                uint16_t _bits_204 = *(uint16_t*)&_bval_204;
                                uint32_t _addr_204 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 3) * 256 + ((d_row >> 3 ^ t0 + 3 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_204), "h"(_bits_204) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_205 = __float2bfloat16_rn(_tmem_load_17[4]);
                                uint16_t _bits_205 = *(uint16_t*)&_bval_205;
                                uint32_t _addr_205 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 4) * 256 + ((d_row >> 3 ^ t0 + 4 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_205), "h"(_bits_205) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_206 = __float2bfloat16_rn(_tmem_load_17[5]);
                                uint16_t _bits_206 = *(uint16_t*)&_bval_206;
                                uint32_t _addr_206 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 5) * 256 + ((d_row >> 3 ^ t0 + 5 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_206), "h"(_bits_206) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_207 = __float2bfloat16_rn(_tmem_load_17[6]);
                                uint16_t _bits_207 = *(uint16_t*)&_bval_207;
                                uint32_t _addr_207 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 6) * 256 + ((d_row >> 3 ^ t0 + 6 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_207), "h"(_bits_207) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_208 = __float2bfloat16_rn(_tmem_load_17[7]);
                                uint16_t _bits_208 = *(uint16_t*)&_bval_208;
                                uint32_t _addr_208 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 7) * 256 + ((d_row >> 3 ^ t0 + 7 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_208), "h"(_bits_208) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_209 = __float2bfloat16_rn(_tmem_load_17[8]);
                                uint16_t _bits_209 = *(uint16_t*)&_bval_209;
                                uint32_t _addr_209 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 8) * 256 + ((d_row >> 3 ^ t0 + 8 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_209), "h"(_bits_209) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_210 = __float2bfloat16_rn(_tmem_load_17[9]);
                                uint16_t _bits_210 = *(uint16_t*)&_bval_210;
                                uint32_t _addr_210 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 9) * 256 + ((d_row >> 3 ^ t0 + 9 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_210), "h"(_bits_210) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_211 = __float2bfloat16_rn(_tmem_load_17[10]);
                                uint16_t _bits_211 = *(uint16_t*)&_bval_211;
                                uint32_t _addr_211 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 10) * 256 + ((d_row >> 3 ^ t0 + 10 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_211), "h"(_bits_211) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_212 = __float2bfloat16_rn(_tmem_load_17[11]);
                                uint16_t _bits_212 = *(uint16_t*)&_bval_212;
                                uint32_t _addr_212 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 11) * 256 + ((d_row >> 3 ^ t0 + 11 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_212), "h"(_bits_212) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_213 = __float2bfloat16_rn(_tmem_load_17[12]);
                                uint16_t _bits_213 = *(uint16_t*)&_bval_213;
                                uint32_t _addr_213 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 12) * 256 + ((d_row >> 3 ^ t0 + 12 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_213), "h"(_bits_213) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_214 = __float2bfloat16_rn(_tmem_load_17[13]);
                                uint16_t _bits_214 = *(uint16_t*)&_bval_214;
                                uint32_t _addr_214 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 13) * 256 + ((d_row >> 3 ^ t0 + 13 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_214), "h"(_bits_214) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_215 = __float2bfloat16_rn(_tmem_load_17[14]);
                                uint16_t _bits_215 = *(uint16_t*)&_bval_215;
                                uint32_t _addr_215 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 14) * 256 + ((d_row >> 3 ^ t0 + 14 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_215), "h"(_bits_215) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_216 = __float2bfloat16_rn(_tmem_load_17[15]);
                                uint16_t _bits_216 = *(uint16_t*)&_bval_216;
                                uint32_t _addr_216 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 15) * 256 + ((d_row >> 3 ^ t0 + 15 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_216), "h"(_bits_216) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_217 = __float2bfloat16_rn(_tmem_load_17[16]);
                                uint16_t _bits_217 = *(uint16_t*)&_bval_217;
                                uint32_t _addr_217 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 16) * 256 + ((d_row >> 3 ^ t0 + 16 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_217), "h"(_bits_217) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_218 = __float2bfloat16_rn(_tmem_load_17[17]);
                                uint16_t _bits_218 = *(uint16_t*)&_bval_218;
                                uint32_t _addr_218 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 17) * 256 + ((d_row >> 3 ^ t0 + 17 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_218), "h"(_bits_218) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_219 = __float2bfloat16_rn(_tmem_load_17[18]);
                                uint16_t _bits_219 = *(uint16_t*)&_bval_219;
                                uint32_t _addr_219 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 18) * 256 + ((d_row >> 3 ^ t0 + 18 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_219), "h"(_bits_219) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_220 = __float2bfloat16_rn(_tmem_load_17[19]);
                                uint16_t _bits_220 = *(uint16_t*)&_bval_220;
                                uint32_t _addr_220 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 19) * 256 + ((d_row >> 3 ^ t0 + 19 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_220), "h"(_bits_220) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_221 = __float2bfloat16_rn(_tmem_load_17[20]);
                                uint16_t _bits_221 = *(uint16_t*)&_bval_221;
                                uint32_t _addr_221 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 20) * 256 + ((d_row >> 3 ^ t0 + 20 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_221), "h"(_bits_221) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_222 = __float2bfloat16_rn(_tmem_load_17[21]);
                                uint16_t _bits_222 = *(uint16_t*)&_bval_222;
                                uint32_t _addr_222 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 21) * 256 + ((d_row >> 3 ^ t0 + 21 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_222), "h"(_bits_222) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_223 = __float2bfloat16_rn(_tmem_load_17[22]);
                                uint16_t _bits_223 = *(uint16_t*)&_bval_223;
                                uint32_t _addr_223 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 22) * 256 + ((d_row >> 3 ^ t0 + 22 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_223), "h"(_bits_223) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_224 = __float2bfloat16_rn(_tmem_load_17[23]);
                                uint16_t _bits_224 = *(uint16_t*)&_bval_224;
                                uint32_t _addr_224 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 23) * 256 + ((d_row >> 3 ^ t0 + 23 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_224), "h"(_bits_224) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_225 = __float2bfloat16_rn(_tmem_load_17[24]);
                                uint16_t _bits_225 = *(uint16_t*)&_bval_225;
                                uint32_t _addr_225 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 24) * 256 + ((d_row >> 3 ^ t0 + 24 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_225), "h"(_bits_225) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_226 = __float2bfloat16_rn(_tmem_load_17[25]);
                                uint16_t _bits_226 = *(uint16_t*)&_bval_226;
                                uint32_t _addr_226 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 25) * 256 + ((d_row >> 3 ^ t0 + 25 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_226), "h"(_bits_226) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_227 = __float2bfloat16_rn(_tmem_load_17[26]);
                                uint16_t _bits_227 = *(uint16_t*)&_bval_227;
                                uint32_t _addr_227 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 26) * 256 + ((d_row >> 3 ^ t0 + 26 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_227), "h"(_bits_227) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_228 = __float2bfloat16_rn(_tmem_load_17[27]);
                                uint16_t _bits_228 = *(uint16_t*)&_bval_228;
                                uint32_t _addr_228 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 27) * 256 + ((d_row >> 3 ^ t0 + 27 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_228), "h"(_bits_228) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_229 = __float2bfloat16_rn(_tmem_load_17[28]);
                                uint16_t _bits_229 = *(uint16_t*)&_bval_229;
                                uint32_t _addr_229 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 28) * 256 + ((d_row >> 3 ^ t0 + 28 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_229), "h"(_bits_229) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_230 = __float2bfloat16_rn(_tmem_load_17[29]);
                                uint16_t _bits_230 = *(uint16_t*)&_bval_230;
                                uint32_t _addr_230 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 29) * 256 + ((d_row >> 3 ^ t0 + 29 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_230), "h"(_bits_230) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_231 = __float2bfloat16_rn(_tmem_load_17[30]);
                                uint16_t _bits_231 = *(uint16_t*)&_bval_231;
                                uint32_t _addr_231 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 30) * 256 + ((d_row >> 3 ^ t0 + 30 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_231), "h"(_bits_231) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_232 = __float2bfloat16_rn(_tmem_load_17[31]);
                                uint16_t _bits_232 = *(uint16_t*)&_bval_232;
                                uint32_t _addr_232 = static_cast<uint32_t>(stage_dq_addr + (unsigned int)((t0 + 31) * 256 + ((d_row >> 3 ^ t0 + 31 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_232), "h"(_bits_232) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_233 = __float2bfloat16_rn(_tmem_load_18[0]);
                                uint16_t _bits_233 = *(uint16_t*)&_bval_233;
                                uint32_t _addr_233 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)(t0 * 256 + ((d_row >> 3 ^ t0 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_233), "h"(_bits_233) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_234 = __float2bfloat16_rn(_tmem_load_18[1]);
                                uint16_t _bits_234 = *(uint16_t*)&_bval_234;
                                uint32_t _addr_234 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 1) * 256 + ((d_row >> 3 ^ t0 + 1 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_234), "h"(_bits_234) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_235 = __float2bfloat16_rn(_tmem_load_18[2]);
                                uint16_t _bits_235 = *(uint16_t*)&_bval_235;
                                uint32_t _addr_235 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 2) * 256 + ((d_row >> 3 ^ t0 + 2 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_235), "h"(_bits_235) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_236 = __float2bfloat16_rn(_tmem_load_18[3]);
                                uint16_t _bits_236 = *(uint16_t*)&_bval_236;
                                uint32_t _addr_236 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 3) * 256 + ((d_row >> 3 ^ t0 + 3 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_236), "h"(_bits_236) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_237 = __float2bfloat16_rn(_tmem_load_18[4]);
                                uint16_t _bits_237 = *(uint16_t*)&_bval_237;
                                uint32_t _addr_237 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 4) * 256 + ((d_row >> 3 ^ t0 + 4 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_237), "h"(_bits_237) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_238 = __float2bfloat16_rn(_tmem_load_18[5]);
                                uint16_t _bits_238 = *(uint16_t*)&_bval_238;
                                uint32_t _addr_238 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 5) * 256 + ((d_row >> 3 ^ t0 + 5 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_238), "h"(_bits_238) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_239 = __float2bfloat16_rn(_tmem_load_18[6]);
                                uint16_t _bits_239 = *(uint16_t*)&_bval_239;
                                uint32_t _addr_239 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 6) * 256 + ((d_row >> 3 ^ t0 + 6 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_239), "h"(_bits_239) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_240 = __float2bfloat16_rn(_tmem_load_18[7]);
                                uint16_t _bits_240 = *(uint16_t*)&_bval_240;
                                uint32_t _addr_240 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 7) * 256 + ((d_row >> 3 ^ t0 + 7 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_240), "h"(_bits_240) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_241 = __float2bfloat16_rn(_tmem_load_18[8]);
                                uint16_t _bits_241 = *(uint16_t*)&_bval_241;
                                uint32_t _addr_241 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 8) * 256 + ((d_row >> 3 ^ t0 + 8 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_241), "h"(_bits_241) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_242 = __float2bfloat16_rn(_tmem_load_18[9]);
                                uint16_t _bits_242 = *(uint16_t*)&_bval_242;
                                uint32_t _addr_242 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 9) * 256 + ((d_row >> 3 ^ t0 + 9 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_242), "h"(_bits_242) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_243 = __float2bfloat16_rn(_tmem_load_18[10]);
                                uint16_t _bits_243 = *(uint16_t*)&_bval_243;
                                uint32_t _addr_243 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 10) * 256 + ((d_row >> 3 ^ t0 + 10 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_243), "h"(_bits_243) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_244 = __float2bfloat16_rn(_tmem_load_18[11]);
                                uint16_t _bits_244 = *(uint16_t*)&_bval_244;
                                uint32_t _addr_244 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 11) * 256 + ((d_row >> 3 ^ t0 + 11 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_244), "h"(_bits_244) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_245 = __float2bfloat16_rn(_tmem_load_18[12]);
                                uint16_t _bits_245 = *(uint16_t*)&_bval_245;
                                uint32_t _addr_245 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 12) * 256 + ((d_row >> 3 ^ t0 + 12 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_245), "h"(_bits_245) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_246 = __float2bfloat16_rn(_tmem_load_18[13]);
                                uint16_t _bits_246 = *(uint16_t*)&_bval_246;
                                uint32_t _addr_246 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 13) * 256 + ((d_row >> 3 ^ t0 + 13 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_246), "h"(_bits_246) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_247 = __float2bfloat16_rn(_tmem_load_18[14]);
                                uint16_t _bits_247 = *(uint16_t*)&_bval_247;
                                uint32_t _addr_247 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 14) * 256 + ((d_row >> 3 ^ t0 + 14 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_247), "h"(_bits_247) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_248 = __float2bfloat16_rn(_tmem_load_18[15]);
                                uint16_t _bits_248 = *(uint16_t*)&_bval_248;
                                uint32_t _addr_248 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 15) * 256 + ((d_row >> 3 ^ t0 + 15 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_248), "h"(_bits_248) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_249 = __float2bfloat16_rn(_tmem_load_18[16]);
                                uint16_t _bits_249 = *(uint16_t*)&_bval_249;
                                uint32_t _addr_249 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 16) * 256 + ((d_row >> 3 ^ t0 + 16 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_249), "h"(_bits_249) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_250 = __float2bfloat16_rn(_tmem_load_18[17]);
                                uint16_t _bits_250 = *(uint16_t*)&_bval_250;
                                uint32_t _addr_250 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 17) * 256 + ((d_row >> 3 ^ t0 + 17 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_250), "h"(_bits_250) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_251 = __float2bfloat16_rn(_tmem_load_18[18]);
                                uint16_t _bits_251 = *(uint16_t*)&_bval_251;
                                uint32_t _addr_251 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 18) * 256 + ((d_row >> 3 ^ t0 + 18 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_251), "h"(_bits_251) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_252 = __float2bfloat16_rn(_tmem_load_18[19]);
                                uint16_t _bits_252 = *(uint16_t*)&_bval_252;
                                uint32_t _addr_252 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 19) * 256 + ((d_row >> 3 ^ t0 + 19 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_252), "h"(_bits_252) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_253 = __float2bfloat16_rn(_tmem_load_18[20]);
                                uint16_t _bits_253 = *(uint16_t*)&_bval_253;
                                uint32_t _addr_253 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 20) * 256 + ((d_row >> 3 ^ t0 + 20 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_253), "h"(_bits_253) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_254 = __float2bfloat16_rn(_tmem_load_18[21]);
                                uint16_t _bits_254 = *(uint16_t*)&_bval_254;
                                uint32_t _addr_254 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 21) * 256 + ((d_row >> 3 ^ t0 + 21 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_254), "h"(_bits_254) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_255 = __float2bfloat16_rn(_tmem_load_18[22]);
                                uint16_t _bits_255 = *(uint16_t*)&_bval_255;
                                uint32_t _addr_255 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 22) * 256 + ((d_row >> 3 ^ t0 + 22 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_255), "h"(_bits_255) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_256 = __float2bfloat16_rn(_tmem_load_18[23]);
                                uint16_t _bits_256 = *(uint16_t*)&_bval_256;
                                uint32_t _addr_256 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 23) * 256 + ((d_row >> 3 ^ t0 + 23 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_256), "h"(_bits_256) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_257 = __float2bfloat16_rn(_tmem_load_18[24]);
                                uint16_t _bits_257 = *(uint16_t*)&_bval_257;
                                uint32_t _addr_257 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 24) * 256 + ((d_row >> 3 ^ t0 + 24 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_257), "h"(_bits_257) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_258 = __float2bfloat16_rn(_tmem_load_18[25]);
                                uint16_t _bits_258 = *(uint16_t*)&_bval_258;
                                uint32_t _addr_258 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 25) * 256 + ((d_row >> 3 ^ t0 + 25 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_258), "h"(_bits_258) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_259 = __float2bfloat16_rn(_tmem_load_18[26]);
                                uint16_t _bits_259 = *(uint16_t*)&_bval_259;
                                uint32_t _addr_259 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 26) * 256 + ((d_row >> 3 ^ t0 + 26 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_259), "h"(_bits_259) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_260 = __float2bfloat16_rn(_tmem_load_18[27]);
                                uint16_t _bits_260 = *(uint16_t*)&_bval_260;
                                uint32_t _addr_260 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 27) * 256 + ((d_row >> 3 ^ t0 + 27 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_260), "h"(_bits_260) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_261 = __float2bfloat16_rn(_tmem_load_18[28]);
                                uint16_t _bits_261 = *(uint16_t*)&_bval_261;
                                uint32_t _addr_261 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 28) * 256 + ((d_row >> 3 ^ t0 + 28 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_261), "h"(_bits_261) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_262 = __float2bfloat16_rn(_tmem_load_18[29]);
                                uint16_t _bits_262 = *(uint16_t*)&_bval_262;
                                uint32_t _addr_262 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 29) * 256 + ((d_row >> 3 ^ t0 + 29 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_262), "h"(_bits_262) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_263 = __float2bfloat16_rn(_tmem_load_18[30]);
                                uint16_t _bits_263 = *(uint16_t*)&_bval_263;
                                uint32_t _addr_263 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 30) * 256 + ((d_row >> 3 ^ t0 + 30 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_263), "h"(_bits_263) : "memory");
                            }
                            {
                                __nv_bfloat16 _bval_264 = __float2bfloat16_rn(_tmem_load_18[31]);
                                uint16_t _bits_264 = *(uint16_t*)&_bval_264;
                                uint32_t _addr_264 = static_cast<uint32_t>(stage_dk_addr + (unsigned int)((t0 + 31) * 256 + ((d_row >> 3 ^ t0 + 31 & 7) << 4) + (d_row & 7) * 2));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_264), "h"(_bits_264) : "memory");
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile("barrier.sync 1, 256;" ::: "memory");
                            int t_0 = tid_1 >> 3;
                            int t1_0 = t_0 + 32;
                            int c0_0 = tid_1 & 7;
                            if (t_0 < clen) {
                                long long base_1 = (long long)(trow + t_0) * (long long)dqk_tstride + (long long)(hq * 128 + c0_0 * 8);
                                int chunk_2 = c0_0 ^ t_0 & 7;
                                unsigned int q0_1_2[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q0_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_2[(0) + 3]))
                                    : "r"(stage_dq_addr + (unsigned int)(t_0 * 256 + (chunk_2 << 4))));
                                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dq + base_1))[0] = reinterpret_cast<int4*>(q0_1_2)[0];
                                int chunk_3 = c0_0 + 8 ^ t_0 & 7;
                                unsigned int q0_1_3[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q0_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_3[(0) + 3]))
                                    : "r"(stage_dq_addr + (unsigned int)(t_0 * 256 + (chunk_3 << 4))));
                                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dq + (base_1 + 64)))[0] = reinterpret_cast<int4*>(q0_1_3)[0];
                            }
                            if (t1_0 < clen) {
                                long long base1_1 = (long long)(trow + t1_0) * (long long)dqk_tstride + (long long)(hq * 128 + c0_0 * 8);
                                int chunk1_2 = c0_0 ^ t1_0 & 7;
                                unsigned int q2_3_2[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q2_3_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_2[(0) + 3]))
                                    : "r"(stage_dq_addr + (unsigned int)(t1_0 * 256 + (chunk1_2 << 4))));
                                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dq + base1_1))[0] = reinterpret_cast<int4*>(q2_3_2)[0];
                                int chunk1_3 = c0_0 + 8 ^ t1_0 & 7;
                                unsigned int q2_3_3[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q2_3_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_3[(0) + 3]))
                                    : "r"(stage_dq_addr + (unsigned int)(t1_0 * 256 + (chunk1_3 << 4))));
                                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dq + (base1_1 + 64)))[0] = reinterpret_cast<int4*>(q2_3_3)[0];
                            }
                            int t_1 = tid_1 >> 3;
                            int t1_1 = t_1 + 32;
                            int c0_1 = tid_1 & 7;
                            if (t_1 < clen) {
                                long long base_2 = (long long)(trow + t_1) * (long long)dqk_tstride + (long long)(hq * 128 + c0_1 * 8);
                                int chunk_4 = c0_1 ^ t_1 & 7;
                                unsigned int q0_1_4[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q0_1_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_4[(0) + 3]))
                                    : "r"(stage_dk_addr + (unsigned int)(t_1 * 256 + (chunk_4 << 4))));
                                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dk + base_2))[0] = reinterpret_cast<int4*>(q0_1_4)[0];
                                int chunk_5 = c0_1 + 8 ^ t_1 & 7;
                                unsigned int q0_1_5[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q0_1_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q0_1_5[(0) + 3]))
                                    : "r"(stage_dk_addr + (unsigned int)(t_1 * 256 + (chunk_5 << 4))));
                                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dk + (base_2 + 64)))[0] = reinterpret_cast<int4*>(q0_1_5)[0];
                            }
                            if (t1_1 < clen) {
                                long long base1_2 = (long long)(trow + t1_1) * (long long)dqk_tstride + (long long)(hq * 128 + c0_1 * 8);
                                int chunk1_4 = c0_1 ^ t1_1 & 7;
                                unsigned int q2_3_4[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q2_3_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_4[(0) + 3]))
                                    : "r"(stage_dk_addr + (unsigned int)(t1_1 * 256 + (chunk1_4 << 4))));
                                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dk + base1_2))[0] = reinterpret_cast<int4*>(q2_3_4)[0];
                                int chunk1_5 = c0_1 + 8 ^ t1_1 & 7;
                                unsigned int q2_3_5[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q2_3_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q2_3_5[(0) + 3]))
                                    : "r"(stage_dk_addr + (unsigned int)(t1_1 * 256 + (chunk1_5 << 4))));
                                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dk + (base1_2 + 64)))[0] = reinterpret_cast<int4*>(q2_3_5)[0];
                            }
                        }
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(e_drained_addr);
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            int r_l = num_heads / num_qk_heads;
            int ngroups_l = seq_chunk_start[num_seqs] * num_qk_heads;
            const int grid_l = (int)num_bids;
            unsigned int lk_stage = 0;
            unsigned int _phase_tk_empty = 1;
            unsigned int _phase_a_committed_0 = 1;
            unsigned int _phase_b_committed_0 = 1;
            unsigned int _phase_dv_stored_0 = 1;
            unsigned int _phase_d_committed_0 = 1;
            unsigned int _phase_e_committed_0 = 1;
            unsigned int _phase_e_drained_0 = 1;
            if (elect_sync()) {
                const unsigned int tk_limit = (unsigned int)(ngroups_l + grid_l - 1);
                unsigned int drawn = (unsigned int)bid;
                if (ngroups_l > grid_l) {
                    uint32_t _atomic_fetch_inc_0;
                    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(_atomic_fetch_inc_0) : "l"(ticket), "r"(static_cast<uint32_t>(tk_limit)) : "memory");
                    drawn = _atomic_fetch_inc_0;
                }
                unsigned int tkw[1];
                #pragma unroll 1
                for (int _j_1 = 0; _j_1 < ngroups_l + 1; _j_1++) {
                    mbarrier_wait(tk_empty_addr + (lk_stage) * 8, _phase_tk_empty);
                    tkw[0] = drawn;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(tk_s_addr + lk_stage * 4), "r"(tkw[0]) : "memory");
                    mbarrier_arrive(tk_full_addr + (lk_stage) * 8);
                    lk_stage += 1;
                    if (lk_stage == 2) { lk_stage = 0; _phase_tk_empty ^= 1; }
                    int u_1 = (int)drawn;
                    if (u_1 >= ngroups_l) {
                        break;
                    }
                    if (ngroups_l > grid_l) {
                        uint32_t _atomic_fetch_inc_1;
                        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_1) : "l"(ticket), "r"(static_cast<uint32_t>(tk_limit)) : "memory");
                        drawn = _atomic_fetch_inc_1;
                    } else {
                        drawn = (unsigned int)(u_1 + grid_l);
                    }
                    unsigned int g_1 = (unsigned int)u_1;
                    int chunk_6 = g_1 / (unsigned int)num_qk_heads;
                    int hq_1 = g_1 % (unsigned int)num_qk_heads;
                    int row0_1 = chunk_6 * 64;
                    int trow_1 = chunk_bos[chunk_6];
                    int clen_1 = chunk_len[chunk_6];
                    #pragma unroll 1
                    for (int hl_1 = 0; hl_1 < r_l; hl_1++) {
                        int hv_1 = hq_1 * r_l + hl_1;
                        int hrow0 = (chunk_6 * num_heads + hv_1) * 128;
                        mbarrier_wait(a_committed_addr, _phase_a_committed_0);
                        _phase_a_committed_0 ^= 1;
                        mbarrier_arrive_expect_tx(dovn_full_addr, 32768);
                        #pragma unroll
                        for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                            int c0_2 = seg_1 * 64;
                            tma_3d_gmem2smem(smem_v0_addr_2 + (unsigned int)(seg_1 * 8192), (&do_tma), c0_2, hv_1, trow_1, dovn_full_addr);
                            tma_3d_gmem2smem(smem_v0_addr_3 + (unsigned int)(seg_1 * 8192), (&vn_tma), c0_2, hv_1, row0_1, dovn_full_addr);
                        }
                        mbarrier_wait(b_committed_addr, _phase_b_committed_0);
                        _phase_b_committed_0 ^= 1;
                        mbarrier_arrive_expect_tx(h_full_addr, 32768);
                        #pragma unroll
                        for (int seg_2 = 0; seg_2 < 2; seg_2++) {
                            int c1 = seg_2 * 64;
                            tma_2d_gmem2smem(h_p_addr + (unsigned int)(seg_2 * 16384), (&h_tma), c1, hrow0, h_full_addr);
                        }
                        mbarrier_arrive_expect_tx(dh_full_addr, 32768);
                        #pragma unroll
                        for (int seg_3 = 0; seg_3 < 2; seg_3++) {
                            int c4 = seg_3 * 64;
                            tma_2d_gmem2smem(dh_p_addr + (unsigned int)(seg_3 * 16384), (&dh_tma), c4, hrow0, dh_full_addr);
                        }
                        mbarrier_wait(dv_stored_addr, _phase_dv_stored_0);
                        _phase_dv_stored_0 ^= 1;
                        mbarrier_arrive_expect_tx(v_full_addr, 16384);
                        mbarrier_arrive_expect_tx(dvn_full_addr, 16384);
                        #pragma unroll
                        for (int seg_4 = 0; seg_4 < 2; seg_4++) {
                            int c4_1 = seg_4 * 64;
                            tma_3d_gmem2smem(smem_v0_addr_1 + (unsigned int)(seg_4 * 8192), (&v_tma), c4_1, hv_1, trow_1, v_full_addr);
                            tma_3d_gmem2smem(smem_v0_addr_4 + (unsigned int)(seg_4 * 8192), (&dvn_tma), c4_1, hv_1, row0_1, dvn_full_addr);
                        }
                        mbarrier_wait(d_committed_addr, _phase_d_committed_0);
                        _phase_d_committed_0 ^= 1;
                        mbarrier_arrive_expect_tx(akk_full_addr, 8192);
                        tma_3d_gmem2smem(akk_s_addr, (&akk_tma), 0, hv_1, row0_1, akk_full_addr);
                        mbarrier_wait(e_committed_addr, _phase_e_committed_0);
                        _phase_e_committed_0 ^= 1;
                        if (hl_1 == r_l - 1) {
                            int un = (int)drawn;
                            if (un < ngroups_l) {
                                int chunk_n = drawn / (unsigned int)num_qk_heads;
                                int hq_n = drawn % (unsigned int)num_qk_heads;
                                int trow_n = chunk_bos[chunk_n];
                                #pragma unroll
                                for (int seg_5 = 0; seg_5 < 2; seg_5++) {
                                    int c5 = seg_5 * 64;
                                    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&q_tma))), "r"((int)(c5)), "r"((int)(hq_n)), "r"((int)(trow_n)) : "memory");
                                    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&k_tma))), "r"((int)(c5)), "r"((int)(hq_n)), "r"((int)(trow_n)) : "memory");
                                }
                            }
                        }
                        mbarrier_wait(e_drained_addr, _phase_e_drained_0);
                        _phase_e_drained_0 ^= 1;
                        if (hl_1 == 0) {
                            mbarrier_arrive_expect_tx(qk_full_addr, 32768);
                            #pragma unroll
                            for (int seg_6 = 0; seg_6 < 2; seg_6++) {
                                int c3 = seg_6 * 64;
                                tma_3d_gmem2smem(smem_v0_addr + (unsigned int)(seg_6 * 8192), (&q_tma), c3, hq_1, trow_1, qk_full_addr);
                                tma_3d_gmem2smem(smem_v0_addr_0 + (unsigned int)(seg_6 * 8192), (&k_tma), c3, hq_1, trow_1, qk_full_addr);
                            }
                        } else {
                            mbarrier_arrive(qk_full_addr);
                        }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            int r_m = num_heads / num_qk_heads;
            int ngroups_m = seq_chunk_start[num_seqs] * num_qk_heads;
            unsigned int _phase_tk_full_1 = 0;
            unsigned int _phase_h_full_0_1 = 0;
            unsigned int _phase_dh_full_0_1 = 0;
            unsigned int _phase_dovn_full_0 = 0;
            unsigned int _phase_dvn_full_0_1 = 0;
            unsigned int _phase_v_full_0_1 = 0;
            unsigned int _phase_e_drained_0_1 = 1;
            unsigned int _phase_daq_drained_0 = 0;
            unsigned int _phase_akk_full_0 = 0;
            unsigned int _phase_a_drained_0 = 0;
            unsigned int _phase_qk_full_0_1 = 0;
            unsigned int _phase_b_drained_0 = 0;
            unsigned int _phase_c_drained_0 = 0;
            unsigned int _phase_d_drained_0 = 0;
            if (elect_sync()) {
                unsigned int mk_stage = 0;
                #pragma unroll 1
                for (int _j_2 = 0; _j_2 < ngroups_m + 1; _j_2++) {
                    mbarrier_wait(tk_full_addr + (mk_stage) * 8, _phase_tk_full_1);
                    unsigned int tku_1[1];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tku_1[0])) : "r"(tk_s_addr + mk_stage * 4) : "memory");
                    int u_2 = (int)tku_1[0];
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(tk_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 2) { mk_stage = 0; _phase_tk_full_1 ^= 1; }
                    if (u_2 >= ngroups_m) {
                        break;
                    }
                    unsigned int g_2 = (unsigned int)u_2;
                    int chunk_7 = g_2 / (unsigned int)num_qk_heads;
                    int hq_2 = g_2 % (unsigned int)num_qk_heads;
                    int row0_2 = chunk_7 * 64;
                    int trow_2 = chunk_bos[chunk_7];
                    int clen_2 = chunk_len[chunk_7];
                    #pragma unroll 1
                    for (int hl_2 = 0; hl_2 < r_m; hl_2++) {
                        mbarrier_wait(h_full_addr, _phase_h_full_0_1);
                        _phase_h_full_0_1 ^= 1;
                        mbarrier_wait(dh_full_addr, _phase_dh_full_0_1);
                        _phase_dh_full_0_1 ^= 1;
                        mbarrier_wait(dovn_full_addr, _phase_dovn_full_0);
                        _phase_dovn_full_0 ^= 1;
                        mbarrier_wait(dvn_full_addr, _phase_dvn_full_0_1);
                        _phase_dvn_full_0_1 ^= 1;
                        mbarrier_wait(v_full_addr, _phase_v_full_0_1);
                        _phase_v_full_0_1 ^= 1;
                        if (clen_2 < 64) {
                            float zq[4];
                            zq[0] = 0.0f;
                            zq[1] = 0.0f;
                            zq[2] = 0.0f;
                            zq[3] = 0.0f;
                            int n16 = (64 - clen_2) * 8;
                            #pragma unroll
                            for (int p = 0; p < 2; p++) {
                                int start = p * 8192 + clen_2 * 128;
                                #pragma unroll 8
                                for (int idx = 0; idx < n16; idx++) {
                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v6_addr_1 + (unsigned int)(start + idx * 16)), "f"(zq[0]), "f"(zq[1]), "f"(zq[2]), "f"(zq[3]) : "memory");
                                }
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        }
                        mbarrier_wait(e_drained_addr, _phase_e_drained_0_1);
                        _phase_e_drained_0_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_0 = ((h_k_addr) >> 4) & 0x7FFF;
                        int _mma_b_lo_0 = ((smem_v6_addr_2) >> 4) & 0x7FFF;
#else
                        int _mma_a_lo_0 = ((h_k_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_0 = ((smem_v6_addr_2) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 135267472;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc_dq), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_1 = ((dh_k_addr) >> 4) & 0x7FFF;
                        int _mma_b_lo_1 = ((smem_v6_addr_3) >> 4) & 0x7FFF;
#else
                        int _mma_a_lo_1 = ((dh_k_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_1 = ((smem_v6_addr_3) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 135267472;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc_dk), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_b_lo_2 = ((smem_v6_addr_4) >> 4) & 0x7FFF;
#else
                        int _mma_b_lo_2 = ((smem_v6_addr_4) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 135267472;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_2), "r"(tmem_acc_dw), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_0 = ((smem_v6_addr_1) >> 4) & 0x7FFF;
#else
                        int _smem_desc_lo_0 = ((smem_v6_addr_1) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_b_lo_2), "r"(_smem_desc_lo_0), "r"(tmem_acc_da), "r"(0) : "memory");
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_b_lo_0), "r"(_mma_b_lo_1), "r"(tmem_acc_daq), "r"(0) : "memory");
                        tcgen05_commit(a_ready_addr);
                        tcgen05_commit(a_committed_addr);
                        mbarrier_wait(daq_drained_addr, _phase_daq_drained_0);
                        _phase_daq_drained_0 ^= 1;
                        mbarrier_wait(akk_full_addr, _phase_akk_full_0);
                        _phase_akk_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_2 = (((smem_v9_addr_1) >> 4) & 0x7FFF) | 0x2000000;
                        int _mma_b_lo_3 = (((smem_v21_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _mma_a_lo_2 = (((smem_v9_addr_1) >> 4) & 0x3FFF) | 0x2000000;
                        int _mma_b_lo_3 = (((smem_v21_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_3), "r"(tmem_acc_dvb), "r"(0) : "memory");
                        tcgen05_commit(b1_ready_addr);
                        mbarrier_wait(a_drained_addr, _phase_a_drained_0);
                        _phase_a_drained_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_1 = ((dwbt_k_addr) >> 4) & 0x7FFF;
#else
                        int _smem_desc_lo_1 = ((dwbt_k_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 135333008;\n\t"
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
                    :: "r"(_smem_desc_lo_1), "r"(_mma_b_lo_3), "r"(tmem_acc_dkgb), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_2 = (((dwbt_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
                        int _smem_desc_lo_3 = (((smem_v6_addr_5) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _smem_desc_lo_2 = (((dwbt_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
                        int _smem_desc_lo_3 = (((smem_v6_addr_5) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 68191376;\n\t"
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
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc_da), "r"(1) : "memory");
                        tcgen05_commit(b_ready_addr);
                        tcgen05_commit(b_committed_addr);
                        mbarrier_wait(qk_full_addr, _phase_qk_full_0_1);
                        _phase_qk_full_0_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_3e = (((smem_v9_addr_0) >> 4) & 0x7FFF) | 0x2000000;
                        int _smem_desc_lo_4 = (((smem_v20_addr_2) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _mma_a_lo_3e = (((smem_v9_addr_0) >> 4) & 0x3FFF) | 0x2000000;
                        int _smem_desc_lo_4 = (((smem_v20_addr_2) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_mma_a_lo_3e), "r"(_smem_desc_lo_4), "r"(tmem_acc_dq), "r"(1) : "memory");
                        tcgen05_commit(e1_ready_addr);
                        mbarrier_wait(b_drained_addr, _phase_b_drained_0);
                        _phase_b_drained_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_5 = ((smem_v20_addr_0) >> 4) & 0x7FFF;
                        int _smem_desc_lo_6 = ((smem_v20_addr) >> 4) & 0x7FFF;
#else
                        int _smem_desc_lo_5 = ((smem_v20_addr_0) >> 4) & 0x3FFF;
                        int _smem_desc_lo_6 = ((smem_v20_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    :: "r"(_smem_desc_lo_5), "r"(_smem_desc_lo_6), "r"(tmem_acc_da1), "r"(0) : "memory");
                        tcgen05_commit(c_ready_addr);
                        mbarrier_wait(c_drained_addr, _phase_c_drained_0);
                        _phase_c_drained_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_7 = (((smem_v21_addr_0) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _smem_desc_lo_7 = (((smem_v21_addr_0) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_mma_b_lo_3), "r"(_smem_desc_lo_7), "r"(tmem_acc_da2), "r"(0) : "memory");
                        tcgen05_commit(d_ready_addr);
                        tcgen05_commit(d_committed_addr);
                        mbarrier_wait(d_drained_addr, _phase_d_drained_0);
                        _phase_d_drained_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_3 = (((smem_v9_addr_0) >> 4) & 0x7FFF) | 0x2000000;
                        int _smem_desc_lo_8 = (((smem_v9_addr) >> 4) & 0x7FFF) | 0x2000000;
                        int _smem_desc_lo_9 = (((smem_v21_addr_1) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _mma_a_lo_3 = (((smem_v9_addr_0) >> 4) & 0x3FFF) | 0x2000000;
                        int _smem_desc_lo_8 = (((smem_v9_addr) >> 4) & 0x3FFF) | 0x2000000;
                        int _smem_desc_lo_9 = (((smem_v21_addr_1) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"(tmem_acc_dksrc), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_10 = (((smem_v9_addr_2) >> 4) & 0x7FFF) | 0x2000000;
                        int _smem_desc_lo_11 = (((smem_v21_addr_2) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _smem_desc_lo_10 = (((smem_v9_addr_2) >> 4) & 0x3FFF) | 0x2000000;
                        int _smem_desc_lo_11 = (((smem_v21_addr_2) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"(tmem_acc_dksrc), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_12 = (((smem_v20_addr_3) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _smem_desc_lo_12 = (((smem_v20_addr_3) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_mma_a_lo_3), "r"(_smem_desc_lo_12), "r"(tmem_acc_x), "r"(0) : "memory");
                        tcgen05_commit(e_ready_addr);
                        tcgen05_commit(e_committed_addr);
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
