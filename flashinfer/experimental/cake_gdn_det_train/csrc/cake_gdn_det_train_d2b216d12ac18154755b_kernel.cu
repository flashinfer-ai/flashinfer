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
#define TMEM_ACC_COLS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_S_KN_OFF 0
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 16384
#define SMEM_S_V_OFF 16384
#define SMEM_S_V_STAGE_BYTES 16384
#define SMEM_S_V_STRIDE 16384
#define SMEM_S_DW_OFF 32768
#define SMEM_S_DW_STAGE_BYTES 16384
#define SMEM_S_DW_STRIDE 16384
#define SMEM_S_DU_OFF 49152
#define SMEM_S_DU_STAGE_BYTES 16384
#define SMEM_S_DU_STRIDE 16384
#define SMEM_S_A_OFF 65536
#define SMEM_S_A_STAGE_BYTES 8192
#define SMEM_S_A_STRIDE 8192
#define SMEM_S_DA_OFF 49152
#define SMEM_S_DA_STAGE_BYTES 8192
#define SMEM_S_DA_STRIDE 8192
#define SMEM_S_DA2T_OFF 57344
#define SMEM_S_DA2T_STAGE_BYTES 8192
#define SMEM_S_DA2T_STRIDE 8192
#define SMEM_S_DA3B_OFF 16384
#define SMEM_S_DA3B_STAGE_BYTES 8192
#define SMEM_S_DA3B_STRIDE 8192
#define SMEM_S_AT_OFF 65536
#define SMEM_S_AT_STAGE_BYTES 8192
#define SMEM_S_AT_STRIDE 8192
#define SMEM_S_DUT_OFF 49152
#define SMEM_S_DUT_STAGE_BYTES 16384
#define SMEM_S_DUT_STRIDE 16384
#define SMEM_S_DWT_OFF 32768
#define SMEM_S_DWT_STAGE_BYTES 16384
#define SMEM_S_DWT_STRIDE 16384
#define SMEM_S_KNT_OFF 0
#define SMEM_S_KNT_STAGE_BYTES 16384
#define SMEM_S_KNT_STRIDE 16384
#define SMEM_S_DA3BT_OFF 16384
#define SMEM_S_DA3BT_STAGE_BYTES 8192
#define SMEM_S_DA3BT_STRIDE 8192
#define SMEM_S_G_OFF 73728
#define SMEM_S_G_STAGE_BYTES 256
#define SMEM_S_G_STRIDE 256
#define SMEM_S_B_OFF 73984
#define SMEM_S_B_STAGE_BYTES 256
#define SMEM_S_B_STRIDE 256
#define SMEM_S_GEXP_OFF 74240
#define SMEM_S_GEXP_STAGE_BYTES 256
#define SMEM_S_GEXP_STRIDE 256
#define SMEM_S_COL_OFF 74496
#define SMEM_S_COL_STAGE_BYTES 1024
#define SMEM_S_COL_STRIDE 1024
#define SMEM_S_DBP_OFF 49152
#define SMEM_S_DBP_STAGE_BYTES 512
#define SMEM_S_DBP_STRIDE 512
#define SMEM_S_DGP_OFF 49664
#define SMEM_S_DGP_STAGE_BYTES 512
#define SMEM_S_DGP_STRIDE 512
#define SMEM_TOTAL 75648
#define THREADS 256
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







__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}




__device__ __forceinline__ void tcgen05_commit_leader(
    int mbar_addr, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %1, 0;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(leader) : "memory");
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_det_train_d2b216d12ac18154755b(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ dw, __nv_bfloat16* __restrict__ du, float* __restrict__ g_cs, float* __restrict__ beta, __nv_bfloat16* __restrict__ dk2_out, __nv_bfloat16* __restrict__ dv_out, float* __restrict__ dbeta_out, float* __restrict__ dg2_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads)
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

    const int mbar_base = smem + 75520;
    #define m1a_addr (mbar_base + 0)
    #define m1b_addr (mbar_base + 8)
    #define m2_addr (mbar_base + 16)
    #define m2b_addr (mbar_base + 24)
    #define m3b_addr (mbar_base + 32)
    #define m3_addr (mbar_base + 40)
    #define m4_addr (mbar_base + 48)
    #define m5_addr (mbar_base + 56)
    #define l1_addr (mbar_base + 64)
    #define l2_addr (mbar_base + 72)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_kn_addr = smem + 0;
    __nv_bfloat16* s_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_v_addr = smem + 16384;
    __nv_bfloat16* s_dw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_dw_addr = smem + 32768;
    __nv_bfloat16* s_du = reinterpret_cast<__nv_bfloat16*>(smem_raw + 49152);
    const int s_du_addr = smem + 49152;
    __nv_bfloat16* s_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_a_addr = smem + 65536;
    __nv_bfloat16* s_da = reinterpret_cast<__nv_bfloat16*>(smem_raw + 49152);
    const int s_da_addr = smem + 49152;
    __nv_bfloat16* s_da2T = reinterpret_cast<__nv_bfloat16*>(smem_raw + 57344);
    const int s_da2T_addr = smem + 57344;
    __nv_bfloat16* s_da3b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_da3b_addr = smem + 16384;
    __nv_bfloat16* s_aT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_aT_addr = smem + 65536;
    __nv_bfloat16* s_duT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 49152);
    const int s_duT_addr = smem + 49152;
    __nv_bfloat16* s_dwT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_dwT_addr = smem + 32768;
    __nv_bfloat16* s_knT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_knT_addr = smem + 0;
    __nv_bfloat16* s_da3bT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_da3bT_addr = smem + 16384;
    float* s_g = reinterpret_cast<float*>(smem_raw + 73728);
    const int s_g_addr = smem + 73728;
    float* s_b = reinterpret_cast<float*>(smem_raw + 73984);
    const int s_b_addr = smem + 73984;
    float* s_gexp = reinterpret_cast<float*>(smem_raw + 74240);
    const int s_gexp_addr = smem + 74240;
    float* s_col = reinterpret_cast<float*>(smem_raw + 74496);
    const int s_col_addr = smem + 74496;
    float* s_dbp = reinterpret_cast<float*>(smem_raw + 49152);
    const int s_dbp_addr = smem + 49152;
    float* s_dgp = reinterpret_cast<float*>(smem_raw + 49664);
    const int s_dgp_addr = smem + 49664;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[75520..75600)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // m1a: 1 barriers, init_count=1
            mbarrier_init(smem + 75520, 1);
            // m1b: 1 barriers, init_count=1
            mbarrier_init(smem + 75528, 1);
            // m2: 1 barriers, init_count=1
            mbarrier_init(smem + 75536, 1);
            // m2b: 1 barriers, init_count=1
            mbarrier_init(smem + 75544, 1);
            // m3b: 1 barriers, init_count=1
            mbarrier_init(smem + 75552, 1);
            // m3: 1 barriers, init_count=1
            mbarrier_init(smem + 75560, 1);
            // m4: 1 barriers, init_count=1
            mbarrier_init(smem + 75568, 1);
            // m5: 1 barriers, init_count=1
            mbarrier_init(smem + 75576, 1);
            // l1: 1 barriers, init_count=256
            mbarrier_init(smem + 75584, 256);
            // l2: 1 barriers, init_count=256
            mbarrier_init(smem + 75592, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 75600);
    if (warp == 0) {
        int _tmem_hold = smem + 75600;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_cols = taddr;

    // === Task calls (declaration order) ===
    int chunk = blockIdx.x;
    int hv = blockIdx.y;
    int tok0 = chunk_start[chunk];
    int n_valid = chunk_len[chunk];
    int lane_0 = lane;
    int warp_1 = warp;
    int tid_2 = tid;
    int group = num_v_heads / num_heads;
    int hq = hv / group;
    int r0 = tid / 8;
    int c0 = tid % 8 * 8;
    long long row_stride = (long long)num_v_heads * 64;
    long long src_row = (long long)(tok0 + r0) * row_stride + (long long)hv * 64 + (long long)c0;
    unsigned int dst0 = s_a_addr + (unsigned int)(c0 / 64 * 8192 + (r0 * 128 + c0 % 64 * 2 ^ (r0 * 128 + c0 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0), "l"(A + src_row), "r"((n_valid > r0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 4096), "l"(A + (src_row + 32 * row_stride)), "r"((n_valid > r0 + 32) ? 16 : 0));
    int r0_0 = tid / 16;
    int c0_1 = tid % 16 * 8;
    long long row_stride_2 = (long long)num_v_heads * 128;
    long long src_row_3 = (long long)(tok0 + r0_0) * row_stride_2 + (long long)hv * 128 + (long long)c0_1;
    unsigned int dst0_4 = s_du_addr + (unsigned int)(c0_1 / 64 * 8192 + (r0_0 * 128 + c0_1 % 64 * 2 ^ (r0_0 * 128 + c0_1 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_4), "l"(du + src_row_3), "r"((n_valid > r0_0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_4 + 2048), "l"(du + (src_row_3 + 16 * row_stride_2)), "r"((n_valid > r0_0 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_4 + 4096), "l"(du + (src_row_3 + 32 * row_stride_2)), "r"((n_valid > r0_0 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_4 + 6144), "l"(du + (src_row_3 + 48 * row_stride_2)), "r"((n_valid > r0_0 + 48) ? 16 : 0));
    int r0_5 = tid / 16;
    int c0_6 = tid % 16 * 8;
    long long row_stride_7 = (long long)num_v_heads * 128;
    long long src_row_8 = (long long)(tok0 + r0_5) * row_stride_7 + (long long)hv * 128 + (long long)c0_6;
    unsigned int dst0_9 = s_v_addr + (unsigned int)(c0_6 / 64 * 8192 + (r0_5 * 128 + c0_6 % 64 * 2 ^ (r0_5 * 128 + c0_6 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_9), "l"(v + src_row_8), "r"((n_valid > r0_5) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_9 + 2048), "l"(v + (src_row_8 + 16 * row_stride_7)), "r"((n_valid > r0_5 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_9 + 4096), "l"(v + (src_row_8 + 32 * row_stride_7)), "r"((n_valid > r0_5 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_9 + 6144), "l"(v + (src_row_8 + 48 * row_stride_7)), "r"((n_valid > r0_5 + 48) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    int r0_10 = tid / 16;
    int c0_11 = tid % 16 * 8;
    long long row_stride_12 = (long long)num_heads * 128;
    long long src_row_13 = (long long)(tok0 + r0_10) * row_stride_12 + (long long)hq * 128 + (long long)c0_11;
    unsigned int dst0_14 = s_kn_addr + (unsigned int)(c0_11 / 64 * 8192 + (r0_10 * 128 + c0_11 % 64 * 2 ^ (r0_10 * 128 + c0_11 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_14), "l"(kn + src_row_13), "r"((n_valid > r0_10) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_14 + 2048), "l"(kn + (src_row_13 + 16 * row_stride_12)), "r"((n_valid > r0_10 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_14 + 4096), "l"(kn + (src_row_13 + 32 * row_stride_12)), "r"((n_valid > r0_10 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_14 + 6144), "l"(kn + (src_row_13 + 48 * row_stride_12)), "r"((n_valid > r0_10 + 48) ? 16 : 0));
    int r0_15 = tid / 16;
    int c0_16 = tid % 16 * 8;
    long long row_stride_17 = (long long)num_v_heads * 128;
    long long src_row_18 = (long long)(tok0 + r0_15) * row_stride_17 + (long long)hv * 128 + (long long)c0_16;
    unsigned int dst0_19 = s_dw_addr + (unsigned int)(c0_16 / 64 * 8192 + (r0_15 * 128 + c0_16 % 64 * 2 ^ (r0_15 * 128 + c0_16 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_19), "l"(dw + src_row_18), "r"((n_valid > r0_15) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_19 + 2048), "l"(dw + (src_row_18 + 16 * row_stride_17)), "r"((n_valid > r0_15 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_19 + 4096), "l"(dw + (src_row_18 + 32 * row_stride_17)), "r"((n_valid > r0_15 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_19 + 6144), "l"(dw + (src_row_18 + 48 * row_stride_17)), "r"((n_valid > r0_15 + 48) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    int rb = warp_1 % 4;
    int nh = warp_1 / 4;
    int n0 = nh * 64;
    int s0 = nh * 32;
    int row_a = rb * 16 + lane_0 / 4;
    int row_b = row_a + 8;
    int col_q = lane_0 % 4 * 2;
    int taddr_1 = tmem_acc_cols + (rb * 32 << 16);
    int ld_w = taddr_1 + n0;
    int ld_n = taddr_1 + s0;
    int st_row = rb * 16 + lane_0 / 8 % 2 * 8 + lane_0 % 8;
    int st_col = s0 + lane_0 / 16 * 8;
    int stT_row = s0 + lane_0 / 16 * 8 + lane_0 % 8;
    int stT_col = rb * 16 + lane_0 / 8 % 2 * 8;
    float acc[32];
    float acc_s[32];
    float acc2[16];
    float acc7[16];
    float vals[8];
    float valsb[8];
    unsigned int pk[4];
    unsigned int pair_a[1];
    unsigned int pair_b[1];
    float gq[2];
    float bq[2];
    float col_part[8];
    float db_a = 0.0f;
    float db_b = 0.0f;
    float dg_a = 0.0f;
    float dg_b = 0.0f;
    long long out_a = ((long long)(tok0 + row_a) * (long long)num_v_heads + (long long)hv) * 128 + (long long)n0 + (long long)col_q;
    long long out_b = ((long long)(tok0 + row_b) * (long long)num_v_heads + (long long)hv) * 128 + (long long)n0 + (long long)col_q;
    if (tid_2 < 64) {
        float gate_val = 0.0f;
        float beta_val = 0.0f;
        if (tid_2 < n_valid) {
            long long gate_index = (long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv;
            gate_val = g_cs[gate_index];
            beta_val = beta[gate_index];
        }
        s_g[tid_2] = gate_val;
        s_b[tid_2] = beta_val;
        float _exp2_0 = approx_exp2(gate_val);
        s_gexp[tid_2] = _exp2_0;
    }
    asm volatile("cp.async.wait_group 1;");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned long long _mbarrier_arrival_token_1;
    asm volatile(
        "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
        : "=l"(_mbarrier_arrival_token_1)
        : "l"(smem_raw + ((l1_addr) - smem)), "r"((uint32_t)(1)) : "memory");
    __syncthreads();
    float g_a = s_g[row_a];
    float g_b = s_g[row_b];
    float beta_a = s_b[row_a];
    float beta_b = s_b[row_b];
    unsigned int _phase_l1_0 = 0;
    mbarrier_wait(l1_addr, _phase_l1_0);
    _phase_l1_0 ^= 1;
    if (warp == 0) {
        uint32_t _elected_2 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_2));
        int _smem_desc_lo_3 = (((s_aT_addr) >> 4) & 0x3FFF) | 0x2000000;
        int _smem_desc_lo_4 = (((s_duT_addr) >> 4) & 0x3FFF) | 0x2000000;
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "setp.ne.b32 leader, %4, 0;\n\t"
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
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_3), "r"(_smem_desc_lo_4), "r"(tmem_acc_cols), "r"(0), "r"(_elected_2) : "memory");
        tcgen05_commit_leader(m1a_addr, _elected_2);
        int _smem_desc_lo_5 = ((s_du_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_6 = ((s_v_addr) >> 4) & 0x3FFF;
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "setp.ne.b32 leader, %4, 0;\n\t"
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
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 506;\n\t"
            "add.u32 blo, blo, 506;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_5), "r"(_smem_desc_lo_6), "r"((tmem_acc_cols + (128))), "r"(0), "r"(_elected_2) : "memory");
        tcgen05_commit_leader(m1b_addr, _elected_2);
    }
    asm volatile("cp.async.wait_group 0;");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    unsigned long long _mbarrier_arrival_token_7;
    asm volatile(
        "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
        : "=l"(_mbarrier_arrival_token_7)
        : "l"(smem_raw + ((l2_addr) - smem)), "r"((uint32_t)(1)) : "memory");
    __syncthreads();
    unsigned int _phase_l2_0 = 0;
    mbarrier_wait(l2_addr, _phase_l2_0);
    _phase_l2_0 ^= 1;
    if (warp == 1) {
        uint32_t _elected_8 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_8));
        int _smem_desc_lo_9 = ((s_dw_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_10 = ((s_kn_addr) >> 4) & 0x3FFF;
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "setp.ne.b32 leader, %4, 0;\n\t"
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
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 506;\n\t"
            "add.u32 blo, blo, 506;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_9), "r"(_smem_desc_lo_10), "r"((tmem_acc_cols + (1048704))), "r"(0), "r"(_elected_8) : "memory");
        tcgen05_commit_leader(m2_addr, _elected_8);
    }
    if (warp == 2) {
        uint32_t _elected_11 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_11));
        int _mma_a_lo_0 = ((s_kn_addr) >> 4) & 0x3FFF;
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "setp.ne.b32 leader, %4, 0;\n\t"
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
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 506;\n\t"
            "add.u32 blo, blo, 506;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_0), "r"(_mma_a_lo_0), "r"((tmem_acc_cols + (192))), "r"(0), "r"(_elected_11) : "memory");
        tcgen05_commit_leader(m2b_addr, _elected_11);
    }
    if (warp == 3) {
        uint32_t _elected_12 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_12));
        int _smem_desc_lo_13 = (((s_aT_addr) >> 4) & 0x3FFF) | 0x2000000;
        int _smem_desc_lo_14 = (((s_dwT_addr) >> 4) & 0x3FFF) | 0x2000000;
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "setp.ne.b32 leader, %4, 0;\n\t"
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
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_13), "r"(_smem_desc_lo_14), "r"((tmem_acc_cols + (1048576))), "r"(0), "r"(_elected_12) : "memory");
        tcgen05_commit_leader(m3b_addr, _elected_12);
    }
    unsigned int _phase_m1a_0 = 0;
    mbarrier_wait(m1a_addr, _phase_m1a_0);
    _phase_m1a_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15])), "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
        : "r"(ld_w));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    #pragma unroll
    for (int nb6 = 0; nb6 < 8; nb6++) {
        int vcol = n0 + nb6 * 8 + col_q;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(s_v_addr + (unsigned int)(vcol / 64 * 8192 + (row_a * 128 + vcol % 64 * 2 ^ (row_a * 128 + vcol % 64 * 2 >> 7 & 7) << 4))));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(s_v_addr + (unsigned int)(vcol / 64 * 8192 + (row_b * 128 + vcol % 64 * 2 ^ (row_b * 128 + vcol % 64 * 2 >> 7 & 7) << 4))));
        db_a = __fmaf_rn(acc[nb6 * 4], __uint_as_float(pair_a[0] << 16), db_a);
        db_b = __fmaf_rn(acc[nb6 * 4 + 2], __uint_as_float(pair_b[0] << 16), db_b);
        db_a = __fmaf_rn(acc[nb6 * 4 + 1], __uint_as_float(pair_a[0] & 4294901760u), db_a);
        db_b = __fmaf_rn(acc[nb6 * 4 + 3], __uint_as_float(pair_b[0] & 4294901760u), db_b);
        acc[nb6 * 4] = acc[nb6 * 4] * beta_a;
        acc[nb6 * 4 + 1] = acc[nb6 * 4 + 1] * beta_a;
        acc[nb6 * 4 + 2] = acc[nb6 * 4 + 2] * beta_b;
        acc[nb6 * 4 + 3] = acc[nb6 * 4 + 3] * beta_b;
        if (row_a < n_valid) {
            unsigned int _packed_0[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + nb6 * 4)[_lp*2 + 0], (acc + nb6 * 4)[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(dv_out + (out_a + (long long)(nb6 * 8)))[0] = reinterpret_cast<unsigned*>(_packed_0)[0];
        }
        if (row_b < n_valid) {
            unsigned int _packed_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + nb6 * 4 + 2)[_lp*2 + 0], (acc + nb6 * 4 + 2)[_lp*2+1 + 0]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(dv_out + (out_b + (long long)(nb6 * 8)))[0] = reinterpret_cast<unsigned*>(_packed_1)[0];
        }
    }
    unsigned int _phase_m2_0 = 0;
    mbarrier_wait(m2_addr, _phase_m2_0);
    _phase_m2_0 ^= 1;
    unsigned int _phase_m1b_0 = 0;
    mbarrier_wait(m1b_addr, _phase_m1b_0);
    _phase_m1b_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[15]))
        : "r"(ld_n + 1048704));
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc2[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[15]))
        : "r"(ld_n + 128));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    #pragma unroll
    for (int p = 0; p < 2; p++) {
        #pragma unroll
        for (int q = 0; q < 2; q++) {
            int col0 = s0 + p * 16 + q * 8 + col_q;
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gq[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq[(0) + 1]))
                : "r"(s_gexp_addr + (unsigned int)(col0 * 4)));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bq[0])), "=r"(*reinterpret_cast<uint32_t*>(&bq[(0) + 1]))
                : "r"(s_b_addr + (unsigned int)(col0 * 4)));
            float da_a0 = __fmaf_rn(acc_s[(2 * p + q) * 4], gq[0], acc2[(2 * p + q) * 4]) * bq[0];
            float da_a1 = __fmaf_rn(acc_s[(2 * p + q) * 4 + 1], gq[1], acc2[(2 * p + q) * 4 + 1]) * bq[1];
            float da_b0 = __fmaf_rn(acc_s[(2 * p + q) * 4 + 2], gq[0], acc2[(2 * p + q) * 4 + 2]) * bq[0];
            float da_b1 = __fmaf_rn(acc_s[(2 * p + q) * 4 + 3], gq[1], acc2[(2 * p + q) * 4 + 3]) * bq[1];
            float val_a0 = 0.0f;
            float val_a1 = 0.0f;
            float val_b0 = 0.0f;
            float val_b1 = 0.0f;
            if (col0 < row_a) {
                if (row_a < n_valid) {
                    val_a0 = da_a0;
                }
            }
            if (row_a > col0 + 1) {
                if (row_a < n_valid) {
                    val_a1 = da_a1;
                }
            }
            if (col0 < row_b) {
                if (row_b < n_valid) {
                    val_b0 = da_b0;
                }
            }
            if (row_b > col0 + 1) {
                if (row_b < n_valid) {
                    val_b1 = da_b1;
                }
            }
            vals[q * 4] = val_a0;
            vals[q * 4 + 1] = val_a1;
            vals[q * 4 + 2] = val_b0;
            vals[q * 4 + 3] = val_b1;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
            pk[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(s_da_addr + (unsigned int)((st_col + p * 16) / 64 * 8192 + (st_row * 128 + (st_col + p * 16) % 64 * 2 ^ (st_row * 128 + (st_col + p * 16) % 64 * 2 >> 7 & 7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3]))
            : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        uint32_t _elected_16 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_16));
        int _smem_desc_lo_17 = ((s_da_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_18 = ((s_a_addr) >> 4) & 0x3FFF;
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "setp.ne.b32 leader, %4, 0;\n\t"
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
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_17), "r"(_smem_desc_lo_18), "r"((tmem_acc_cols + (128))), "r"(0), "r"(_elected_16) : "memory");
        tcgen05_commit_leader(m3_addr, _elected_16);
    }
    unsigned int _phase_m3_0 = 0;
    mbarrier_wait(m3_addr, _phase_m3_0);
    _phase_m3_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc2[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[15]))
        : "r"(ld_n + 128));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    #pragma unroll
    for (int p_1 = 0; p_1 < 2; p_1++) {
        #pragma unroll
        for (int e4 = 0; e4 < 8; e4++) {
            vals[e4] = acc2[p_1 * 8 + e4];
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
            pk[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(s_da2T_addr + (unsigned int)(stT_col / 64 * 8192 + ((stT_row + p_1 * 16) * 128 + stT_col % 64 * 2 ^ ((stT_row + p_1 * 16) * 128 + stT_col % 64 * 2 >> 7 & 7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3]))
            : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        uint32_t _elected_20 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_20));
        int _smem_desc_lo_21 = (((s_aT_addr) >> 4) & 0x3FFF) | 0x2000000;
        int _smem_desc_lo_22 = (((s_da2T_addr) >> 4) & 0x3FFF) | 0x2000000;
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "setp.ne.b32 leader, %4, 0;\n\t"
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
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_21), "r"(_smem_desc_lo_22), "r"((tmem_acc_cols + (1048704))), "r"(0), "r"(_elected_20) : "memory");
        tcgen05_commit_leader(m4_addr, _elected_20);
    }
    unsigned int _phase_m4_0 = 0;
    mbarrier_wait(m4_addr, _phase_m4_0);
    _phase_m4_0 ^= 1;
    unsigned int _phase_m2b_0 = 0;
    mbarrier_wait(m2b_addr, _phase_m2b_0);
    _phase_m2b_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[15]))
        : "r"(ld_n + 1048704));
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc7[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc7[15]))
        : "r"(ld_n + 192));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    col_part[0] = 0.0f;
    col_part[1] = 0.0f;
    col_part[2] = 0.0f;
    col_part[3] = 0.0f;
    col_part[4] = 0.0f;
    col_part[5] = 0.0f;
    col_part[6] = 0.0f;
    col_part[7] = 0.0f;
    #pragma unroll
    for (int p_2 = 0; p_2 < 2; p_2++) {
        #pragma unroll
        for (int q_1 = 0; q_1 < 2; q_1++) {
            int col0_1 = s0 + p_2 * 16 + q_1 * 8 + col_q;
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gq[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq[(0) + 1]))
                : "r"(s_g_addr + (unsigned int)(col0_1 * 4)));
            float _exp2_23 = approx_exp2(g_a - gq[0]);
            float va0 = (-acc_s[(2 * p_2 + q_1) * 4]) * _exp2_23;
            float _exp2_24 = approx_exp2(g_a - gq[1]);
            float va1 = (-acc_s[(2 * p_2 + q_1) * 4 + 1]) * _exp2_24;
            float _exp2_25 = approx_exp2(g_b - gq[0]);
            float vb0 = (-acc_s[(2 * p_2 + q_1) * 4 + 2]) * _exp2_25;
            float _exp2_26 = approx_exp2(g_b - gq[1]);
            float vb1 = (-acc_s[(2 * p_2 + q_1) * 4 + 3]) * _exp2_26;
            float val_a0_1 = 0.0f;
            float val_a1_1 = 0.0f;
            float val_b0_1 = 0.0f;
            float val_b1_1 = 0.0f;
            if (col0_1 < row_a) {
                if (row_a < n_valid) {
                    val_a0_1 = va0;
                }
            }
            if (row_a > col0_1 + 1) {
                if (row_a < n_valid) {
                    val_a1_1 = va1;
                }
            }
            if (col0_1 < row_b) {
                if (row_b < n_valid) {
                    val_b0_1 = vb0;
                }
            }
            if (row_b > col0_1 + 1) {
                if (row_b < n_valid) {
                    val_b1_1 = vb1;
                }
            }
            float t_a0 = val_a0_1 * acc7[(2 * p_2 + q_1) * 4];
            float t_a1 = val_a1_1 * acc7[(2 * p_2 + q_1) * 4 + 1];
            float t_b0 = val_b0_1 * acc7[(2 * p_2 + q_1) * 4 + 2];
            float t_b1 = val_b1_1 * acc7[(2 * p_2 + q_1) * 4 + 3];
            db_a = db_a + t_a0;
            db_a = db_a + t_a1;
            db_b = db_b + t_b0;
            db_b = db_b + t_b1;
            float ada_a0 = t_a0 * beta_a;
            float ada_b0 = t_b0 * beta_b;
            dg_a = dg_a + ada_a0;
            dg_b = dg_b + ada_b0;
            col_part[(2 * p_2 + q_1) * 2] = ada_a0 + ada_b0;
            float ada_a1 = t_a1 * beta_a;
            float ada_b1 = t_b1 * beta_b;
            dg_a = dg_a + ada_a1;
            dg_b = dg_b + ada_b1;
            col_part[(2 * p_2 + q_1) * 2 + 1] = ada_a1 + ada_b1;
            valsb[q_1 * 4] = val_a0_1 * beta_a;
            valsb[q_1 * 4 + 1] = val_a1_1 * beta_a;
            valsb[q_1 * 4 + 2] = val_b0_1 * beta_b;
            valsb[q_1 * 4 + 3] = val_b1_1 * beta_b;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(valsb[_lp*2 + 0], valsb[_lp*2+1 + 0]));
            pk[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_27 = static_cast<uint32_t>(s_da3b_addr + (unsigned int)((st_col + p_2 * 16) / 64 * 8192 + (st_row * 128 + (st_col + p_2 * 16) % 64 * 2 ^ (st_row * 128 + (st_col + p_2 * 16) % 64 * 2 >> 7 & 7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_27), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3]))
            : "memory");
    }
    #pragma unroll
    for (int ci = 0; ci < 8; ci++) {
        float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, col_part[ci], 4);
        col_part[ci] = col_part[ci] + _shfl_xor_sync_28;
        float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, col_part[ci], 8);
        col_part[ci] = col_part[ci] + _shfl_xor_sync_29;
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, col_part[ci], 16);
        col_part[ci] = col_part[ci] + _shfl_xor_sync_30;
    }
    if (lane_0 < 4) {
        #pragma unroll
        for (int nb14 = 0; nb14 < 4; nb14++) {
            #pragma unroll
            for (int e7 = 0; e7 < 2; e7++) {
                s_col[rb * 64 + s0 + nb14 * 8 + col_q + e7] = col_part[nb14 * 2 + e7];
            }
        }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        uint32_t _elected_31 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_31));
        int _mma_a_lo_1 = ((s_da3b_addr) >> 4) & 0x3FFF;
        int _mma_b_lo_0 = (((s_knT_addr) >> 4) & 0x3FFF) | 0x2000000;
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "setp.ne.b32 leader, %4, 0;\n\t"
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
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_0), "r"(tmem_acc_cols), "r"(0), "r"(_elected_31) : "memory");
        int _smem_desc_lo_32 = (((s_da3bT_addr) >> 4) & 0x3FFF) | 0x2000000;
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "setp.ne.b32 leader, %4, 0;\n\t"
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
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_32), "r"(_mma_b_lo_0), "r"(tmem_acc_cols), "r"(1), "r"(_elected_31) : "memory");
        tcgen05_commit_leader(m5_addr, _elected_31);
    }
    unsigned int _phase_m5_0 = 0;
    mbarrier_wait(m5_addr, _phase_m5_0);
    _phase_m5_0 ^= 1;
    unsigned int _phase_m3b_0 = 0;
    mbarrier_wait(m3b_addr, _phase_m3b_0);
    _phase_m3b_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    float g5_a = s_g[row_a];
    float g5_b = s_g[row_b];
    float gexp_a = approx_exp2(g5_a);
    float gexp_b = approx_exp2(g5_b);
    float bg_a = gexp_a * beta_a;
    float bg_b = gexp_b * beta_b;
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
        : "r"(ld_w));
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[15]))
        : "r"(ld_w + 1048576));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int nb4 = 0; nb4 < 4; nb4++) {
        int kcol = n0 + nb4 * 8 + col_q;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(s_kn_addr + (unsigned int)(kcol / 64 * 8192 + (row_a * 128 + kcol % 64 * 2 ^ (row_a * 128 + kcol % 64 * 2 >> 7 & 7) << 4))));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(s_kn_addr + (unsigned int)(kcol / 64 * 8192 + (row_b * 128 + kcol % 64 * 2 ^ (row_b * 128 + kcol % 64 * 2 >> 7 & 7) << 4))));
        float k_a0 = __uint_as_float(pair_a[0] << 16);
        float k_b0 = __uint_as_float(pair_b[0] << 16);
        float k_a1 = __uint_as_float(pair_a[0] & 4294901760u);
        float k_b1 = __uint_as_float(pair_b[0] & 4294901760u);
        db_a = __fmaf_rn(acc_s[nb4 * 4], k_a0 * gexp_a, db_a);
        db_b = __fmaf_rn(acc_s[nb4 * 4 + 2], k_b0 * gexp_b, db_b);
        dg_a = __fmaf_rn(acc_s[nb4 * 4], k_a0 * bg_a, dg_a);
        dg_b = __fmaf_rn(acc_s[nb4 * 4 + 2], k_b0 * bg_b, dg_b);
        db_a = __fmaf_rn(acc_s[nb4 * 4 + 1], k_a1 * gexp_a, db_a);
        db_b = __fmaf_rn(acc_s[nb4 * 4 + 3], k_b1 * gexp_b, db_b);
        dg_a = __fmaf_rn(acc_s[nb4 * 4 + 1], k_a1 * bg_a, dg_a);
        dg_b = __fmaf_rn(acc_s[nb4 * 4 + 3], k_b1 * bg_b, dg_b);
        acc[nb4 * 4] = __fmaf_rn(acc_s[nb4 * 4], bg_a, acc[nb4 * 4]);
        acc[nb4 * 4 + 1] = __fmaf_rn(acc_s[nb4 * 4 + 1], bg_a, acc[nb4 * 4 + 1]);
        acc[nb4 * 4 + 2] = __fmaf_rn(acc_s[nb4 * 4 + 2], bg_b, acc[nb4 * 4 + 2]);
        acc[nb4 * 4 + 3] = __fmaf_rn(acc_s[nb4 * 4 + 3], bg_b, acc[nb4 * 4 + 3]);
    }
    #pragma unroll
    for (int nb18 = 0; nb18 < 4; nb18++) {
        if (row_a < n_valid) {
            unsigned int _packed_2[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + nb18 * 4)[_lp*2 + 0], (acc + nb18 * 4)[_lp*2+1 + 0]));
                _packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(dk2_out + (out_a + (long long)(nb18 * 8)))[0] = reinterpret_cast<unsigned*>(_packed_2)[0];
        }
        if (row_b < n_valid) {
            unsigned int _packed_3[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + nb18 * 4 + 2)[_lp*2 + 0], (acc + nb18 * 4 + 2)[_lp*2+1 + 0]));
                _packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(dk2_out + (out_b + (long long)(nb18 * 8)))[0] = reinterpret_cast<unsigned*>(_packed_3)[0];
        }
    }
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
        : "r"(ld_w + 32));
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[15]))
        : "r"(ld_w + 1048608));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    #pragma unroll
    for (int nb4_1 = 4; nb4_1 < 8; nb4_1++) {
        int kcol_1 = n0 + nb4_1 * 8 + col_q;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(s_kn_addr + (unsigned int)(kcol_1 / 64 * 8192 + (row_a * 128 + kcol_1 % 64 * 2 ^ (row_a * 128 + kcol_1 % 64 * 2 >> 7 & 7) << 4))));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(s_kn_addr + (unsigned int)(kcol_1 / 64 * 8192 + (row_b * 128 + kcol_1 % 64 * 2 ^ (row_b * 128 + kcol_1 % 64 * 2 >> 7 & 7) << 4))));
        float k_a0_1 = __uint_as_float(pair_a[0] << 16);
        float k_b0_1 = __uint_as_float(pair_b[0] << 16);
        float k_a1_1 = __uint_as_float(pair_a[0] & 4294901760u);
        float k_b1_1 = __uint_as_float(pair_b[0] & 4294901760u);
        db_a = __fmaf_rn(acc_s[(nb4_1 - 4) * 4], k_a0_1 * gexp_a, db_a);
        db_b = __fmaf_rn(acc_s[(nb4_1 - 4) * 4 + 2], k_b0_1 * gexp_b, db_b);
        dg_a = __fmaf_rn(acc_s[(nb4_1 - 4) * 4], k_a0_1 * bg_a, dg_a);
        dg_b = __fmaf_rn(acc_s[(nb4_1 - 4) * 4 + 2], k_b0_1 * bg_b, dg_b);
        db_a = __fmaf_rn(acc_s[(nb4_1 - 4) * 4 + 1], k_a1_1 * gexp_a, db_a);
        db_b = __fmaf_rn(acc_s[(nb4_1 - 4) * 4 + 3], k_b1_1 * gexp_b, db_b);
        dg_a = __fmaf_rn(acc_s[(nb4_1 - 4) * 4 + 1], k_a1_1 * bg_a, dg_a);
        dg_b = __fmaf_rn(acc_s[(nb4_1 - 4) * 4 + 3], k_b1_1 * bg_b, dg_b);
        acc[(nb4_1 - 4) * 4] = __fmaf_rn(acc_s[(nb4_1 - 4) * 4], bg_a, acc[(nb4_1 - 4) * 4]);
        acc[(nb4_1 - 4) * 4 + 1] = __fmaf_rn(acc_s[(nb4_1 - 4) * 4 + 1], bg_a, acc[(nb4_1 - 4) * 4 + 1]);
        acc[(nb4_1 - 4) * 4 + 2] = __fmaf_rn(acc_s[(nb4_1 - 4) * 4 + 2], bg_b, acc[(nb4_1 - 4) * 4 + 2]);
        acc[(nb4_1 - 4) * 4 + 3] = __fmaf_rn(acc_s[(nb4_1 - 4) * 4 + 3], bg_b, acc[(nb4_1 - 4) * 4 + 3]);
    }
    #pragma unroll
    for (int nb18_1 = 4; nb18_1 < 8; nb18_1++) {
        if (row_a < n_valid) {
            unsigned int _packed_4[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + (nb18_1 - 4) * 4)[_lp*2 + 0], (acc + (nb18_1 - 4) * 4)[_lp*2+1 + 0]));
                _packed_4[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(dk2_out + (out_a + (long long)(nb18_1 * 8)))[0] = reinterpret_cast<unsigned*>(_packed_4)[0];
        }
        if (row_b < n_valid) {
            unsigned int _packed_5[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + (nb18_1 - 4) * 4 + 2)[_lp*2 + 0], (acc + (nb18_1 - 4) * 4 + 2)[_lp*2+1 + 0]));
                _packed_5[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(dk2_out + (out_b + (long long)(nb18_1 * 8)))[0] = reinterpret_cast<unsigned*>(_packed_5)[0];
        }
    }
    float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, db_a, 1);
    db_a = db_a + _shfl_xor_sync_33;
    float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, db_a, 2);
    db_a = db_a + _shfl_xor_sync_34;
    float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, db_b, 1);
    db_b = db_b + _shfl_xor_sync_35;
    float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, db_b, 2);
    db_b = db_b + _shfl_xor_sync_36;
    float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, dg_a, 1);
    dg_a = dg_a + _shfl_xor_sync_37;
    float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, dg_a, 2);
    dg_a = dg_a + _shfl_xor_sync_38;
    float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, dg_b, 1);
    dg_b = dg_b + _shfl_xor_sync_39;
    float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, dg_b, 2);
    dg_b = dg_b + _shfl_xor_sync_40;
    if (lane_0 % 4 == 0) {
        s_dbp[nh * 64 + row_a] = db_a;
        s_dbp[nh * 64 + row_b] = db_b;
        s_dgp[nh * 64 + row_a] = dg_a;
        s_dgp[nh * 64 + row_b] = dg_b;
    }
    __syncthreads();
    if (tid_2 < n_valid) {
        long long gate_row_index = (long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv;
        float colsum = s_col[tid_2] + s_col[64 + tid_2] + s_col[128 + tid_2] + s_col[192 + tid_2];
        float db_row = s_dbp[tid_2] + s_dbp[64 + tid_2];
        float dg_row = s_dgp[tid_2] + s_dgp[64 + tid_2];
        dbeta_out[gate_row_index] = db_row;
        dg2_out[gate_row_index] = dg_row - colsum;
    }
    if (warp_1 == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
}

} // extern "C"
