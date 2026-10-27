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
#define SMEM_S_DO_OFF 0
#define SMEM_S_DO_STAGE_BYTES 16384
#define SMEM_S_DO_STRIDE 16384
#define SMEM_S_V_OFF 0
#define SMEM_S_V_STAGE_BYTES 16384
#define SMEM_S_V_STRIDE 16384
#define SMEM_S_DA3B_OFF 0
#define SMEM_S_DA3B_STAGE_BYTES 8192
#define SMEM_S_DA3B_STRIDE 8192
#define SMEM_S_DA3BT_OFF 0
#define SMEM_S_DA3BT_STAGE_BYTES 8192
#define SMEM_S_DA3BT_STRIDE 8192
#define SMEM_S_VN_OFF 16384
#define SMEM_S_VN_STAGE_BYTES 16384
#define SMEM_S_VN_STRIDE 16384
#define SMEM_S_KN_OFF 16384
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 16384
#define SMEM_S_KNT_OFF 16384
#define SMEM_S_KNT_STAGE_BYTES 16384
#define SMEM_S_KNT_STRIDE 16384
#define SMEM_S_H_OFF 32768
#define SMEM_S_H_STAGE_BYTES 32768
#define SMEM_S_H_STRIDE 32768
#define SMEM_S_HT_OFF 32768
#define SMEM_S_HT_STAGE_BYTES 32768
#define SMEM_S_HT_STRIDE 32768
#define SMEM_S_DW_OFF 32768
#define SMEM_S_DW_STAGE_BYTES 16384
#define SMEM_S_DW_STRIDE 16384
#define SMEM_S_DWT_OFF 32768
#define SMEM_S_DWT_STAGE_BYTES 16384
#define SMEM_S_DWT_STRIDE 16384
#define SMEM_S_DH_OFF 65536
#define SMEM_S_DH_STAGE_BYTES 32768
#define SMEM_S_DH_STRIDE 32768
#define SMEM_S_DHT_OFF 65536
#define SMEM_S_DHT_STAGE_BYTES 32768
#define SMEM_S_DHT_STRIDE 32768
#define SMEM_S_QN_OFF 65536
#define SMEM_S_QN_STAGE_BYTES 16384
#define SMEM_S_QN_STRIDE 16384
#define SMEM_S_QNT_OFF 65536
#define SMEM_S_QNT_STAGE_BYTES 16384
#define SMEM_S_QNT_STRIDE 16384
#define SMEM_S_DA_OFF 65536
#define SMEM_S_DA_STAGE_BYTES 8192
#define SMEM_S_DA_STRIDE 8192
#define SMEM_S_DA2T_OFF 73728
#define SMEM_S_DA2T_STAGE_BYTES 8192
#define SMEM_S_DA2T_STRIDE 8192
#define SMEM_S_DV_OFF 81920
#define SMEM_S_DV_STAGE_BYTES 16384
#define SMEM_S_DV_STRIDE 16384
#define SMEM_S_DUT_OFF 81920
#define SMEM_S_DUT_STAGE_BYTES 16384
#define SMEM_S_DUT_STRIDE 16384
#define SMEM_S_DS_OFF 98304
#define SMEM_S_DS_STAGE_BYTES 8192
#define SMEM_S_DS_STRIDE 8192
#define SMEM_S_DST_OFF 98304
#define SMEM_S_DST_STAGE_BYTES 8192
#define SMEM_S_DST_STRIDE 8192
#define SMEM_S_A_OFF 98304
#define SMEM_S_A_STAGE_BYTES 8192
#define SMEM_S_A_STRIDE 8192
#define SMEM_S_AT_OFF 98304
#define SMEM_S_AT_STAGE_BYTES 8192
#define SMEM_S_AT_STRIDE 8192
#define SMEM_S_B_OFF 49152
#define SMEM_S_B_STAGE_BYTES 256
#define SMEM_S_B_STRIDE 256
#define SMEM_S_GEXP_OFF 49408
#define SMEM_S_GEXP_STAGE_BYTES 256
#define SMEM_S_GEXP_STRIDE 256
#define SMEM_S_COL_OFF 49664
#define SMEM_S_COL_STAGE_BYTES 1024
#define SMEM_S_COL_STRIDE 1024
#define SMEM_S_DBP_OFF 50688
#define SMEM_S_DBP_STAGE_BYTES 512
#define SMEM_S_DBP_STRIDE 512
#define SMEM_S_DGP_OFF 51200
#define SMEM_S_DGP_STAGE_BYTES 512
#define SMEM_S_DGP_STRIDE 512
#define SMEM_S_DG2_OFF 51712
#define SMEM_S_DG2_STAGE_BYTES 256
#define SMEM_S_DG2_STRIDE 256
#define SMEM_S_OWNQ_OFF 0
#define SMEM_S_OWNQ_STAGE_BYTES 8192
#define SMEM_S_OWNQ_STRIDE 8192
#define SMEM_S_OWNK_OFF 8192
#define SMEM_S_OWNK_STAGE_BYTES 8192
#define SMEM_S_OWNK_STRIDE 8192
#define SMEM_S_OWN2_OFF 32768
#define SMEM_S_OWN2_STAGE_BYTES 8192
#define SMEM_S_OWN2_STRIDE 8192
#define SMEM_S_RX_OFF 65536
#define SMEM_S_RX_STAGE_BYTES 24576
#define SMEM_S_RX_STRIDE 24576
#define SMEM_S_QF_OFF 98304
#define SMEM_S_QF_STAGE_BYTES 8192
#define SMEM_S_QF_STRIDE 8192
#define SMEM_S_RS_OFF 52224
#define SMEM_S_RS_STAGE_BYTES 256
#define SMEM_S_RS_STRIDE 256
#define SMEM_S_G_OFF 106496
#define SMEM_S_G_STAGE_BYTES 256
#define SMEM_S_G_STRIDE 256
#define SMEM_S_DGQ_OFF 106752
#define SMEM_S_DGQ_STAGE_BYTES 512
#define SMEM_S_DGQ_STRIDE 512
#define SMEM_S_DGK_OFF 107264
#define SMEM_S_DGK_STAGE_BYTES 512
#define SMEM_S_DGK_STRIDE 512
#define SMEM_S_RED_OFF 107776
#define SMEM_S_RED_STAGE_BYTES 128
#define SMEM_S_RED_STRIDE 128
#define SMEM_TOTAL 108160
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


__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
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






__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_gdn_det_train_34a7a0645157db9e92f4(__nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, float* __restrict__ rstd_q, float* __restrict__ rstd_k, __nv_bfloat16* __restrict__ v_new, __nv_bfloat16* __restrict__ do_, __nv_bfloat16* __restrict__ dv2, __nv_bfloat16* __restrict__ h, __nv_bfloat16* __restrict__ dh, float* __restrict__ g_cs, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ A, float* __restrict__ beta, __nv_bfloat16* __restrict__ dq_out, __nv_bfloat16* __restrict__ dk_out, float* __restrict__ dg_out, __nv_bfloat16* __restrict__ dv_out, __nv_bfloat16* __restrict__ dbeta_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, float scale, int normalize_qk)
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

    const int mbar_base = smem + 107904;
    #define m1_addr (mbar_base + 0)
    #define m2_addr (mbar_base + 8)
    #define m3_addr (mbar_base + 16)
    #define m4_addr (mbar_base + 24)
    #define m5_addr (mbar_base + 32)
    #define m6_addr (mbar_base + 40)
    #define mw1_addr (mbar_base + 48)
    #define mw2_addr (mbar_base + 56)
    #define mw3_addr (mbar_base + 64)
    #define mw4_addr (mbar_base + 72)
    #define mw5_addr (mbar_base + 80)
    #define mw6_addr (mbar_base + 88)
    #define m7_addr (mbar_base + 96)
    #define mw8_addr (mbar_base + 104)
    #define l1_addr (mbar_base + 112)
    #define l2_addr (mbar_base + 120)
    #define l3_addr (mbar_base + 128)
    #define l4_addr (mbar_base + 136)
    #define l5_addr (mbar_base + 144)
    #define l6_addr (mbar_base + 152)
    #define l7_addr (mbar_base + 160)
    #define tx_ok_addr (mbar_base + 168)
    #define rx_full_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* s_do = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_do_addr = smem + 0;
    __nv_bfloat16* s_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_v_addr = smem + 0;
    __nv_bfloat16* s_da3b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_da3b_addr = smem + 0;
    __nv_bfloat16* s_da3bT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_da3bT_addr = smem + 0;
    __nv_bfloat16* s_vn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_vn_addr = smem + 16384;
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_kn_addr = smem + 16384;
    __nv_bfloat16* s_knT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_knT_addr = smem + 16384;
    __nv_bfloat16* s_h = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_h_addr = smem + 32768;
    __nv_bfloat16* s_hT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_hT_addr = smem + 32768;
    __nv_bfloat16* s_dw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_dw_addr = smem + 32768;
    __nv_bfloat16* s_dwT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_dwT_addr = smem + 32768;
    __nv_bfloat16* s_dh = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_dh_addr = smem + 65536;
    __nv_bfloat16* s_dhT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_dhT_addr = smem + 65536;
    __nv_bfloat16* s_qn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_qn_addr = smem + 65536;
    __nv_bfloat16* s_qnT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_qnT_addr = smem + 65536;
    __nv_bfloat16* s_da = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_da_addr = smem + 65536;
    __nv_bfloat16* s_da2T = reinterpret_cast<__nv_bfloat16*>(smem_raw + 73728);
    const int s_da2T_addr = smem + 73728;
    __nv_bfloat16* s_dv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 81920);
    const int s_dv_addr = smem + 81920;
    __nv_bfloat16* s_duT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 81920);
    const int s_duT_addr = smem + 81920;
    __nv_bfloat16* s_ds = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int s_ds_addr = smem + 98304;
    __nv_bfloat16* s_dsT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int s_dsT_addr = smem + 98304;
    __nv_bfloat16* s_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int s_a_addr = smem + 98304;
    __nv_bfloat16* s_aT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int s_aT_addr = smem + 98304;
    float* s_b = reinterpret_cast<float*>(smem_raw + 49152);
    const int s_b_addr = smem + 49152;
    float* s_gexp = reinterpret_cast<float*>(smem_raw + 49408);
    const int s_gexp_addr = smem + 49408;
    float* s_col = reinterpret_cast<float*>(smem_raw + 49664);
    const int s_col_addr = smem + 49664;
    float* s_dbp = reinterpret_cast<float*>(smem_raw + 50688);
    const int s_dbp_addr = smem + 50688;
    float* s_dgp = reinterpret_cast<float*>(smem_raw + 51200);
    const int s_dgp_addr = smem + 51200;
    float* s_dg2 = reinterpret_cast<float*>(smem_raw + 51712);
    const int s_dg2_addr = smem + 51712;
    unsigned int* s_ownq = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int s_ownq_addr = smem + 0;
    unsigned int* s_ownk = reinterpret_cast<unsigned int*>(smem_raw + 8192);
    const int s_ownk_addr = smem + 8192;
    unsigned int* s_own2 = reinterpret_cast<unsigned int*>(smem_raw + 32768);
    const int s_own2_addr = smem + 32768;
    unsigned int* s_rx = reinterpret_cast<unsigned int*>(smem_raw + 65536);
    const int s_rx_addr = smem + 65536;
    unsigned int* s_qf = reinterpret_cast<unsigned int*>(smem_raw + 98304);
    const int s_qf_addr = smem + 98304;
    float* s_rs = reinterpret_cast<float*>(smem_raw + 52224);
    const int s_rs_addr = smem + 52224;
    float* s_g = reinterpret_cast<float*>(smem_raw + 106496);
    const int s_g_addr = smem + 106496;
    float* s_dgq = reinterpret_cast<float*>(smem_raw + 106752);
    const int s_dgq_addr = smem + 106752;
    float* s_dgk = reinterpret_cast<float*>(smem_raw + 107264);
    const int s_dgk_addr = smem + 107264;
    float* s_red = reinterpret_cast<float*>(smem_raw + 107776);
    const int s_red_addr = smem + 107776;

    // Mbarrier init (23 pipeline groups, 0 ordered-sequence groups, 23 barriers)
    // Mbarriers at smem_raw[107904..108088)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // m1: 1 barriers, init_count=1
            mbarrier_init(smem + 107904, 1);
            // m2: 1 barriers, init_count=1
            mbarrier_init(smem + 107912, 1);
            // m3: 1 barriers, init_count=1
            mbarrier_init(smem + 107920, 1);
            // m4: 1 barriers, init_count=1
            mbarrier_init(smem + 107928, 1);
            // m5: 1 barriers, init_count=1
            mbarrier_init(smem + 107936, 1);
            // m6: 1 barriers, init_count=1
            mbarrier_init(smem + 107944, 1);
            // mw1: 1 barriers, init_count=1
            mbarrier_init(smem + 107952, 1);
            // mw2: 1 barriers, init_count=1
            mbarrier_init(smem + 107960, 1);
            // mw3: 1 barriers, init_count=1
            mbarrier_init(smem + 107968, 1);
            // mw4: 1 barriers, init_count=1
            mbarrier_init(smem + 107976, 1);
            // mw5: 1 barriers, init_count=1
            mbarrier_init(smem + 107984, 1);
            // mw6: 1 barriers, init_count=1
            mbarrier_init(smem + 107992, 1);
            // m7: 1 barriers, init_count=1
            mbarrier_init(smem + 108000, 1);
            // mw8: 1 barriers, init_count=1
            mbarrier_init(smem + 108008, 1);
            // l1: 1 barriers, init_count=256
            mbarrier_init(smem + 108016, 256);
            // l2: 1 barriers, init_count=256
            mbarrier_init(smem + 108024, 256);
            // l3: 1 barriers, init_count=256
            mbarrier_init(smem + 108032, 256);
            // l4: 1 barriers, init_count=256
            mbarrier_init(smem + 108040, 256);
            // l5: 1 barriers, init_count=256
            mbarrier_init(smem + 108048, 256);
            // l6: 1 barriers, init_count=256
            mbarrier_init(smem + 108056, 256);
            // l7: 1 barriers, init_count=256
            mbarrier_init(smem + 108064, 256);
            // tx_ok: 1 barriers, init_count=1
            mbarrier_init(smem + 108072, 1);
            // rx_full: 1 barriers, init_count=1
            mbarrier_init(smem + 108080, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 108080, 24576);
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 108088);
    if (warp == 0) {
        int _tmem_hold = smem + 108088;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_cols = taddr;

    // === Task calls (declaration order) ===
    int lane_0 = lane;
    int warp_1 = warp;
    int tid_2 = tid;
    int block_x = blockIdx.x;
    int chunk = block_x / num_v_heads;
    int hv = block_x % num_v_heads;
    int hq = hv / 2;
    int rank = cta_rank;
    int partner = rank ^ 1;
    int tok0 = chunk_start[chunk];
    int n_valid = chunk_len[chunk];
    int r0 = tid / 16;
    int c0 = tid % 16 * 8;
    long long row_stride = (long long)num_v_heads * 128;
    long long src_row = (long long)(tok0 + r0) * row_stride + (long long)hv * 128 + (long long)c0;
    unsigned int dst0 = s_do_addr + (unsigned int)(c0 / 64 * 8192 + (r0 * 128 + c0 % 64 * 2 ^ (r0 * 128 + c0 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0), "l"(do_ + src_row), "r"((n_valid > r0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 2048), "l"(do_ + (src_row + 16 * row_stride)), "r"((n_valid > r0 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 4096), "l"(do_ + (src_row + 32 * row_stride)), "r"((n_valid > r0 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 6144), "l"(do_ + (src_row + 48 * row_stride)), "r"((n_valid > r0 + 48) ? 16 : 0));
    int r0_3 = tid / 16;
    int c0_4 = tid % 16 * 8;
    long long row_stride_5 = (long long)num_v_heads * 128;
    long long src_row_6 = (long long)(tok0 + r0_3) * row_stride_5 + (long long)hv * 128 + (long long)c0_4;
    unsigned int dst0_7 = s_vn_addr + (unsigned int)(c0_4 / 64 * 8192 + (r0_3 * 128 + c0_4 % 64 * 2 ^ (r0_3 * 128 + c0_4 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7), "l"(v_new + src_row_6), "r"((n_valid > r0_3) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 2048), "l"(v_new + (src_row_6 + 16 * row_stride_5)), "r"((n_valid > r0_3 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 4096), "l"(v_new + (src_row_6 + 32 * row_stride_5)), "r"((n_valid > r0_3 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 6144), "l"(v_new + (src_row_6 + 48 * row_stride_5)), "r"((n_valid > r0_3 + 48) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    long long state_base = ((long long)chunk * (long long)num_v_heads + (long long)hv) * 16384;
    #pragma unroll
    for (int it = 0; it < 8; it++) {
        int idx = it * 256 + tid_2;
        int dh_row = idx / 16;
        int dh_col = idx % 16 * 8;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(s_dh_addr + (unsigned int)(dh_col / 64 * 16384 + (dh_row * 128 + dh_col % 64 * 2 ^ (dh_row * 128 + dh_col % 64 * 2 >> 7 & 7) << 4))), "l"(dh + (state_base + (long long)dh_row * 128 + (long long)dh_col)));
    }
    asm volatile("cp.async.commit_group;");
    #pragma unroll
    for (int it1 = 0; it1 < 8; it1++) {
        int idx1 = it1 * 256 + tid_2;
        int h_row = idx1 / 16;
        int h_col = idx1 % 16 * 8;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(s_h_addr + (unsigned int)(h_col / 64 * 16384 + (h_row * 128 + h_col % 64 * 2 ^ (h_row * 128 + h_col % 64 * 2 >> 7 & 7) << 4))), "l"(h + (state_base + (long long)h_row * 128 + (long long)h_col)));
    }
    asm volatile("cp.async.commit_group;");
    float beta_t = 0.0f;
    if (tid_2 < 64) {
        float gate_val = 0.0f;
        if (tid_2 < n_valid) {
            long long gate_index = (long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv;
            gate_val = g_cs[gate_index];
            beta_t = beta[gate_index];
        }
        s_g[tid_2] = gate_val;
    }
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
    float acc_s[16];
    float acc[32];
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
    asm volatile("cp.async.wait_group 2;");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned long long _mbarrier_arrival_token_0;
    asm volatile(
        "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
        : "=l"(_mbarrier_arrival_token_0)
        : "l"(smem_raw + ((l1_addr) - smem)), "r"((uint32_t)(1)) : "memory");
    __syncthreads();
    unsigned int _phase_l1_0 = 0;
    mbarrier_wait(l1_addr, _phase_l1_0);
    _phase_l1_0 ^= 1;
    if (warp == 0) {
        uint32_t _elected_1 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_1));
        int _smem_desc_lo_2 = ((s_do_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_3 = ((s_vn_addr) >> 4) & 0x3FFF;
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
            :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"((tmem_acc_cols + (1048704))), "r"(0), "r"(_elected_1) : "memory");
        tcgen05_commit_leader(m1_addr, _elected_1);
    }
    asm volatile("cp.async.wait_group 1;");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned long long _mbarrier_arrival_token_4;
    asm volatile(
        "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
        : "=l"(_mbarrier_arrival_token_4)
        : "l"(smem_raw + ((l2_addr) - smem)), "r"((uint32_t)(1)) : "memory");
    __syncthreads();
    unsigned int _phase_l2_0 = 0;
    mbarrier_wait(l2_addr, _phase_l2_0);
    _phase_l2_0 ^= 1;
    if (warp == 1) {
        uint32_t _elected_5 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_5));
        int _smem_desc_lo_6 = ((s_vn_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_7 = (((s_dhT_addr) >> 4) & 0x3FFF) | 0x4000000;
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
            "add.u32 alo, alo, 506;\n\t"
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
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_acc_cols), "r"(0), "r"(_elected_5) : "memory");
        tcgen05_commit_leader(m2_addr, _elected_5);
    }
    unsigned int _phase_m1_0 = 0;
    mbarrier_wait(m1_addr, _phase_m1_0);
    _phase_m1_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[15]))
        : "r"(ld_n + 1048704));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    float g_a = s_g[row_a];
    float g_b = s_g[row_b];
    #pragma unroll
    for (int p = 0; p < 2; p++) {
        #pragma unroll
        for (int q = 0; q < 2; q++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int col = s0 + p * 16 + q * 8 + col_q + e;
                float g_col = s_g[col];
                float val_a = 0.0f;
                float val_b = 0.0f;
                if (col <= row_a) {
                    if (row_a < n_valid) {
                        float _exp2_8 = approx_exp2(g_a - g_col);
                        val_a = acc_s[(2 * p + q) * 4 + e] * _exp2_8 * scale;
                    }
                }
                if (col <= row_b) {
                    if (row_b < n_valid) {
                        float _exp2_9 = approx_exp2(g_b - g_col);
                        val_b = acc_s[(2 * p + q) * 4 + 2 + e] * _exp2_9 * scale;
                    }
                }
                vals[q * 4 + e] = val_a;
                vals[q * 4 + 2 + e] = val_b;
            }
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
            pk[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(s_ds_addr + (unsigned int)((st_col + p * 16) / 64 * 8192 + (st_row * 128 + (st_col + p * 16) % 64 * 2 ^ (st_row * 128 + (st_col + p * 16) % 64 * 2 >> 7 & 7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3]))
            : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned long long _mbarrier_arrival_token_11;
    asm volatile(
        "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
        : "=l"(_mbarrier_arrival_token_11)
        : "l"(smem_raw + ((l3_addr) - smem)), "r"((uint32_t)(1)) : "memory");
    __syncthreads();
    unsigned int _phase_l3_0 = 0;
    mbarrier_wait(l3_addr, _phase_l3_0);
    _phase_l3_0 ^= 1;
    if (warp == 2) {
        uint32_t _elected_12 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_12));
        int _smem_desc_lo_13 = ((s_do_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_14 = (((s_hT_addr) >> 4) & 0x3FFF) | 0x4000000;
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
            "add.u32 alo, alo, 506;\n\t"
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
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_13), "r"(_smem_desc_lo_14), "r"((tmem_acc_cols + (1048576))), "r"(0), "r"(_elected_12) : "memory");
        tcgen05_commit_leader(m4_addr, _elected_12);
    }
    unsigned int _phase_m2_0 = 0;
    mbarrier_wait(m2_addr, _phase_m2_0);
    _phase_m2_0 ^= 1;
    int r0_8 = tid / 16;
    int c0_9 = tid % 16 * 8;
    long long row_stride_10 = (long long)num_heads * 128;
    long long src_row_11 = (long long)(tok0 + r0_8) * row_stride_10 + (long long)hq * 128 + (long long)c0_9;
    unsigned int dst0_12 = s_kn_addr + (unsigned int)(c0_9 / 64 * 8192 + (r0_8 * 128 + c0_9 % 64 * 2 ^ (r0_8 * 128 + c0_9 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12), "l"(kn + src_row_11), "r"((n_valid > r0_8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 2048), "l"(kn + (src_row_11 + 16 * row_stride_10)), "r"((n_valid > r0_8 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 4096), "l"(kn + (src_row_11 + 32 * row_stride_10)), "r"((n_valid > r0_8 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 6144), "l"(kn + (src_row_11 + 48 * row_stride_10)), "r"((n_valid > r0_8 + 48) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    float hdh = 0.0f;
    int h_row2 = tid_2 / 2;
    int h_col2 = tid_2 % 2 * 64;
    unsigned int word_h[4];
    unsigned int word_dh[4];
    #pragma unroll
    for (int it2 = 0; it2 < 8; it2++) {
        int e_off = (h_col2 + it2 * 8) / 64 * 16384 + (h_row2 * 128 + (h_col2 + it2 * 8) % 64 * 2 ^ (h_row2 * 128 + (h_col2 + it2 * 8) % 64 * 2 >> 7 & 7) << 4);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&word_h[0])), "=r"(*reinterpret_cast<uint32_t*>(&word_h[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&word_h[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&word_h[(0) + 3]))
            : "r"(s_h_addr + (unsigned int)e_off));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&word_dh[0])), "=r"(*reinterpret_cast<uint32_t*>(&word_dh[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&word_dh[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&word_dh[(0) + 3]))
            : "r"(s_dh_addr + (unsigned int)e_off));
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            hdh = __fmaf_rn(__uint_as_float(word_h[j] << 16), __uint_as_float(word_dh[j] << 16), hdh);
            hdh = __fmaf_rn(__uint_as_float(word_h[j] & 4294901760u), __uint_as_float(word_dh[j] & 4294901760u), hdh);
        }
    }
    float _warp_reduce_15 = hdh;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_15 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_15, offset);
    hdh = _warp_reduce_15;
    if (lane_0 == 0) {
        s_red[warp_1] = hdh;
    }
    __syncthreads();
    int r0_13 = tid / 16;
    int c0_14 = tid % 16 * 8;
    long long row_stride_15 = (long long)num_heads * 128;
    long long src_row_16 = (long long)(tok0 + r0_13) * row_stride_15 + (long long)hq * 128 + (long long)c0_14;
    unsigned int dst0_17 = s_qn_addr + (unsigned int)(c0_14 / 64 * 8192 + (r0_13 * 128 + c0_14 % 64 * 2 ^ (r0_13 * 128 + c0_14 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17), "l"(qn + src_row_16), "r"((n_valid > r0_13) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17 + 2048), "l"(qn + (src_row_16 + 16 * row_stride_15)), "r"((n_valid > r0_13 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17 + 4096), "l"(qn + (src_row_16 + 32 * row_stride_15)), "r"((n_valid > r0_13 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17 + 6144), "l"(qn + (src_row_16 + 48 * row_stride_15)), "r"((n_valid > r0_13 + 48) ? 16 : 0));
    int r0_18 = tid / 16;
    int c0_19 = tid % 16 * 8;
    long long row_stride_20 = (long long)num_v_heads * 128;
    long long src_row_21 = (long long)(tok0 + r0_18) * row_stride_20 + (long long)hv * 128 + (long long)c0_19;
    unsigned int dst0_22 = s_dv_addr + (unsigned int)(c0_19 / 64 * 8192 + (r0_18 * 128 + c0_19 % 64 * 2 ^ (r0_18 * 128 + c0_19 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_22), "l"(dv2 + src_row_21), "r"((n_valid > r0_18) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_22 + 2048), "l"(dv2 + (src_row_21 + 16 * row_stride_20)), "r"((n_valid > r0_18 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_22 + 4096), "l"(dv2 + (src_row_21 + 32 * row_stride_20)), "r"((n_valid > r0_18 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_22 + 6144), "l"(dv2 + (src_row_21 + 48 * row_stride_20)), "r"((n_valid > r0_18 + 48) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    unsigned int _phase_m4_0 = 0;
    mbarrier_wait(m4_addr, _phase_m4_0);
    _phase_m4_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    float g4_a = s_g[row_a];
    float g4_b = s_g[row_b];
    float _exp2_16 = approx_exp2(g4_a);
    float gain_a = _exp2_16 * scale;
    float _exp2_17 = approx_exp2(g4_b);
    float gain_b = _exp2_17 * scale;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
            : "r"(ld_w + 1048576 + half * 32));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            acc[i * 4] = acc[i * 4] * gain_a;
            acc[i * 4 + 1] = acc[i * 4 + 1] * gain_a;
            acc[i * 4 + 2] = acc[i * 4 + 2] * gain_b;
            acc[i * 4 + 3] = acc[i * 4 + 3] * gain_b;
        }
        asm volatile(
            "tcgen05.st.sync.aligned.16x256b.x4.b32"
            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
            :: "r"(ld_w + 1048576 + half * 32), "r"(*reinterpret_cast<const uint32_t*>(&acc[0])), "r"(*reinterpret_cast<const uint32_t*>(&acc[1])), "r"(*reinterpret_cast<const uint32_t*>(&acc[2])), "r"(*reinterpret_cast<const uint32_t*>(&acc[3])), "r"(*reinterpret_cast<const uint32_t*>(&acc[4])), "r"(*reinterpret_cast<const uint32_t*>(&acc[5])), "r"(*reinterpret_cast<const uint32_t*>(&acc[6])), "r"(*reinterpret_cast<const uint32_t*>(&acc[7])), "r"(*reinterpret_cast<const uint32_t*>(&acc[8])), "r"(*reinterpret_cast<const uint32_t*>(&acc[9])), "r"(*reinterpret_cast<const uint32_t*>(&acc[10])), "r"(*reinterpret_cast<const uint32_t*>(&acc[11])), "r"(*reinterpret_cast<const uint32_t*>(&acc[12])), "r"(*reinterpret_cast<const uint32_t*>(&acc[13])), "r"(*reinterpret_cast<const uint32_t*>(&acc[14])), "r"(*reinterpret_cast<const uint32_t*>(&acc[15])));
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    int r0_23 = tid / 16;
    int c0_24 = tid % 16 * 8;
    long long row_stride_25 = (long long)num_v_heads * 128;
    long long src_row_26 = (long long)(tok0 + r0_23) * row_stride_25 + (long long)hv * 128 + (long long)c0_24;
    unsigned int dst0_27 = s_v_addr + (unsigned int)(c0_24 / 64 * 8192 + (r0_23 * 128 + c0_24 % 64 * 2 ^ (r0_23 * 128 + c0_24 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_27), "l"(v + src_row_26), "r"((n_valid > r0_23) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_27 + 2048), "l"(v + (src_row_26 + 16 * row_stride_25)), "r"((n_valid > r0_23 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_27 + 4096), "l"(v + (src_row_26 + 32 * row_stride_25)), "r"((n_valid > r0_23 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_27 + 6144), "l"(v + (src_row_26 + 48 * row_stride_25)), "r"((n_valid > r0_23 + 48) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 2;");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned long long _mbarrier_arrival_token_18;
    asm volatile(
        "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
        : "=l"(_mbarrier_arrival_token_18)
        : "l"(smem_raw + ((l4_addr) - smem)), "r"((uint32_t)(1)) : "memory");
    __syncthreads();
    unsigned int _phase_l4_0 = 0;
    mbarrier_wait(l4_addr, _phase_l4_0);
    _phase_l4_0 ^= 1;
    if (warp == 1) {
        uint32_t _elected_19 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_19));
        int _smem_desc_lo_20 = ((s_ds_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_21 = (((s_knT_addr) >> 4) & 0x3FFF) | 0x2000000;
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
            :: "r"(_smem_desc_lo_20), "r"(_smem_desc_lo_21), "r"((tmem_acc_cols + (1048576))), "r"(1), "r"(_elected_19) : "memory");
        tcgen05_commit_leader(m5_addr, _elected_19);
    }
    asm volatile("tcgen05.fence::after_thread_sync;");
    float g5_a = s_g[row_a];
    float g5_b = s_g[row_b];
    int _max_22 = ((n_valid) > (1) ? (n_valid) : (1));
    float g_last = s_g[_max_22 - 1];
    float decay_a = 0.0f;
    float decay_b = 0.0f;
    if (row_a < n_valid) {
        float _exp2_23 = approx_exp2(g_last - g5_a);
        decay_a = _exp2_23;
    }
    if (row_b < n_valid) {
        float _exp2_24 = approx_exp2(g_last - g5_b);
        decay_b = _exp2_24;
    }
    float dg_last_part = 0.0f;
    #pragma unroll
    for (int half1 = 0; half1 < 2; half1++) {
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
            : "r"(ld_w + half1 * 32));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int i1 = 0; i1 < 4; i1++) {
            acc[i1 * 4] = acc[i1 * 4] * decay_a;
            acc[i1 * 4 + 1] = acc[i1 * 4 + 1] * decay_a;
            acc[i1 * 4 + 2] = acc[i1 * 4 + 2] * decay_b;
            acc[i1 * 4 + 3] = acc[i1 * 4 + 3] * decay_b;
            int kcol = n0 + (half1 * 4 + i1) * 8 + col_q;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(s_kn_addr + (unsigned int)(kcol / 64 * 8192 + (row_a * 128 + kcol % 64 * 2 ^ (row_a * 128 + kcol % 64 * 2 >> 7 & 7) << 4))));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(s_kn_addr + (unsigned int)(kcol / 64 * 8192 + (row_b * 128 + kcol % 64 * 2 ^ (row_b * 128 + kcol % 64 * 2 >> 7 & 7) << 4))));
            dg_last_part = __fmaf_rn(acc[i1 * 4], __uint_as_float(pair_a[0] << 16), dg_last_part);
            dg_last_part = __fmaf_rn(acc[i1 * 4 + 2], __uint_as_float(pair_b[0] << 16), dg_last_part);
            dg_last_part = __fmaf_rn(acc[i1 * 4 + 1], __uint_as_float(pair_a[0] & 4294901760u), dg_last_part);
            dg_last_part = __fmaf_rn(acc[i1 * 4 + 3], __uint_as_float(pair_b[0] & 4294901760u), dg_last_part);
        }
        asm volatile(
            "tcgen05.st.sync.aligned.16x256b.x4.b32"
            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
            :: "r"(ld_w + half1 * 32), "r"(*reinterpret_cast<const uint32_t*>(&acc[0])), "r"(*reinterpret_cast<const uint32_t*>(&acc[1])), "r"(*reinterpret_cast<const uint32_t*>(&acc[2])), "r"(*reinterpret_cast<const uint32_t*>(&acc[3])), "r"(*reinterpret_cast<const uint32_t*>(&acc[4])), "r"(*reinterpret_cast<const uint32_t*>(&acc[5])), "r"(*reinterpret_cast<const uint32_t*>(&acc[6])), "r"(*reinterpret_cast<const uint32_t*>(&acc[7])), "r"(*reinterpret_cast<const uint32_t*>(&acc[8])), "r"(*reinterpret_cast<const uint32_t*>(&acc[9])), "r"(*reinterpret_cast<const uint32_t*>(&acc[10])), "r"(*reinterpret_cast<const uint32_t*>(&acc[11])), "r"(*reinterpret_cast<const uint32_t*>(&acc[12])), "r"(*reinterpret_cast<const uint32_t*>(&acc[13])), "r"(*reinterpret_cast<const uint32_t*>(&acc[14])), "r"(*reinterpret_cast<const uint32_t*>(&acc[15])));
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    float _warp_reduce_25 = dg_last_part;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_25 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_25, offset);
    dg_last_part = _warp_reduce_25;
    if (lane_0 == 0) {
        s_red[8 + warp_1] = dg_last_part;
    }
    asm volatile("cp.async.wait_group 1;");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    unsigned long long _mbarrier_arrival_token_26;
    asm volatile(
        "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
        : "=l"(_mbarrier_arrival_token_26)
        : "l"(smem_raw + ((l5_addr) - smem)), "r"((uint32_t)(1)) : "memory");
    __syncthreads();
    unsigned int _phase_l5_0 = 0;
    mbarrier_wait(l5_addr, _phase_l5_0);
    _phase_l5_0 ^= 1;
    if (warp == 0) {
        uint32_t _elected_27 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_27));
        int _smem_desc_lo_28 = (((s_dsT_addr) >> 4) & 0x3FFF) | 0x2000000;
        int _smem_desc_lo_29 = (((s_qnT_addr) >> 4) & 0x3FFF) | 0x2000000;
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
            :: "r"(_smem_desc_lo_28), "r"(_smem_desc_lo_29), "r"(tmem_acc_cols), "r"(1), "r"(_elected_27) : "memory");
        tcgen05_commit_leader(m3_addr, _elected_27);
    }
    if (warp == 2) {
        uint32_t _elected_30 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_30));
        int _smem_desc_lo_31 = ((s_dv_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_32 = (((s_hT_addr) >> 4) & 0x3FFF) | 0x4000000;
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
            "add.u32 alo, alo, 506;\n\t"
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
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_31), "r"(_smem_desc_lo_32), "r"((tmem_acc_cols + (128))), "r"(0), "r"(_elected_30) : "memory");
        tcgen05_commit_leader(m6_addr, _elected_30);
    }
    unsigned int _phase_m5_0 = 0;
    mbarrier_wait(m5_addr, _phase_m5_0);
    _phase_m5_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    float dgq_a = 0.0f;
    float dgq_b = 0.0f;
    #pragma unroll
    for (int half3 = 0; half3 < 2; half3++) {
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
            : "r"(ld_w + 1048576 + half3 * 32));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int i3 = 0; i3 < 4; i3++) {
            int qcol = n0 + (half3 * 4 + i3) * 8 + col_q;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(s_qn_addr + (unsigned int)(qcol / 64 * 8192 + (row_a * 128 + qcol % 64 * 2 ^ (row_a * 128 + qcol % 64 * 2 >> 7 & 7) << 4))));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(s_qn_addr + (unsigned int)(qcol / 64 * 8192 + (row_b * 128 + qcol % 64 * 2 ^ (row_b * 128 + qcol % 64 * 2 >> 7 & 7) << 4))));
            dgq_a = __fmaf_rn(acc[i3 * 4], __uint_as_float(pair_a[0] << 16), dgq_a);
            dgq_a = __fmaf_rn(acc[i3 * 4 + 1], __uint_as_float(pair_a[0] & 4294901760u), dgq_a);
            dgq_b = __fmaf_rn(acc[i3 * 4 + 2], __uint_as_float(pair_b[0] << 16), dgq_b);
            dgq_b = __fmaf_rn(acc[i3 * 4 + 3], __uint_as_float(pair_b[0] & 4294901760u), dgq_b);
        }
    }
    float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, dgq_a, 1);
    dgq_a = dgq_a + _shfl_xor_sync_33;
    float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, dgq_a, 2);
    dgq_a = dgq_a + _shfl_xor_sync_34;
    float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, dgq_b, 1);
    dgq_b = dgq_b + _shfl_xor_sync_35;
    float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, dgq_b, 2);
    dgq_b = dgq_b + _shfl_xor_sync_36;
    if (lane_0 % 4 == 0) {
        s_dgq[nh * 64 + row_a] = dgq_a;
        s_dgq[nh * 64 + row_b] = dgq_b;
    }
    unsigned int _phase_m3_0 = 0;
    mbarrier_wait(m3_addr, _phase_m3_0);
    _phase_m3_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    float dgk_a = 0.0f;
    float dgk_b = 0.0f;
    #pragma unroll
    for (int half4 = 0; half4 < 2; half4++) {
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
            : "r"(ld_w + half4 * 32));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int i4 = 0; i4 < 4; i4++) {
            int kcol2 = n0 + (half4 * 4 + i4) * 8 + col_q;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(s_kn_addr + (unsigned int)(kcol2 / 64 * 8192 + (row_a * 128 + kcol2 % 64 * 2 ^ (row_a * 128 + kcol2 % 64 * 2 >> 7 & 7) << 4))));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(s_kn_addr + (unsigned int)(kcol2 / 64 * 8192 + (row_b * 128 + kcol2 % 64 * 2 ^ (row_b * 128 + kcol2 % 64 * 2 >> 7 & 7) << 4))));
            dgk_a = __fmaf_rn(acc[i4 * 4], __uint_as_float(pair_a[0] << 16), dgk_a);
            dgk_a = __fmaf_rn(acc[i4 * 4 + 1], __uint_as_float(pair_a[0] & 4294901760u), dgk_a);
            dgk_b = __fmaf_rn(acc[i4 * 4 + 2], __uint_as_float(pair_b[0] << 16), dgk_b);
            dgk_b = __fmaf_rn(acc[i4 * 4 + 3], __uint_as_float(pair_b[0] & 4294901760u), dgk_b);
        }
    }
    float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, dgk_a, 1);
    dgk_a = dgk_a + _shfl_xor_sync_37;
    float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, dgk_a, 2);
    dgk_a = dgk_a + _shfl_xor_sync_38;
    float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, dgk_b, 1);
    dgk_b = dgk_b + _shfl_xor_sync_39;
    float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, dgk_b, 2);
    dgk_b = dgk_b + _shfl_xor_sync_40;
    if (lane_0 % 4 == 0) {
        s_dgk[nh * 64 + row_a] = dgk_a;
        s_dgk[nh * 64 + row_b] = dgk_b;
    }
    int r0_28 = tid / 8;
    int c0_29 = tid % 8 * 8;
    long long row_stride_30 = (long long)num_v_heads * 64;
    long long src_row_31 = (long long)(tok0 + r0_28) * row_stride_30 + (long long)hv * 64 + (long long)c0_29;
    unsigned int dst0_32 = s_a_addr + (unsigned int)(c0_29 / 64 * 8192 + (r0_28 * 128 + c0_29 % 64 * 2 ^ (r0_28 * 128 + c0_29 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_32), "l"(A + src_row_31), "r"((n_valid > r0_28) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_32 + 4096), "l"(A + (src_row_31 + 32 * row_stride_30)), "r"((n_valid > r0_28 + 32) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    unsigned int _phase_m6_0 = 0;
    mbarrier_wait(m6_addr, _phase_m6_0);
    _phase_m6_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    #pragma unroll
    for (int half5 = 0; half5 < 2; half5++) {
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
            : "r"(ld_w + 128 + half5 * 32));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        const float2 _scale2_41 = {-1.0f, -1.0f};
        #pragma unroll
        for (int _ls = 0; _ls < 8; _ls++)
            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc)[_ls], _scale2_41);
        #pragma unroll
        for (int p5 = 0; p5 < 2; p5++) {
            #pragma unroll
            for (int q5 = 0; q5 < 2; q5++) {
                #pragma unroll
                for (int e5 = 0; e5 < 2; e5++) {
                    float dw_a = 0.0f;
                    float dw_b = 0.0f;
                    if (row_a < n_valid) {
                        dw_a = acc[(2 * p5 + q5) * 4 + e5];
                    }
                    if (row_b < n_valid) {
                        dw_b = acc[(2 * p5 + q5) * 4 + 2 + e5];
                    }
                    vals[q5 * 4 + e5] = dw_a;
                    vals[q5 * 4 + 2 + e5] = dw_b;
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                pk[_lp] = *(uint32_t*)&_bf2;
            }
            uint32_t _stmatrix_addr_42 = static_cast<uint32_t>(s_dw_addr + (unsigned int)((n0 + half5 * 32 + p5 * 16 + lane_0 / 16 * 8) / 64 * 8192 + (st_row * 128 + (n0 + half5 * 32 + p5 * 16 + lane_0 / 16 * 8) % 64 * 2 ^ (st_row * 128 + (n0 + half5 * 32 + p5 * 16 + lane_0 / 16 * 8) % 64 * 2 >> 7 & 7) << 4)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_42), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3]))
                : "memory");
        }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (tid_2 < n_valid) {
        float hdh_sum = 0.0f;
        float part_sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            hdh_sum = hdh_sum + s_red[r];
        }
        #pragma unroll
        for (int r2 = 0; r2 < 8; r2++) {
            part_sum = part_sum + s_red[8 + r2];
        }
        int _max_43 = ((n_valid) > (1) ? (n_valid) : (1));
        float g6_last = s_g[_max_43 - 1];
        float _exp2_44 = approx_exp2(g6_last);
        float dg_last = __fmaf_rn(hdh_sum, _exp2_44, part_sum);
        float dg_val = s_dgq[tid_2] + s_dgq[64 + tid_2] - (s_dgk[tid_2] + s_dgk[64 + tid_2]);
        if (tid_2 == n_valid - 1) {
            dg_val = dg_val + dg_last;
        }
        s_dgq[tid_2] = dg_val;
    }
    if (tid_2 < 64) {
        s_b[tid_2] = beta_t;
        float _exp2_45 = approx_exp2(s_g[tid_2]);
        s_gexp[tid_2] = _exp2_45;
    }
    asm volatile("cp.async.wait_group 1;");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned long long _mbarrier_arrival_token_46;
    asm volatile(
        "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
        : "=l"(_mbarrier_arrival_token_46)
        : "l"(smem_raw + ((l6_addr) - smem)), "r"((uint32_t)(1)) : "memory");
    __syncthreads();
    unsigned int _phase_l6_0 = 0;
    mbarrier_wait(l6_addr, _phase_l6_0);
    _phase_l6_0 ^= 1;
    if (warp == 1) {
        uint32_t _elected_47 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_47));
        int _smem_desc_lo_48 = ((s_dw_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_49 = ((s_kn_addr) >> 4) & 0x3FFF;
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
            :: "r"(_smem_desc_lo_48), "r"(_smem_desc_lo_49), "r"((tmem_acc_cols + (1048704))), "r"(0), "r"(_elected_47) : "memory");
        tcgen05_commit_leader(mw1_addr, _elected_47);
    }
    if (warp == 2) {
        uint32_t _elected_50 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_50));
        int _smem_desc_lo_51 = ((s_dv_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_52 = ((s_v_addr) >> 4) & 0x3FFF;
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
            :: "r"(_smem_desc_lo_51), "r"(_smem_desc_lo_52), "r"((tmem_acc_cols + (1048768))), "r"(0), "r"(_elected_50) : "memory");
        tcgen05_commit_leader(mw2_addr, _elected_50);
    }
    float beta_a = s_b[row_a];
    float beta_b = s_b[row_b];
    float db_a = 0.0f;
    float db_b = 0.0f;
    float dg_a = 0.0f;
    float dg_b = 0.0f;
    asm volatile("cp.async.wait_group 0;");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned long long _mbarrier_arrival_token_53;
    asm volatile(
        "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
        : "=l"(_mbarrier_arrival_token_53)
        : "l"(smem_raw + ((l7_addr) - smem)), "r"((uint32_t)(1)) : "memory");
    __syncthreads();
    unsigned int _phase_l7_0 = 0;
    mbarrier_wait(l7_addr, _phase_l7_0);
    _phase_l7_0 ^= 1;
    if (warp == 0) {
        uint32_t _elected_54 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_54));
        int _smem_desc_lo_55 = (((s_aT_addr) >> 4) & 0x3FFF) | 0x2000000;
        int _smem_desc_lo_56 = (((s_duT_addr) >> 4) & 0x3FFF) | 0x2000000;
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
            :: "r"(_smem_desc_lo_55), "r"(_smem_desc_lo_56), "r"((tmem_acc_cols + (128))), "r"(0), "r"(_elected_54) : "memory");
        tcgen05_commit_leader(mw4_addr, _elected_54);
    }
    unsigned int _phase_mw1_0 = 0;
    mbarrier_wait(mw1_addr, _phase_mw1_0);
    _phase_mw1_0 ^= 1;
    unsigned int _phase_mw2_0 = 0;
    mbarrier_wait(mw2_addr, _phase_mw2_0);
    _phase_mw2_0 ^= 1;
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
        : "r"(ld_n + 1048768));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    #pragma unroll
    for (int p_1 = 0; p_1 < 2; p_1++) {
        #pragma unroll
        for (int q_1 = 0; q_1 < 2; q_1++) {
            int col0 = s0 + p_1 * 16 + q_1 * 8 + col_q;
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gq[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq[(0) + 1]))
                : "r"(s_gexp_addr + (unsigned int)(col0 * 4)));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bq[0])), "=r"(*reinterpret_cast<uint32_t*>(&bq[(0) + 1]))
                : "r"(s_b_addr + (unsigned int)(col0 * 4)));
            float da_a0 = __fmaf_rn(acc_s[(2 * p_1 + q_1) * 4], gq[0], acc2[(2 * p_1 + q_1) * 4]) * bq[0];
            float da_a1 = __fmaf_rn(acc_s[(2 * p_1 + q_1) * 4 + 1], gq[1], acc2[(2 * p_1 + q_1) * 4 + 1]) * bq[1];
            float da_b0 = __fmaf_rn(acc_s[(2 * p_1 + q_1) * 4 + 2], gq[0], acc2[(2 * p_1 + q_1) * 4 + 2]) * bq[0];
            float da_b1 = __fmaf_rn(acc_s[(2 * p_1 + q_1) * 4 + 3], gq[1], acc2[(2 * p_1 + q_1) * 4 + 3]) * bq[1];
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
            vals[q_1 * 4] = val_a0;
            vals[q_1 * 4 + 1] = val_a1;
            vals[q_1 * 4 + 2] = val_b0;
            vals[q_1 * 4 + 3] = val_b1;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
            pk[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_57 = static_cast<uint32_t>(s_da_addr + (unsigned int)((st_col + p_1 * 16) / 64 * 8192 + (st_row * 128 + (st_col + p_1 * 16) % 64 * 2 ^ (st_row * 128 + (st_col + p_1 * 16) % 64 * 2 >> 7 & 7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_57), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3]))
            : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        uint32_t _elected_58 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_58));
        int _smem_desc_lo_59 = ((s_da_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_60 = ((s_a_addr) >> 4) & 0x3FFF;
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
            :: "r"(_smem_desc_lo_59), "r"(_smem_desc_lo_60), "r"((tmem_acc_cols + (1048704))), "r"(0), "r"(_elected_58) : "memory");
        tcgen05_commit_leader(mw5_addr, _elected_58);
    }
    if (warp == 2) {
        uint32_t _elected_61 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_61));
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
            :: "r"(_mma_a_lo_0), "r"(_mma_a_lo_0), "r"((tmem_acc_cols + (1048768))), "r"(0), "r"(_elected_61) : "memory");
        tcgen05_commit_leader(m7_addr, _elected_61);
    }
    unsigned int _phase_mw4_0 = 0;
    mbarrier_wait(mw4_addr, _phase_mw4_0);
    _phase_mw4_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15])), "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
        : "r"(ld_w + 128));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    long long out_a = ((long long)(tok0 + row_a) * (long long)num_v_heads + (long long)hv) * 128 + (long long)n0 + (long long)col_q;
    long long out_b = ((long long)(tok0 + row_b) * (long long)num_v_heads + (long long)hv) * 128 + (long long)n0 + (long long)col_q;
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
    __syncthreads();
    if (warp == 3) {
        uint32_t _elected_62 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_62));
        int _smem_desc_lo_63 = (((s_aT_addr) >> 4) & 0x3FFF) | 0x2000000;
        int _smem_desc_lo_64 = (((s_dwT_addr) >> 4) & 0x3FFF) | 0x2000000;
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
            :: "r"(_smem_desc_lo_63), "r"(_smem_desc_lo_64), "r"((tmem_acc_cols + (128))), "r"(0), "r"(_elected_62) : "memory");
        tcgen05_commit_leader(mw3_addr, _elected_62);
    }
    unsigned int _phase_mw5_0 = 0;
    mbarrier_wait(mw5_addr, _phase_mw5_0);
    _phase_mw5_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc2[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc2[15]))
        : "r"(ld_n + 1048704));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    #pragma unroll
    for (int p_2 = 0; p_2 < 2; p_2++) {
        #pragma unroll
        for (int e4 = 0; e4 < 8; e4++) {
            vals[e4] = acc2[p_2 * 8 + e4];
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
            pk[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_65 = static_cast<uint32_t>(s_da2T_addr + (unsigned int)(stT_col / 64 * 8192 + ((stT_row + p_2 * 16) * 128 + stT_col % 64 * 2 ^ ((stT_row + p_2 * 16) * 128 + stT_col % 64 * 2 >> 7 & 7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_65), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3]))
            : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        uint32_t _elected_66 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_66));
        int _smem_desc_lo_67 = (((s_aT_addr) >> 4) & 0x3FFF) | 0x2000000;
        int _smem_desc_lo_68 = (((s_da2T_addr) >> 4) & 0x3FFF) | 0x2000000;
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
            :: "r"(_smem_desc_lo_67), "r"(_smem_desc_lo_68), "r"((tmem_acc_cols + (1048704))), "r"(0), "r"(_elected_66) : "memory");
        tcgen05_commit_leader(mw6_addr, _elected_66);
    }
    unsigned int _phase_mw6_0 = 0;
    mbarrier_wait(mw6_addr, _phase_mw6_0);
    _phase_mw6_0 ^= 1;
    unsigned int _phase_m7_0 = 0;
    mbarrier_wait(m7_addr, _phase_m7_0);
    _phase_m7_0 ^= 1;
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
        : "r"(ld_n + 1048768));
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
    for (int p_3 = 0; p_3 < 2; p_3++) {
        #pragma unroll
        for (int q_2 = 0; q_2 < 2; q_2++) {
            int col0_1 = s0 + p_3 * 16 + q_2 * 8 + col_q;
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gq[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq[(0) + 1]))
                : "r"(s_g_addr + (unsigned int)(col0_1 * 4)));
            float _exp2_69 = approx_exp2(g_a - gq[0]);
            float va0 = (-acc_s[(2 * p_3 + q_2) * 4]) * _exp2_69;
            float _exp2_70 = approx_exp2(g_a - gq[1]);
            float va1 = (-acc_s[(2 * p_3 + q_2) * 4 + 1]) * _exp2_70;
            float _exp2_71 = approx_exp2(g_b - gq[0]);
            float vb0 = (-acc_s[(2 * p_3 + q_2) * 4 + 2]) * _exp2_71;
            float _exp2_72 = approx_exp2(g_b - gq[1]);
            float vb1 = (-acc_s[(2 * p_3 + q_2) * 4 + 3]) * _exp2_72;
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
            float t_a0 = val_a0_1 * acc7[(2 * p_3 + q_2) * 4];
            float t_a1 = val_a1_1 * acc7[(2 * p_3 + q_2) * 4 + 1];
            float t_b0 = val_b0_1 * acc7[(2 * p_3 + q_2) * 4 + 2];
            float t_b1 = val_b1_1 * acc7[(2 * p_3 + q_2) * 4 + 3];
            db_a = db_a + t_a0;
            db_a = db_a + t_a1;
            db_b = db_b + t_b0;
            db_b = db_b + t_b1;
            float ada_a0 = t_a0 * beta_a;
            float ada_b0 = t_b0 * beta_b;
            dg_a = dg_a + ada_a0;
            dg_b = dg_b + ada_b0;
            col_part[(2 * p_3 + q_2) * 2] = ada_a0 + ada_b0;
            float ada_a1 = t_a1 * beta_a;
            float ada_b1 = t_b1 * beta_b;
            dg_a = dg_a + ada_a1;
            dg_b = dg_b + ada_b1;
            col_part[(2 * p_3 + q_2) * 2 + 1] = ada_a1 + ada_b1;
            valsb[q_2 * 4] = val_a0_1 * beta_a;
            valsb[q_2 * 4 + 1] = val_a1_1 * beta_a;
            valsb[q_2 * 4 + 2] = val_b0_1 * beta_b;
            valsb[q_2 * 4 + 3] = val_b1_1 * beta_b;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(valsb[_lp*2 + 0], valsb[_lp*2+1 + 0]));
            pk[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_73 = static_cast<uint32_t>(s_da3b_addr + (unsigned int)((st_col + p_3 * 16) / 64 * 8192 + (st_row * 128 + (st_col + p_3 * 16) % 64 * 2 ^ (st_row * 128 + (st_col + p_3 * 16) % 64 * 2 >> 7 & 7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_73), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3]))
            : "memory");
    }
    #pragma unroll
    for (int ci = 0; ci < 8; ci++) {
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, col_part[ci], 4);
        col_part[ci] = col_part[ci] + _shfl_xor_sync_74;
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, col_part[ci], 8);
        col_part[ci] = col_part[ci] + _shfl_xor_sync_75;
        float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, col_part[ci], 16);
        col_part[ci] = col_part[ci] + _shfl_xor_sync_76;
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
    if (tid_2 == 32) {
        asm volatile(
            "{\n\t"
            ".reg .b32 remAddr32;\n\t"
            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
            "}"
            :: "r"(tx_ok_addr), "r"(partner) : "memory");
    }
    if (tid_2 == 0) {
        mbarrier_arrive(rx_full_addr);
    }
    if (warp == 0) {
        uint32_t _elected_77 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_77));
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
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_0), "r"((tmem_acc_cols + (1048704))), "r"(0), "r"(_elected_77) : "memory");
        int _smem_desc_lo_78 = (((s_da3bT_addr) >> 4) & 0x3FFF) | 0x2000000;
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
            :: "r"(_smem_desc_lo_78), "r"(_mma_b_lo_0), "r"((tmem_acc_cols + (1048704))), "r"(1), "r"(_elected_77) : "memory");
        tcgen05_commit_leader(mw8_addr, _elected_77);
    }
    unsigned int _phase_mw8_0 = 0;
    mbarrier_wait(mw8_addr, _phase_mw8_0);
    _phase_mw8_0 ^= 1;
    unsigned int _phase_mw3_0 = 0;
    mbarrier_wait(mw3_addr, _phase_mw3_0);
    _phase_mw3_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    #pragma unroll
    for (int k6 = 0; k6 < 2; k6++) {
        int qr6 = tid_2 / 16 + k6 * 16;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_qf_addr + (unsigned int)(qr6 * 256 + tid_2 % 16 * 16)), "l"(qn + (((long long)(tok0 + rank * 32 + qr6) * (long long)num_heads + (long long)hq) * 128 + (long long)(tid_2 % 16 * 8))), "r"((n_valid > rank * 32 + qr6) ? 16 : 0));
    }
    long long rs_row = (long long)(tok0 + rank * 32 + tid_2 % 32) * (long long)num_heads + (long long)hq;
    if (tid_2 < 32) {
        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
            :: "r"(s_rs_addr + (unsigned int)(tid_2 * 4)), "l"(rstd_q + rs_row), "r"((n_valid > rank * 32 + tid_2) ? 4 : 0));
    } else if (tid_2 < 64) {
        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
            :: "r"(s_rs_addr + (unsigned int)(tid_2 * 4)), "l"(rstd_k + rs_row), "r"((n_valid > rank * 32 + tid_2 - 32) ? 4 : 0));
    }
    asm volatile("cp.async.commit_group;");
    float gw_a = s_g[row_a];
    float gw_b = s_g[row_b];
    float gexp_a = approx_exp2(gw_a);
    float gexp_b = approx_exp2(gw_b);
    float bg_a = gexp_a * beta_a;
    float bg_b = gexp_b * beta_b;
    int mine = ((rb / 2 == rank) ? 1 : 0);
    int lr_a = row_a % 32;
    int lr_b = row_b % 32;
    uint32_t rx_peer;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(rx_peer) : "r"(s_rx_addr), "r"(partner));
    unsigned int _phase_tx_ok_0 = 0;
    mbarrier_wait_cluster_hint(tx_ok_addr, _phase_tx_ok_0, 10000000);
    _phase_tx_ok_0 ^= 1;
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
        : "r"(ld_w + 1048704));
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[15]))
        : "r"(ld_w + 128));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int nb4 = 0; nb4 < 4; nb4++) {
        int kcol3 = n0 + (0 + nb4) * 8 + col_q;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(s_kn_addr + (unsigned int)(kcol3 / 64 * 8192 + (row_a * 128 + kcol3 % 64 * 2 ^ (row_a * 128 + kcol3 % 64 * 2 >> 7 & 7) << 4))));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(s_kn_addr + (unsigned int)(kcol3 / 64 * 8192 + (row_b * 128 + kcol3 % 64 * 2 ^ (row_b * 128 + kcol3 % 64 * 2 >> 7 & 7) << 4))));
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
        int ch = nh * 8 + 0 + nb18;
        unsigned int off_a = (unsigned int)(lr_a * 256 + ((ch ^ lr_a & 7) << 4)) + (unsigned int)(col_q * 2);
        unsigned int off_b = (unsigned int)(lr_b * 256 + ((ch ^ lr_b & 7) << 4)) + (unsigned int)(col_q * 2);
        unsigned int _packed_2[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + nb18 * 4)[_lp*2 + 0], (acc + nb18 * 4)[_lp*2+1 + 0]));
            _packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        unsigned int _packed_3[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + nb18 * 4 + 2)[_lp*2 + 0], (acc + nb18 * 4 + 2)[_lp*2+1 + 0]));
            _packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        if (mine != 0) {
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_own2_addr + off_a), "r"(_packed_2[0]) : "memory");
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_own2_addr + off_b), "r"(_packed_3[0]) : "memory");
        } else {
            uint32_t _mbar_peer_79;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_79) : "r"(rx_full_addr), "r"(partner));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(rx_peer + (16384 + off_a)), "r"(_packed_2[0]), "r"(_mbar_peer_79) : "memory");
            uint32_t _mbar_peer_80;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_80) : "r"(rx_full_addr), "r"(partner));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(rx_peer + (16384 + off_b)), "r"(_packed_3[0]), "r"(_mbar_peer_80) : "memory");
        }
    }
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
        : "r"(ld_w + 1048736));
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[15]))
        : "r"(ld_w + 160));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    #pragma unroll
    for (int nb4_1 = 0; nb4_1 < 4; nb4_1++) {
        int kcol3_1 = n0 + (4 + nb4_1) * 8 + col_q;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(s_kn_addr + (unsigned int)(kcol3_1 / 64 * 8192 + (row_a * 128 + kcol3_1 % 64 * 2 ^ (row_a * 128 + kcol3_1 % 64 * 2 >> 7 & 7) << 4))));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(s_kn_addr + (unsigned int)(kcol3_1 / 64 * 8192 + (row_b * 128 + kcol3_1 % 64 * 2 ^ (row_b * 128 + kcol3_1 % 64 * 2 >> 7 & 7) << 4))));
        float k_a0_1 = __uint_as_float(pair_a[0] << 16);
        float k_b0_1 = __uint_as_float(pair_b[0] << 16);
        float k_a1_1 = __uint_as_float(pair_a[0] & 4294901760u);
        float k_b1_1 = __uint_as_float(pair_b[0] & 4294901760u);
        db_a = __fmaf_rn(acc_s[nb4_1 * 4], k_a0_1 * gexp_a, db_a);
        db_b = __fmaf_rn(acc_s[nb4_1 * 4 + 2], k_b0_1 * gexp_b, db_b);
        dg_a = __fmaf_rn(acc_s[nb4_1 * 4], k_a0_1 * bg_a, dg_a);
        dg_b = __fmaf_rn(acc_s[nb4_1 * 4 + 2], k_b0_1 * bg_b, dg_b);
        db_a = __fmaf_rn(acc_s[nb4_1 * 4 + 1], k_a1_1 * gexp_a, db_a);
        db_b = __fmaf_rn(acc_s[nb4_1 * 4 + 3], k_b1_1 * gexp_b, db_b);
        dg_a = __fmaf_rn(acc_s[nb4_1 * 4 + 1], k_a1_1 * bg_a, dg_a);
        dg_b = __fmaf_rn(acc_s[nb4_1 * 4 + 3], k_b1_1 * bg_b, dg_b);
        acc[nb4_1 * 4] = __fmaf_rn(acc_s[nb4_1 * 4], bg_a, acc[nb4_1 * 4]);
        acc[nb4_1 * 4 + 1] = __fmaf_rn(acc_s[nb4_1 * 4 + 1], bg_a, acc[nb4_1 * 4 + 1]);
        acc[nb4_1 * 4 + 2] = __fmaf_rn(acc_s[nb4_1 * 4 + 2], bg_b, acc[nb4_1 * 4 + 2]);
        acc[nb4_1 * 4 + 3] = __fmaf_rn(acc_s[nb4_1 * 4 + 3], bg_b, acc[nb4_1 * 4 + 3]);
    }
    #pragma unroll
    for (int nb18_1 = 0; nb18_1 < 4; nb18_1++) {
        int ch_1 = nh * 8 + 4 + nb18_1;
        unsigned int off_a_1 = (unsigned int)(lr_a * 256 + ((ch_1 ^ lr_a & 7) << 4)) + (unsigned int)(col_q * 2);
        unsigned int off_b_1 = (unsigned int)(lr_b * 256 + ((ch_1 ^ lr_b & 7) << 4)) + (unsigned int)(col_q * 2);
        unsigned int _packed_2_1[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + nb18_1 * 4)[_lp*2 + 0], (acc + nb18_1 * 4)[_lp*2+1 + 0]));
            _packed_2_1[_lp] = *(uint32_t*)&_bf2;
        }
        unsigned int _packed_3_1[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + nb18_1 * 4 + 2)[_lp*2 + 0], (acc + nb18_1 * 4 + 2)[_lp*2+1 + 0]));
            _packed_3_1[_lp] = *(uint32_t*)&_bf2;
        }
        if (mine != 0) {
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_own2_addr + off_a_1), "r"(_packed_2_1[0]) : "memory");
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_own2_addr + off_b_1), "r"(_packed_3_1[0]) : "memory");
        } else {
            uint32_t _mbar_peer_81;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_81) : "r"(rx_full_addr), "r"(partner));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(rx_peer + (16384 + off_a_1)), "r"(_packed_2_1[0]), "r"(_mbar_peer_81) : "memory");
            uint32_t _mbar_peer_82;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_82) : "r"(rx_full_addr), "r"(partner));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(rx_peer + (16384 + off_b_1)), "r"(_packed_3_1[0]), "r"(_mbar_peer_82) : "memory");
        }
    }
    float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, db_a, 1);
    db_a = db_a + _shfl_xor_sync_83;
    float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, db_a, 2);
    db_a = db_a + _shfl_xor_sync_84;
    float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, db_b, 1);
    db_b = db_b + _shfl_xor_sync_85;
    float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, db_b, 2);
    db_b = db_b + _shfl_xor_sync_86;
    float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, dg_a, 1);
    dg_a = dg_a + _shfl_xor_sync_87;
    float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, dg_a, 2);
    dg_a = dg_a + _shfl_xor_sync_88;
    float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, dg_b, 1);
    dg_b = dg_b + _shfl_xor_sync_89;
    float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, dg_b, 2);
    dg_b = dg_b + _shfl_xor_sync_90;
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
        dbeta_out[gate_row_index] = (__nv_bfloat16)db_row;
        s_dg2[tid_2] = dg_row - colsum;
    }
    float f0[32];
    unsigned int f0p[16];
    int lr_f = (rb * 16 + lane_0 % 16) % 32;
    unsigned int t_off = ((lane_0 < 16) ? 8192 : 0);
    unsigned int own_base = ((lane_0 < 16) ? s_ownk_addr : s_ownq_addr);
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(f0[0]), "=f"(f0[1]), "=f"(f0[2]), "=f"(f0[3]), "=f"(f0[4]), "=f"(f0[5]), "=f"(f0[6]), "=f"(f0[7]), "=f"(f0[8]), "=f"(f0[9]), "=f"(f0[10]), "=f"(f0[11]), "=f"(f0[12]), "=f"(f0[13]), "=f"(f0[14]), "=f"(f0[15]), "=f"(f0[16]), "=f"(f0[17]), "=f"(f0[18]), "=f"(f0[19]), "=f"(f0[20]), "=f"(f0[21]), "=f"(f0[22]), "=f"(f0[23]), "=f"(f0[24]), "=f"(f0[25]), "=f"(f0[26]), "=f"(f0[27]), "=f"(f0[28]), "=f"(f0[29]), "=f"(f0[30]), "=f"(f0[31])
        : "r"(taddr_1 + n0 + 0));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 16; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(f0[_lp*2 + 0], f0[_lp*2+1 + 0]));
        f0p[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        unsigned int off_f = (unsigned int)(lr_f * 256 + ((nh * 8 + 0 + k ^ lr_f & 7) << 4));
        if (mine != 0) {
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(own_base + off_f), "r"(f0p[4 * k]), "r"(f0p[4 * k + 1]), "r"(f0p[4 * k + 2]), "r"(f0p[4 * k + 3]) : "memory");
        } else {
            uint32_t _mbar_peer_91;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_91) : "r"(rx_full_addr), "r"(partner));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(rx_peer + (t_off + off_f)), "r"(f0p[4 * k]), "r"(f0p[4 * k + 1]), "r"(f0p[4 * k + 2]), "r"(f0p[4 * k + 3]), "r"(_mbar_peer_91) : "memory");
        }
    }
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(f0[0]), "=f"(f0[1]), "=f"(f0[2]), "=f"(f0[3]), "=f"(f0[4]), "=f"(f0[5]), "=f"(f0[6]), "=f"(f0[7]), "=f"(f0[8]), "=f"(f0[9]), "=f"(f0[10]), "=f"(f0[11]), "=f"(f0[12]), "=f"(f0[13]), "=f"(f0[14]), "=f"(f0[15]), "=f"(f0[16]), "=f"(f0[17]), "=f"(f0[18]), "=f"(f0[19]), "=f"(f0[20]), "=f"(f0[21]), "=f"(f0[22]), "=f"(f0[23]), "=f"(f0[24]), "=f"(f0[25]), "=f"(f0[26]), "=f"(f0[27]), "=f"(f0[28]), "=f"(f0[29]), "=f"(f0[30]), "=f"(f0[31])
        : "r"(taddr_1 + n0 + 32));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 16; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(f0[_lp*2 + 0], f0[_lp*2+1 + 0]));
        f0p[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int k_1 = 0; k_1 < 4; k_1++) {
        unsigned int off_f_1 = (unsigned int)(lr_f * 256 + ((nh * 8 + 4 + k_1 ^ lr_f & 7) << 4));
        if (mine != 0) {
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(own_base + off_f_1), "r"(f0p[4 * k_1]), "r"(f0p[4 * k_1 + 1]), "r"(f0p[4 * k_1 + 2]), "r"(f0p[4 * k_1 + 3]) : "memory");
        } else {
            uint32_t _mbar_peer_92;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_92) : "r"(rx_full_addr), "r"(partner));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(rx_peer + (t_off + off_f_1)), "r"(f0p[4 * k_1]), "r"(f0p[4 * k_1 + 1]), "r"(f0p[4 * k_1 + 2]), "r"(f0p[4 * k_1 + 3]), "r"(_mbar_peer_92) : "memory");
        }
    }
    asm volatile("cp.async.wait_group 0;");
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (warp_1 == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
    unsigned int _phase_rx_full_0 = 0;
    asm volatile("{ .reg .pred P1; WAIT: mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P1, [%0], %1, %2; @P1 bra.uni DONE; bra.uni WAIT; DONE: }" :: "r"(rx_full_addr), "r"(uint32_t(_phase_rx_full_0)), "r"(uint32_t(10000000)) : "memory");
    asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
    _phase_rx_full_0 ^= 1;
    unsigned int base_q0 = ((rank == 0) ? s_ownq_addr : s_rx_addr);
    unsigned int base_k0 = ((rank == 0) ? s_ownk_addr : s_rx_addr + 8192);
    unsigned int base_20 = ((rank == 0) ? s_own2_addr : s_rx_addr + 16384);
    unsigned int base_q1 = ((rank == 0) ? s_rx_addr : s_ownq_addr);
    unsigned int base_k1 = ((rank == 0) ? s_rx_addr + 8192 : s_ownk_addr);
    unsigned int base_21 = ((rank == 0) ? s_rx_addr + 16384 : s_own2_addr);
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; i_1++) {
        int lr = warp_1 * 4 + i_1;
        int row = rank * 32 + lr;
        if (row < n_valid) {
            long long tok = (long long)(tok0 + row);
            long long qbase = (tok * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4);
            unsigned int so = (unsigned int)(lr * 256 + ((lane_0 >> 1 ^ lr & 7) << 4) + (lane_0 & 1) * 8);
            unsigned int yw[4];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&yw[0])), "=r"(*reinterpret_cast<uint32_t*>(&yw[(0) + 1]))
                : "r"(s_qf_addr + (unsigned int)(lr * 256 + lane_0 * 8)));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&yw[2])), "=r"(*reinterpret_cast<uint32_t*>(&yw[(2) + 1]))
                : "r"(s_kn_addr + (unsigned int)(lane_0 * 4 / 64 * 8192 + (row * 128 + lane_0 * 4 % 64 * 2 ^ (row * 128 + lane_0 * 4 % 64 * 2 >> 7 & 7) << 4))));
            float yq[4];
            float yk[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&yq[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw[_pair]) << 16);
                (&yq[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&yk[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw[2 + _pair]) << 16);
                (&yk[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw[2 + _pair]) & 0xffff0000u);
            }
            float rq = s_rs[lr];
            float rk = s_rs[32 + lr];
            unsigned int w6[12];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w6[0])), "=r"(*reinterpret_cast<uint32_t*>(&w6[(0) + 1]))
                : "r"(base_q0 + so));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w6[2])), "=r"(*reinterpret_cast<uint32_t*>(&w6[(2) + 1]))
                : "r"(base_k0 + so));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w6[4])), "=r"(*reinterpret_cast<uint32_t*>(&w6[(4) + 1]))
                : "r"(base_20 + so));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w6[6])), "=r"(*reinterpret_cast<uint32_t*>(&w6[(6) + 1]))
                : "r"(base_q1 + so));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w6[8])), "=r"(*reinterpret_cast<uint32_t*>(&w6[(8) + 1]))
                : "r"(base_k1 + so));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w6[10])), "=r"(*reinterpret_cast<uint32_t*>(&w6[(10) + 1]))
                : "r"(base_21 + so));
            float pq0[4];
            float pk0[4];
            float p20[4];
            float pq1[4];
            float pk1[4];
            float p21[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&pq0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w6[_pair]) << 16);
                (&pq0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w6[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&pk0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w6[2 + _pair]) << 16);
                (&pk0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w6[2 + _pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&p20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w6[4 + _pair]) << 16);
                (&p20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w6[4 + _pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&pq1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w6[6 + _pair]) << 16);
                (&pq1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w6[6 + _pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&pk1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w6[8 + _pair]) << 16);
                (&pk1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w6[8 + _pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&p21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w6[10 + _pair]) << 16);
                (&p21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w6[10 + _pair]) & 0xffff0000u);
            }
            float dq_r[4];
            float dk_r[4];
            dq_r[0] = 0.0f;
            dq_r[1] = 0.0f;
            dq_r[2] = 0.0f;
            dq_r[3] = 0.0f;
            dk_r[0] = 0.0f;
            dk_r[1] = 0.0f;
            dk_r[2] = 0.0f;
            dk_r[3] = 0.0f;
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dq_r[_la] = dq_r[_la] + pq0[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_r[_la] = dk_r[_la] + pk0[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_r[_la] = dk_r[_la] + p20[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dq_r[_la] = dq_r[_la] + pq1[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_r[_la] = dk_r[_la] + pk1[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_r[_la] = dk_r[_la] + p21[_la];
            float inner_q = 0.0f;
            float inner_k = 0.0f;
            #pragma unroll
            for (int i5 = 0; i5 < 4; i5++) {
                inner_q = __fmaf_rn(dq_r[i5], yq[i5], inner_q);
                inner_k = __fmaf_rn(dk_r[i5], yk[i5], inner_k);
            }
            float _warp_reduce_93 = inner_q;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_93 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_93, offset);
            inner_q = _warp_reduce_93;
            float _warp_reduce_94 = inner_k;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_94 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_94, offset);
            inner_k = _warp_reduce_94;
            if (normalize_qk != 0) {
                #pragma unroll
                for (int i6 = 0; i6 < 4; i6++) {
                    dq_r[i6] = __fmaf_rn(dq_r[i6], rq, -(inner_q * yq[i6] * rq));
                    dk_r[i6] = __fmaf_rn(dk_r[i6], rk, -(inner_k * yk[i6] * rk));
                }
            }
            unsigned int _packed_4[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r[_lp*2 + 0], dq_r[_lp*2+1 + 0]));
                _packed_4[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r[_lp*2 + 2], dq_r[_lp*2+1 + 2]));
                _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dq_out + (qbase))[0] = reinterpret_cast<int2*>(_packed_4)[0];
            unsigned int _packed_5[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r[_lp*2 + 0], dk_r[_lp*2+1 + 0]));
                _packed_5[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r[_lp*2 + 2], dk_r[_lp*2+1 + 2]));
                _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dk_out + (qbase))[0] = reinterpret_cast<int2*>(_packed_5)[0];
        }
    }
    if (warp_1 == 7) {
        float d_lo = 0.0f;
        float d_hi = 0.0f;
        long long gate_index_lo = (long long)(tok0 + lane_0) * (long long)num_v_heads + (long long)hv;
        long long gate_index_hi = (long long)(tok0 + lane_0 + 32) * (long long)num_v_heads + (long long)hv;
        if (lane_0 < n_valid) {
            d_lo = s_dgq[lane_0] + s_dg2[lane_0];
        }
        if (n_valid > lane_0 + 32) {
            d_hi = s_dgq[lane_0 + 32] + s_dg2[lane_0 + 32];
        }
        float self_lo = d_lo;
        float self_hi = d_hi;
        #pragma unroll
        for (int delta = 0; delta < 5; delta++) {
            const int shift = 1 << delta;
            float up_lo = __shfl_up_sync(0xFFFFFFFF, d_lo, shift, 32);
            float up_hi = __shfl_up_sync(0xFFFFFFFF, d_hi, shift, 32);
            if (lane_0 >= shift) {
                d_lo = d_lo + up_lo;
                d_hi = d_hi + up_hi;
            }
        }
        float total_lo = __shfl_sync(0xFFFFFFFF, d_lo, 31);
        float total_hi = __shfl_sync(0xFFFFFFFF, d_hi, 31);
        float total = total_lo + total_hi;
        if (lane_0 < n_valid) {
            dg_out[gate_index_lo] = total - d_lo + self_lo;
        }
        if (n_valid > lane_0 + 32) {
            dg_out[gate_index_hi] = total - total_lo - d_hi + self_hi;
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
}

} // extern "C"
