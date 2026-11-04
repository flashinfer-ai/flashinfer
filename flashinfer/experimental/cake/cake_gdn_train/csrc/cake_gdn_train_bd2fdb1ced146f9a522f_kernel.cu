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
#define SMEM_S_IQ_OFF 51968
#define SMEM_S_IQ_STAGE_BYTES 256
#define SMEM_S_IQ_STRIDE 256
#define SMEM_S_RS_OFF 52224
#define SMEM_S_RS_STAGE_BYTES 512
#define SMEM_S_RS_STRIDE 512
#define SMEM_S_IK_OFF 52736
#define SMEM_S_IK_STAGE_BYTES 512
#define SMEM_S_IK_STRIDE 512
#define SMEM_S_FLAG_OFF 53248
#define SMEM_S_FLAG_STAGE_BYTES 16
#define SMEM_S_FLAG_STRIDE 16
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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_bd2fdb1ced146f9a522f(__nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, float* __restrict__ rstd_q, float* __restrict__ rstd_k, __nv_bfloat16* __restrict__ v_new, __nv_bfloat16* __restrict__ do_, __nv_bfloat16* __restrict__ dv2, __nv_bfloat16* __restrict__ h, __nv_bfloat16* __restrict__ dh, float* __restrict__ g_cs, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ A, float* __restrict__ beta, __nv_bfloat16* __restrict__ dq_out, __nv_bfloat16* __restrict__ dk_out, float* __restrict__ dg_out, __nv_bfloat16* __restrict__ dv_out, float* __restrict__ dbeta_out, float* __restrict__ ws, unsigned int* __restrict__ ws_count, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, float scale, int normalize_qk, int ws_tiles)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

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
    float* s_iq = reinterpret_cast<float*>(smem_raw + 51968);
    const int s_iq_addr = smem + 51968;
    float* s_rs = reinterpret_cast<float*>(smem_raw + 52224);
    const int s_rs_addr = smem + 52224;
    float* s_ik = reinterpret_cast<float*>(smem_raw + 52736);
    const int s_ik_addr = smem + 52736;
    unsigned int* s_flag = reinterpret_cast<unsigned int*>(smem_raw + 53248);
    const int s_flag_addr = smem + 53248;
    float* s_g = reinterpret_cast<float*>(smem_raw + 106496);
    const int s_g_addr = smem + 106496;
    float* s_dgq = reinterpret_cast<float*>(smem_raw + 106752);
    const int s_dgq_addr = smem + 106752;
    float* s_dgk = reinterpret_cast<float*>(smem_raw + 107264);
    const int s_dgk_addr = smem + 107264;
    float* s_red = reinterpret_cast<float*>(smem_raw + 107776);
    const int s_red_addr = smem + 107776;

    // Mbarrier init (21 pipeline groups, 0 ordered-sequence groups, 21 barriers)
    // Mbarriers at smem_raw[107904..108072)

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
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 108072);
    if (warp == 0) {
        int _tmem_hold = smem + 108072;
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
    int lane_0 = lane;
    int warp_1 = warp;
    int tid_2 = tid;
    int block_x = blockIdx.x;
    int chunk = block_x / num_v_heads;
    int hv = block_x % num_v_heads;
    int hq = hv * num_heads / num_v_heads;
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
    int r0_0 = tid / 16;
    int c0_0 = tid % 16 * 8;
    long long row_stride_0 = (long long)num_v_heads * 128;
    long long src_row_0 = (long long)(tok0 + r0_0) * row_stride_0 + (long long)hv * 128 + (long long)c0_0;
    unsigned int dst0_0 = s_vn_addr + (unsigned int)(c0_0 / 64 * 8192 + (r0_0 * 128 + c0_0 % 64 * 2 ^ (r0_0 * 128 + c0_0 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_0), "l"(v_new + src_row_0), "r"((n_valid > r0_0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_0 + 2048), "l"(v_new + (src_row_0 + 16 * row_stride_0)), "r"((n_valid > r0_0 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_0 + 4096), "l"(v_new + (src_row_0 + 32 * row_stride_0)), "r"((n_valid > r0_0 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_0 + 6144), "l"(v_new + (src_row_0 + 48 * row_stride_0)), "r"((n_valid > r0_0 + 48) ? 16 : 0));
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_2 = ((s_do_addr) >> 4) & 0x7FFF;
        int _smem_desc_lo_3 = ((s_vn_addr) >> 4) & 0x7FFF;
#else
        int _smem_desc_lo_2 = ((s_do_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_3 = ((s_vn_addr) >> 4) & 0x3FFF;
#endif
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_6 = ((s_vn_addr) >> 4) & 0x7FFF;
        int _smem_desc_lo_7 = (((s_dhT_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
        int _smem_desc_lo_6 = ((s_vn_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_7 = (((s_dhT_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_13 = ((s_do_addr) >> 4) & 0x7FFF;
        int _smem_desc_lo_14 = (((s_hT_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
        int _smem_desc_lo_13 = ((s_do_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_14 = (((s_hT_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
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
    int r0_1 = tid / 16;
    int c0_1 = tid % 16 * 8;
    long long row_stride_1 = (long long)num_heads * 128;
    long long src_row_1 = (long long)(tok0 + r0_1) * row_stride_1 + (long long)hq * 128 + (long long)c0_1;
    unsigned int dst0_1 = s_kn_addr + (unsigned int)(c0_1 / 64 * 8192 + (r0_1 * 128 + c0_1 % 64 * 2 ^ (r0_1 * 128 + c0_1 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_1), "l"(kn + src_row_1), "r"((n_valid > r0_1) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_1 + 2048), "l"(kn + (src_row_1 + 16 * row_stride_1)), "r"((n_valid > r0_1 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_1 + 4096), "l"(kn + (src_row_1 + 32 * row_stride_1)), "r"((n_valid > r0_1 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_1 + 6144), "l"(kn + (src_row_1 + 48 * row_stride_1)), "r"((n_valid > r0_1 + 48) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    float hdh = 0.0f;
    int h_row2 = tid_2 / 2;
    int h_col2 = tid_2 % 2 * 64;
    unsigned int word_h[4];
    unsigned int word_dh[4];
    int h_odd = tid_2 % 2;
    if (h_odd == 0) {
        #pragma unroll
        for (int it2 = 0; it2 < 4; it2++) {
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
    }
    #pragma unroll
    for (int s2 = 4; s2 < 8; s2++) {
        int it2_1 = s2 - 4 * h_odd;
        int e_off_1 = (h_col2 + it2_1 * 8) / 64 * 16384 + (h_row2 * 128 + (h_col2 + it2_1 * 8) % 64 * 2 ^ (h_row2 * 128 + (h_col2 + it2_1 * 8) % 64 * 2 >> 7 & 7) << 4);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&word_h[0])), "=r"(*reinterpret_cast<uint32_t*>(&word_h[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&word_h[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&word_h[(0) + 3]))
            : "r"(s_h_addr + (unsigned int)e_off_1));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&word_dh[0])), "=r"(*reinterpret_cast<uint32_t*>(&word_dh[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&word_dh[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&word_dh[(0) + 3]))
            : "r"(s_dh_addr + (unsigned int)e_off_1));
        #pragma unroll
        for (int j_1 = 0; j_1 < 4; j_1++) {
            hdh = __fmaf_rn(__uint_as_float(word_h[j_1] << 16), __uint_as_float(word_dh[j_1] << 16), hdh);
            hdh = __fmaf_rn(__uint_as_float(word_h[j_1] & 4294901760u), __uint_as_float(word_dh[j_1] & 4294901760u), hdh);
        }
    }
    if (h_odd != 0) {
        #pragma unroll
        for (int it2_2 = 4; it2_2 < 8; it2_2++) {
            int e_off_2 = (h_col2 + it2_2 * 8) / 64 * 16384 + (h_row2 * 128 + (h_col2 + it2_2 * 8) % 64 * 2 ^ (h_row2 * 128 + (h_col2 + it2_2 * 8) % 64 * 2 >> 7 & 7) << 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&word_h[0])), "=r"(*reinterpret_cast<uint32_t*>(&word_h[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&word_h[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&word_h[(0) + 3]))
                : "r"(s_h_addr + (unsigned int)e_off_2));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&word_dh[0])), "=r"(*reinterpret_cast<uint32_t*>(&word_dh[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&word_dh[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&word_dh[(0) + 3]))
                : "r"(s_dh_addr + (unsigned int)e_off_2));
            #pragma unroll
            for (int j_2 = 0; j_2 < 4; j_2++) {
                hdh = __fmaf_rn(__uint_as_float(word_h[j_2] << 16), __uint_as_float(word_dh[j_2] << 16), hdh);
                hdh = __fmaf_rn(__uint_as_float(word_h[j_2] & 4294901760u), __uint_as_float(word_dh[j_2] & 4294901760u), hdh);
            }
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
    int r0_2 = tid / 16;
    int c0_2 = tid % 16 * 8;
    long long row_stride_2 = (long long)num_heads * 128;
    long long src_row_2 = (long long)(tok0 + r0_2) * row_stride_2 + (long long)hq * 128 + (long long)c0_2;
    unsigned int dst0_2 = s_qn_addr + (unsigned int)(c0_2 / 64 * 8192 + (r0_2 * 128 + c0_2 % 64 * 2 ^ (r0_2 * 128 + c0_2 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_2), "l"(qn + src_row_2), "r"((n_valid > r0_2) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_2 + 2048), "l"(qn + (src_row_2 + 16 * row_stride_2)), "r"((n_valid > r0_2 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_2 + 4096), "l"(qn + (src_row_2 + 32 * row_stride_2)), "r"((n_valid > r0_2 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_2 + 6144), "l"(qn + (src_row_2 + 48 * row_stride_2)), "r"((n_valid > r0_2 + 48) ? 16 : 0));
    int r0_3 = tid / 16;
    int c0_3 = tid % 16 * 8;
    long long row_stride_3 = (long long)num_v_heads * 128;
    long long src_row_3 = (long long)(tok0 + r0_3) * row_stride_3 + (long long)hv * 128 + (long long)c0_3;
    unsigned int dst0_3 = s_dv_addr + (unsigned int)(c0_3 / 64 * 8192 + (r0_3 * 128 + c0_3 % 64 * 2 ^ (r0_3 * 128 + c0_3 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_3), "l"(dv2 + src_row_3), "r"((n_valid > r0_3) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_3 + 2048), "l"(dv2 + (src_row_3 + 16 * row_stride_3)), "r"((n_valid > r0_3 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_3 + 4096), "l"(dv2 + (src_row_3 + 32 * row_stride_3)), "r"((n_valid > r0_3 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_3 + 6144), "l"(dv2 + (src_row_3 + 48 * row_stride_3)), "r"((n_valid > r0_3 + 48) ? 16 : 0));
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
    int r0_4 = tid / 16;
    int c0_4 = tid % 16 * 8;
    long long row_stride_4 = (long long)num_v_heads * 128;
    long long src_row_4 = (long long)(tok0 + r0_4) * row_stride_4 + (long long)hv * 128 + (long long)c0_4;
    unsigned int dst0_4 = s_v_addr + (unsigned int)(c0_4 / 64 * 8192 + (r0_4 * 128 + c0_4 % 64 * 2 ^ (r0_4 * 128 + c0_4 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_4), "l"(v + src_row_4), "r"((n_valid > r0_4) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_4 + 2048), "l"(v + (src_row_4 + 16 * row_stride_4)), "r"((n_valid > r0_4 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_4 + 4096), "l"(v + (src_row_4 + 32 * row_stride_4)), "r"((n_valid > r0_4 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_4 + 6144), "l"(v + (src_row_4 + 48 * row_stride_4)), "r"((n_valid > r0_4 + 48) ? 16 : 0));
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_20 = ((s_ds_addr) >> 4) & 0x7FFF;
        int _smem_desc_lo_21 = (((s_knT_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
        int _smem_desc_lo_20 = ((s_ds_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_21 = (((s_knT_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_28 = (((s_dsT_addr) >> 4) & 0x7FFF) | 0x2000000;
        int _smem_desc_lo_29 = (((s_qnT_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
        int _smem_desc_lo_28 = (((s_dsT_addr) >> 4) & 0x3FFF) | 0x2000000;
        int _smem_desc_lo_29 = (((s_qnT_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_31 = ((s_dv_addr) >> 4) & 0x7FFF;
        int _smem_desc_lo_32 = (((s_hT_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
        int _smem_desc_lo_31 = ((s_dv_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_32 = (((s_hT_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
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
    int r0_5 = tid / 8;
    int c0_5 = tid % 8 * 8;
    long long row_stride_5 = (long long)num_v_heads * 64;
    long long src_row_5 = (long long)(tok0 + r0_5) * row_stride_5 + (long long)hv * 64 + (long long)c0_5;
    unsigned int dst0_5 = s_a_addr + (unsigned int)(c0_5 / 64 * 8192 + (r0_5 * 128 + c0_5 % 64 * 2 ^ (r0_5 * 128 + c0_5 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_5), "l"(A + src_row_5), "r"((n_valid > r0_5) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_5 + 4096), "l"(A + (src_row_5 + 32 * row_stride_5)), "r"((n_valid > r0_5 + 32) ? 16 : 0));
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
        s_iq[tid_2] = s_dgq[tid_2] + s_dgq[64 + tid_2];
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_48 = ((s_dw_addr) >> 4) & 0x7FFF;
        int _smem_desc_lo_49 = ((s_kn_addr) >> 4) & 0x7FFF;
#else
        int _smem_desc_lo_48 = ((s_dw_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_49 = ((s_kn_addr) >> 4) & 0x3FFF;
#endif
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_51 = ((s_dv_addr) >> 4) & 0x7FFF;
        int _smem_desc_lo_52 = ((s_v_addr) >> 4) & 0x7FFF;
#else
        int _smem_desc_lo_51 = ((s_dv_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_52 = ((s_v_addr) >> 4) & 0x3FFF;
#endif
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_55 = (((s_aT_addr) >> 4) & 0x7FFF) | 0x2000000;
        int _smem_desc_lo_56 = (((s_duT_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
        int _smem_desc_lo_55 = (((s_aT_addr) >> 4) & 0x3FFF) | 0x2000000;
        int _smem_desc_lo_56 = (((s_duT_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_59 = ((s_da_addr) >> 4) & 0x7FFF;
        int _smem_desc_lo_60 = ((s_a_addr) >> 4) & 0x7FFF;
#else
        int _smem_desc_lo_59 = ((s_da_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_60 = ((s_a_addr) >> 4) & 0x3FFF;
#endif
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
#if __CUDA_ARCH__ == 1070
        int _mma_a_lo_0 = ((s_kn_addr) >> 4) & 0x7FFF;
#else
        int _mma_a_lo_0 = ((s_kn_addr) >> 4) & 0x3FFF;
#endif
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_63 = (((s_aT_addr) >> 4) & 0x7FFF) | 0x2000000;
        int _smem_desc_lo_64 = (((s_dwT_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
        int _smem_desc_lo_63 = (((s_aT_addr) >> 4) & 0x3FFF) | 0x2000000;
        int _smem_desc_lo_64 = (((s_dwT_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_67 = (((s_aT_addr) >> 4) & 0x7FFF) | 0x2000000;
        int _smem_desc_lo_68 = (((s_da2T_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
        int _smem_desc_lo_67 = (((s_aT_addr) >> 4) & 0x3FFF) | 0x2000000;
        int _smem_desc_lo_68 = (((s_da2T_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
    if (warp == 0) {
        uint32_t _elected_77 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_77));
#if __CUDA_ARCH__ == 1070
        int _mma_a_lo_1 = ((s_da3b_addr) >> 4) & 0x7FFF;
        int _mma_b_lo_0 = (((s_knT_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
        int _mma_a_lo_1 = ((s_da3b_addr) >> 4) & 0x3FFF;
        int _mma_b_lo_0 = (((s_knT_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_78 = (((s_da3bT_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
        int _smem_desc_lo_78 = (((s_da3bT_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
    int group = num_v_heads / num_heads;
    int tile = block_x / group;
    if (group > 1) {
        if (tile >= ws_tiles) {
            asm volatile("trap;" ::: "memory");
        }
    }
    long long ws_q = (long long)block_x * 16384;
    long long ws_k = ws_q + 8192;
    long long rs_row = (long long)(tok0 + tid_2 % 64) * (long long)num_heads + (long long)hq;
    if (tid_2 < 64) {
        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
            :: "r"(s_rs_addr + (unsigned int)(tid_2 * 4)), "l"(rstd_q + rs_row), "r"((n_valid > tid_2) ? 4 : 0));
    } else if (tid_2 < 128) {
        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
            :: "r"(s_rs_addr + (unsigned int)(tid_2 * 4)), "l"(rstd_k + rs_row), "r"((n_valid > tid_2 - 64) ? 4 : 0));
    }
    asm volatile("cp.async.commit_group;");
    float gw_a = s_g[row_a];
    float gw_b = s_g[row_b];
    float gexp_a = approx_exp2(gw_a);
    float gexp_b = approx_exp2(gw_b);
    float bg_a = gexp_a * beta_a;
    float bg_b = gexp_b * beta_b;
    float ik_a = 0.0f;
    float ik_b = 0.0f;
    unsigned int kn_row = s_kn_addr + (unsigned int)(n0 / 64 * 8192 + row_a * 128 + col_q * 2);
    int kn_swz = row_a & 7;
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
    for (int nb4_0 = 0; nb4_0 < 4; nb4_0++) {
        unsigned int kn_at_0 = kn_row + (unsigned int)((nb4_0 ^ kn_swz) << 4);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(kn_at_0));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(kn_at_0 + 1024));
        float k_a0_0 = __uint_as_float(pair_a[0] << 16);
        float k_b0_0 = __uint_as_float(pair_b[0] << 16);
        float k_a1_0 = __uint_as_float(pair_a[0] & 4294901760u);
        float k_b1_0 = __uint_as_float(pair_b[0] & 4294901760u);
        db_a = __fmaf_rn(acc_s[nb4_0 * 4], k_a0_0 * gexp_a, db_a);
        db_b = __fmaf_rn(acc_s[nb4_0 * 4 + 2], k_b0_0 * gexp_b, db_b);
        dg_a = __fmaf_rn(acc_s[nb4_0 * 4], k_a0_0 * bg_a, dg_a);
        dg_b = __fmaf_rn(acc_s[nb4_0 * 4 + 2], k_b0_0 * bg_b, dg_b);
        db_a = __fmaf_rn(acc_s[nb4_0 * 4 + 1], k_a1_0 * gexp_a, db_a);
        db_b = __fmaf_rn(acc_s[nb4_0 * 4 + 3], k_b1_0 * gexp_b, db_b);
        dg_a = __fmaf_rn(acc_s[nb4_0 * 4 + 1], k_a1_0 * bg_a, dg_a);
        dg_b = __fmaf_rn(acc_s[nb4_0 * 4 + 3], k_b1_0 * bg_b, dg_b);
        acc[nb4_0 * 4] = __fmaf_rn(acc_s[nb4_0 * 4], bg_a, acc[nb4_0 * 4]);
        acc[nb4_0 * 4 + 1] = __fmaf_rn(acc_s[nb4_0 * 4 + 1], bg_a, acc[nb4_0 * 4 + 1]);
        acc[nb4_0 * 4 + 2] = __fmaf_rn(acc_s[nb4_0 * 4 + 2], bg_b, acc[nb4_0 * 4 + 2]);
        acc[nb4_0 * 4 + 3] = __fmaf_rn(acc_s[nb4_0 * 4 + 3], bg_b, acc[nb4_0 * 4 + 3]);
        ik_a = __fmaf_rn(acc[nb4_0 * 4], k_a0_0, ik_a);
        ik_a = __fmaf_rn(acc[nb4_0 * 4 + 1], k_a1_0, ik_a);
        ik_b = __fmaf_rn(acc[nb4_0 * 4 + 2], k_b0_0, ik_b);
        ik_b = __fmaf_rn(acc[nb4_0 * 4 + 3], k_b1_0, ik_b);
    }
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[15]))
        : "r"(ld_w));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int _la = 0; _la < 16; _la++)
        acc[_la] = acc[_la] + acc_s[_la];
    if (group > 1) {
        #pragma unroll
        for (int iw = 0; iw < 4; iw++) {
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.f32 [%0], {%1, %2}; }" :: "l"(ws + (ws_k + (long long)(row_a * 128 + (n0 + col_q) + iw * 8))), "f"(acc[(iw * 4) + 0]), "f"(acc[(iw * 4) + 1]), "r"((unsigned int)(row_a < n_valid)) : "memory");
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.f32 [%0], {%1, %2}; }" :: "l"(ws + (ws_k + (long long)(row_b * 128 + (n0 + col_q) + iw * 8))), "f"(acc[(iw * 4 + 2) + 0]), "f"(acc[(iw * 4 + 2) + 1]), "r"((unsigned int)(row_b < n_valid)) : "memory");
        }
    } else {
        asm volatile(
            "tcgen05.st.sync.aligned.16x256b.x4.b32"
            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
            :: "r"(ld_w), "r"(*reinterpret_cast<const uint32_t*>(&acc[0])), "r"(*reinterpret_cast<const uint32_t*>(&acc[1])), "r"(*reinterpret_cast<const uint32_t*>(&acc[2])), "r"(*reinterpret_cast<const uint32_t*>(&acc[3])), "r"(*reinterpret_cast<const uint32_t*>(&acc[4])), "r"(*reinterpret_cast<const uint32_t*>(&acc[5])), "r"(*reinterpret_cast<const uint32_t*>(&acc[6])), "r"(*reinterpret_cast<const uint32_t*>(&acc[7])), "r"(*reinterpret_cast<const uint32_t*>(&acc[8])), "r"(*reinterpret_cast<const uint32_t*>(&acc[9])), "r"(*reinterpret_cast<const uint32_t*>(&acc[10])), "r"(*reinterpret_cast<const uint32_t*>(&acc[11])), "r"(*reinterpret_cast<const uint32_t*>(&acc[12])), "r"(*reinterpret_cast<const uint32_t*>(&acc[13])), "r"(*reinterpret_cast<const uint32_t*>(&acc[14])), "r"(*reinterpret_cast<const uint32_t*>(&acc[15])));
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
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
    #pragma unroll
    for (int nb4_1 = 0; nb4_1 < 4; nb4_1++) {
        unsigned int kn_at_1 = kn_row + (unsigned int)((4 + nb4_1 ^ kn_swz) << 4);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(kn_at_1));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(kn_at_1 + 1024));
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
        ik_a = __fmaf_rn(acc[nb4_1 * 4], k_a0_1, ik_a);
        ik_a = __fmaf_rn(acc[nb4_1 * 4 + 1], k_a1_1, ik_a);
        ik_b = __fmaf_rn(acc[nb4_1 * 4 + 2], k_b0_1, ik_b);
        ik_b = __fmaf_rn(acc[nb4_1 * 4 + 3], k_b1_1, ik_b);
    }
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[15]))
        : "r"(ld_w + 32));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int _la = 0; _la < 16; _la++)
        acc[_la] = acc[_la] + acc_s[_la];
    if (group > 1) {
        #pragma unroll
        for (int iw_1 = 0; iw_1 < 4; iw_1++) {
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.f32 [%0], {%1, %2}; }" :: "l"(ws + (ws_k + (long long)(row_a * 128 + (n0 + 32 + col_q) + iw_1 * 8))), "f"(acc[(iw_1 * 4) + 0]), "f"(acc[(iw_1 * 4) + 1]), "r"((unsigned int)(row_a < n_valid)) : "memory");
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.f32 [%0], {%1, %2}; }" :: "l"(ws + (ws_k + (long long)(row_b * 128 + (n0 + 32 + col_q) + iw_1 * 8))), "f"(acc[(iw_1 * 4 + 2) + 0]), "f"(acc[(iw_1 * 4 + 2) + 1]), "r"((unsigned int)(row_b < n_valid)) : "memory");
        }
    } else {
        asm volatile(
            "tcgen05.st.sync.aligned.16x256b.x4.b32"
            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
            :: "r"(ld_w + 32), "r"(*reinterpret_cast<const uint32_t*>(&acc[0])), "r"(*reinterpret_cast<const uint32_t*>(&acc[1])), "r"(*reinterpret_cast<const uint32_t*>(&acc[2])), "r"(*reinterpret_cast<const uint32_t*>(&acc[3])), "r"(*reinterpret_cast<const uint32_t*>(&acc[4])), "r"(*reinterpret_cast<const uint32_t*>(&acc[5])), "r"(*reinterpret_cast<const uint32_t*>(&acc[6])), "r"(*reinterpret_cast<const uint32_t*>(&acc[7])), "r"(*reinterpret_cast<const uint32_t*>(&acc[8])), "r"(*reinterpret_cast<const uint32_t*>(&acc[9])), "r"(*reinterpret_cast<const uint32_t*>(&acc[10])), "r"(*reinterpret_cast<const uint32_t*>(&acc[11])), "r"(*reinterpret_cast<const uint32_t*>(&acc[12])), "r"(*reinterpret_cast<const uint32_t*>(&acc[13])), "r"(*reinterpret_cast<const uint32_t*>(&acc[14])), "r"(*reinterpret_cast<const uint32_t*>(&acc[15])));
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    if (group > 1) {
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
            : "r"(ld_w + 1048576));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int iw_2 = 0; iw_2 < 4; iw_2++) {
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.f32 [%0], {%1, %2}; }" :: "l"(ws + (ws_q + (long long)(row_a * 128 + (n0 + col_q) + iw_2 * 8))), "f"(acc[(iw_2 * 4) + 0]), "f"(acc[(iw_2 * 4) + 1]), "r"((unsigned int)(row_a < n_valid)) : "memory");
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.f32 [%0], {%1, %2}; }" :: "l"(ws + (ws_q + (long long)(row_b * 128 + (n0 + col_q) + iw_2 * 8))), "f"(acc[(iw_2 * 4 + 2) + 0]), "f"(acc[(iw_2 * 4 + 2) + 1]), "r"((unsigned int)(row_b < n_valid)) : "memory");
        }
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
            : "r"(ld_w + 1048608));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int iw_3 = 0; iw_3 < 4; iw_3++) {
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.f32 [%0], {%1, %2}; }" :: "l"(ws + (ws_q + (long long)(row_a * 128 + (n0 + 32 + col_q) + iw_3 * 8))), "f"(acc[(iw_3 * 4) + 0]), "f"(acc[(iw_3 * 4) + 1]), "r"((unsigned int)(row_a < n_valid)) : "memory");
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.f32 [%0], {%1, %2}; }" :: "l"(ws + (ws_q + (long long)(row_b * 128 + (n0 + 32 + col_q) + iw_3 * 8))), "f"(acc[(iw_3 * 4 + 2) + 0]), "f"(acc[(iw_3 * 4 + 2) + 1]), "r"((unsigned int)(row_b < n_valid)) : "memory");
        }
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, db_a, 1);
    db_a = db_a + _shfl_xor_sync_79;
    float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, db_a, 2);
    db_a = db_a + _shfl_xor_sync_80;
    float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, db_b, 1);
    db_b = db_b + _shfl_xor_sync_81;
    float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, db_b, 2);
    db_b = db_b + _shfl_xor_sync_82;
    float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, dg_a, 1);
    dg_a = dg_a + _shfl_xor_sync_83;
    float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, dg_a, 2);
    dg_a = dg_a + _shfl_xor_sync_84;
    float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, dg_b, 1);
    dg_b = dg_b + _shfl_xor_sync_85;
    float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, dg_b, 2);
    dg_b = dg_b + _shfl_xor_sync_86;
    float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, ik_a, 1);
    ik_a = ik_a + _shfl_xor_sync_87;
    float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, ik_a, 2);
    ik_a = ik_a + _shfl_xor_sync_88;
    float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, ik_b, 1);
    ik_b = ik_b + _shfl_xor_sync_89;
    float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, ik_b, 2);
    ik_b = ik_b + _shfl_xor_sync_90;
    if (lane_0 % 4 == 0) {
        s_dbp[nh * 64 + row_a] = db_a;
        s_dbp[nh * 64 + row_b] = db_b;
        s_dgp[nh * 64 + row_a] = dg_a;
        s_dgp[nh * 64 + row_b] = dg_b;
        s_ik[nh * 64 + row_a] = ik_a;
        s_ik[nh * 64 + row_b] = ik_b;
    }
    __syncthreads();
    if (tid_2 < n_valid) {
        long long gate_row_index = (long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv;
        float colsum = s_col[tid_2] + s_col[64 + tid_2] + s_col[128 + tid_2] + s_col[192 + tid_2];
        float db_row = s_dbp[tid_2] + s_dbp[64 + tid_2];
        float dg_row = s_dgp[tid_2] + s_dgp[64 + tid_2];
        dbeta_out[gate_row_index] = db_row;
        s_dg2[tid_2] = dg_row - colsum;
    }
    asm volatile("cp.async.wait_group 0;");
    if (group == 1) {
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;");
        float iq_a = s_iq[row_a];
        float iq_b = s_iq[row_b];
        float ikr_a = s_dgk[row_a] + s_dgk[64 + row_a] + (s_ik[row_a] + s_ik[64 + row_a]);
        float ikr_b = s_dgk[row_b] + s_dgk[64 + row_b] + (s_ik[row_b] + s_ik[64 + row_b]);
        float rq_a = s_rs[row_a];
        float rq_b = s_rs[row_b];
        float rk_a = s_rs[64 + row_a];
        float rk_b = s_rs[64 + row_b];
        long long ob_a = ((long long)(tok0 + row_a) * (long long)num_heads + (long long)hq) * 128;
        long long ob_b = ((long long)(tok0 + row_b) * (long long)num_heads + (long long)hq) * 128;
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
            : "r"(ld_w + 1048576));
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(*reinterpret_cast<uint32_t*>(&acc_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[15]))
            : "r"(ld_w));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int iv = 0; iv < 4; iv++) {
            long long ycol = (long long)(n0 + col_q + iv * 8);
            unsigned int y_a[1] = {0};
            unsigned int y_b[1] = {0};
            if (row_a < n_valid) {
                {
                    y_a[0] = *reinterpret_cast<const unsigned int*>(qn + (ob_a + ycol) + 0);
                }
            }
            if (row_b < n_valid) {
                {
                    y_b[0] = *reinterpret_cast<const unsigned int*>(qn + (ob_b + ycol) + 0);
                }
            }
            if (normalize_qk != 0) {
                acc[iv * 4] = __fmaf_rn(acc[iv * 4], rq_a, -(iq_a * __uint_as_float(y_a[0] << 16) * rq_a));
                acc[iv * 4 + 1] = __fmaf_rn(acc[iv * 4 + 1], rq_a, -(iq_a * __uint_as_float(y_a[0] & 4294901760u) * rq_a));
                acc[iv * 4 + 2] = __fmaf_rn(acc[iv * 4 + 2], rq_b, -(iq_b * __uint_as_float(y_b[0] << 16) * rq_b));
                acc[iv * 4 + 3] = __fmaf_rn(acc[iv * 4 + 3], rq_b, -(iq_b * __uint_as_float(y_b[0] & 4294901760u) * rq_b));
            }
            if (row_a < n_valid) {
                unsigned int o_a[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + iv * 4)[_lp*2 + 0], (acc + iv * 4)[_lp*2+1 + 0]));
                    o_a[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dq_out + (ob_a + ycol))[0] = reinterpret_cast<unsigned*>(o_a)[0];
            }
            if (row_b < n_valid) {
                unsigned int o_b[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + iv * 4 + 2)[_lp*2 + 0], (acc + iv * 4 + 2)[_lp*2+1 + 0]));
                    o_b[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dq_out + (ob_b + ycol))[0] = reinterpret_cast<unsigned*>(o_b)[0];
            }
        }
        #pragma unroll
        for (int iv_1 = 0; iv_1 < 4; iv_1++) {
            long long ycol_1 = (long long)(n0 + col_q + iv_1 * 8);
            unsigned int y_a_1[1] = {0};
            unsigned int y_b_1[1] = {0};
            if (row_a < n_valid) {
                {
                    y_a_1[0] = *reinterpret_cast<const unsigned int*>(kn + (ob_a + ycol_1) + 0);
                }
            }
            if (row_b < n_valid) {
                {
                    y_b_1[0] = *reinterpret_cast<const unsigned int*>(kn + (ob_b + ycol_1) + 0);
                }
            }
            if (normalize_qk != 0) {
                acc_s[iv_1 * 4] = __fmaf_rn(acc_s[iv_1 * 4], rk_a, -(ikr_a * __uint_as_float(y_a_1[0] << 16) * rk_a));
                acc_s[iv_1 * 4 + 1] = __fmaf_rn(acc_s[iv_1 * 4 + 1], rk_a, -(ikr_a * __uint_as_float(y_a_1[0] & 4294901760u) * rk_a));
                acc_s[iv_1 * 4 + 2] = __fmaf_rn(acc_s[iv_1 * 4 + 2], rk_b, -(ikr_b * __uint_as_float(y_b_1[0] << 16) * rk_b));
                acc_s[iv_1 * 4 + 3] = __fmaf_rn(acc_s[iv_1 * 4 + 3], rk_b, -(ikr_b * __uint_as_float(y_b_1[0] & 4294901760u) * rk_b));
            }
            if (row_a < n_valid) {
                unsigned int o_a_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc_s + iv_1 * 4)[_lp*2 + 0], (acc_s + iv_1 * 4)[_lp*2+1 + 0]));
                    o_a_1[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dk_out + (ob_a + ycol_1))[0] = reinterpret_cast<unsigned*>(o_a_1)[0];
            }
            if (row_b < n_valid) {
                unsigned int o_b_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc_s + iv_1 * 4 + 2)[_lp*2 + 0], (acc_s + iv_1 * 4 + 2)[_lp*2+1 + 0]));
                    o_b_1[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dk_out + (ob_b + ycol_1))[0] = reinterpret_cast<unsigned*>(o_b_1)[0];
            }
        }
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
            : "r"(ld_w + 1048608));
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(*reinterpret_cast<uint32_t*>(&acc_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc_s[15]))
            : "r"(ld_w + 32));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int iv_2 = 0; iv_2 < 4; iv_2++) {
            long long ycol_2 = (long long)(n0 + 32 + col_q + iv_2 * 8);
            unsigned int y_a_2[1] = {0};
            unsigned int y_b_2[1] = {0};
            if (row_a < n_valid) {
                {
                    y_a_2[0] = *reinterpret_cast<const unsigned int*>(qn + (ob_a + ycol_2) + 0);
                }
            }
            if (row_b < n_valid) {
                {
                    y_b_2[0] = *reinterpret_cast<const unsigned int*>(qn + (ob_b + ycol_2) + 0);
                }
            }
            if (normalize_qk != 0) {
                acc[iv_2 * 4] = __fmaf_rn(acc[iv_2 * 4], rq_a, -(iq_a * __uint_as_float(y_a_2[0] << 16) * rq_a));
                acc[iv_2 * 4 + 1] = __fmaf_rn(acc[iv_2 * 4 + 1], rq_a, -(iq_a * __uint_as_float(y_a_2[0] & 4294901760u) * rq_a));
                acc[iv_2 * 4 + 2] = __fmaf_rn(acc[iv_2 * 4 + 2], rq_b, -(iq_b * __uint_as_float(y_b_2[0] << 16) * rq_b));
                acc[iv_2 * 4 + 3] = __fmaf_rn(acc[iv_2 * 4 + 3], rq_b, -(iq_b * __uint_as_float(y_b_2[0] & 4294901760u) * rq_b));
            }
            if (row_a < n_valid) {
                unsigned int o_a_2[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + iv_2 * 4)[_lp*2 + 0], (acc + iv_2 * 4)[_lp*2+1 + 0]));
                    o_a_2[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dq_out + (ob_a + ycol_2))[0] = reinterpret_cast<unsigned*>(o_a_2)[0];
            }
            if (row_b < n_valid) {
                unsigned int o_b_2[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + iv_2 * 4 + 2)[_lp*2 + 0], (acc + iv_2 * 4 + 2)[_lp*2+1 + 0]));
                    o_b_2[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dq_out + (ob_b + ycol_2))[0] = reinterpret_cast<unsigned*>(o_b_2)[0];
            }
        }
        #pragma unroll
        for (int iv_3 = 0; iv_3 < 4; iv_3++) {
            long long ycol_3 = (long long)(n0 + 32 + col_q + iv_3 * 8);
            unsigned int y_a_3[1] = {0};
            unsigned int y_b_3[1] = {0};
            if (row_a < n_valid) {
                {
                    y_a_3[0] = *reinterpret_cast<const unsigned int*>(kn + (ob_a + ycol_3) + 0);
                }
            }
            if (row_b < n_valid) {
                {
                    y_b_3[0] = *reinterpret_cast<const unsigned int*>(kn + (ob_b + ycol_3) + 0);
                }
            }
            if (normalize_qk != 0) {
                acc_s[iv_3 * 4] = __fmaf_rn(acc_s[iv_3 * 4], rk_a, -(ikr_a * __uint_as_float(y_a_3[0] << 16) * rk_a));
                acc_s[iv_3 * 4 + 1] = __fmaf_rn(acc_s[iv_3 * 4 + 1], rk_a, -(ikr_a * __uint_as_float(y_a_3[0] & 4294901760u) * rk_a));
                acc_s[iv_3 * 4 + 2] = __fmaf_rn(acc_s[iv_3 * 4 + 2], rk_b, -(ikr_b * __uint_as_float(y_b_3[0] << 16) * rk_b));
                acc_s[iv_3 * 4 + 3] = __fmaf_rn(acc_s[iv_3 * 4 + 3], rk_b, -(ikr_b * __uint_as_float(y_b_3[0] & 4294901760u) * rk_b));
            }
            if (row_a < n_valid) {
                unsigned int o_a_3[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc_s + iv_3 * 4)[_lp*2 + 0], (acc_s + iv_3 * 4)[_lp*2+1 + 0]));
                    o_a_3[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dk_out + (ob_a + ycol_3))[0] = reinterpret_cast<unsigned*>(o_a_3)[0];
            }
            if (row_b < n_valid) {
                unsigned int o_b_3[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc_s + iv_3 * 4 + 2)[_lp*2 + 0], (acc_s + iv_3 * 4 + 2)[_lp*2+1 + 0]));
                    o_b_3[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dk_out + (ob_b + ycol_3))[0] = reinterpret_cast<unsigned*>(o_b_3)[0];
            }
        }
    } else {
        if (tid_2 == 255) {
            uint32_t done_old;
            asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                : "=r"(done_old) : "l"(&ws_count[tile]), "r"(static_cast<uint32_t>((unsigned int)(group - 1))) : "memory");
            s_flag[0] = ((done_old == (unsigned int)(group - 1)) ? 1 : 0);
        }
        __syncthreads();
        unsigned int last_cta = s_flag[0];
        if (last_cta != 0) {
            asm volatile("fence.acquire.gpu;" ::: "memory");
            float fq_0[16];
            float fk_0[16];
            unsigned int yq4_0[8] = {0};
            int yrow_0 = warp_1 * 8;
            if (yrow_0 < n_valid) {
                {
                    const uint2* _ivptr_91 = reinterpret_cast<const uint2*>(qn + (((long long)(tok0 + yrow_0) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4)) + 0);
                    uint2 _ivld_91;
                    _ivld_91 = *_ivptr_91;
                    yq4_0[0 + 0] = _ivld_91.x;
                    yq4_0[0 + 1] = _ivld_91.y;
                }
            }
            int yrow_1 = warp_1 * 8 + 1;
            if (yrow_1 < n_valid) {
                {
                    const uint2* _ivptr_92 = reinterpret_cast<const uint2*>(qn + (((long long)(tok0 + yrow_1) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4)) + 0);
                    uint2 _ivld_92;
                    _ivld_92 = *_ivptr_92;
                    yq4_0[2 + 0] = _ivld_92.x;
                    yq4_0[2 + 1] = _ivld_92.y;
                }
            }
            int yrow_2 = warp_1 * 8 + 2;
            if (yrow_2 < n_valid) {
                {
                    const uint2* _ivptr_93 = reinterpret_cast<const uint2*>(qn + (((long long)(tok0 + yrow_2) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4)) + 0);
                    uint2 _ivld_93;
                    _ivld_93 = *_ivptr_93;
                    yq4_0[4 + 0] = _ivld_93.x;
                    yq4_0[4 + 1] = _ivld_93.y;
                }
            }
            int yrow_3 = warp_1 * 8 + 3;
            if (yrow_3 < n_valid) {
                {
                    const uint2* _ivptr_94 = reinterpret_cast<const uint2*>(qn + (((long long)(tok0 + yrow_3) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4)) + 0);
                    uint2 _ivld_94;
                    _ivld_94 = *_ivptr_94;
                    yq4_0[6 + 0] = _ivld_94.x;
                    yq4_0[6 + 1] = _ivld_94.y;
                }
            }
            fq_0[0] = 0.0f;
            fk_0[0] = 0.0f;
            fq_0[1] = 0.0f;
            fk_0[1] = 0.0f;
            fq_0[2] = 0.0f;
            fk_0[2] = 0.0f;
            fq_0[3] = 0.0f;
            fk_0[3] = 0.0f;
            fq_0[4] = 0.0f;
            fk_0[4] = 0.0f;
            fq_0[5] = 0.0f;
            fk_0[5] = 0.0f;
            fq_0[6] = 0.0f;
            fk_0[6] = 0.0f;
            fq_0[7] = 0.0f;
            fk_0[7] = 0.0f;
            fq_0[8] = 0.0f;
            fk_0[8] = 0.0f;
            fq_0[9] = 0.0f;
            fk_0[9] = 0.0f;
            fq_0[10] = 0.0f;
            fk_0[10] = 0.0f;
            fq_0[11] = 0.0f;
            fk_0[11] = 0.0f;
            fq_0[12] = 0.0f;
            fk_0[12] = 0.0f;
            fq_0[13] = 0.0f;
            fk_0[13] = 0.0f;
            fq_0[14] = 0.0f;
            fk_0[14] = 0.0f;
            fq_0[15] = 0.0f;
            fk_0[15] = 0.0f;
            for (int j_0 = 0; j_0 < group; j_0++) {
                long long slot_q_0 = (long long)(tile * group + j_0) * 16384;
                float tq_0[16] = {0};
                float tk_0[16] = {0};
                int lrow_0 = warp_1 * 8;
                if (lrow_0 < n_valid) {
                    {
                        unsigned _v4_95_0;
                        unsigned _v4_95_1;
                        unsigned _v4_95_2;
                        unsigned _v4_95_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_95_0), "=r"(_v4_95_1), "=r"(_v4_95_2), "=r"(_v4_95_3) : "l"((const void*)(ws + (slot_q_0 + (long long)(lrow_0 * 128 + lane_0 * 4)))) : "memory");
                        tq_0[0 + 0] = __uint_as_float(_v4_95_0);
                        tq_0[0 + 1] = __uint_as_float(_v4_95_1);
                        tq_0[0 + 2] = __uint_as_float(_v4_95_2);
                        tq_0[0 + 3] = __uint_as_float(_v4_95_3);
                    }
                    {
                        unsigned _v4_96_0;
                        unsigned _v4_96_1;
                        unsigned _v4_96_2;
                        unsigned _v4_96_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_96_0), "=r"(_v4_96_1), "=r"(_v4_96_2), "=r"(_v4_96_3) : "l"((const void*)(ws + (slot_q_0 + (long long)(8192 + lrow_0 * 128 + lane_0 * 4)))) : "memory");
                        tk_0[0 + 0] = __uint_as_float(_v4_96_0);
                        tk_0[0 + 1] = __uint_as_float(_v4_96_1);
                        tk_0[0 + 2] = __uint_as_float(_v4_96_2);
                        tk_0[0 + 3] = __uint_as_float(_v4_96_3);
                    }
                }
                int lrow_1 = warp_1 * 8 + 1;
                if (lrow_1 < n_valid) {
                    {
                        unsigned _v4_97_0;
                        unsigned _v4_97_1;
                        unsigned _v4_97_2;
                        unsigned _v4_97_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_97_0), "=r"(_v4_97_1), "=r"(_v4_97_2), "=r"(_v4_97_3) : "l"((const void*)(ws + (slot_q_0 + (long long)(lrow_1 * 128 + lane_0 * 4)))) : "memory");
                        tq_0[4 + 0] = __uint_as_float(_v4_97_0);
                        tq_0[4 + 1] = __uint_as_float(_v4_97_1);
                        tq_0[4 + 2] = __uint_as_float(_v4_97_2);
                        tq_0[4 + 3] = __uint_as_float(_v4_97_3);
                    }
                    {
                        unsigned _v4_98_0;
                        unsigned _v4_98_1;
                        unsigned _v4_98_2;
                        unsigned _v4_98_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_98_0), "=r"(_v4_98_1), "=r"(_v4_98_2), "=r"(_v4_98_3) : "l"((const void*)(ws + (slot_q_0 + (long long)(8192 + lrow_1 * 128 + lane_0 * 4)))) : "memory");
                        tk_0[4 + 0] = __uint_as_float(_v4_98_0);
                        tk_0[4 + 1] = __uint_as_float(_v4_98_1);
                        tk_0[4 + 2] = __uint_as_float(_v4_98_2);
                        tk_0[4 + 3] = __uint_as_float(_v4_98_3);
                    }
                }
                int lrow_2 = warp_1 * 8 + 2;
                if (lrow_2 < n_valid) {
                    {
                        unsigned _v4_99_0;
                        unsigned _v4_99_1;
                        unsigned _v4_99_2;
                        unsigned _v4_99_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_99_0), "=r"(_v4_99_1), "=r"(_v4_99_2), "=r"(_v4_99_3) : "l"((const void*)(ws + (slot_q_0 + (long long)(lrow_2 * 128 + lane_0 * 4)))) : "memory");
                        tq_0[8 + 0] = __uint_as_float(_v4_99_0);
                        tq_0[8 + 1] = __uint_as_float(_v4_99_1);
                        tq_0[8 + 2] = __uint_as_float(_v4_99_2);
                        tq_0[8 + 3] = __uint_as_float(_v4_99_3);
                    }
                    {
                        unsigned _v4_100_0;
                        unsigned _v4_100_1;
                        unsigned _v4_100_2;
                        unsigned _v4_100_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_100_0), "=r"(_v4_100_1), "=r"(_v4_100_2), "=r"(_v4_100_3) : "l"((const void*)(ws + (slot_q_0 + (long long)(8192 + lrow_2 * 128 + lane_0 * 4)))) : "memory");
                        tk_0[8 + 0] = __uint_as_float(_v4_100_0);
                        tk_0[8 + 1] = __uint_as_float(_v4_100_1);
                        tk_0[8 + 2] = __uint_as_float(_v4_100_2);
                        tk_0[8 + 3] = __uint_as_float(_v4_100_3);
                    }
                }
                int lrow_3 = warp_1 * 8 + 3;
                if (lrow_3 < n_valid) {
                    {
                        unsigned _v4_101_0;
                        unsigned _v4_101_1;
                        unsigned _v4_101_2;
                        unsigned _v4_101_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_101_0), "=r"(_v4_101_1), "=r"(_v4_101_2), "=r"(_v4_101_3) : "l"((const void*)(ws + (slot_q_0 + (long long)(lrow_3 * 128 + lane_0 * 4)))) : "memory");
                        tq_0[12 + 0] = __uint_as_float(_v4_101_0);
                        tq_0[12 + 1] = __uint_as_float(_v4_101_1);
                        tq_0[12 + 2] = __uint_as_float(_v4_101_2);
                        tq_0[12 + 3] = __uint_as_float(_v4_101_3);
                    }
                    {
                        unsigned _v4_102_0;
                        unsigned _v4_102_1;
                        unsigned _v4_102_2;
                        unsigned _v4_102_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_102_0), "=r"(_v4_102_1), "=r"(_v4_102_2), "=r"(_v4_102_3) : "l"((const void*)(ws + (slot_q_0 + (long long)(8192 + lrow_3 * 128 + lane_0 * 4)))) : "memory");
                        tk_0[12 + 0] = __uint_as_float(_v4_102_0);
                        tk_0[12 + 1] = __uint_as_float(_v4_102_1);
                        tk_0[12 + 2] = __uint_as_float(_v4_102_2);
                        tk_0[12 + 3] = __uint_as_float(_v4_102_3);
                    }
                }
                #pragma unroll
                for (int _la = 0; _la < 16; _la++)
                    fq_0[_la] = fq_0[_la] + tq_0[_la];
                #pragma unroll
                for (int _la = 0; _la < 16; _la++)
                    fk_0[_la] = fk_0[_la] + tk_0[_la];
            }
            int frow_0 = warp_1 * 8;
            if (frow_0 < n_valid) {
                unsigned int yw_0[4];
                yw_0[0] = yq4_0[0];
                yw_0[1] = yq4_0[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&yw_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&yw_0[(2) + 1]))
                    : "r"(s_kn_addr + (unsigned int)(lane_0 * 4 / 64 * 8192 + (frow_0 * 128 + lane_0 * 4 % 64 * 2 ^ (frow_0 * 128 + lane_0 * 4 % 64 * 2 >> 7 & 7) << 4))));
                float yq_0[4];
                float yk_0[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yq_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_0[_pair]) << 16);
                    (&yq_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_0[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yk_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_0[2 + _pair]) << 16);
                    (&yk_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_0[2 + _pair]) & 0xffff0000u);
                }
                float rq_0 = s_rs[frow_0];
                float rk_0 = s_rs[64 + frow_0];
                float dq_r_0[4];
                float dk_r_0[4];
                dq_r_0[0] = fq_0[0];
                dq_r_0[1] = fq_0[1];
                dq_r_0[2] = fq_0[2];
                dq_r_0[3] = fq_0[3];
                dk_r_0[0] = fk_0[0];
                dk_r_0[1] = fk_0[1];
                dk_r_0[2] = fk_0[2];
                dk_r_0[3] = fk_0[3];
                float inner_q_0 = 0.0f;
                float inner_k_0 = 0.0f;
                #pragma unroll
                for (int i5_0 = 0; i5_0 < 4; i5_0++) {
                    inner_q_0 = __fmaf_rn(dq_r_0[i5_0], yq_0[i5_0], inner_q_0);
                    inner_k_0 = __fmaf_rn(dk_r_0[i5_0], yk_0[i5_0], inner_k_0);
                }
                float _warp_reduce_103 = inner_q_0;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_103 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_103, offset);
                inner_q_0 = _warp_reduce_103;
                float _warp_reduce_104 = inner_k_0;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_104 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_104, offset);
                inner_k_0 = _warp_reduce_104;
                if (normalize_qk != 0) {
                    #pragma unroll
                    for (int i6_0 = 0; i6_0 < 4; i6_0++) {
                        dq_r_0[i6_0] = __fmaf_rn(dq_r_0[i6_0], rq_0, -(inner_q_0 * yq_0[i6_0] * rq_0));
                        dk_r_0[i6_0] = __fmaf_rn(dk_r_0[i6_0], rk_0, -(inner_k_0 * yk_0[i6_0] * rk_0));
                    }
                }
                long long qbase_0 = ((long long)(tok0 + frow_0) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4);
                unsigned int _packed_4_0[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_0[_lp*2 + 0], dq_r_0[_lp*2+1 + 0]));
                    _packed_4_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_0[_lp*2 + 2], dq_r_0[_lp*2+1 + 2]));
                    _packed_4_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dq_out + (qbase_0))[0] = reinterpret_cast<int2*>(_packed_4_0)[0];
                unsigned int _packed_5_0[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_0[_lp*2 + 0], dk_r_0[_lp*2+1 + 0]));
                    _packed_5_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_0[_lp*2 + 2], dk_r_0[_lp*2+1 + 2]));
                    _packed_5_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dk_out + (qbase_0))[0] = reinterpret_cast<int2*>(_packed_5_0)[0];
            }
            int frow_1 = warp_1 * 8 + 1;
            if (frow_1 < n_valid) {
                unsigned int yw_1[4];
                yw_1[0] = yq4_0[2];
                yw_1[1] = yq4_0[3];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&yw_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&yw_1[(2) + 1]))
                    : "r"(s_kn_addr + (unsigned int)(lane_0 * 4 / 64 * 8192 + (frow_1 * 128 + lane_0 * 4 % 64 * 2 ^ (frow_1 * 128 + lane_0 * 4 % 64 * 2 >> 7 & 7) << 4))));
                float yq_1[4];
                float yk_1[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yq_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_1[_pair]) << 16);
                    (&yq_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yk_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_1[2 + _pair]) << 16);
                    (&yk_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_1[2 + _pair]) & 0xffff0000u);
                }
                float rq_1 = s_rs[frow_1];
                float rk_1 = s_rs[64 + frow_1];
                float dq_r_1[4];
                float dk_r_1[4];
                dq_r_1[0] = fq_0[4];
                dq_r_1[1] = fq_0[5];
                dq_r_1[2] = fq_0[6];
                dq_r_1[3] = fq_0[7];
                dk_r_1[0] = fk_0[4];
                dk_r_1[1] = fk_0[5];
                dk_r_1[2] = fk_0[6];
                dk_r_1[3] = fk_0[7];
                float inner_q_1 = 0.0f;
                float inner_k_1 = 0.0f;
                #pragma unroll
                for (int i5_1 = 0; i5_1 < 4; i5_1++) {
                    inner_q_1 = __fmaf_rn(dq_r_1[i5_1], yq_1[i5_1], inner_q_1);
                    inner_k_1 = __fmaf_rn(dk_r_1[i5_1], yk_1[i5_1], inner_k_1);
                }
                float _warp_reduce_105 = inner_q_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_105 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_105, offset);
                inner_q_1 = _warp_reduce_105;
                float _warp_reduce_106 = inner_k_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_106 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_106, offset);
                inner_k_1 = _warp_reduce_106;
                if (normalize_qk != 0) {
                    #pragma unroll
                    for (int i6_1 = 0; i6_1 < 4; i6_1++) {
                        dq_r_1[i6_1] = __fmaf_rn(dq_r_1[i6_1], rq_1, -(inner_q_1 * yq_1[i6_1] * rq_1));
                        dk_r_1[i6_1] = __fmaf_rn(dk_r_1[i6_1], rk_1, -(inner_k_1 * yk_1[i6_1] * rk_1));
                    }
                }
                long long qbase_1 = ((long long)(tok0 + frow_1) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4);
                unsigned int _packed_4_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_1[_lp*2 + 0], dq_r_1[_lp*2+1 + 0]));
                    _packed_4_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_1[_lp*2 + 2], dq_r_1[_lp*2+1 + 2]));
                    _packed_4_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dq_out + (qbase_1))[0] = reinterpret_cast<int2*>(_packed_4_1)[0];
                unsigned int _packed_5_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_1[_lp*2 + 0], dk_r_1[_lp*2+1 + 0]));
                    _packed_5_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_1[_lp*2 + 2], dk_r_1[_lp*2+1 + 2]));
                    _packed_5_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dk_out + (qbase_1))[0] = reinterpret_cast<int2*>(_packed_5_1)[0];
            }
            int frow_2 = warp_1 * 8 + 2;
            if (frow_2 < n_valid) {
                unsigned int yw_2[4];
                yw_2[0] = yq4_0[4];
                yw_2[1] = yq4_0[5];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&yw_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&yw_2[(2) + 1]))
                    : "r"(s_kn_addr + (unsigned int)(lane_0 * 4 / 64 * 8192 + (frow_2 * 128 + lane_0 * 4 % 64 * 2 ^ (frow_2 * 128 + lane_0 * 4 % 64 * 2 >> 7 & 7) << 4))));
                float yq_2[4];
                float yk_2[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yq_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_2[_pair]) << 16);
                    (&yq_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_2[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yk_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_2[2 + _pair]) << 16);
                    (&yk_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_2[2 + _pair]) & 0xffff0000u);
                }
                float rq_2 = s_rs[frow_2];
                float rk_2 = s_rs[64 + frow_2];
                float dq_r_2[4];
                float dk_r_2[4];
                dq_r_2[0] = fq_0[8];
                dq_r_2[1] = fq_0[9];
                dq_r_2[2] = fq_0[10];
                dq_r_2[3] = fq_0[11];
                dk_r_2[0] = fk_0[8];
                dk_r_2[1] = fk_0[9];
                dk_r_2[2] = fk_0[10];
                dk_r_2[3] = fk_0[11];
                float inner_q_2 = 0.0f;
                float inner_k_2 = 0.0f;
                #pragma unroll
                for (int i5_2 = 0; i5_2 < 4; i5_2++) {
                    inner_q_2 = __fmaf_rn(dq_r_2[i5_2], yq_2[i5_2], inner_q_2);
                    inner_k_2 = __fmaf_rn(dk_r_2[i5_2], yk_2[i5_2], inner_k_2);
                }
                float _warp_reduce_107 = inner_q_2;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_107 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_107, offset);
                inner_q_2 = _warp_reduce_107;
                float _warp_reduce_108 = inner_k_2;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_108 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_108, offset);
                inner_k_2 = _warp_reduce_108;
                if (normalize_qk != 0) {
                    #pragma unroll
                    for (int i6_2 = 0; i6_2 < 4; i6_2++) {
                        dq_r_2[i6_2] = __fmaf_rn(dq_r_2[i6_2], rq_2, -(inner_q_2 * yq_2[i6_2] * rq_2));
                        dk_r_2[i6_2] = __fmaf_rn(dk_r_2[i6_2], rk_2, -(inner_k_2 * yk_2[i6_2] * rk_2));
                    }
                }
                long long qbase_2 = ((long long)(tok0 + frow_2) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4);
                unsigned int _packed_4_2[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_2[_lp*2 + 0], dq_r_2[_lp*2+1 + 0]));
                    _packed_4_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_2[_lp*2 + 2], dq_r_2[_lp*2+1 + 2]));
                    _packed_4_2[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dq_out + (qbase_2))[0] = reinterpret_cast<int2*>(_packed_4_2)[0];
                unsigned int _packed_5_2[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_2[_lp*2 + 0], dk_r_2[_lp*2+1 + 0]));
                    _packed_5_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_2[_lp*2 + 2], dk_r_2[_lp*2+1 + 2]));
                    _packed_5_2[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dk_out + (qbase_2))[0] = reinterpret_cast<int2*>(_packed_5_2)[0];
            }
            int frow_3 = warp_1 * 8 + 3;
            if (frow_3 < n_valid) {
                unsigned int yw_3[4];
                yw_3[0] = yq4_0[6];
                yw_3[1] = yq4_0[7];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&yw_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&yw_3[(2) + 1]))
                    : "r"(s_kn_addr + (unsigned int)(lane_0 * 4 / 64 * 8192 + (frow_3 * 128 + lane_0 * 4 % 64 * 2 ^ (frow_3 * 128 + lane_0 * 4 % 64 * 2 >> 7 & 7) << 4))));
                float yq_3[4];
                float yk_3[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yq_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_3[_pair]) << 16);
                    (&yq_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_3[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yk_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_3[2 + _pair]) << 16);
                    (&yk_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_3[2 + _pair]) & 0xffff0000u);
                }
                float rq_3 = s_rs[frow_3];
                float rk_3 = s_rs[64 + frow_3];
                float dq_r_3[4];
                float dk_r_3[4];
                dq_r_3[0] = fq_0[12];
                dq_r_3[1] = fq_0[13];
                dq_r_3[2] = fq_0[14];
                dq_r_3[3] = fq_0[15];
                dk_r_3[0] = fk_0[12];
                dk_r_3[1] = fk_0[13];
                dk_r_3[2] = fk_0[14];
                dk_r_3[3] = fk_0[15];
                float inner_q_3 = 0.0f;
                float inner_k_3 = 0.0f;
                #pragma unroll
                for (int i5_3 = 0; i5_3 < 4; i5_3++) {
                    inner_q_3 = __fmaf_rn(dq_r_3[i5_3], yq_3[i5_3], inner_q_3);
                    inner_k_3 = __fmaf_rn(dk_r_3[i5_3], yk_3[i5_3], inner_k_3);
                }
                float _warp_reduce_109 = inner_q_3;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_109 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_109, offset);
                inner_q_3 = _warp_reduce_109;
                float _warp_reduce_110 = inner_k_3;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_110 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_110, offset);
                inner_k_3 = _warp_reduce_110;
                if (normalize_qk != 0) {
                    #pragma unroll
                    for (int i6_3 = 0; i6_3 < 4; i6_3++) {
                        dq_r_3[i6_3] = __fmaf_rn(dq_r_3[i6_3], rq_3, -(inner_q_3 * yq_3[i6_3] * rq_3));
                        dk_r_3[i6_3] = __fmaf_rn(dk_r_3[i6_3], rk_3, -(inner_k_3 * yk_3[i6_3] * rk_3));
                    }
                }
                long long qbase_3 = ((long long)(tok0 + frow_3) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4);
                unsigned int _packed_4_3[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_3[_lp*2 + 0], dq_r_3[_lp*2+1 + 0]));
                    _packed_4_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_3[_lp*2 + 2], dq_r_3[_lp*2+1 + 2]));
                    _packed_4_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dq_out + (qbase_3))[0] = reinterpret_cast<int2*>(_packed_4_3)[0];
                unsigned int _packed_5_3[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_3[_lp*2 + 0], dk_r_3[_lp*2+1 + 0]));
                    _packed_5_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_3[_lp*2 + 2], dk_r_3[_lp*2+1 + 2]));
                    _packed_5_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dk_out + (qbase_3))[0] = reinterpret_cast<int2*>(_packed_5_3)[0];
            }
            float fq_1[16];
            float fk_1[16];
            unsigned int yq4_1[8] = {0};
            int yrow_4 = warp_1 * 8 + 4;
            if (yrow_4 < n_valid) {
                {
                    const uint2* _ivptr_111 = reinterpret_cast<const uint2*>(qn + (((long long)(tok0 + yrow_4) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4)) + 0);
                    uint2 _ivld_111;
                    _ivld_111 = *_ivptr_111;
                    yq4_1[0 + 0] = _ivld_111.x;
                    yq4_1[0 + 1] = _ivld_111.y;
                }
            }
            int yrow_5 = warp_1 * 8 + 4 + 1;
            if (yrow_5 < n_valid) {
                {
                    const uint2* _ivptr_112 = reinterpret_cast<const uint2*>(qn + (((long long)(tok0 + yrow_5) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4)) + 0);
                    uint2 _ivld_112;
                    _ivld_112 = *_ivptr_112;
                    yq4_1[2 + 0] = _ivld_112.x;
                    yq4_1[2 + 1] = _ivld_112.y;
                }
            }
            int yrow_6 = warp_1 * 8 + 4 + 2;
            if (yrow_6 < n_valid) {
                {
                    const uint2* _ivptr_113 = reinterpret_cast<const uint2*>(qn + (((long long)(tok0 + yrow_6) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4)) + 0);
                    uint2 _ivld_113;
                    _ivld_113 = *_ivptr_113;
                    yq4_1[4 + 0] = _ivld_113.x;
                    yq4_1[4 + 1] = _ivld_113.y;
                }
            }
            int yrow_7 = warp_1 * 8 + 4 + 3;
            if (yrow_7 < n_valid) {
                {
                    const uint2* _ivptr_114 = reinterpret_cast<const uint2*>(qn + (((long long)(tok0 + yrow_7) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4)) + 0);
                    uint2 _ivld_114;
                    _ivld_114 = *_ivptr_114;
                    yq4_1[6 + 0] = _ivld_114.x;
                    yq4_1[6 + 1] = _ivld_114.y;
                }
            }
            fq_1[0] = 0.0f;
            fk_1[0] = 0.0f;
            fq_1[1] = 0.0f;
            fk_1[1] = 0.0f;
            fq_1[2] = 0.0f;
            fk_1[2] = 0.0f;
            fq_1[3] = 0.0f;
            fk_1[3] = 0.0f;
            fq_1[4] = 0.0f;
            fk_1[4] = 0.0f;
            fq_1[5] = 0.0f;
            fk_1[5] = 0.0f;
            fq_1[6] = 0.0f;
            fk_1[6] = 0.0f;
            fq_1[7] = 0.0f;
            fk_1[7] = 0.0f;
            fq_1[8] = 0.0f;
            fk_1[8] = 0.0f;
            fq_1[9] = 0.0f;
            fk_1[9] = 0.0f;
            fq_1[10] = 0.0f;
            fk_1[10] = 0.0f;
            fq_1[11] = 0.0f;
            fk_1[11] = 0.0f;
            fq_1[12] = 0.0f;
            fk_1[12] = 0.0f;
            fq_1[13] = 0.0f;
            fk_1[13] = 0.0f;
            fq_1[14] = 0.0f;
            fk_1[14] = 0.0f;
            fq_1[15] = 0.0f;
            fk_1[15] = 0.0f;
            for (int j_1_1 = 0; j_1_1 < group; j_1_1++) {
                long long slot_q_1 = (long long)(tile * group + j_1_1) * 16384;
                float tq_1[16] = {0};
                float tk_1[16] = {0};
                int lrow_4 = warp_1 * 8 + 4;
                if (lrow_4 < n_valid) {
                    {
                        unsigned _v4_115_0;
                        unsigned _v4_115_1;
                        unsigned _v4_115_2;
                        unsigned _v4_115_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_115_0), "=r"(_v4_115_1), "=r"(_v4_115_2), "=r"(_v4_115_3) : "l"((const void*)(ws + (slot_q_1 + (long long)(lrow_4 * 128 + lane_0 * 4)))) : "memory");
                        tq_1[0 + 0] = __uint_as_float(_v4_115_0);
                        tq_1[0 + 1] = __uint_as_float(_v4_115_1);
                        tq_1[0 + 2] = __uint_as_float(_v4_115_2);
                        tq_1[0 + 3] = __uint_as_float(_v4_115_3);
                    }
                    {
                        unsigned _v4_116_0;
                        unsigned _v4_116_1;
                        unsigned _v4_116_2;
                        unsigned _v4_116_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_116_0), "=r"(_v4_116_1), "=r"(_v4_116_2), "=r"(_v4_116_3) : "l"((const void*)(ws + (slot_q_1 + (long long)(8192 + lrow_4 * 128 + lane_0 * 4)))) : "memory");
                        tk_1[0 + 0] = __uint_as_float(_v4_116_0);
                        tk_1[0 + 1] = __uint_as_float(_v4_116_1);
                        tk_1[0 + 2] = __uint_as_float(_v4_116_2);
                        tk_1[0 + 3] = __uint_as_float(_v4_116_3);
                    }
                }
                int lrow_5 = warp_1 * 8 + 4 + 1;
                if (lrow_5 < n_valid) {
                    {
                        unsigned _v4_117_0;
                        unsigned _v4_117_1;
                        unsigned _v4_117_2;
                        unsigned _v4_117_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_117_0), "=r"(_v4_117_1), "=r"(_v4_117_2), "=r"(_v4_117_3) : "l"((const void*)(ws + (slot_q_1 + (long long)(lrow_5 * 128 + lane_0 * 4)))) : "memory");
                        tq_1[4 + 0] = __uint_as_float(_v4_117_0);
                        tq_1[4 + 1] = __uint_as_float(_v4_117_1);
                        tq_1[4 + 2] = __uint_as_float(_v4_117_2);
                        tq_1[4 + 3] = __uint_as_float(_v4_117_3);
                    }
                    {
                        unsigned _v4_118_0;
                        unsigned _v4_118_1;
                        unsigned _v4_118_2;
                        unsigned _v4_118_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_118_0), "=r"(_v4_118_1), "=r"(_v4_118_2), "=r"(_v4_118_3) : "l"((const void*)(ws + (slot_q_1 + (long long)(8192 + lrow_5 * 128 + lane_0 * 4)))) : "memory");
                        tk_1[4 + 0] = __uint_as_float(_v4_118_0);
                        tk_1[4 + 1] = __uint_as_float(_v4_118_1);
                        tk_1[4 + 2] = __uint_as_float(_v4_118_2);
                        tk_1[4 + 3] = __uint_as_float(_v4_118_3);
                    }
                }
                int lrow_6 = warp_1 * 8 + 4 + 2;
                if (lrow_6 < n_valid) {
                    {
                        unsigned _v4_119_0;
                        unsigned _v4_119_1;
                        unsigned _v4_119_2;
                        unsigned _v4_119_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_119_0), "=r"(_v4_119_1), "=r"(_v4_119_2), "=r"(_v4_119_3) : "l"((const void*)(ws + (slot_q_1 + (long long)(lrow_6 * 128 + lane_0 * 4)))) : "memory");
                        tq_1[8 + 0] = __uint_as_float(_v4_119_0);
                        tq_1[8 + 1] = __uint_as_float(_v4_119_1);
                        tq_1[8 + 2] = __uint_as_float(_v4_119_2);
                        tq_1[8 + 3] = __uint_as_float(_v4_119_3);
                    }
                    {
                        unsigned _v4_120_0;
                        unsigned _v4_120_1;
                        unsigned _v4_120_2;
                        unsigned _v4_120_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_120_0), "=r"(_v4_120_1), "=r"(_v4_120_2), "=r"(_v4_120_3) : "l"((const void*)(ws + (slot_q_1 + (long long)(8192 + lrow_6 * 128 + lane_0 * 4)))) : "memory");
                        tk_1[8 + 0] = __uint_as_float(_v4_120_0);
                        tk_1[8 + 1] = __uint_as_float(_v4_120_1);
                        tk_1[8 + 2] = __uint_as_float(_v4_120_2);
                        tk_1[8 + 3] = __uint_as_float(_v4_120_3);
                    }
                }
                int lrow_7 = warp_1 * 8 + 4 + 3;
                if (lrow_7 < n_valid) {
                    {
                        unsigned _v4_121_0;
                        unsigned _v4_121_1;
                        unsigned _v4_121_2;
                        unsigned _v4_121_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_121_0), "=r"(_v4_121_1), "=r"(_v4_121_2), "=r"(_v4_121_3) : "l"((const void*)(ws + (slot_q_1 + (long long)(lrow_7 * 128 + lane_0 * 4)))) : "memory");
                        tq_1[12 + 0] = __uint_as_float(_v4_121_0);
                        tq_1[12 + 1] = __uint_as_float(_v4_121_1);
                        tq_1[12 + 2] = __uint_as_float(_v4_121_2);
                        tq_1[12 + 3] = __uint_as_float(_v4_121_3);
                    }
                    {
                        unsigned _v4_122_0;
                        unsigned _v4_122_1;
                        unsigned _v4_122_2;
                        unsigned _v4_122_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_122_0), "=r"(_v4_122_1), "=r"(_v4_122_2), "=r"(_v4_122_3) : "l"((const void*)(ws + (slot_q_1 + (long long)(8192 + lrow_7 * 128 + lane_0 * 4)))) : "memory");
                        tk_1[12 + 0] = __uint_as_float(_v4_122_0);
                        tk_1[12 + 1] = __uint_as_float(_v4_122_1);
                        tk_1[12 + 2] = __uint_as_float(_v4_122_2);
                        tk_1[12 + 3] = __uint_as_float(_v4_122_3);
                    }
                }
                #pragma unroll
                for (int _la = 0; _la < 16; _la++)
                    fq_1[_la] = fq_1[_la] + tq_1[_la];
                #pragma unroll
                for (int _la = 0; _la < 16; _la++)
                    fk_1[_la] = fk_1[_la] + tk_1[_la];
            }
            int frow_4 = warp_1 * 8 + 4;
            if (frow_4 < n_valid) {
                unsigned int yw_4[4];
                yw_4[0] = yq4_1[0];
                yw_4[1] = yq4_1[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&yw_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&yw_4[(2) + 1]))
                    : "r"(s_kn_addr + (unsigned int)(lane_0 * 4 / 64 * 8192 + (frow_4 * 128 + lane_0 * 4 % 64 * 2 ^ (frow_4 * 128 + lane_0 * 4 % 64 * 2 >> 7 & 7) << 4))));
                float yq_4[4];
                float yk_4[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yq_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_4[_pair]) << 16);
                    (&yq_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_4[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yk_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_4[2 + _pair]) << 16);
                    (&yk_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_4[2 + _pair]) & 0xffff0000u);
                }
                float rq_4 = s_rs[frow_4];
                float rk_4 = s_rs[64 + frow_4];
                float dq_r_4[4];
                float dk_r_4[4];
                dq_r_4[0] = fq_1[0];
                dq_r_4[1] = fq_1[1];
                dq_r_4[2] = fq_1[2];
                dq_r_4[3] = fq_1[3];
                dk_r_4[0] = fk_1[0];
                dk_r_4[1] = fk_1[1];
                dk_r_4[2] = fk_1[2];
                dk_r_4[3] = fk_1[3];
                float inner_q_4 = 0.0f;
                float inner_k_4 = 0.0f;
                #pragma unroll
                for (int i5_4 = 0; i5_4 < 4; i5_4++) {
                    inner_q_4 = __fmaf_rn(dq_r_4[i5_4], yq_4[i5_4], inner_q_4);
                    inner_k_4 = __fmaf_rn(dk_r_4[i5_4], yk_4[i5_4], inner_k_4);
                }
                float _warp_reduce_123 = inner_q_4;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_123 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_123, offset);
                inner_q_4 = _warp_reduce_123;
                float _warp_reduce_124 = inner_k_4;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_124 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_124, offset);
                inner_k_4 = _warp_reduce_124;
                if (normalize_qk != 0) {
                    #pragma unroll
                    for (int i6_4 = 0; i6_4 < 4; i6_4++) {
                        dq_r_4[i6_4] = __fmaf_rn(dq_r_4[i6_4], rq_4, -(inner_q_4 * yq_4[i6_4] * rq_4));
                        dk_r_4[i6_4] = __fmaf_rn(dk_r_4[i6_4], rk_4, -(inner_k_4 * yk_4[i6_4] * rk_4));
                    }
                }
                long long qbase_4 = ((long long)(tok0 + frow_4) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4);
                unsigned int _packed_4_4[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_4[_lp*2 + 0], dq_r_4[_lp*2+1 + 0]));
                    _packed_4_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_4[_lp*2 + 2], dq_r_4[_lp*2+1 + 2]));
                    _packed_4_4[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dq_out + (qbase_4))[0] = reinterpret_cast<int2*>(_packed_4_4)[0];
                unsigned int _packed_5_4[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_4[_lp*2 + 0], dk_r_4[_lp*2+1 + 0]));
                    _packed_5_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_4[_lp*2 + 2], dk_r_4[_lp*2+1 + 2]));
                    _packed_5_4[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dk_out + (qbase_4))[0] = reinterpret_cast<int2*>(_packed_5_4)[0];
            }
            int frow_5 = warp_1 * 8 + 4 + 1;
            if (frow_5 < n_valid) {
                unsigned int yw_5[4];
                yw_5[0] = yq4_1[2];
                yw_5[1] = yq4_1[3];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&yw_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&yw_5[(2) + 1]))
                    : "r"(s_kn_addr + (unsigned int)(lane_0 * 4 / 64 * 8192 + (frow_5 * 128 + lane_0 * 4 % 64 * 2 ^ (frow_5 * 128 + lane_0 * 4 % 64 * 2 >> 7 & 7) << 4))));
                float yq_5[4];
                float yk_5[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yq_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_5[_pair]) << 16);
                    (&yq_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_5[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yk_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_5[2 + _pair]) << 16);
                    (&yk_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_5[2 + _pair]) & 0xffff0000u);
                }
                float rq_5 = s_rs[frow_5];
                float rk_5 = s_rs[64 + frow_5];
                float dq_r_5[4];
                float dk_r_5[4];
                dq_r_5[0] = fq_1[4];
                dq_r_5[1] = fq_1[5];
                dq_r_5[2] = fq_1[6];
                dq_r_5[3] = fq_1[7];
                dk_r_5[0] = fk_1[4];
                dk_r_5[1] = fk_1[5];
                dk_r_5[2] = fk_1[6];
                dk_r_5[3] = fk_1[7];
                float inner_q_5 = 0.0f;
                float inner_k_5 = 0.0f;
                #pragma unroll
                for (int i5_5 = 0; i5_5 < 4; i5_5++) {
                    inner_q_5 = __fmaf_rn(dq_r_5[i5_5], yq_5[i5_5], inner_q_5);
                    inner_k_5 = __fmaf_rn(dk_r_5[i5_5], yk_5[i5_5], inner_k_5);
                }
                float _warp_reduce_125 = inner_q_5;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_125 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_125, offset);
                inner_q_5 = _warp_reduce_125;
                float _warp_reduce_126 = inner_k_5;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_126 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_126, offset);
                inner_k_5 = _warp_reduce_126;
                if (normalize_qk != 0) {
                    #pragma unroll
                    for (int i6_5 = 0; i6_5 < 4; i6_5++) {
                        dq_r_5[i6_5] = __fmaf_rn(dq_r_5[i6_5], rq_5, -(inner_q_5 * yq_5[i6_5] * rq_5));
                        dk_r_5[i6_5] = __fmaf_rn(dk_r_5[i6_5], rk_5, -(inner_k_5 * yk_5[i6_5] * rk_5));
                    }
                }
                long long qbase_5 = ((long long)(tok0 + frow_5) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4);
                unsigned int _packed_4_5[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_5[_lp*2 + 0], dq_r_5[_lp*2+1 + 0]));
                    _packed_4_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_5[_lp*2 + 2], dq_r_5[_lp*2+1 + 2]));
                    _packed_4_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dq_out + (qbase_5))[0] = reinterpret_cast<int2*>(_packed_4_5)[0];
                unsigned int _packed_5_5[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_5[_lp*2 + 0], dk_r_5[_lp*2+1 + 0]));
                    _packed_5_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_5[_lp*2 + 2], dk_r_5[_lp*2+1 + 2]));
                    _packed_5_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dk_out + (qbase_5))[0] = reinterpret_cast<int2*>(_packed_5_5)[0];
            }
            int frow_6 = warp_1 * 8 + 4 + 2;
            if (frow_6 < n_valid) {
                unsigned int yw_6[4];
                yw_6[0] = yq4_1[4];
                yw_6[1] = yq4_1[5];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&yw_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&yw_6[(2) + 1]))
                    : "r"(s_kn_addr + (unsigned int)(lane_0 * 4 / 64 * 8192 + (frow_6 * 128 + lane_0 * 4 % 64 * 2 ^ (frow_6 * 128 + lane_0 * 4 % 64 * 2 >> 7 & 7) << 4))));
                float yq_6[4];
                float yk_6[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yq_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_6[_pair]) << 16);
                    (&yq_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_6[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yk_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_6[2 + _pair]) << 16);
                    (&yk_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_6[2 + _pair]) & 0xffff0000u);
                }
                float rq_6 = s_rs[frow_6];
                float rk_6 = s_rs[64 + frow_6];
                float dq_r_6[4];
                float dk_r_6[4];
                dq_r_6[0] = fq_1[8];
                dq_r_6[1] = fq_1[9];
                dq_r_6[2] = fq_1[10];
                dq_r_6[3] = fq_1[11];
                dk_r_6[0] = fk_1[8];
                dk_r_6[1] = fk_1[9];
                dk_r_6[2] = fk_1[10];
                dk_r_6[3] = fk_1[11];
                float inner_q_6 = 0.0f;
                float inner_k_6 = 0.0f;
                #pragma unroll
                for (int i5_6 = 0; i5_6 < 4; i5_6++) {
                    inner_q_6 = __fmaf_rn(dq_r_6[i5_6], yq_6[i5_6], inner_q_6);
                    inner_k_6 = __fmaf_rn(dk_r_6[i5_6], yk_6[i5_6], inner_k_6);
                }
                float _warp_reduce_127 = inner_q_6;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_127 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_127, offset);
                inner_q_6 = _warp_reduce_127;
                float _warp_reduce_128 = inner_k_6;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_128 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_128, offset);
                inner_k_6 = _warp_reduce_128;
                if (normalize_qk != 0) {
                    #pragma unroll
                    for (int i6_6 = 0; i6_6 < 4; i6_6++) {
                        dq_r_6[i6_6] = __fmaf_rn(dq_r_6[i6_6], rq_6, -(inner_q_6 * yq_6[i6_6] * rq_6));
                        dk_r_6[i6_6] = __fmaf_rn(dk_r_6[i6_6], rk_6, -(inner_k_6 * yk_6[i6_6] * rk_6));
                    }
                }
                long long qbase_6 = ((long long)(tok0 + frow_6) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4);
                unsigned int _packed_4_6[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_6[_lp*2 + 0], dq_r_6[_lp*2+1 + 0]));
                    _packed_4_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_6[_lp*2 + 2], dq_r_6[_lp*2+1 + 2]));
                    _packed_4_6[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dq_out + (qbase_6))[0] = reinterpret_cast<int2*>(_packed_4_6)[0];
                unsigned int _packed_5_6[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_6[_lp*2 + 0], dk_r_6[_lp*2+1 + 0]));
                    _packed_5_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_6[_lp*2 + 2], dk_r_6[_lp*2+1 + 2]));
                    _packed_5_6[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dk_out + (qbase_6))[0] = reinterpret_cast<int2*>(_packed_5_6)[0];
            }
            int frow_7 = warp_1 * 8 + 4 + 3;
            if (frow_7 < n_valid) {
                unsigned int yw_7[4];
                yw_7[0] = yq4_1[6];
                yw_7[1] = yq4_1[7];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&yw_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&yw_7[(2) + 1]))
                    : "r"(s_kn_addr + (unsigned int)(lane_0 * 4 / 64 * 8192 + (frow_7 * 128 + lane_0 * 4 % 64 * 2 ^ (frow_7 * 128 + lane_0 * 4 % 64 * 2 >> 7 & 7) << 4))));
                float yq_7[4];
                float yk_7[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yq_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_7[_pair]) << 16);
                    (&yq_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_7[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yk_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(yw_7[2 + _pair]) << 16);
                    (&yk_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(yw_7[2 + _pair]) & 0xffff0000u);
                }
                float rq_7 = s_rs[frow_7];
                float rk_7 = s_rs[64 + frow_7];
                float dq_r_7[4];
                float dk_r_7[4];
                dq_r_7[0] = fq_1[12];
                dq_r_7[1] = fq_1[13];
                dq_r_7[2] = fq_1[14];
                dq_r_7[3] = fq_1[15];
                dk_r_7[0] = fk_1[12];
                dk_r_7[1] = fk_1[13];
                dk_r_7[2] = fk_1[14];
                dk_r_7[3] = fk_1[15];
                float inner_q_7 = 0.0f;
                float inner_k_7 = 0.0f;
                #pragma unroll
                for (int i5_7 = 0; i5_7 < 4; i5_7++) {
                    inner_q_7 = __fmaf_rn(dq_r_7[i5_7], yq_7[i5_7], inner_q_7);
                    inner_k_7 = __fmaf_rn(dk_r_7[i5_7], yk_7[i5_7], inner_k_7);
                }
                float _warp_reduce_129 = inner_q_7;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_129 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_129, offset);
                inner_q_7 = _warp_reduce_129;
                float _warp_reduce_130 = inner_k_7;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_130 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_130, offset);
                inner_k_7 = _warp_reduce_130;
                if (normalize_qk != 0) {
                    #pragma unroll
                    for (int i6_7 = 0; i6_7 < 4; i6_7++) {
                        dq_r_7[i6_7] = __fmaf_rn(dq_r_7[i6_7], rq_7, -(inner_q_7 * yq_7[i6_7] * rq_7));
                        dk_r_7[i6_7] = __fmaf_rn(dk_r_7[i6_7], rk_7, -(inner_k_7 * yk_7[i6_7] * rk_7));
                    }
                }
                long long qbase_7 = ((long long)(tok0 + frow_7) * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4);
                unsigned int _packed_4_7[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_7[_lp*2 + 0], dq_r_7[_lp*2+1 + 0]));
                    _packed_4_7[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_r_7[_lp*2 + 2], dq_r_7[_lp*2+1 + 2]));
                    _packed_4_7[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dq_out + (qbase_7))[0] = reinterpret_cast<int2*>(_packed_4_7)[0];
                unsigned int _packed_5_7[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_7[_lp*2 + 0], dk_r_7[_lp*2+1 + 0]));
                    _packed_5_7[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_r_7[_lp*2 + 2], dk_r_7[_lp*2+1 + 2]));
                    _packed_5_7[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dk_out + (qbase_7))[0] = reinterpret_cast<int2*>(_packed_5_7)[0];
            }
        }
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (warp_1 == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
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
    __syncthreads();
}

} // extern "C"
