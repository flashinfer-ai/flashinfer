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
#define TMEM_NCOLS 448
#define TMEM_ACC_DQ_OFFSET 0
#define TMEM_ACC_DK_OFFSET 128
#define TMEM_ACC_DW_OFFSET 256
#define TMEM_ACC_DA_OFFSET 384
#define TMEM_ACC_DVB_OFFSET 0
#define TMEM_ACC_DKGB_OFFSET 128
#define TMEM_ACC_DA1_OFFSET 256
#define TMEM_ACC_DA2_OFFSET 320
#define NUM_ONE_STAGES 1
#define SMEM_SMEM_V0_OFF 1024
#define SMEM_SMEM_V0_STAGE_BYTES 16384
#define SMEM_SMEM_V0_STRIDE 57344
#define SMEM_SMEM_V1_OFF 17408
#define SMEM_SMEM_V1_STAGE_BYTES 16384
#define SMEM_SMEM_V1_STRIDE 57344
#define SMEM_SMEM_V2_OFF 33792
#define SMEM_SMEM_V2_STAGE_BYTES 16384
#define SMEM_SMEM_V2_STRIDE 57344
#define SMEM_SMEM_V3_OFF 1024
#define SMEM_SMEM_V3_STAGE_BYTES 16384
#define SMEM_SMEM_V3_STRIDE 57344
#define SMEM_SMEM_V4_OFF 17408
#define SMEM_SMEM_V4_STAGE_BYTES 16384
#define SMEM_SMEM_V4_STRIDE 57344
#define SMEM_SMEM_V5_OFF 33792
#define SMEM_SMEM_V5_STAGE_BYTES 16384
#define SMEM_SMEM_V5_STRIDE 57344
#define SMEM_DV2_MN_OFF 33792
#define SMEM_DV2_MN_STAGE_BYTES 16384
#define SMEM_DV2_MN_STRIDE 57344
#define SMEM_AKK_K_OFF 50176
#define SMEM_AKK_K_STAGE_BYTES 8192
#define SMEM_AKK_K_STRIDE 57344
#define SMEM_AKK_MN_OFF 50176
#define SMEM_AKK_MN_STAGE_BYTES 8192
#define SMEM_AKK_MN_STRIDE 57344
#define SMEM_DAB_K_OFF 58368
#define SMEM_DAB_K_STAGE_BYTES 8192
#define SMEM_DAB_K_STRIDE 8192
#define SMEM_DA1B_MN_OFF 66560
#define SMEM_DA1B_MN_STAGE_BYTES 8192
#define SMEM_DA1B_MN_STRIDE 8192
#define SMEM_DA1B_W_OFF 66560
#define SMEM_DA1B_W_STAGE_BYTES 8192
#define SMEM_DA1B_W_STRIDE 8192
#define SMEM_RING_F32_OFF 1024
#define SMEM_RING_F32_STAGE_BYTES 57344
#define SMEM_RING_F32_STRIDE 57344
#define SMEM_V_P_OFF 58368
#define SMEM_V_P_STAGE_BYTES 8192
#define SMEM_V_P_STRIDE 8192
#define SMEM_V_K_OFF 58368
#define SMEM_V_K_STAGE_BYTES 16384
#define SMEM_V_K_STRIDE 16384
#define SMEM_H_P_OFF 74752
#define SMEM_H_P_STAGE_BYTES 16384
#define SMEM_H_P_STRIDE 16384
#define SMEM_DH_P_OFF 107520
#define SMEM_DH_P_STAGE_BYTES 16384
#define SMEM_DH_P_STRIDE 16384
#define SMEM_H_K_OFF 74752
#define SMEM_H_K_STAGE_BYTES 32768
#define SMEM_H_K_STRIDE 32768
#define SMEM_DH_K_OFF 107520
#define SMEM_DH_K_STAGE_BYTES 32768
#define SMEM_DH_K_STRIDE 32768
#define SMEM_DWB_K_OFF 140288
#define SMEM_DWB_K_STAGE_BYTES 16384
#define SMEM_DWB_K_STRIDE 16384
#define SMEM_DWB_MN_OFF 140288
#define SMEM_DWB_MN_STAGE_BYTES 16384
#define SMEM_DWB_MN_STRIDE 16384
#define SMEM_KGB_K_OFF 156672
#define SMEM_KGB_K_STAGE_BYTES 16384
#define SMEM_KGB_K_STRIDE 16384
#define SMEM_GK_P_OFF 173056
#define SMEM_GK_P_STAGE_BYTES 8192
#define SMEM_GK_P_STRIDE 8192
#define SMEM_GK_S_OFF 173056
#define SMEM_GK_S_STAGE_BYTES 32768
#define SMEM_GK_S_STRIDE 32768
#define SMEM_K_P_OFF 205824
#define SMEM_K_P_STAGE_BYTES 8192
#define SMEM_K_P_STRIDE 8192
#define SMEM_K_S_OFF 205824
#define SMEM_K_S_STAGE_BYTES 16384
#define SMEM_K_S_STRIDE 16384
#define SMEM_Q_P_OFF 74752
#define SMEM_Q_P_STAGE_BYTES 8192
#define SMEM_Q_P_STRIDE 8192
#define SMEM_Q_S_OFF 74752
#define SMEM_Q_S_STAGE_BYTES 16384
#define SMEM_Q_S_STRIDE 16384
#define SMEM_BETA_S_OFF 223744
#define SMEM_BETA_S_STAGE_BYTES 512
#define SMEM_BETA_S_STRIDE 512
#define SMEM_REC_S_OFF 224256
#define SMEM_REC_S_STAGE_BYTES 16
#define SMEM_REC_S_STRIDE 16
#define SMEM_DGK_S_OFF 222208
#define SMEM_DGK_S_STAGE_BYTES 512
#define SMEM_DGK_S_STRIDE 512
#define SMEM_DBP_S_OFF 222720
#define SMEM_DBP_S_STAGE_BYTES 1024
#define SMEM_DBP_S_STRIDE 1024
#define SMEM_TOTAL 224384
#define THREADS 576
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



__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}



// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




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

__global__ __launch_bounds__(576, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_6df24a76cec0e85a8c75(const __grid_constant__ CUtensorMap do_tma, const __grid_constant__ CUtensorMap vn_tma, const __grid_constant__ CUtensorMap dv2_tma, const __grid_constant__ CUtensorMap v_tma, const __grid_constant__ CUtensorMap h_tma, const __grid_constant__ CUtensorMap dh_tma, const __grid_constant__ CUtensorMap akk_tma, const __grid_constant__ CUtensorMap gk_tma, const __grid_constant__ CUtensorMap k_tma, const __grid_constant__ CUtensorMap q_tma, float* __restrict__ beta, float* __restrict__ dq_out, float* __restrict__ dk_out, float* __restrict__ dg_out, float* __restrict__ db_out, float* __restrict__ dAkk_out, __nv_bfloat16* __restrict__ dv_out, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float scale, float* __restrict__ dAqk_out)
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
    #define dovn_full_addr (mbar_base + 0)
    #define dv2_full_addr (mbar_base + 8)
    #define h_full_addr (mbar_base + 16)
    #define dh_full_addr (mbar_base + 24)
    #define ops_full_addr (mbar_base + 32)
    #define akkv_full_addr (mbar_base + 40)
    #define q_full_addr (mbar_base + 48)
    #define a_committed_addr (mbar_base + 56)
    #define hd_read_done_addr (mbar_base + 64)
    #define dv2_free_addr (mbar_base + 72)
    #define hd_empty_addr (mbar_base + 80)
    #define d_committed_addr (mbar_base + 88)
    #define a_ready_addr (mbar_base + 96)
    #define a_drained_addr (mbar_base + 104)
    #define b_ready_addr (mbar_base + 112)
    #define b_drained_addr (mbar_base + 120)
    #define c_ready_addr (mbar_base + 128)
    #define c_drained_addr (mbar_base + 136)
    #define d_ready_addr (mbar_base + 144)
    #define d_drained_addr (mbar_base + 152)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_v0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v0_addr = smem + 1024;
    __nv_bfloat16* smem_v1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_v1_addr = smem + 17408;
    __nv_bfloat16* smem_v2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_v2_addr = smem + 33792;
    __nv_bfloat16* smem_v3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v3_addr = smem + 1024;
    __nv_bfloat16* smem_v4 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_v4_addr = smem + 17408;
    __nv_bfloat16* smem_v5 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_v5_addr = smem + 33792;
    __nv_bfloat16* dv2_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int dv2_mn_addr = smem + 33792;
    __nv_bfloat16* akk_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int akk_k_addr = smem + 50176;
    __nv_bfloat16* akk_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int akk_mn_addr = smem + 50176;
    __nv_bfloat16* dab_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 58368);
    const int dab_k_addr = smem + 58368;
    __nv_bfloat16* da1b_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int da1b_mn_addr = smem + 66560;
    __nv_bfloat16* da1b_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int da1b_w_addr = smem + 66560;
    float* ring_f32 = reinterpret_cast<float*>(smem_raw + 1024);
    const int ring_f32_addr = smem + 1024;
    __nv_bfloat16* v_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 58368);
    const int v_p_addr = smem + 58368;
    __nv_bfloat16* v_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 58368);
    const int v_k_addr = smem + 58368;
    __nv_bfloat16* h_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int h_p_addr = smem + 74752;
    __nv_bfloat16* dh_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 107520);
    const int dh_p_addr = smem + 107520;
    __nv_bfloat16* h_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int h_k_addr = smem + 74752;
    __nv_bfloat16* dh_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 107520);
    const int dh_k_addr = smem + 107520;
    __nv_bfloat16* dwb_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int dwb_k_addr = smem + 140288;
    __nv_bfloat16* dwb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int dwb_mn_addr = smem + 140288;
    __nv_bfloat16* kgb_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 156672);
    const int kgb_k_addr = smem + 156672;
    float* gk_p = reinterpret_cast<float*>(smem_raw + 173056);
    const int gk_p_addr = smem + 173056;
    float* gk_s = reinterpret_cast<float*>(smem_raw + 173056);
    const int gk_s_addr = smem + 173056;
    __nv_bfloat16* k_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int k_p_addr = smem + 205824;
    __nv_bfloat16* k_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int k_s_addr = smem + 205824;
    __nv_bfloat16* q_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int q_p_addr = smem + 74752;
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int q_s_addr = smem + 74752;
    float* beta_s = reinterpret_cast<float*>(smem_raw + 223744);
    const int beta_s_addr = smem + 223744;
    int* rec_s = reinterpret_cast<int*>(smem_raw + 224256);
    const int rec_s_addr = smem + 224256;
    float* dgk_s = reinterpret_cast<float*>(smem_raw + 222208);
    const int dgk_s_addr = smem + 222208;
    float* dbp_s = reinterpret_cast<float*>(smem_raw + 222720);
    const int dbp_s_addr = smem + 222720;

    // Mbarrier init (20 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'one' ---
            // dovn_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // dv2_full: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // h_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // dh_full: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // ops_full: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // akkv_full: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // a_committed: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // hd_read_done: 1 barriers, init_count=512
            mbarrier_init(smem + 64, 512);
            // dv2_free: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // hd_empty: 1 barriers, init_count=512
            mbarrier_init(smem + 80, 512);
            // d_committed: 1 barriers, init_count=513
            mbarrier_init(smem + 88, 513);
            // a_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // a_drained: 1 barriers, init_count=512
            mbarrier_init(smem + 104, 512);
            // b_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // b_drained: 1 barriers, init_count=512
            mbarrier_init(smem + 120, 512);
            // c_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // c_drained: 1 barriers, init_count=512
            mbarrier_init(smem + 136, 512);
            // d_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // d_drained: 1 barriers, init_count=512
            mbarrier_init(smem + 152, 512);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 448 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_dq = taddr;
    const int tmem_acc_dk = taddr + 128;
    const int tmem_acc_dw = taddr + 256;
    const int tmem_acc_da = taddr + 384;
    const int tmem_acc_dvb = taddr;
    const int tmem_acc_dkgb = taddr + 128;
    const int tmem_acc_da1 = taddr + 256;
    const int tmem_acc_da2 = taddr + 320;

    // ---- Role: compute ----
    if (warp <= 15) {
        { // compute_main
            int wq = warp % 4;
            int cblk = warp / 4;
            int lane_half = lane >> 4;
            int t_row = wq * 16 + (lane & 15);
            int row_base = wq * 32 << 16;
            int tid_1 = warp * 32 + lane;
            int lane_c = lane;
            int it_i = 0;
            int nc_c = seq_chunk_start[num_seqs];
            int num_items_c = nc_c * num_heads;
            unsigned int first_c = (unsigned int)bid;
            if (first_c < (unsigned int)num_items_c) {
                int chunk = first_c / (unsigned int)num_heads;
                int head = first_c % (unsigned int)num_heads;
                int row0 = chunk * 64;
                int hrow0 = (chunk * num_heads + head) * 128;
                if (tid_1 < 64) {
                    beta_s[tid_1] = beta[(row0 + tid_1) * num_heads + head];
                }
                if (tid_1 == 64) {
                    rec_s[0] = chunk_bos[chunk];
                }
                if (tid_1 == 65) {
                    rec_s[1] = chunk_len[chunk];
                }
            }
            asm volatile("barrier.sync 1, 512;" ::: "memory");
            unsigned int _phase_h_full_0 = 0;
            unsigned int _phase_dh_full_0 = 0;
            unsigned int _phase_dovn_full_0 = 0;
            unsigned int _phase_a_ready_0 = 0;
            unsigned int _phase_ops_full_0 = 0;
            unsigned int _phase_akkv_full_0 = 0;
            unsigned int _phase_b_ready_0 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_c_ready_0 = 0;
            unsigned int _phase_d_ready_0 = 0;
            #pragma unroll 1
            for (unsigned int it = bid; it < num_items_c; it += num_bids) {
                int chunk_1 = it / (unsigned int)num_heads;
                int head_1 = it % (unsigned int)num_heads;
                int row0_1 = chunk_1 * 64;
                int hrow0_1 = (chunk_1 * num_heads + head_1) * 128;
                int grow = row0_1 + t_row;
                long long ibase = (long long)(row0_1 * num_heads + head_1) * 128;
                int roff = t_row * num_heads * 128;
                float* dq_item = dq_out + ibase;
                float* dk_item = dk_out + ibase;
                float* dg_item = dg_out + ibase;
                long long gn_base = ibase + (long long)(63 * num_heads * 128);
                int bbase = (it_i & 1) * 64;
                float beta_t = beta_s[t_row + bbase];
                int cbos = rec_s[(it_i & 1) * 2];
                __nv_bfloat16* dv_item = dv_out + ((long long)(cbos * num_heads + head_1) * 128);
                int clen = rec_s[(it_i & 1) * 2 + 1];
                float dgk_inter = 0.0f;
                mbarrier_wait(h_full_addr, _phase_h_full_0);
                _phase_h_full_0 ^= 1;
                mbarrier_wait(dh_full_addr, _phase_dh_full_0);
                _phase_dh_full_0 ^= 1;
                if (tid_1 < 128) {
                    #pragma unroll
                    for (int seg = 0; seg < 8; seg++) {
                        unsigned int hraw[8];
                        unsigned int dhraw[8];
                        #pragma unroll
                        for (int q = 0; q < 2; q++) {
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&hraw[q * 4])), "=r"(*reinterpret_cast<uint32_t*>(&hraw[(q * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&hraw[(q * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&hraw[(q * 4) + 3]))
                                : "r"((h_k_addr + (unsigned int)((seg * 16 + q * 8) / 64 * 16384 + tid_1 * 128 + (seg * 16 + q * 8) % 64 * 2 ^ ((seg * 16 + q * 8) / 64 * 16384 + tid_1 * 128 + (seg * 16 + q * 8) % 64 * 2 >> 7 & 7) << 4))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&dhraw[q * 4])), "=r"(*reinterpret_cast<uint32_t*>(&dhraw[(q * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dhraw[(q * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dhraw[(q * 4) + 3]))
                                : "r"((dh_k_addr + (unsigned int)((seg * 16 + q * 8) / 64 * 16384 + tid_1 * 128 + (seg * 16 + q * 8) % 64 * 2 ^ ((seg * 16 + q * 8) / 64 * 16384 + tid_1 * 128 + (seg * 16 + q * 8) % 64 * 2 >> 7 & 7) << 4))));
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
                        #pragma unroll
                        for (int j = 0; j < 16; j++) {
                            dgk_inter = dgk_inter + (float)(__nv_bfloat16)(_unpack_bf16_0[j] * _unpack_bf16_1[j]);
                        }
                    }
                }
                mbarrier_arrive(hd_read_done_addr);
                mbarrier_wait(dovn_full_addr, _phase_dovn_full_0);
                _phase_dovn_full_0 ^= 1;
                mbarrier_wait(a_ready_addr, _phase_a_ready_0);
                _phase_a_ready_0 ^= 1;
                mbarrier_wait(ops_full_addr, _phase_ops_full_0);
                _phase_ops_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (clen < 64) {
                    float zq[4];
                    zq[0] = 0.0f;
                    zq[1] = 0.0f;
                    zq[2] = 0.0f;
                    zq[3] = 0.0f;
                    int row = wq * 16 + (lane_c & 15);
                    #pragma unroll
                    for (int i = 0; i < 2; i++) {
                        int qc = (lane_c >> 4) * 2 + i;
                        if (row >= clen) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((k_s_addr + (unsigned int)((cblk * 32 + qc * 8) / 64 * 8192 + row * 128 + (cblk * 32 + qc * 8) % 64 * 2 ^ ((cblk * 32 + qc * 8) / 64 * 8192 + row * 128 + (cblk * 32 + qc * 8) % 64 * 2 >> 7 & 7) << 4))), "f"(zq[0]), "f"(zq[1]), "f"(zq[2]), "f"(zq[3]) : "memory");
                        }
                    }
                }
                float dqc[16];
                #pragma unroll
                for (int _one_a = 0; _one_a < 1; _one_a++) {
                    int blk = cblk;
                    int col = blk * 32 + lane_half * 16;
                    float _tmem_load_0[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                        : "r"(taddr + (unsigned int)row_base + (unsigned int)(blk * 32)));
                    float _tmem_load_1[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                        : "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)(blk * 32)));
                    float _tmem_load_2[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                        : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(blk * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float gkv[16];
                    float gnv[16];
                    unsigned int kraw[8];
                    #pragma unroll
                    for (int q_1 = 0; q_1 < 4; q_1++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gkv[q_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&gkv[(q_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gkv[(q_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gkv[(q_1 * 4) + 3]))
                            : "r"((gk_s_addr + (unsigned int)((col + q_1 * 4) / 32 * 8192 + t_row * 128 + (col + q_1 * 4) % 32 * 4 ^ ((col + q_1 * 4) / 32 * 8192 + t_row * 128 + (col + q_1 * 4) % 32 * 4 >> 7 & 7) << 4))));
                    }
                    #pragma unroll
                    for (int q_2 = 0; q_2 < 4; q_2++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gnv[q_2 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&gnv[(q_2 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gnv[(q_2 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gnv[(q_2 * 4) + 3]))
                            : "r"((gk_s_addr + (unsigned int)((col + q_2 * 4) / 32 * 8192 + 8064 + (col + q_2 * 4) % 32 * 4 ^ ((col + q_2 * 4) / 32 * 8192 + 8064 + (col + q_2 * 4) % 32 * 4 >> 7 & 7) << 4))));
                    }
                    #pragma unroll
                    for (int q_3 = 0; q_3 < 2; q_3++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&kraw[q_3 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&kraw[(q_3 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw[(q_3 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw[(q_3 * 4) + 3]))
                            : "r"((k_s_addr + (unsigned int)((col + q_3 * 8) / 64 * 8192 + t_row * 128 + (col + q_3 * 8) % 64 * 2 ^ ((col + q_3 * 8) / 64 * 8192 + t_row * 128 + (col + q_3 * 8) % 64 * 2 >> 7 & 7) << 4))));
                    }
                    float _unpack_bf16_2[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kraw[_pair]) << 16);
                        (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kraw[_pair]) & 0xffff0000u);
                    }
                    float dkv[16];
                    float dwbv[16];
                    float kgv[16];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 16; j_1++) {
                        float e_g = approx_exp2(gkv[j_1]);
                        dqc[j_1] = _tmem_load_0[j_1] * e_g * scale;
                        float _exp2_0 = approx_exp2(gnv[j_1] - gkv[j_1]);
                        dkv[j_1] = _tmem_load_1[j_1] * _exp2_0;
                        dwbv[j_1] = -(float)(__nv_bfloat16)_tmem_load_2[j_1];
                        kgv[j_1] = _unpack_bf16_2[j_1] * e_g;
                    }
                    float dqv_lo[8];
                    float dqv_hi[8];
                    #pragma unroll
                    for (int jj = 0; jj < 8; jj++) {
                        dqv_lo[jj] = dqc[jj];
                        dqv_hi[jj] = dqc[jj + 8];
                    }
                    {
                        unsigned _stv8_1_0 = __float_as_uint(dqv_lo[0 + 0]);
                        unsigned _stv8_1_1 = __float_as_uint(dqv_lo[0 + 1]);
                        unsigned _stv8_1_2 = __float_as_uint(dqv_lo[0 + 2]);
                        unsigned _stv8_1_3 = __float_as_uint(dqv_lo[0 + 3]);
                        unsigned _stv8_1_4 = __float_as_uint(dqv_lo[0 + 4]);
                        unsigned _stv8_1_5 = __float_as_uint(dqv_lo[0 + 5]);
                        unsigned _stv8_1_6 = __float_as_uint(dqv_lo[0 + 6]);
                        unsigned _stv8_1_7 = __float_as_uint(dqv_lo[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dq_item + (roff + col) + (0))), "r"(_stv8_1_0), "r"(_stv8_1_1), "r"(_stv8_1_2), "r"(_stv8_1_3), "r"(_stv8_1_4), "r"(_stv8_1_5), "r"(_stv8_1_6), "r"(_stv8_1_7) : "memory");
                    }
                    {
                        unsigned _stv8_2_0 = __float_as_uint(dqv_hi[0 + 0]);
                        unsigned _stv8_2_1 = __float_as_uint(dqv_hi[0 + 1]);
                        unsigned _stv8_2_2 = __float_as_uint(dqv_hi[0 + 2]);
                        unsigned _stv8_2_3 = __float_as_uint(dqv_hi[0 + 3]);
                        unsigned _stv8_2_4 = __float_as_uint(dqv_hi[0 + 4]);
                        unsigned _stv8_2_5 = __float_as_uint(dqv_hi[0 + 5]);
                        unsigned _stv8_2_6 = __float_as_uint(dqv_hi[0 + 6]);
                        unsigned _stv8_2_7 = __float_as_uint(dqv_hi[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dq_item + (roff + col + 8) + (0))), "r"(_stv8_2_0), "r"(_stv8_2_1), "r"(_stv8_2_2), "r"(_stv8_2_3), "r"(_stv8_2_4), "r"(_stv8_2_5), "r"(_stv8_2_6), "r"(_stv8_2_7) : "memory");
                    }
                    int kchunk = col >> 2;
                    #pragma unroll
                    for (int q_4 = 0; q_4 < 4; q_4++) {
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ring_f32_addr + (unsigned int)(t_row * 128 * 4) + (unsigned int)((kchunk + q_4 ^ t_row & 7) << 4)), "f"(dkv[q_4 * 4]), "f"(dkv[q_4 * 4 + 1]), "f"(dkv[q_4 * 4 + 2]), "f"(dkv[q_4 * 4 + 3]) : "memory");
                    }
                    unsigned int dwbv_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dwbv[_lp*2 + 0], dwbv[_lp*2+1 + 0]));
                        dwbv_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int kgv_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kgv[_lp*2 + 0], kgv[_lp*2+1 + 0]));
                        kgv_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_5 = 0; q_5 < 2; q_5++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dwb_k_addr + (unsigned int)((col + q_5 * 8) / 64 * 8192 + t_row * 128 + (col + q_5 * 8) % 64 * 2 ^ ((col + q_5 * 8) / 64 * 8192 + t_row * 128 + (col + q_5 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(dwbv_bf16[q_5 * 4]), "r"(dwbv_bf16[q_5 * 4 + 1]), "r"(dwbv_bf16[q_5 * 4 + 2]), "r"(dwbv_bf16[q_5 * 4 + 3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((kgb_k_addr + (unsigned int)((col + q_5 * 8) / 64 * 8192 + t_row * 128 + (col + q_5 * 8) % 64 * 2 ^ ((col + q_5 * 8) / 64 * 8192 + t_row * 128 + (col + q_5 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(kgv_bf16[q_5 * 4]), "r"(kgv_bf16[q_5 * 4 + 1]), "r"(kgv_bf16[q_5 * 4 + 2]), "r"(kgv_bf16[q_5 * 4 + 3]) : "memory");
                    }
                }
                mbarrier_wait(akkv_full_addr, _phase_akkv_full_0);
                _phase_akkv_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (clen < 64) {
                    float zq_1[4];
                    zq_1[0] = 0.0f;
                    zq_1[1] = 0.0f;
                    zq_1[2] = 0.0f;
                    zq_1[3] = 0.0f;
                    int row_1 = wq * 16 + (lane_c & 15);
                    #pragma unroll
                    for (int i_1 = 0; i_1 < 2; i_1++) {
                        int qc_1 = (lane_c >> 4) * 2 + i_1;
                        if (row_1 >= clen) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((v_k_addr + (unsigned int)((cblk * 32 + qc_1 * 8) / 64 * 8192 + row_1 * 128 + (cblk * 32 + qc_1 * 8) % 64 * 2 ^ ((cblk * 32 + qc_1 * 8) / 64 * 8192 + row_1 * 128 + (cblk * 32 + qc_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "f"(zq_1[0]), "f"(zq_1[1]), "f"(zq_1[2]), "f"(zq_1[3]) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(a_drained_addr);
                asm volatile("barrier.sync 1, 512;" ::: "memory");
                if (tid_1 < 128) {
                    float colsum = 0.0f;
                    unsigned int kshift = (tid_1 & 1 ^ 1) * 16;
                    #pragma unroll 16
                    for (int t = 0; t < 64; t += 2) {
                        unsigned int kw0[1];
                        unsigned int kw1[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw0[0])) : "r"((k_s_addr + (unsigned int)((tid_1 & -2) / 64 * 8192 + t * 128 + (tid_1 & -2) % 64 * 2 ^ ((tid_1 & -2) / 64 * 8192 + t * 128 + (tid_1 & -2) % 64 * 2 >> 7 & 7) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw1[0])) : "r"((k_s_addr + (unsigned int)((tid_1 & -2) / 64 * 8192 + (t + 1) * 128 + (tid_1 & -2) % 64 * 2 ^ ((tid_1 & -2) / 64 * 8192 + (t + 1) * 128 + (tid_1 & -2) % 64 * 2 >> 7 & 7) << 4))));
                        unsigned int kb0 = kw0[0] << kshift & 4294901760u;
                        unsigned int kb1 = kw1[0] << kshift & 4294901760u;
                        float dk0 = ring_f32[t * 128 + (((tid_1 >> 2 ^ t & 7) << 2) + (tid_1 & 3))];
                        float dk1 = ring_f32[(t + 1) * 128 + (((tid_1 >> 2 ^ t + 1 & 7) << 2) + (tid_1 & 3))];
                        float2 _f32x2_mul_a_3 = make_float2(__uint_as_float(kb0), __uint_as_float(kb1));
                        float2 _f32x2_mul_b_4 = make_float2(dk0, dk1);
                        float2 kdk2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&kdk2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_4));
                        colsum = colsum + kdk2.x;
                        colsum = colsum + kdk2.y;
                    }
                    float gn1[1];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&gn1[0])) : "r"((gk_s_addr + (unsigned int)(tid_1 / 32 * 8192 + 8064 + tid_1 % 32 * 4 ^ (tid_1 / 32 * 8192 + 8064 + tid_1 % 32 * 4 >> 7 & 7) << 4))));
                    float _exp2_5 = approx_exp2(gn1[0]);
                    dgk_inter = dgk_inter * _exp2_5;
                    dgk_s[tid_1] = dgk_inter + colsum;
                }
                asm volatile("barrier.sync 1, 512;" ::: "memory");
                unsigned int nxt_c = it + (unsigned int)num_bids;
                float nbv = 0.0f;
                int nbos = 0;
                int nlen = 0;
                if (nxt_c < (unsigned int)num_items_c) {
                    int chunk_0 = nxt_c / (unsigned int)num_heads;
                    int head_1_1 = nxt_c % (unsigned int)num_heads;
                    int row0_2 = chunk_0 * 64;
                    int hrow0_3 = (chunk_0 * num_heads + head_1_1) * 128;
                    if (tid_1 < 64) {
                        nbv = beta[(row0_2 + tid_1) * num_heads + head_1_1];
                    }
                    if (tid_1 == 64) {
                        nbos = chunk_bos[chunk_0];
                        nlen = chunk_len[chunk_0];
                    }
                }
                mbarrier_wait(b_ready_addr, _phase_b_ready_0);
                _phase_b_ready_0 ^= 1;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (clen < 64) {
                    float zq_2[4];
                    zq_2[0] = 0.0f;
                    zq_2[1] = 0.0f;
                    zq_2[2] = 0.0f;
                    zq_2[3] = 0.0f;
                    int row_2 = wq * 16 + (lane_c & 15);
                    #pragma unroll
                    for (int i_2 = 0; i_2 < 2; i_2++) {
                        int qc_2 = (lane_c >> 4) * 2 + i_2;
                        if (row_2 >= clen) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((q_s_addr + (unsigned int)((cblk * 32 + qc_2 * 8) / 64 * 8192 + row_2 * 128 + (cblk * 32 + qc_2 * 8) % 64 * 2 ^ ((cblk * 32 + qc_2 * 8) / 64 * 8192 + row_2 * 128 + (cblk * 32 + qc_2 * 8) % 64 * 2 >> 7 & 7) << 4))), "f"(zq_2[0]), "f"(zq_2[1]), "f"(zq_2[2]), "f"(zq_2[3]) : "memory");
                        }
                    }
                }
                float db_acc = 0.0f;
                #pragma unroll
                for (int _one_b = 0; _one_b < 1; _one_b++) {
                    int blk_1 = cblk;
                    int col_1 = blk_1 * 32 + lane_half * 16;
                    float _tmem_load_3[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                        : "r"(taddr + (unsigned int)row_base + (unsigned int)(blk_1 * 32)));
                    float _tmem_load_4[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                        : "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)(blk_1 * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float gkv2[16];
                    unsigned int kraw2[8];
                    float dk2[16];
                    #pragma unroll
                    for (int q_6 = 0; q_6 < 4; q_6++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gkv2[q_6 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&gkv2[(q_6 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gkv2[(q_6 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gkv2[(q_6 * 4) + 3]))
                            : "r"((gk_s_addr + (unsigned int)((col_1 + q_6 * 4) / 32 * 8192 + t_row * 128 + (col_1 + q_6 * 4) % 32 * 4 ^ ((col_1 + q_6 * 4) / 32 * 8192 + t_row * 128 + (col_1 + q_6 * 4) % 32 * 4 >> 7 & 7) << 4))));
                    }
                    #pragma unroll
                    for (int q_7 = 0; q_7 < 2; q_7++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&kraw2[q_7 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&kraw2[(q_7 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw2[(q_7 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw2[(q_7 * 4) + 3]))
                            : "r"((k_s_addr + (unsigned int)((col_1 + q_7 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_7 * 8) % 64 * 2 ^ ((col_1 + q_7 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_7 * 8) % 64 * 2 >> 7 & 7) << 4))));
                    }
                    float _unpack_bf16_3[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kraw2[_pair]) << 16);
                        (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kraw2[_pair]) & 0xffff0000u);
                    }
                    unsigned int qraw2[8];
                    unsigned int vraw2[8];
                    #pragma unroll
                    for (int q_8 = 0; q_8 < 2; q_8++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qraw2[q_8 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&qraw2[(q_8 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qraw2[(q_8 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qraw2[(q_8 * 4) + 3]))
                            : "r"((q_s_addr + (unsigned int)((col_1 + q_8 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_8 * 8) % 64 * 2 ^ ((col_1 + q_8 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_8 * 8) % 64 * 2 >> 7 & 7) << 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&vraw2[q_8 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&vraw2[(q_8 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vraw2[(q_8 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vraw2[(q_8 * 4) + 3]))
                            : "r"((v_k_addr + (unsigned int)((col_1 + q_8 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_8 * 8) % 64 * 2 ^ ((col_1 + q_8 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_8 * 8) % 64 * 2 >> 7 & 7) << 4))));
                    }
                    float _unpack_bf16_4[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qraw2[_pair]) << 16);
                        (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qraw2[_pair]) & 0xffff0000u);
                    }
                    float _unpack_bf16_5[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(vraw2[_pair]) << 16);
                        (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(vraw2[_pair]) & 0xffff0000u);
                    }
                    int kchunk2 = col_1 >> 2;
                    #pragma unroll
                    for (int q_9 = 0; q_9 < 4; q_9++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&dk2[q_9 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&dk2[(q_9 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dk2[(q_9 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dk2[(q_9 * 4) + 3]))
                            : "r"(ring_f32_addr + (unsigned int)(t_row * 128 * 4) + (unsigned int)((kchunk2 + q_9 ^ t_row & 7) << 4)));
                    }
                    float dvo[16];
                    float dgv[16];
                    float dkf[16];
                    float e_g2[16];
                    float kgf2[2];
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 16; j_2 += 2) {
                        float _exp2_6 = approx_exp2(gkv2[j_2]);
                        e_g2[j_2] = _exp2_6;
                        float _exp2_7 = approx_exp2(gkv2[j_2 + 1]);
                        e_g2[j_2 + 1] = _exp2_7;
                        kgf2[0] = _unpack_bf16_3[j_2] * e_g2[j_2];
                        kgf2[1] = _unpack_bf16_3[j_2 + 1] * e_g2[j_2 + 1];
                        dvo[j_2] = _tmem_load_3[j_2] * beta_t;
                        dvo[j_2 + 1] = _tmem_load_3[j_2 + 1] * beta_t;
                        #pragma unroll
                        for (int jj_1 = 0; jj_1 < 2; jj_1++) {
                            float kgf = kgf2[jj_1];
                            db_acc = __fmaf_rn(_tmem_load_3[j_2 + jj_1], _unpack_bf16_5[j_2 + jj_1], db_acc);
                            db_acc = __fmaf_rn(_tmem_load_4[j_2 + jj_1], kgf, db_acc);
                            float dgj = __fmaf_rn(kgf * _tmem_load_4[j_2 + jj_1], beta_t, __fmaf_rn(_unpack_bf16_4[j_2 + jj_1], dqc[j_2 + jj_1], -(_unpack_bf16_3[j_2 + jj_1] * dk2[j_2 + jj_1])));
                            if (t_row == 63) {
                                dgj = dgj + dgk_s[col_1 + j_2 + jj_1];
                            }
                            dgv[j_2 + jj_1] = dgj;
                            dkf[j_2 + jj_1] = __fmaf_rn(_tmem_load_4[j_2 + jj_1] * e_g2[j_2 + jj_1], beta_t, dk2[j_2 + jj_1]);
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(hd_empty_addr);
                    if (nxt_c < (unsigned int)num_items_c) {
                        int nb = it_i + 1 & 1;
                        if (tid_1 < 64) {
                            beta_s[nb * 64 + tid_1] = nbv;
                        }
                        if (tid_1 == 64) {
                            rec_s[nb * 2] = nbos;
                            rec_s[nb * 2 + 1] = nlen;
                        }
                    }
                    if (t_row < clen) {
                        unsigned int _packed_0[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dvo[_lp*2 + 0], dvo[_lp*2+1 + 0]));
                            _packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dvo[_lp*2 + 2], dvo[_lp*2+1 + 2]));
                            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dvo[_lp*2 + 4], dvo[_lp*2+1 + 4]));
                            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dvo[_lp*2 + 6], dvo[_lp*2+1 + 6]));
                            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dvo[_lp*2 + 8], dvo[_lp*2+1 + 8]));
                            _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dvo[_lp*2 + 10], dvo[_lp*2+1 + 10]));
                            _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dvo[_lp*2 + 12], dvo[_lp*2+1 + 12]));
                            _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dvo[_lp*2 + 14], dvo[_lp*2+1 + 14]));
                            _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_8 = reinterpret_cast<const unsigned*>(_packed_0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(dv_item + (roff + col_1))), "r"(_raw_stv8_8[0]), "r"(_raw_stv8_8[1]), "r"(_raw_stv8_8[2]), "r"(_raw_stv8_8[3]), "r"(_raw_stv8_8[4]), "r"(_raw_stv8_8[5]), "r"(_raw_stv8_8[6]), "r"(_raw_stv8_8[7]) : "memory");
                        }
                    }
                    float dgv_lo[8];
                    float dgv_hi[8];
                    #pragma unroll
                    for (int jj_2 = 0; jj_2 < 8; jj_2++) {
                        dgv_lo[jj_2] = dgv[jj_2];
                        dgv_hi[jj_2] = dgv[jj_2 + 8];
                    }
                    {
                        unsigned _stv8_9_0 = __float_as_uint(dgv_lo[0 + 0]);
                        unsigned _stv8_9_1 = __float_as_uint(dgv_lo[0 + 1]);
                        unsigned _stv8_9_2 = __float_as_uint(dgv_lo[0 + 2]);
                        unsigned _stv8_9_3 = __float_as_uint(dgv_lo[0 + 3]);
                        unsigned _stv8_9_4 = __float_as_uint(dgv_lo[0 + 4]);
                        unsigned _stv8_9_5 = __float_as_uint(dgv_lo[0 + 5]);
                        unsigned _stv8_9_6 = __float_as_uint(dgv_lo[0 + 6]);
                        unsigned _stv8_9_7 = __float_as_uint(dgv_lo[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dg_item + (roff + col_1) + (0))), "r"(_stv8_9_0), "r"(_stv8_9_1), "r"(_stv8_9_2), "r"(_stv8_9_3), "r"(_stv8_9_4), "r"(_stv8_9_5), "r"(_stv8_9_6), "r"(_stv8_9_7) : "memory");
                    }
                    {
                        unsigned _stv8_10_0 = __float_as_uint(dgv_hi[0 + 0]);
                        unsigned _stv8_10_1 = __float_as_uint(dgv_hi[0 + 1]);
                        unsigned _stv8_10_2 = __float_as_uint(dgv_hi[0 + 2]);
                        unsigned _stv8_10_3 = __float_as_uint(dgv_hi[0 + 3]);
                        unsigned _stv8_10_4 = __float_as_uint(dgv_hi[0 + 4]);
                        unsigned _stv8_10_5 = __float_as_uint(dgv_hi[0 + 5]);
                        unsigned _stv8_10_6 = __float_as_uint(dgv_hi[0 + 6]);
                        unsigned _stv8_10_7 = __float_as_uint(dgv_hi[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dg_item + (roff + col_1 + 8) + (0))), "r"(_stv8_10_0), "r"(_stv8_10_1), "r"(_stv8_10_2), "r"(_stv8_10_3), "r"(_stv8_10_4), "r"(_stv8_10_5), "r"(_stv8_10_6), "r"(_stv8_10_7) : "memory");
                    }
                    float dkf_lo[8];
                    float dkf_hi[8];
                    #pragma unroll
                    for (int jj_3 = 0; jj_3 < 8; jj_3++) {
                        dkf_lo[jj_3] = dkf[jj_3];
                        dkf_hi[jj_3] = dkf[jj_3 + 8];
                    }
                    {
                        unsigned _stv8_11_0 = __float_as_uint(dkf_lo[0 + 0]);
                        unsigned _stv8_11_1 = __float_as_uint(dkf_lo[0 + 1]);
                        unsigned _stv8_11_2 = __float_as_uint(dkf_lo[0 + 2]);
                        unsigned _stv8_11_3 = __float_as_uint(dkf_lo[0 + 3]);
                        unsigned _stv8_11_4 = __float_as_uint(dkf_lo[0 + 4]);
                        unsigned _stv8_11_5 = __float_as_uint(dkf_lo[0 + 5]);
                        unsigned _stv8_11_6 = __float_as_uint(dkf_lo[0 + 6]);
                        unsigned _stv8_11_7 = __float_as_uint(dkf_lo[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dk_item + (roff + col_1) + (0))), "r"(_stv8_11_0), "r"(_stv8_11_1), "r"(_stv8_11_2), "r"(_stv8_11_3), "r"(_stv8_11_4), "r"(_stv8_11_5), "r"(_stv8_11_6), "r"(_stv8_11_7) : "memory");
                    }
                    {
                        unsigned _stv8_12_0 = __float_as_uint(dkf_hi[0 + 0]);
                        unsigned _stv8_12_1 = __float_as_uint(dkf_hi[0 + 1]);
                        unsigned _stv8_12_2 = __float_as_uint(dkf_hi[0 + 2]);
                        unsigned _stv8_12_3 = __float_as_uint(dkf_hi[0 + 3]);
                        unsigned _stv8_12_4 = __float_as_uint(dkf_hi[0 + 4]);
                        unsigned _stv8_12_5 = __float_as_uint(dkf_hi[0 + 5]);
                        unsigned _stv8_12_6 = __float_as_uint(dkf_hi[0 + 6]);
                        unsigned _stv8_12_7 = __float_as_uint(dkf_hi[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dk_item + (roff + col_1 + 8) + (0))), "r"(_stv8_12_0), "r"(_stv8_12_1), "r"(_stv8_12_2), "r"(_stv8_12_3), "r"(_stv8_12_4), "r"(_stv8_12_5), "r"(_stv8_12_6), "r"(_stv8_12_7) : "memory");
                    }
                }
                float db_other = __shfl_xor_sync(0xFFFFFFFF, db_acc, 16);
                float db_row = db_acc + db_other;
                if (lane_half == 0) {
                    dbp_s[cblk * 64 + t_row] = db_row;
                }
                asm volatile("barrier.sync 1, 512;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (cblk == 0) {
                    if (lane_half == 0) {
                        float db_sum = dbp_s[t_row];
                        db_sum = db_sum + dbp_s[64 + t_row];
                        db_sum = db_sum + dbp_s[128 + t_row];
                        db_sum = db_sum + dbp_s[192 + t_row];
                        db_out[grow * num_heads + head_1] = db_sum;
                    }
                }
                if (cblk < 2) {
                    int blk_2 = cblk;
                    int col_2 = blk_2 * 32 + lane_half * 16;
                    float _tmem_load_5[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                        : "r"(taddr + 384 + (unsigned int)row_base + (unsigned int)(blk_2 * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float dab[16];
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 16; j_3++) {
                        int s_col = col_2 + j_3;
                        float val = _tmem_load_5[j_3] * beta_s[s_col + bbase];
                        if (s_col >= t_row) {
                            val = 0.0f;
                        }
                        dab[j_3] = val;
                    }
                    unsigned int dab_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dab[_lp*2 + 0], dab[_lp*2+1 + 0]));
                        dab_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_10 = 0; q_10 < 2; q_10++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dab_k_addr + (unsigned int)((col_2 + q_10 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_10 * 8) % 64 * 2 ^ ((col_2 + q_10 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_10 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(dab_bf16[q_10 * 4]), "r"(dab_bf16[q_10 * 4 + 1]), "r"(dab_bf16[q_10 * 4 + 2]), "r"(dab_bf16[q_10 * 4 + 3]) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(b_drained_addr);
                mbarrier_wait(c_ready_addr, _phase_c_ready_0);
                _phase_c_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (cblk < 2) {
                    int blk_3 = cblk;
                    int col_3 = blk_3 * 32 + lane_half * 16;
                    float _tmem_load_6[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                        : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(blk_3 * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int _tmem_load_6_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                        _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_11 = 0; q_11 < 2; q_11++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((da1b_w_addr + (unsigned int)((col_3 + q_11 * 8) / 64 * 8192 + t_row * 128 + (col_3 + q_11 * 8) % 64 * 2 ^ ((col_3 + q_11 * 8) / 64 * 8192 + t_row * 128 + (col_3 + q_11 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[q_11 * 4]), "r"(_tmem_load_6_bf16[q_11 * 4 + 1]), "r"(_tmem_load_6_bf16[q_11 * 4 + 2]), "r"(_tmem_load_6_bf16[q_11 * 4 + 3]) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(c_drained_addr);
                mbarrier_arrive(d_committed_addr);
                mbarrier_wait(d_ready_addr, _phase_d_ready_0);
                _phase_d_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int abase = (grow * num_heads + head_1) * 64;
                if (cblk < 2) {
                    int blk_4 = cblk;
                    int col_4 = blk_4 * 32 + lane_half * 16;
                    float _tmem_load_7[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                        : "r"(taddr + 320 + (unsigned int)row_base + (unsigned int)(blk_4 * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float outv[16];
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 16; j_4++) {
                        int s_col2 = col_4 + j_4;
                        float v2 = -_tmem_load_7[j_4];
                        if (s_col2 >= t_row) {
                            v2 = 0.0f;
                        }
                        outv[j_4] = v2;
                    }
                    float outv_lo[8];
                    float outv_hi[8];
                    #pragma unroll
                    for (int jj_4 = 0; jj_4 < 8; jj_4++) {
                        outv_lo[jj_4] = outv[jj_4];
                        outv_hi[jj_4] = outv[jj_4 + 8];
                    }
                    {
                        unsigned _stv8_13_0 = __float_as_uint(outv_lo[0 + 0]);
                        unsigned _stv8_13_1 = __float_as_uint(outv_lo[0 + 1]);
                        unsigned _stv8_13_2 = __float_as_uint(outv_lo[0 + 2]);
                        unsigned _stv8_13_3 = __float_as_uint(outv_lo[0 + 3]);
                        unsigned _stv8_13_4 = __float_as_uint(outv_lo[0 + 4]);
                        unsigned _stv8_13_5 = __float_as_uint(outv_lo[0 + 5]);
                        unsigned _stv8_13_6 = __float_as_uint(outv_lo[0 + 6]);
                        unsigned _stv8_13_7 = __float_as_uint(outv_lo[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dAkk_out + (abase + col_4) + (0))), "r"(_stv8_13_0), "r"(_stv8_13_1), "r"(_stv8_13_2), "r"(_stv8_13_3), "r"(_stv8_13_4), "r"(_stv8_13_5), "r"(_stv8_13_6), "r"(_stv8_13_7) : "memory");
                    }
                    {
                        unsigned _stv8_14_0 = __float_as_uint(outv_hi[0 + 0]);
                        unsigned _stv8_14_1 = __float_as_uint(outv_hi[0 + 1]);
                        unsigned _stv8_14_2 = __float_as_uint(outv_hi[0 + 2]);
                        unsigned _stv8_14_3 = __float_as_uint(outv_hi[0 + 3]);
                        unsigned _stv8_14_4 = __float_as_uint(outv_hi[0 + 4]);
                        unsigned _stv8_14_5 = __float_as_uint(outv_hi[0 + 5]);
                        unsigned _stv8_14_6 = __float_as_uint(outv_hi[0 + 6]);
                        unsigned _stv8_14_7 = __float_as_uint(outv_hi[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dAkk_out + (abase + col_4 + 8) + (0))), "r"(_stv8_14_0), "r"(_stv8_14_1), "r"(_stv8_14_2), "r"(_stv8_14_3), "r"(_stv8_14_4), "r"(_stv8_14_5), "r"(_stv8_14_6), "r"(_stv8_14_7) : "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(d_drained_addr);
                it_i += 1;
            }
            asm volatile("barrier.sync 1, 512;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 16) {
        { // load_main
            int nc_l = seq_chunk_start[num_seqs];
            int num_items_l = nc_l * num_heads;
            unsigned int _phase_dv2_free_0 = 1;
            unsigned int _phase_hd_empty_0 = 1;
            unsigned int _phase_d_committed_0 = 1;
            unsigned int _phase_a_committed_0 = 0;
            unsigned int _phase_hd_read_done_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int it_1 = bid; it_1 < num_items_l; it_1 += num_bids) {
                    int chunk_2 = it_1 / (unsigned int)num_heads;
                    int head_2 = it_1 % (unsigned int)num_heads;
                    int row0_3 = chunk_2 * 64;
                    int hrow0_2 = (chunk_2 * num_heads + head_2) * 128;
                    int trow = chunk_bos[chunk_2];
                    mbarrier_wait(dv2_free_addr, _phase_dv2_free_0);
                    _phase_dv2_free_0 ^= 1;
                    mbarrier_arrive_expect_tx(dv2_full_addr, 16384);
                    #pragma unroll
                    for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                        int c0 = seg_1 * 64;
                        if (num_heads >= 16) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_v2_addr + (unsigned int)(seg_1 * 8192)), "l"((&dv2_tma)), "r"(c0), "r"(head_2), "r"(row0_3),
                                   "r"(dv2_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                        } else {
                            tma_3d_gmem2smem(smem_v2_addr + (unsigned int)(seg_1 * 8192), (&dv2_tma), c0, head_2, row0_3, dv2_full_addr);
                        }
                    }
                    mbarrier_wait(hd_empty_addr, _phase_hd_empty_0);
                    _phase_hd_empty_0 ^= 1;
                    mbarrier_arrive_expect_tx(h_full_addr, 32768);
                    #pragma unroll
                    for (int seg_2 = 0; seg_2 < 2; seg_2++) {
                        int c1 = seg_2 * 64;
                        tma_2d_gmem2smem(h_p_addr + (unsigned int)(seg_2 * 16384), (&h_tma), c1, hrow0_2, h_full_addr);
                    }
                    if ((it_1 - (unsigned int)bid) / (unsigned int)num_bids == 0) {
                        mbarrier_arrive_expect_tx(dh_full_addr, 32768);
                        #pragma unroll
                        for (int seg_3 = 0; seg_3 < 2; seg_3++) {
                            int c2 = seg_3 * 64;
                            if (num_heads >= 16) {
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3}], [%4], %5;"
                                    :: "r"(dh_p_addr + (unsigned int)(seg_3 * 16384)), "l"((&dh_tma)), "r"(c2), "r"(hrow0_2),
                                       "r"(dh_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                            } else {
                                tma_2d_gmem2smem(dh_p_addr + (unsigned int)(seg_3 * 16384), (&dh_tma), c2, hrow0_2, dh_full_addr);
                            }
                        }
                    }
                    mbarrier_arrive_expect_tx(dovn_full_addr, 32768);
                    #pragma unroll
                    for (int seg_4 = 0; seg_4 < 2; seg_4++) {
                        int c5 = seg_4 * 64;
                        if (num_heads >= 16) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_v0_addr + (unsigned int)(seg_4 * 8192)), "l"((&do_tma)), "r"(c5), "r"(head_2), "r"(row0_3),
                                   "r"(dovn_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                        } else {
                            tma_3d_gmem2smem(smem_v0_addr + (unsigned int)(seg_4 * 8192), (&do_tma), c5, head_2, row0_3, dovn_full_addr);
                        }
                        if (num_heads >= 16) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_v1_addr + (unsigned int)(seg_4 * 8192)), "l"((&vn_tma)), "r"(c5), "r"(head_2), "r"(row0_3),
                                   "r"(dovn_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                        } else {
                            tma_3d_gmem2smem(smem_v1_addr + (unsigned int)(seg_4 * 8192), (&vn_tma), c5, head_2, row0_3, dovn_full_addr);
                        }
                    }
                    mbarrier_arrive_expect_tx(ops_full_addr, 49152);
                    #pragma unroll
                    for (int sl = 0; sl < 4; sl++) {
                        int c3 = sl * 32;
                        if (num_heads >= 16) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(gk_p_addr + (unsigned int)(sl * 8192)), "l"((&gk_tma)), "r"(c3), "r"(head_2), "r"(row0_3),
                                   "r"(ops_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                        } else {
                            tma_3d_gmem2smem(gk_p_addr + (unsigned int)(sl * 8192), (&gk_tma), c3, head_2, row0_3, ops_full_addr);
                        }
                    }
                    #pragma unroll
                    for (int seg_5 = 0; seg_5 < 2; seg_5++) {
                        int c4 = seg_5 * 64;
                        if (num_heads >= 16) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(k_p_addr + (unsigned int)(seg_5 * 8192)), "l"((&k_tma)), "r"(c4), "r"(head_2), "r"(trow),
                                   "r"(ops_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                        } else {
                            tma_3d_gmem2smem(k_p_addr + (unsigned int)(seg_5 * 8192), (&k_tma), c4, head_2, trow, ops_full_addr);
                        }
                    }
                    mbarrier_wait(d_committed_addr, _phase_d_committed_0);
                    _phase_d_committed_0 ^= 1;
                    mbarrier_arrive_expect_tx(akkv_full_addr, 24576);
                    if (num_heads >= 16) {
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(akk_k_addr), "l"((&akk_tma)), "r"(0), "r"(head_2), "r"(row0_3),
                               "r"(akkv_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    } else {
                        tma_3d_gmem2smem(akk_k_addr, (&akk_tma), 0, head_2, row0_3, akkv_full_addr);
                    }
                    #pragma unroll
                    for (int seg_6 = 0; seg_6 < 2; seg_6++) {
                        int c6 = seg_6 * 64;
                        if (num_heads >= 16) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(v_p_addr + (unsigned int)(seg_6 * 8192)), "l"((&v_tma)), "r"(c6), "r"(head_2), "r"(trow),
                                   "r"(akkv_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                        } else {
                            tma_3d_gmem2smem(v_p_addr + (unsigned int)(seg_6 * 8192), (&v_tma), c6, head_2, trow, akkv_full_addr);
                        }
                    }
                    mbarrier_wait(a_committed_addr, _phase_a_committed_0);
                    _phase_a_committed_0 ^= 1;
                    mbarrier_wait(hd_read_done_addr, _phase_hd_read_done_0);
                    _phase_hd_read_done_0 ^= 1;
                    mbarrier_arrive_expect_tx(q_full_addr, 16384);
                    #pragma unroll
                    for (int seg_7 = 0; seg_7 < 2; seg_7++) {
                        int c7 = seg_7 * 64;
                        if (num_heads >= 16) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(q_p_addr + (unsigned int)(seg_7 * 8192)), "l"((&q_tma)), "r"(c7), "r"(head_2), "r"(trow),
                                   "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                        } else {
                            tma_3d_gmem2smem(q_p_addr + (unsigned int)(seg_7 * 8192), (&q_tma), c7, head_2, trow, q_full_addr);
                        }
                    }
                    unsigned int nxt = it_1 + (unsigned int)num_bids;
                    if (nxt < (unsigned int)num_items_l) {
                        int chunk_0_1 = nxt / (unsigned int)num_heads;
                        int head_1_2 = nxt % (unsigned int)num_heads;
                        int row0_2_1 = chunk_0_1 * 64;
                        int hrow0_3_1 = (chunk_0_1 * num_heads + head_1_2) * 128;
                        mbarrier_arrive_expect_tx(dh_full_addr, 32768);
                        #pragma unroll
                        for (int seg_8 = 0; seg_8 < 2; seg_8++) {
                            int c8 = seg_8 * 64;
                            if (num_heads >= 16) {
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3}], [%4], %5;"
                                    :: "r"(dh_p_addr + (unsigned int)(seg_8 * 16384)), "l"((&dh_tma)), "r"(c8), "r"(hrow0_3_1),
                                       "r"(dh_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                            } else {
                                tma_2d_gmem2smem(dh_p_addr + (unsigned int)(seg_8 * 16384), (&dh_tma), c8, hrow0_3_1, dh_full_addr);
                            }
                        }
                    }
                    if (num_heads >= 16) {
                        unsigned int pf = it_1 + (unsigned int)num_bids;
                        if (pf < (unsigned int)num_items_l) {
                            int chunk_0_2 = pf / (unsigned int)num_heads;
                            int head_1_3 = pf % (unsigned int)num_heads;
                            int row0_2_2 = chunk_0_2 * 64;
                            int hrow0_3_2 = (chunk_0_2 * num_heads + head_1_3) * 128;
                            int ptrow = chunk_bos[chunk_0_2];
                            #pragma unroll
                            for (int seg_9 = 0; seg_9 < 2; seg_9++) {
                                int cp = seg_9 * 64;
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&dv2_tma))), "r"((int)(cp)), "r"((int)(head_1_3)), "r"((int)(row0_2_2)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&h_tma))), "r"((int)(cp)), "r"((int)(hrow0_3_2)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&k_tma))), "r"((int)(cp)), "r"((int)(head_1_3)), "r"((int)(ptrow)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&do_tma))), "r"((int)(cp)), "r"((int)(head_1_3)), "r"((int)(row0_2_2)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&vn_tma))), "r"((int)(cp)), "r"((int)(head_1_3)), "r"((int)(row0_2_2)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&v_tma))), "r"((int)(cp)), "r"((int)(head_1_3)), "r"((int)(ptrow)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&q_tma))), "r"((int)(cp)), "r"((int)(head_1_3)), "r"((int)(ptrow)) : "memory");
                            }
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&akk_tma))), "r"((int)(0)), "r"((int)(head_1_3)), "r"((int)(row0_2_2)) : "memory");
                            #pragma unroll
                            for (int sl_1 = 0; sl_1 < 4; sl_1++) {
                                int cg = sl_1 * 32;
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&gk_tma))), "r"((int)(cg)), "r"((int)(head_1_3)), "r"((int)(row0_2_2)) : "memory");
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 17) {
        { // mma_main
            int nc_m = seq_chunk_start[num_seqs];
            int num_items_m = nc_m * num_heads;
            unsigned int _phase_dovn_full_0_1 = 0;
            unsigned int _phase_dv2_full_0 = 0;
            unsigned int _phase_h_full_0_1 = 0;
            unsigned int _phase_dh_full_0_1 = 0;
            unsigned int _phase_d_drained_0 = 1;
            unsigned int _phase_a_drained_0 = 0;
            unsigned int _phase_akkv_full_0_1 = 0;
            unsigned int _phase_b_drained_0 = 0;
            unsigned int _phase_c_drained_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int it_2 = bid; it_2 < num_items_m; it_2 += num_bids) {
                    mbarrier_wait(dovn_full_addr, _phase_dovn_full_0_1);
                    _phase_dovn_full_0_1 ^= 1;
                    mbarrier_wait(dv2_full_addr, _phase_dv2_full_0);
                    _phase_dv2_full_0 ^= 1;
                    mbarrier_wait(h_full_addr, _phase_h_full_0_1);
                    _phase_h_full_0_1 ^= 1;
                    mbarrier_wait(dh_full_addr, _phase_dh_full_0_1);
                    _phase_dh_full_0_1 ^= 1;
                    mbarrier_wait(d_drained_addr, _phase_d_drained_0);
                    _phase_d_drained_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_0 = (((smem_v3_addr) >> 4) & 0x7FFF) + (0) * 3584;
                    int _mma_b_lo_0 = ((h_k_addr) >> 4) & 0x7FFF;
#else
                    int _mma_a_lo_0 = (((smem_v3_addr) >> 4) & 0x3FFF) + (0) * 3584;
                    int _mma_b_lo_0 = ((h_k_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 69207184;\n\t"
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
                    "add.u32 blo, blo, 1018;\n\t"
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
                    int _smem_desc_lo_0 = (((smem_v4_addr) >> 4) & 0x7FFF) + (0) * 3584;
                    int _smem_desc_lo_1 = ((dh_k_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_0 = (((smem_v4_addr) >> 4) & 0x3FFF) + (0) * 3584;
                    int _smem_desc_lo_1 = ((dh_k_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 69207184;\n\t"
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
                    "add.u32 blo, blo, 1018;\n\t"
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc_dk), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = (((smem_v5_addr) >> 4) & 0x7FFF) + (0) * 3584;
#else
                    int _smem_desc_lo_2 = (((smem_v5_addr) >> 4) & 0x3FFF) + (0) * 3584;
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
                    "mov.b32 id, 69207184;\n\t"
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
                    "add.u32 blo, blo, 1018;\n\t"
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
                    :: "r"(_smem_desc_lo_2), "r"(_mma_b_lo_0), "r"(tmem_acc_dw), "r"(0) : "memory");
                    tcgen05_commit(a_ready_addr);
                    tcgen05_commit(a_committed_addr);
                    mbarrier_wait(a_drained_addr, _phase_a_drained_0);
                    _phase_a_drained_0 ^= 1;
                    mbarrier_wait(akkv_full_addr, _phase_akkv_full_0_1);
                    _phase_akkv_full_0_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_3 = (((smem_v5_addr) >> 4) & 0x7FFF) + (0) * 3584;
                    int _smem_desc_lo_4 = ((v_k_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_3 = (((smem_v5_addr) >> 4) & 0x3FFF) + (0) * 3584;
                    int _smem_desc_lo_4 = ((v_k_addr) >> 4) & 0x3FFF;
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
                    :: "r"(_smem_desc_lo_3), "r"(_smem_desc_lo_4), "r"(tmem_acc_da), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_5 = ((((akk_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (0) * 3584;
                    int _smem_desc_lo_6 = ((((dv2_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (0) * 3584;
#else
                    int _smem_desc_lo_5 = ((((akk_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 3584;
                    int _smem_desc_lo_6 = ((((dv2_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 3584;
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
                    :: "r"(_smem_desc_lo_5), "r"(_smem_desc_lo_6), "r"(tmem_acc_dvb), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_7 = ((((akk_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (0) * 3584;
                    int _smem_desc_lo_8 = (((dwb_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_7 = ((((akk_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 3584;
                    int _smem_desc_lo_8 = (((dwb_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_7), "r"(_smem_desc_lo_8), "r"(tmem_acc_dkgb), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_9 = ((dwb_k_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_10 = ((kgb_k_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_9 = ((dwb_k_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_10 = ((kgb_k_addr) >> 4) & 0x3FFF;
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
                    :: "r"(_smem_desc_lo_9), "r"(_smem_desc_lo_10), "r"(tmem_acc_da), "r"(1) : "memory");
                    tcgen05_commit(b_ready_addr);
                    tcgen05_commit(dv2_free_addr);
                    mbarrier_wait(b_drained_addr, _phase_b_drained_0);
                    _phase_b_drained_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_11 = ((dab_k_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_12 = (((akk_k_addr) >> 4) & 0x7FFF) + (0) * 3584;
#else
                    int _smem_desc_lo_11 = ((dab_k_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_12 = (((akk_k_addr) >> 4) & 0x3FFF) + (0) * 3584;
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
                    :: "r"(_smem_desc_lo_11), "r"(_smem_desc_lo_12), "r"(tmem_acc_da1), "r"(0) : "memory");
                    tcgen05_commit(c_ready_addr);
                    mbarrier_wait(c_drained_addr, _phase_c_drained_0);
                    _phase_c_drained_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_13 = ((((akk_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (0) * 3584;
                    int _smem_desc_lo_14 = (((da1b_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_13 = ((((akk_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 3584;
                    int _smem_desc_lo_14 = (((da1b_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_13), "r"(_smem_desc_lo_14), "r"(tmem_acc_da2), "r"(0) : "memory");
                    tcgen05_commit(d_ready_addr);
                    tcgen05_commit(d_committed_addr);
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
