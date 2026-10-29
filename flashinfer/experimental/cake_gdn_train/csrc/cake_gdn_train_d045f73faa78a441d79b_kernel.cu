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
#define TMEM_ACC_OFFSET 0
#define NUM_VRING_STAGES 2
#define NUM_ONE_STAGES 1
#define SMEM_Q_P_OFF 1024
#define SMEM_Q_P_STAGE_BYTES 16384
#define SMEM_Q_P_STRIDE 16384
#define SMEM_K_P_OFF 17408
#define SMEM_K_P_STAGE_BYTES 16384
#define SMEM_K_P_STRIDE 16384
#define SMEM_V_P_OFF 33792
#define SMEM_V_P_STAGE_BYTES 16384
#define SMEM_V_P_STRIDE 16384
#define SMEM_Q_K_OFF 1024
#define SMEM_Q_K_STAGE_BYTES 16384
#define SMEM_Q_K_STRIDE 16384
#define SMEM_K_K_OFF 17408
#define SMEM_K_K_STAGE_BYTES 16384
#define SMEM_K_K_STRIDE 16384
#define SMEM_KB_MN_OFF 1024
#define SMEM_KB_MN_STAGE_BYTES 16384
#define SMEM_KB_MN_STRIDE 16384
#define SMEM_V_T_OFF 33792
#define SMEM_V_T_STAGE_BYTES 16384
#define SMEM_V_T_STRIDE 16384
#define SMEM_V_MN_OFF 33792
#define SMEM_V_MN_STAGE_BYTES 16384
#define SMEM_V_MN_STRIDE 16384
#define SMEM_AKK_T_OFF 105472
#define SMEM_AKK_T_STAGE_BYTES 8192
#define SMEM_AKK_T_STRIDE 8192
#define SMEM_L_S_OFF 66560
#define SMEM_L_S_STAGE_BYTES 17408
#define SMEM_L_S_STRIDE 17408
#define SMEM_A_S_OFF 83968
#define SMEM_A_S_STAGE_BYTES 17408
#define SMEM_A_S_STRIDE 17408
#define SMEM_X_S_OFF 101376
#define SMEM_X_S_STAGE_BYTES 3072
#define SMEM_X_S_STRIDE 3072
#define SMEM_GS_S_OFF 104448
#define SMEM_GS_S_STAGE_BYTES 256
#define SMEM_GS_S_STRIDE 256
#define SMEM_GC_S_OFF 104704
#define SMEM_GC_S_STAGE_BYTES 256
#define SMEM_GC_S_STRIDE 256
#define SMEM_EG_S_OFF 104960
#define SMEM_EG_S_STAGE_BYTES 256
#define SMEM_EG_S_STRIDE 256
#define SMEM_BETA_S_OFF 105216
#define SMEM_BETA_S_STAGE_BYTES 256
#define SMEM_BETA_S_STRIDE 256
#define SMEM_TOTAL 113664
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
kernel_cake_gdn_train_d045f73faa78a441d79b(const __grid_constant__ CUtensorMap q_tma, const __grid_constant__ CUtensorMap k_tma, const __grid_constant__ CUtensorMap v_tma, float* __restrict__ g, float* __restrict__ beta, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ w_out, __nv_bfloat16* __restrict__ u_out, __nv_bfloat16* __restrict__ aqk_out, __nv_bfloat16* __restrict__ akk_out, float* __restrict__ g_cum, float* __restrict__ rowscale, float* __restrict__ qscale, float* __restrict__ gamma, int num_heads, int num_qk_heads, float scale)
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
    #define gram_full_addr (mbar_base + 8)
    #define v_full_addr (mbar_base + 16)
    #define v_empty_addr (mbar_base + 32)
    #define op_ready_addr (mbar_base + 48)
    #define wu_full_addr (mbar_base + 56)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* q_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_p_addr = smem + 1024;
    __nv_bfloat16* k_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int k_p_addr = smem + 17408;
    __nv_bfloat16* v_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int v_p_addr = smem + 33792;
    __nv_bfloat16* q_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_k_addr = smem + 1024;
    __nv_bfloat16* k_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int k_k_addr = smem + 17408;
    __nv_bfloat16* kb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int kb_mn_addr = smem + 1024;
    __nv_bfloat16* v_t = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int v_t_addr = smem + 33792;
    __nv_bfloat16* v_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int v_mn_addr = smem + 33792;
    __nv_bfloat16* akk_t = reinterpret_cast<__nv_bfloat16*>(smem_raw + 105472);
    const int akk_t_addr = smem + 105472;
    float* l_s = reinterpret_cast<float*>(smem_raw + 66560);
    const int l_s_addr = smem + 66560;
    float* a_s = reinterpret_cast<float*>(smem_raw + 83968);
    const int a_s_addr = smem + 83968;
    float* x_s = reinterpret_cast<float*>(smem_raw + 101376);
    const int x_s_addr = smem + 101376;
    float* gs_s = reinterpret_cast<float*>(smem_raw + 104448);
    const int gs_s_addr = smem + 104448;
    float* gc_s = reinterpret_cast<float*>(smem_raw + 104704);
    const int gc_s_addr = smem + 104704;
    float* eg_s = reinterpret_cast<float*>(smem_raw + 104960);
    const int eg_s_addr = smem + 104960;
    float* beta_s = reinterpret_cast<float*>(smem_raw + 105216);
    const int beta_s_addr = smem + 105216;
    int chunk = blockIdx.x;
    int hq = blockIdx.y;
    int ext0 = chunk_bos[chunk];
    int clen = chunk_len[chunk];
    int group = num_heads / num_qk_heads;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'one' ---
            // qk_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // gram_full: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'vring' ---
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'one' ---
            // op_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            // wu_full: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 64);
    if (warp == 0) {
        int _tmem_hold = smem + 64;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int tid_0 = tid;
            int w_id = warp;
            int lane_1 = lane;
            int half = lane_1 >> 4;
            int t_row = w_id * 16 + (lane_1 & 15);
            int row_base = w_id * 32 << 16;
            int l_base = (int)l_s_addr;
            int a_base = (int)a_s_addr;
            int x_base = (int)x_s_addr;
            int gs_base = (int)gs_s_addr;
            int gc_base = (int)gc_s_addr;
            int eg_base = (int)eg_s_addr;
            int beta_base = (int)beta_s_addr;
            int irow0 = chunk * 64;
            unsigned int vst_c = 0;
            float zero16[16];
            zero16[0] = 0.0f;
            zero16[1] = 0.0f;
            zero16[2] = 0.0f;
            zero16[3] = 0.0f;
            zero16[4] = 0.0f;
            zero16[5] = 0.0f;
            zero16[6] = 0.0f;
            zero16[7] = 0.0f;
            zero16[8] = 0.0f;
            zero16[9] = 0.0f;
            zero16[10] = 0.0f;
            zero16[11] = 0.0f;
            zero16[12] = 0.0f;
            zero16[13] = 0.0f;
            zero16[14] = 0.0f;
            zero16[15] = 0.0f;
            unsigned int _phase_qk_full_0 = 0;
            unsigned int _phase_gram_full_0 = 0;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_wu_full_0 = 0;
            if (clen > 0) {
                float sqk[32];
                float skk[32];
                float gv_n = 0.0f;
                float bv_n = 0.0f;
                if (tid_0 < clen) {
                    long long tok0 = (long long)(ext0 + tid_0) * (long long)num_heads + (long long)(hq * group);
                    gv_n = g[tok0];
                    bv_n = beta[tok0];
                }
                mbarrier_wait(qk_full_addr, _phase_qk_full_0);
                _phase_qk_full_0 ^= 1;
                mbarrier_wait(gram_full_addr, _phase_gram_full_0);
                _phase_gram_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sqk[0])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[1])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[2])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[3])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[4])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[5])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[6])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[7])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[8])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[9])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[10])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[11])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[12])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[13])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[14])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[15]))
                    : "r"(taddr + (unsigned int)row_base));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&skk[0])), "=r"(*reinterpret_cast<uint32_t*>(&skk[1])), "=r"(*reinterpret_cast<uint32_t*>(&skk[2])), "=r"(*reinterpret_cast<uint32_t*>(&skk[3])), "=r"(*reinterpret_cast<uint32_t*>(&skk[4])), "=r"(*reinterpret_cast<uint32_t*>(&skk[5])), "=r"(*reinterpret_cast<uint32_t*>(&skk[6])), "=r"(*reinterpret_cast<uint32_t*>(&skk[7])), "=r"(*reinterpret_cast<uint32_t*>(&skk[8])), "=r"(*reinterpret_cast<uint32_t*>(&skk[9])), "=r"(*reinterpret_cast<uint32_t*>(&skk[10])), "=r"(*reinterpret_cast<uint32_t*>(&skk[11])), "=r"(*reinterpret_cast<uint32_t*>(&skk[12])), "=r"(*reinterpret_cast<uint32_t*>(&skk[13])), "=r"(*reinterpret_cast<uint32_t*>(&skk[14])), "=r"(*reinterpret_cast<uint32_t*>(&skk[15]))
                    : "r"(taddr + (unsigned int)row_base + 64));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sqk[16])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[17])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[18])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[19])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[20])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[21])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[22])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[23])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[24])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[25])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[26])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[27])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[28])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[29])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[30])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[31]))
                    : "r"(taddr + (unsigned int)row_base + 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&skk[16])), "=r"(*reinterpret_cast<uint32_t*>(&skk[17])), "=r"(*reinterpret_cast<uint32_t*>(&skk[18])), "=r"(*reinterpret_cast<uint32_t*>(&skk[19])), "=r"(*reinterpret_cast<uint32_t*>(&skk[20])), "=r"(*reinterpret_cast<uint32_t*>(&skk[21])), "=r"(*reinterpret_cast<uint32_t*>(&skk[22])), "=r"(*reinterpret_cast<uint32_t*>(&skk[23])), "=r"(*reinterpret_cast<uint32_t*>(&skk[24])), "=r"(*reinterpret_cast<uint32_t*>(&skk[25])), "=r"(*reinterpret_cast<uint32_t*>(&skk[26])), "=r"(*reinterpret_cast<uint32_t*>(&skk[27])), "=r"(*reinterpret_cast<uint32_t*>(&skk[28])), "=r"(*reinterpret_cast<uint32_t*>(&skk[29])), "=r"(*reinterpret_cast<uint32_t*>(&skk[30])), "=r"(*reinterpret_cast<uint32_t*>(&skk[31]))
                    : "r"(taddr + (unsigned int)row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int row_valid = 0;
                if (t_row < clen) {
                    row_valid = 1;
                }
                #pragma unroll 1
                for (int j = 0; j < group + 1; j++) {
                    int hv = hq * group + j;
                    if (j > 1) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    if (group > j) {
                        if (tid_0 < 64) {
                            gs_s[tid_0] = gv_n;
                            beta_s[tid_0] = bv_n;
                            if (group > j + 1) {
                                if (tid_0 < clen) {
                                    long long tok_n = (long long)(ext0 + tid_0) * (long long)num_heads + (long long)(hv + 1);
                                    gv_n = g[tok_n];
                                    bv_n = beta[tok_n];
                                }
                            }
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (w_id == 0) {
                            float g0[1];
                            float g1[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&g0[0])) : "r"(gs_base + lane_1 * 4));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&g1[0])) : "r"(gs_base + (32 + lane_1) * 4));
                            float x0 = g0[0];
                            float x1 = g1[0];
                            float y0_1_0 = __shfl_up_sync(0xFFFFFFFF, x0, 1, 32);
                            float y1_2_0 = __shfl_up_sync(0xFFFFFFFF, x1, 1, 32);
                            x0 = x0 + ((lane_1 >= 1) ? y0_1_0 : 0.0f);
                            x1 = x1 + ((lane_1 >= 1) ? y1_2_0 : 0.0f);
                            float y0_1_1 = __shfl_up_sync(0xFFFFFFFF, x0, 2, 32);
                            float y1_2_1 = __shfl_up_sync(0xFFFFFFFF, x1, 2, 32);
                            x0 = x0 + ((lane_1 >= 2) ? y0_1_1 : 0.0f);
                            x1 = x1 + ((lane_1 >= 2) ? y1_2_1 : 0.0f);
                            float y0_1_2 = __shfl_up_sync(0xFFFFFFFF, x0, 4, 32);
                            float y1_2_2 = __shfl_up_sync(0xFFFFFFFF, x1, 4, 32);
                            x0 = x0 + ((lane_1 >= 4) ? y0_1_2 : 0.0f);
                            x1 = x1 + ((lane_1 >= 4) ? y1_2_2 : 0.0f);
                            float y0_1_3 = __shfl_up_sync(0xFFFFFFFF, x0, 8, 32);
                            float y1_2_3 = __shfl_up_sync(0xFFFFFFFF, x1, 8, 32);
                            x0 = x0 + ((lane_1 >= 8) ? y0_1_3 : 0.0f);
                            x1 = x1 + ((lane_1 >= 8) ? y1_2_3 : 0.0f);
                            float y0_1_4 = __shfl_up_sync(0xFFFFFFFF, x0, 16, 32);
                            float y1_2_4 = __shfl_up_sync(0xFFFFFFFF, x1, 16, 32);
                            x0 = x0 + ((lane_1 >= 16) ? y0_1_4 : 0.0f);
                            x1 = x1 + ((lane_1 >= 16) ? y1_2_4 : 0.0f);
                            float tot0 = __shfl_sync(0xFFFFFFFF, x0, 31);
                            x1 = x1 + tot0;
                            gc_s[lane_1] = x0;
                            gc_s[32 + lane_1] = x1;
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid_0 < 64) {
                            float gt_r[1];
                            float gl_r[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&gt_r[0])) : "r"(gc_base + tid_0 * 4));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&gl_r[0])) : "r"(gc_base + (clen - 1) * 4));
                            float gt = gt_r[0];
                            float gl = gl_r[0];
                            float eg = approx_exp2(gt * 1.442695f);
                            float rs = approx_exp2((gl - gt) * 1.442695f);
                            int valid = 0;
                            if (tid_0 < clen) {
                                valid = 1;
                            }
                            float eg_v = ((valid != 0) ? eg : 0.0f);
                            float rs_v = ((valid != 0) ? rs : 0.0f);
                            float gt_v = ((valid != 0) ? gt : 0.0f);
                            eg_s[tid_0] = eg_v;
                            long long tape = (long long)(irow0 + tid_0) * (long long)num_heads + (long long)hv;
                            *(reinterpret_cast<float*>(g_cum + tape) + (0)) = gt_v;
                            *(reinterpret_cast<float*>(rowscale + tape) + (0)) = rs_v;
                            *(reinterpret_cast<float*>(qscale + tape) + (0)) = eg_v * scale;
                            if (tid_0 == 0) {
                                float gam = approx_exp2(gl * 1.442695f);
                                *(reinterpret_cast<float*>(gamma + ((long long)chunk * (long long)num_heads + (long long)hv)) + (0)) = gam;
                            }
                        }
                        float bt_r[1];
                        float gtr_r[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bt_r[0])) : "r"(beta_base + t_row * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&gtr_r[0])) : "r"(gc_base + t_row * 4));
                        float bt = bt_r[0];
                        float gtr = gtr_r[0];
                        long long aqk_base = ((long long)(irow0 + t_row) * (long long)num_heads + (long long)hv) * 64;
                        int s0_1_0 = half * 16;
                        float gsv_1_0[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(0) + 3]))
                            : "r"(gc_base + s0_1_0 * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(4) + 3]))
                            : "r"(gc_base + (s0_1_0 + 4) * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(8) + 3]))
                            : "r"(gc_base + (s0_1_0 + 8) * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(12) + 3]))
                            : "r"(gc_base + (s0_1_0 + 12) * 4));
                        float aq_1_0[16];
                        float lvs_1_0[16];
                        int s_1_0 = s0_1_0;
                        float fac_2_0 = approx_exp2(fminf(gtr - gsv_1_0[0], 0.0f) * 1.442695f);
                        int on_aqk_1_0 = 0;
                        int on_l_1_0 = 0;
                        if (s_1_0 <= t_row) {
                            on_aqk_1_0 = row_valid;
                        }
                        if (s_1_0 < t_row) {
                            on_l_1_0 = row_valid;
                        }
                        aq_1_0[0] = ((on_aqk_1_0 != 0) ? sqk[0] * scale * fac_2_0 : 0.0f);
                        lvs_1_0[0] = ((on_l_1_0 != 0) ? bt * skk[0] * fac_2_0 : 0.0f);
                        int s_1_1 = s0_1_0 + 1;
                        float fac_2_1 = approx_exp2(fminf(gtr - gsv_1_0[1], 0.0f) * 1.442695f);
                        int on_aqk_1_1 = 0;
                        int on_l_1_1 = 0;
                        if (s_1_1 <= t_row) {
                            on_aqk_1_1 = row_valid;
                        }
                        if (s_1_1 < t_row) {
                            on_l_1_1 = row_valid;
                        }
                        aq_1_0[1] = ((on_aqk_1_1 != 0) ? sqk[1] * scale * fac_2_1 : 0.0f);
                        lvs_1_0[1] = ((on_l_1_1 != 0) ? bt * skk[1] * fac_2_1 : 0.0f);
                        int s_1_2 = s0_1_0 + 2;
                        float fac_2_2 = approx_exp2(fminf(gtr - gsv_1_0[2], 0.0f) * 1.442695f);
                        int on_aqk_1_2 = 0;
                        int on_l_1_2 = 0;
                        if (s_1_2 <= t_row) {
                            on_aqk_1_2 = row_valid;
                        }
                        if (s_1_2 < t_row) {
                            on_l_1_2 = row_valid;
                        }
                        aq_1_0[2] = ((on_aqk_1_2 != 0) ? sqk[2] * scale * fac_2_2 : 0.0f);
                        lvs_1_0[2] = ((on_l_1_2 != 0) ? bt * skk[2] * fac_2_2 : 0.0f);
                        int s_1_3 = s0_1_0 + 3;
                        float fac_2_3 = approx_exp2(fminf(gtr - gsv_1_0[3], 0.0f) * 1.442695f);
                        int on_aqk_1_3 = 0;
                        int on_l_1_3 = 0;
                        if (s_1_3 <= t_row) {
                            on_aqk_1_3 = row_valid;
                        }
                        if (s_1_3 < t_row) {
                            on_l_1_3 = row_valid;
                        }
                        aq_1_0[3] = ((on_aqk_1_3 != 0) ? sqk[3] * scale * fac_2_3 : 0.0f);
                        lvs_1_0[3] = ((on_l_1_3 != 0) ? bt * skk[3] * fac_2_3 : 0.0f);
                        int s_1_4 = s0_1_0 + 4;
                        float fac_2_4 = approx_exp2(fminf(gtr - gsv_1_0[4], 0.0f) * 1.442695f);
                        int on_aqk_1_4 = 0;
                        int on_l_1_4 = 0;
                        if (s_1_4 <= t_row) {
                            on_aqk_1_4 = row_valid;
                        }
                        if (s_1_4 < t_row) {
                            on_l_1_4 = row_valid;
                        }
                        aq_1_0[4] = ((on_aqk_1_4 != 0) ? sqk[4] * scale * fac_2_4 : 0.0f);
                        lvs_1_0[4] = ((on_l_1_4 != 0) ? bt * skk[4] * fac_2_4 : 0.0f);
                        int s_1_5 = s0_1_0 + 5;
                        float fac_2_5 = approx_exp2(fminf(gtr - gsv_1_0[5], 0.0f) * 1.442695f);
                        int on_aqk_1_5 = 0;
                        int on_l_1_5 = 0;
                        if (s_1_5 <= t_row) {
                            on_aqk_1_5 = row_valid;
                        }
                        if (s_1_5 < t_row) {
                            on_l_1_5 = row_valid;
                        }
                        aq_1_0[5] = ((on_aqk_1_5 != 0) ? sqk[5] * scale * fac_2_5 : 0.0f);
                        lvs_1_0[5] = ((on_l_1_5 != 0) ? bt * skk[5] * fac_2_5 : 0.0f);
                        int s_1_6 = s0_1_0 + 6;
                        float fac_2_6 = approx_exp2(fminf(gtr - gsv_1_0[6], 0.0f) * 1.442695f);
                        int on_aqk_1_6 = 0;
                        int on_l_1_6 = 0;
                        if (s_1_6 <= t_row) {
                            on_aqk_1_6 = row_valid;
                        }
                        if (s_1_6 < t_row) {
                            on_l_1_6 = row_valid;
                        }
                        aq_1_0[6] = ((on_aqk_1_6 != 0) ? sqk[6] * scale * fac_2_6 : 0.0f);
                        lvs_1_0[6] = ((on_l_1_6 != 0) ? bt * skk[6] * fac_2_6 : 0.0f);
                        int s_1_7 = s0_1_0 + 7;
                        float fac_2_7 = approx_exp2(fminf(gtr - gsv_1_0[7], 0.0f) * 1.442695f);
                        int on_aqk_1_7 = 0;
                        int on_l_1_7 = 0;
                        if (s_1_7 <= t_row) {
                            on_aqk_1_7 = row_valid;
                        }
                        if (s_1_7 < t_row) {
                            on_l_1_7 = row_valid;
                        }
                        aq_1_0[7] = ((on_aqk_1_7 != 0) ? sqk[7] * scale * fac_2_7 : 0.0f);
                        lvs_1_0[7] = ((on_l_1_7 != 0) ? bt * skk[7] * fac_2_7 : 0.0f);
                        int s_1_8 = s0_1_0 + 8;
                        float fac_2_8 = approx_exp2(fminf(gtr - gsv_1_0[8], 0.0f) * 1.442695f);
                        int on_aqk_1_8 = 0;
                        int on_l_1_8 = 0;
                        if (s_1_8 <= t_row) {
                            on_aqk_1_8 = row_valid;
                        }
                        if (s_1_8 < t_row) {
                            on_l_1_8 = row_valid;
                        }
                        aq_1_0[8] = ((on_aqk_1_8 != 0) ? sqk[8] * scale * fac_2_8 : 0.0f);
                        lvs_1_0[8] = ((on_l_1_8 != 0) ? bt * skk[8] * fac_2_8 : 0.0f);
                        int s_1_9 = s0_1_0 + 9;
                        float fac_2_9 = approx_exp2(fminf(gtr - gsv_1_0[9], 0.0f) * 1.442695f);
                        int on_aqk_1_9 = 0;
                        int on_l_1_9 = 0;
                        if (s_1_9 <= t_row) {
                            on_aqk_1_9 = row_valid;
                        }
                        if (s_1_9 < t_row) {
                            on_l_1_9 = row_valid;
                        }
                        aq_1_0[9] = ((on_aqk_1_9 != 0) ? sqk[9] * scale * fac_2_9 : 0.0f);
                        lvs_1_0[9] = ((on_l_1_9 != 0) ? bt * skk[9] * fac_2_9 : 0.0f);
                        int s_1_10 = s0_1_0 + 10;
                        float fac_2_10 = approx_exp2(fminf(gtr - gsv_1_0[10], 0.0f) * 1.442695f);
                        int on_aqk_1_10 = 0;
                        int on_l_1_10 = 0;
                        if (s_1_10 <= t_row) {
                            on_aqk_1_10 = row_valid;
                        }
                        if (s_1_10 < t_row) {
                            on_l_1_10 = row_valid;
                        }
                        aq_1_0[10] = ((on_aqk_1_10 != 0) ? sqk[10] * scale * fac_2_10 : 0.0f);
                        lvs_1_0[10] = ((on_l_1_10 != 0) ? bt * skk[10] * fac_2_10 : 0.0f);
                        int s_1_11 = s0_1_0 + 11;
                        float fac_2_11 = approx_exp2(fminf(gtr - gsv_1_0[11], 0.0f) * 1.442695f);
                        int on_aqk_1_11 = 0;
                        int on_l_1_11 = 0;
                        if (s_1_11 <= t_row) {
                            on_aqk_1_11 = row_valid;
                        }
                        if (s_1_11 < t_row) {
                            on_l_1_11 = row_valid;
                        }
                        aq_1_0[11] = ((on_aqk_1_11 != 0) ? sqk[11] * scale * fac_2_11 : 0.0f);
                        lvs_1_0[11] = ((on_l_1_11 != 0) ? bt * skk[11] * fac_2_11 : 0.0f);
                        int s_1_12 = s0_1_0 + 12;
                        float fac_2_12 = approx_exp2(fminf(gtr - gsv_1_0[12], 0.0f) * 1.442695f);
                        int on_aqk_1_12 = 0;
                        int on_l_1_12 = 0;
                        if (s_1_12 <= t_row) {
                            on_aqk_1_12 = row_valid;
                        }
                        if (s_1_12 < t_row) {
                            on_l_1_12 = row_valid;
                        }
                        aq_1_0[12] = ((on_aqk_1_12 != 0) ? sqk[12] * scale * fac_2_12 : 0.0f);
                        lvs_1_0[12] = ((on_l_1_12 != 0) ? bt * skk[12] * fac_2_12 : 0.0f);
                        int s_1_13 = s0_1_0 + 13;
                        float fac_2_13 = approx_exp2(fminf(gtr - gsv_1_0[13], 0.0f) * 1.442695f);
                        int on_aqk_1_13 = 0;
                        int on_l_1_13 = 0;
                        if (s_1_13 <= t_row) {
                            on_aqk_1_13 = row_valid;
                        }
                        if (s_1_13 < t_row) {
                            on_l_1_13 = row_valid;
                        }
                        aq_1_0[13] = ((on_aqk_1_13 != 0) ? sqk[13] * scale * fac_2_13 : 0.0f);
                        lvs_1_0[13] = ((on_l_1_13 != 0) ? bt * skk[13] * fac_2_13 : 0.0f);
                        int s_1_14 = s0_1_0 + 14;
                        float fac_2_14 = approx_exp2(fminf(gtr - gsv_1_0[14], 0.0f) * 1.442695f);
                        int on_aqk_1_14 = 0;
                        int on_l_1_14 = 0;
                        if (s_1_14 <= t_row) {
                            on_aqk_1_14 = row_valid;
                        }
                        if (s_1_14 < t_row) {
                            on_l_1_14 = row_valid;
                        }
                        aq_1_0[14] = ((on_aqk_1_14 != 0) ? sqk[14] * scale * fac_2_14 : 0.0f);
                        lvs_1_0[14] = ((on_l_1_14 != 0) ? bt * skk[14] * fac_2_14 : 0.0f);
                        int s_1_15 = s0_1_0 + 15;
                        float fac_2_15 = approx_exp2(fminf(gtr - gsv_1_0[15], 0.0f) * 1.442695f);
                        int on_aqk_1_15 = 0;
                        int on_l_1_15 = 0;
                        if (s_1_15 <= t_row) {
                            on_aqk_1_15 = row_valid;
                        }
                        if (s_1_15 < t_row) {
                            on_l_1_15 = row_valid;
                        }
                        aq_1_0[15] = ((on_aqk_1_15 != 0) ? sqk[15] * scale * fac_2_15 : 0.0f);
                        lvs_1_0[15] = ((on_l_1_15 != 0) ? bt * skk[15] * fac_2_15 : 0.0f);
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0_1_0) * 4), "f"(lvs_1_0[0]), "f"(lvs_1_0[1]), "f"(lvs_1_0[2]), "f"(lvs_1_0[3]) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0_1_0 + 4) * 4), "f"(lvs_1_0[4]), "f"(lvs_1_0[5]), "f"(lvs_1_0[6]), "f"(lvs_1_0[7]) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0_1_0 + 8) * 4), "f"(lvs_1_0[8]), "f"(lvs_1_0[9]), "f"(lvs_1_0[10]), "f"(lvs_1_0[11]) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0_1_0 + 12) * 4), "f"(lvs_1_0[12]), "f"(lvs_1_0[13]), "f"(lvs_1_0[14]), "f"(lvs_1_0[15]) : "memory");
                        unsigned int _packed_0[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 0], aq_1_0[_lp*2+1 + 0]));
                            _packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 2], aq_1_0[_lp*2+1 + 2]));
                            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 4], aq_1_0[_lp*2+1 + 4]));
                            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 6], aq_1_0[_lp*2+1 + 6]));
                            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 8], aq_1_0[_lp*2+1 + 8]));
                            _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 10], aq_1_0[_lp*2+1 + 10]));
                            _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 12], aq_1_0[_lp*2+1 + 12]));
                            _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 14], aq_1_0[_lp*2+1 + 14]));
                            _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(_packed_0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (aqk_base + (long long)s0_1_0))), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
                        }
                        int s0_1_1 = 32 + half * 16;
                        float gsv_1_1[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(0) + 3]))
                            : "r"(gc_base + s0_1_1 * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(4) + 3]))
                            : "r"(gc_base + (s0_1_1 + 4) * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(8) + 3]))
                            : "r"(gc_base + (s0_1_1 + 8) * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(12) + 3]))
                            : "r"(gc_base + (s0_1_1 + 12) * 4));
                        float aq_1_1[16];
                        float lvs_1_1[16];
                        int s_1_16 = s0_1_1;
                        float fac_2_16 = approx_exp2(fminf(gtr - gsv_1_1[0], 0.0f) * 1.442695f);
                        int on_aqk_1_16 = 0;
                        int on_l_1_16 = 0;
                        if (s_1_16 <= t_row) {
                            on_aqk_1_16 = row_valid;
                        }
                        if (s_1_16 < t_row) {
                            on_l_1_16 = row_valid;
                        }
                        aq_1_1[0] = ((on_aqk_1_16 != 0) ? sqk[16] * scale * fac_2_16 : 0.0f);
                        lvs_1_1[0] = ((on_l_1_16 != 0) ? bt * skk[16] * fac_2_16 : 0.0f);
                        int s_1_17 = s0_1_1 + 1;
                        float fac_2_17 = approx_exp2(fminf(gtr - gsv_1_1[1], 0.0f) * 1.442695f);
                        int on_aqk_1_17 = 0;
                        int on_l_1_17 = 0;
                        if (s_1_17 <= t_row) {
                            on_aqk_1_17 = row_valid;
                        }
                        if (s_1_17 < t_row) {
                            on_l_1_17 = row_valid;
                        }
                        aq_1_1[1] = ((on_aqk_1_17 != 0) ? sqk[17] * scale * fac_2_17 : 0.0f);
                        lvs_1_1[1] = ((on_l_1_17 != 0) ? bt * skk[17] * fac_2_17 : 0.0f);
                        int s_1_18 = s0_1_1 + 2;
                        float fac_2_18 = approx_exp2(fminf(gtr - gsv_1_1[2], 0.0f) * 1.442695f);
                        int on_aqk_1_18 = 0;
                        int on_l_1_18 = 0;
                        if (s_1_18 <= t_row) {
                            on_aqk_1_18 = row_valid;
                        }
                        if (s_1_18 < t_row) {
                            on_l_1_18 = row_valid;
                        }
                        aq_1_1[2] = ((on_aqk_1_18 != 0) ? sqk[18] * scale * fac_2_18 : 0.0f);
                        lvs_1_1[2] = ((on_l_1_18 != 0) ? bt * skk[18] * fac_2_18 : 0.0f);
                        int s_1_19 = s0_1_1 + 3;
                        float fac_2_19 = approx_exp2(fminf(gtr - gsv_1_1[3], 0.0f) * 1.442695f);
                        int on_aqk_1_19 = 0;
                        int on_l_1_19 = 0;
                        if (s_1_19 <= t_row) {
                            on_aqk_1_19 = row_valid;
                        }
                        if (s_1_19 < t_row) {
                            on_l_1_19 = row_valid;
                        }
                        aq_1_1[3] = ((on_aqk_1_19 != 0) ? sqk[19] * scale * fac_2_19 : 0.0f);
                        lvs_1_1[3] = ((on_l_1_19 != 0) ? bt * skk[19] * fac_2_19 : 0.0f);
                        int s_1_20 = s0_1_1 + 4;
                        float fac_2_20 = approx_exp2(fminf(gtr - gsv_1_1[4], 0.0f) * 1.442695f);
                        int on_aqk_1_20 = 0;
                        int on_l_1_20 = 0;
                        if (s_1_20 <= t_row) {
                            on_aqk_1_20 = row_valid;
                        }
                        if (s_1_20 < t_row) {
                            on_l_1_20 = row_valid;
                        }
                        aq_1_1[4] = ((on_aqk_1_20 != 0) ? sqk[20] * scale * fac_2_20 : 0.0f);
                        lvs_1_1[4] = ((on_l_1_20 != 0) ? bt * skk[20] * fac_2_20 : 0.0f);
                        int s_1_21 = s0_1_1 + 5;
                        float fac_2_21 = approx_exp2(fminf(gtr - gsv_1_1[5], 0.0f) * 1.442695f);
                        int on_aqk_1_21 = 0;
                        int on_l_1_21 = 0;
                        if (s_1_21 <= t_row) {
                            on_aqk_1_21 = row_valid;
                        }
                        if (s_1_21 < t_row) {
                            on_l_1_21 = row_valid;
                        }
                        aq_1_1[5] = ((on_aqk_1_21 != 0) ? sqk[21] * scale * fac_2_21 : 0.0f);
                        lvs_1_1[5] = ((on_l_1_21 != 0) ? bt * skk[21] * fac_2_21 : 0.0f);
                        int s_1_22 = s0_1_1 + 6;
                        float fac_2_22 = approx_exp2(fminf(gtr - gsv_1_1[6], 0.0f) * 1.442695f);
                        int on_aqk_1_22 = 0;
                        int on_l_1_22 = 0;
                        if (s_1_22 <= t_row) {
                            on_aqk_1_22 = row_valid;
                        }
                        if (s_1_22 < t_row) {
                            on_l_1_22 = row_valid;
                        }
                        aq_1_1[6] = ((on_aqk_1_22 != 0) ? sqk[22] * scale * fac_2_22 : 0.0f);
                        lvs_1_1[6] = ((on_l_1_22 != 0) ? bt * skk[22] * fac_2_22 : 0.0f);
                        int s_1_23 = s0_1_1 + 7;
                        float fac_2_23 = approx_exp2(fminf(gtr - gsv_1_1[7], 0.0f) * 1.442695f);
                        int on_aqk_1_23 = 0;
                        int on_l_1_23 = 0;
                        if (s_1_23 <= t_row) {
                            on_aqk_1_23 = row_valid;
                        }
                        if (s_1_23 < t_row) {
                            on_l_1_23 = row_valid;
                        }
                        aq_1_1[7] = ((on_aqk_1_23 != 0) ? sqk[23] * scale * fac_2_23 : 0.0f);
                        lvs_1_1[7] = ((on_l_1_23 != 0) ? bt * skk[23] * fac_2_23 : 0.0f);
                        int s_1_24 = s0_1_1 + 8;
                        float fac_2_24 = approx_exp2(fminf(gtr - gsv_1_1[8], 0.0f) * 1.442695f);
                        int on_aqk_1_24 = 0;
                        int on_l_1_24 = 0;
                        if (s_1_24 <= t_row) {
                            on_aqk_1_24 = row_valid;
                        }
                        if (s_1_24 < t_row) {
                            on_l_1_24 = row_valid;
                        }
                        aq_1_1[8] = ((on_aqk_1_24 != 0) ? sqk[24] * scale * fac_2_24 : 0.0f);
                        lvs_1_1[8] = ((on_l_1_24 != 0) ? bt * skk[24] * fac_2_24 : 0.0f);
                        int s_1_25 = s0_1_1 + 9;
                        float fac_2_25 = approx_exp2(fminf(gtr - gsv_1_1[9], 0.0f) * 1.442695f);
                        int on_aqk_1_25 = 0;
                        int on_l_1_25 = 0;
                        if (s_1_25 <= t_row) {
                            on_aqk_1_25 = row_valid;
                        }
                        if (s_1_25 < t_row) {
                            on_l_1_25 = row_valid;
                        }
                        aq_1_1[9] = ((on_aqk_1_25 != 0) ? sqk[25] * scale * fac_2_25 : 0.0f);
                        lvs_1_1[9] = ((on_l_1_25 != 0) ? bt * skk[25] * fac_2_25 : 0.0f);
                        int s_1_26 = s0_1_1 + 10;
                        float fac_2_26 = approx_exp2(fminf(gtr - gsv_1_1[10], 0.0f) * 1.442695f);
                        int on_aqk_1_26 = 0;
                        int on_l_1_26 = 0;
                        if (s_1_26 <= t_row) {
                            on_aqk_1_26 = row_valid;
                        }
                        if (s_1_26 < t_row) {
                            on_l_1_26 = row_valid;
                        }
                        aq_1_1[10] = ((on_aqk_1_26 != 0) ? sqk[26] * scale * fac_2_26 : 0.0f);
                        lvs_1_1[10] = ((on_l_1_26 != 0) ? bt * skk[26] * fac_2_26 : 0.0f);
                        int s_1_27 = s0_1_1 + 11;
                        float fac_2_27 = approx_exp2(fminf(gtr - gsv_1_1[11], 0.0f) * 1.442695f);
                        int on_aqk_1_27 = 0;
                        int on_l_1_27 = 0;
                        if (s_1_27 <= t_row) {
                            on_aqk_1_27 = row_valid;
                        }
                        if (s_1_27 < t_row) {
                            on_l_1_27 = row_valid;
                        }
                        aq_1_1[11] = ((on_aqk_1_27 != 0) ? sqk[27] * scale * fac_2_27 : 0.0f);
                        lvs_1_1[11] = ((on_l_1_27 != 0) ? bt * skk[27] * fac_2_27 : 0.0f);
                        int s_1_28 = s0_1_1 + 12;
                        float fac_2_28 = approx_exp2(fminf(gtr - gsv_1_1[12], 0.0f) * 1.442695f);
                        int on_aqk_1_28 = 0;
                        int on_l_1_28 = 0;
                        if (s_1_28 <= t_row) {
                            on_aqk_1_28 = row_valid;
                        }
                        if (s_1_28 < t_row) {
                            on_l_1_28 = row_valid;
                        }
                        aq_1_1[12] = ((on_aqk_1_28 != 0) ? sqk[28] * scale * fac_2_28 : 0.0f);
                        lvs_1_1[12] = ((on_l_1_28 != 0) ? bt * skk[28] * fac_2_28 : 0.0f);
                        int s_1_29 = s0_1_1 + 13;
                        float fac_2_29 = approx_exp2(fminf(gtr - gsv_1_1[13], 0.0f) * 1.442695f);
                        int on_aqk_1_29 = 0;
                        int on_l_1_29 = 0;
                        if (s_1_29 <= t_row) {
                            on_aqk_1_29 = row_valid;
                        }
                        if (s_1_29 < t_row) {
                            on_l_1_29 = row_valid;
                        }
                        aq_1_1[13] = ((on_aqk_1_29 != 0) ? sqk[29] * scale * fac_2_29 : 0.0f);
                        lvs_1_1[13] = ((on_l_1_29 != 0) ? bt * skk[29] * fac_2_29 : 0.0f);
                        int s_1_30 = s0_1_1 + 14;
                        float fac_2_30 = approx_exp2(fminf(gtr - gsv_1_1[14], 0.0f) * 1.442695f);
                        int on_aqk_1_30 = 0;
                        int on_l_1_30 = 0;
                        if (s_1_30 <= t_row) {
                            on_aqk_1_30 = row_valid;
                        }
                        if (s_1_30 < t_row) {
                            on_l_1_30 = row_valid;
                        }
                        aq_1_1[14] = ((on_aqk_1_30 != 0) ? sqk[30] * scale * fac_2_30 : 0.0f);
                        lvs_1_1[14] = ((on_l_1_30 != 0) ? bt * skk[30] * fac_2_30 : 0.0f);
                        int s_1_31 = s0_1_1 + 15;
                        float fac_2_31 = approx_exp2(fminf(gtr - gsv_1_1[15], 0.0f) * 1.442695f);
                        int on_aqk_1_31 = 0;
                        int on_l_1_31 = 0;
                        if (s_1_31 <= t_row) {
                            on_aqk_1_31 = row_valid;
                        }
                        if (s_1_31 < t_row) {
                            on_l_1_31 = row_valid;
                        }
                        aq_1_1[15] = ((on_aqk_1_31 != 0) ? sqk[31] * scale * fac_2_31 : 0.0f);
                        lvs_1_1[15] = ((on_l_1_31 != 0) ? bt * skk[31] * fac_2_31 : 0.0f);
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0_1_1) * 4), "f"(lvs_1_1[0]), "f"(lvs_1_1[1]), "f"(lvs_1_1[2]), "f"(lvs_1_1[3]) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0_1_1 + 4) * 4), "f"(lvs_1_1[4]), "f"(lvs_1_1[5]), "f"(lvs_1_1[6]), "f"(lvs_1_1[7]) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0_1_1 + 8) * 4), "f"(lvs_1_1[8]), "f"(lvs_1_1[9]), "f"(lvs_1_1[10]), "f"(lvs_1_1[11]) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0_1_1 + 12) * 4), "f"(lvs_1_1[12]), "f"(lvs_1_1[13]), "f"(lvs_1_1[14]), "f"(lvs_1_1[15]) : "memory");
                        unsigned int _packed_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 0], aq_1_1[_lp*2+1 + 0]));
                            _packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 2], aq_1_1[_lp*2+1 + 2]));
                            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 4], aq_1_1[_lp*2+1 + 4]));
                            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 6], aq_1_1[_lp*2+1 + 6]));
                            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 8], aq_1_1[_lp*2+1 + 8]));
                            _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 10], aq_1_1[_lp*2+1 + 10]));
                            _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 12], aq_1_1[_lp*2+1 + 12]));
                            _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 14], aq_1_1[_lp*2+1 + 14]));
                            _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(_packed_1);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (aqk_base + (long long)s0_1_1))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
                        }
                        int dcol0 = (t_row >> 4) * 16 + half * 8;
                        float dvs[8];
                        dvs[0] = ((dcol0 == t_row) ? 1.0f : 0.0f);
                        dvs[1] = ((dcol0 + 1 == t_row) ? 1.0f : 0.0f);
                        dvs[2] = ((dcol0 + 2 == t_row) ? 1.0f : 0.0f);
                        dvs[3] = ((dcol0 + 3 == t_row) ? 1.0f : 0.0f);
                        dvs[4] = ((dcol0 + 4 == t_row) ? 1.0f : 0.0f);
                        dvs[5] = ((dcol0 + 5 == t_row) ? 1.0f : 0.0f);
                        dvs[6] = ((dcol0 + 6 == t_row) ? 1.0f : 0.0f);
                        dvs[7] = ((dcol0 + 7 == t_row) ? 1.0f : 0.0f);
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(a_base + (t_row * 68 + dcol0) * 4), "f"(dvs[0]), "f"(dvs[1]), "f"(dvs[2]), "f"(dvs[3]) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(a_base + (t_row * 68 + dcol0 + 4) * 4), "f"(dvs[4]), "f"(dvs[5]), "f"(dvs[6]), "f"(dvs[7]) : "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid_0 < 64) {
                            int bb = tid_0 >> 4;
                            int jj = tid_0 & 15;
                            int l_row0 = l_base + (bb * 16 * 68 + bb * 16) * 4;
                            int a_col = a_base + (bb * 16 * 68 + bb * 16 + jj) * 4;
                            float a_vals[16];
                            a_vals[0] = ((0 == jj) ? 1.0f : 0.0f);
                            a_vals[1] = ((1 == jj) ? 1.0f : 0.0f);
                            a_vals[2] = ((2 == jj) ? 1.0f : 0.0f);
                            a_vals[3] = ((3 == jj) ? 1.0f : 0.0f);
                            a_vals[4] = ((4 == jj) ? 1.0f : 0.0f);
                            a_vals[5] = ((5 == jj) ? 1.0f : 0.0f);
                            a_vals[6] = ((6 == jj) ? 1.0f : 0.0f);
                            a_vals[7] = ((7 == jj) ? 1.0f : 0.0f);
                            a_vals[8] = ((8 == jj) ? 1.0f : 0.0f);
                            a_vals[9] = ((9 == jj) ? 1.0f : 0.0f);
                            a_vals[10] = ((10 == jj) ? 1.0f : 0.0f);
                            a_vals[11] = ((11 == jj) ? 1.0f : 0.0f);
                            a_vals[12] = ((12 == jj) ? 1.0f : 0.0f);
                            a_vals[13] = ((13 == jj) ? 1.0f : 0.0f);
                            a_vals[14] = ((14 == jj) ? 1.0f : 0.0f);
                            a_vals[15] = ((15 == jj) ? 1.0f : 0.0f);
                            float lrow[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow[(0) + 3]))
                                : "r"(l_row0 + 272));
                            float accv = 0.0f;
                            accv = __fmaf_rn(lrow[0], a_vals[0], accv);
                            float negv = -accv;
                            a_vals[1] = ((jj < 1) ? negv : a_vals[1]);
                            if (jj < 1) {
                                {
                                    uint32_t _addr_2 = static_cast<uint32_t>(a_col + 272);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_2), "f"(negv) : "memory");
                                }
                            }
                            float lrow_r_0[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_0[(0) + 3]))
                                : "r"(l_row0 + 544));
                            float accv_r_0 = 0.0f;
                            accv_r_0 = __fmaf_rn(lrow_r_0[0], a_vals[0], accv_r_0);
                            accv_r_0 = __fmaf_rn(lrow_r_0[1], a_vals[1], accv_r_0);
                            float negv_r_0 = -accv_r_0;
                            a_vals[2] = ((jj < 2) ? negv_r_0 : a_vals[2]);
                            if (jj < 2) {
                                {
                                    uint32_t _addr_3 = static_cast<uint32_t>(a_col + 544);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3), "f"(negv_r_0) : "memory");
                                }
                            }
                            float lrow_r_1[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_1[(0) + 3]))
                                : "r"(l_row0 + 816));
                            float accv_r_1 = 0.0f;
                            accv_r_1 = __fmaf_rn(lrow_r_1[0], a_vals[0], accv_r_1);
                            accv_r_1 = __fmaf_rn(lrow_r_1[1], a_vals[1], accv_r_1);
                            accv_r_1 = __fmaf_rn(lrow_r_1[2], a_vals[2], accv_r_1);
                            float negv_r_1 = -accv_r_1;
                            a_vals[3] = ((jj < 3) ? negv_r_1 : a_vals[3]);
                            if (jj < 3) {
                                {
                                    uint32_t _addr_4 = static_cast<uint32_t>(a_col + 816);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_4), "f"(negv_r_1) : "memory");
                                }
                            }
                            float lrow_r_2[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_2[(0) + 3]))
                                : "r"(l_row0 + 1088));
                            float accv_r_2 = 0.0f;
                            accv_r_2 = __fmaf_rn(lrow_r_2[0], a_vals[0], accv_r_2);
                            accv_r_2 = __fmaf_rn(lrow_r_2[1], a_vals[1], accv_r_2);
                            accv_r_2 = __fmaf_rn(lrow_r_2[2], a_vals[2], accv_r_2);
                            accv_r_2 = __fmaf_rn(lrow_r_2[3], a_vals[3], accv_r_2);
                            float negv_r_2 = -accv_r_2;
                            a_vals[4] = ((jj < 4) ? negv_r_2 : a_vals[4]);
                            if (jj < 4) {
                                {
                                    uint32_t _addr_5 = static_cast<uint32_t>(a_col + 1088);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_5), "f"(negv_r_2) : "memory");
                                }
                            }
                            float lrow_r_r_0[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_0[(0) + 3]))
                                : "r"(l_row0 + 1360));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_0[(4) + 3]))
                                : "r"(l_row0 + 1360 + 16));
                            float accv_r_r_0 = 0.0f;
                            accv_r_r_0 = __fmaf_rn(lrow_r_r_0[0], a_vals[0], accv_r_r_0);
                            accv_r_r_0 = __fmaf_rn(lrow_r_r_0[1], a_vals[1], accv_r_r_0);
                            accv_r_r_0 = __fmaf_rn(lrow_r_r_0[2], a_vals[2], accv_r_r_0);
                            accv_r_r_0 = __fmaf_rn(lrow_r_r_0[3], a_vals[3], accv_r_r_0);
                            accv_r_r_0 = __fmaf_rn(lrow_r_r_0[4], a_vals[4], accv_r_r_0);
                            float negv_r_r_0 = -accv_r_r_0;
                            a_vals[5] = ((jj < 5) ? negv_r_r_0 : a_vals[5]);
                            if (jj < 5) {
                                {
                                    uint32_t _addr_6 = static_cast<uint32_t>(a_col + 1360);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_6), "f"(negv_r_r_0) : "memory");
                                }
                            }
                            float lrow_r_r_1[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_1[(0) + 3]))
                                : "r"(l_row0 + 1632));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_1[(4) + 3]))
                                : "r"(l_row0 + 1632 + 16));
                            float accv_r_r_1 = 0.0f;
                            accv_r_r_1 = __fmaf_rn(lrow_r_r_1[0], a_vals[0], accv_r_r_1);
                            accv_r_r_1 = __fmaf_rn(lrow_r_r_1[1], a_vals[1], accv_r_r_1);
                            accv_r_r_1 = __fmaf_rn(lrow_r_r_1[2], a_vals[2], accv_r_r_1);
                            accv_r_r_1 = __fmaf_rn(lrow_r_r_1[3], a_vals[3], accv_r_r_1);
                            accv_r_r_1 = __fmaf_rn(lrow_r_r_1[4], a_vals[4], accv_r_r_1);
                            accv_r_r_1 = __fmaf_rn(lrow_r_r_1[5], a_vals[5], accv_r_r_1);
                            float negv_r_r_1 = -accv_r_r_1;
                            a_vals[6] = ((jj < 6) ? negv_r_r_1 : a_vals[6]);
                            if (jj < 6) {
                                {
                                    uint32_t _addr_7 = static_cast<uint32_t>(a_col + 1632);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(negv_r_r_1) : "memory");
                                }
                            }
                            float lrow_r_r_2[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_2[(0) + 3]))
                                : "r"(l_row0 + 1904));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_2[(4) + 3]))
                                : "r"(l_row0 + 1904 + 16));
                            float accv_r_r_2 = 0.0f;
                            accv_r_r_2 = __fmaf_rn(lrow_r_r_2[0], a_vals[0], accv_r_r_2);
                            accv_r_r_2 = __fmaf_rn(lrow_r_r_2[1], a_vals[1], accv_r_r_2);
                            accv_r_r_2 = __fmaf_rn(lrow_r_r_2[2], a_vals[2], accv_r_r_2);
                            accv_r_r_2 = __fmaf_rn(lrow_r_r_2[3], a_vals[3], accv_r_r_2);
                            accv_r_r_2 = __fmaf_rn(lrow_r_r_2[4], a_vals[4], accv_r_r_2);
                            accv_r_r_2 = __fmaf_rn(lrow_r_r_2[5], a_vals[5], accv_r_r_2);
                            accv_r_r_2 = __fmaf_rn(lrow_r_r_2[6], a_vals[6], accv_r_r_2);
                            float negv_r_r_2 = -accv_r_r_2;
                            a_vals[7] = ((jj < 7) ? negv_r_r_2 : a_vals[7]);
                            if (jj < 7) {
                                {
                                    uint32_t _addr_8 = static_cast<uint32_t>(a_col + 1904);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_8), "f"(negv_r_r_2) : "memory");
                                }
                            }
                            float lrow_r_r_3[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_3[(0) + 3]))
                                : "r"(l_row0 + 2176));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_3[(4) + 3]))
                                : "r"(l_row0 + 2176 + 16));
                            float accv_r_r_3 = 0.0f;
                            accv_r_r_3 = __fmaf_rn(lrow_r_r_3[0], a_vals[0], accv_r_r_3);
                            accv_r_r_3 = __fmaf_rn(lrow_r_r_3[1], a_vals[1], accv_r_r_3);
                            accv_r_r_3 = __fmaf_rn(lrow_r_r_3[2], a_vals[2], accv_r_r_3);
                            accv_r_r_3 = __fmaf_rn(lrow_r_r_3[3], a_vals[3], accv_r_r_3);
                            accv_r_r_3 = __fmaf_rn(lrow_r_r_3[4], a_vals[4], accv_r_r_3);
                            accv_r_r_3 = __fmaf_rn(lrow_r_r_3[5], a_vals[5], accv_r_r_3);
                            accv_r_r_3 = __fmaf_rn(lrow_r_r_3[6], a_vals[6], accv_r_r_3);
                            accv_r_r_3 = __fmaf_rn(lrow_r_r_3[7], a_vals[7], accv_r_r_3);
                            float negv_r_r_3 = -accv_r_r_3;
                            a_vals[8] = ((jj < 8) ? negv_r_r_3 : a_vals[8]);
                            if (jj < 8) {
                                {
                                    uint32_t _addr_9 = static_cast<uint32_t>(a_col + 2176);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(negv_r_r_3) : "memory");
                                }
                            }
                            float lrow_r_r_r_0[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_0[(0) + 3]))
                                : "r"(l_row0 + 2448));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_0[(4) + 3]))
                                : "r"(l_row0 + 2448 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_0[(8) + 3]))
                                : "r"(l_row0 + 2448 + 32));
                            float accv_r_r_r_0 = 0.0f;
                            accv_r_r_r_0 = __fmaf_rn(lrow_r_r_r_0[0], a_vals[0], accv_r_r_r_0);
                            accv_r_r_r_0 = __fmaf_rn(lrow_r_r_r_0[1], a_vals[1], accv_r_r_r_0);
                            accv_r_r_r_0 = __fmaf_rn(lrow_r_r_r_0[2], a_vals[2], accv_r_r_r_0);
                            accv_r_r_r_0 = __fmaf_rn(lrow_r_r_r_0[3], a_vals[3], accv_r_r_r_0);
                            accv_r_r_r_0 = __fmaf_rn(lrow_r_r_r_0[4], a_vals[4], accv_r_r_r_0);
                            accv_r_r_r_0 = __fmaf_rn(lrow_r_r_r_0[5], a_vals[5], accv_r_r_r_0);
                            accv_r_r_r_0 = __fmaf_rn(lrow_r_r_r_0[6], a_vals[6], accv_r_r_r_0);
                            accv_r_r_r_0 = __fmaf_rn(lrow_r_r_r_0[7], a_vals[7], accv_r_r_r_0);
                            accv_r_r_r_0 = __fmaf_rn(lrow_r_r_r_0[8], a_vals[8], accv_r_r_r_0);
                            float negv_r_r_r_0 = -accv_r_r_r_0;
                            a_vals[9] = ((jj < 9) ? negv_r_r_r_0 : a_vals[9]);
                            if (jj < 9) {
                                {
                                    uint32_t _addr_10 = static_cast<uint32_t>(a_col + 2448);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(negv_r_r_r_0) : "memory");
                                }
                            }
                            float lrow_r_r_r_1[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_1[(0) + 3]))
                                : "r"(l_row0 + 2720));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_1[(4) + 3]))
                                : "r"(l_row0 + 2720 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_1[(8) + 3]))
                                : "r"(l_row0 + 2720 + 32));
                            float accv_r_r_r_1 = 0.0f;
                            accv_r_r_r_1 = __fmaf_rn(lrow_r_r_r_1[0], a_vals[0], accv_r_r_r_1);
                            accv_r_r_r_1 = __fmaf_rn(lrow_r_r_r_1[1], a_vals[1], accv_r_r_r_1);
                            accv_r_r_r_1 = __fmaf_rn(lrow_r_r_r_1[2], a_vals[2], accv_r_r_r_1);
                            accv_r_r_r_1 = __fmaf_rn(lrow_r_r_r_1[3], a_vals[3], accv_r_r_r_1);
                            accv_r_r_r_1 = __fmaf_rn(lrow_r_r_r_1[4], a_vals[4], accv_r_r_r_1);
                            accv_r_r_r_1 = __fmaf_rn(lrow_r_r_r_1[5], a_vals[5], accv_r_r_r_1);
                            accv_r_r_r_1 = __fmaf_rn(lrow_r_r_r_1[6], a_vals[6], accv_r_r_r_1);
                            accv_r_r_r_1 = __fmaf_rn(lrow_r_r_r_1[7], a_vals[7], accv_r_r_r_1);
                            accv_r_r_r_1 = __fmaf_rn(lrow_r_r_r_1[8], a_vals[8], accv_r_r_r_1);
                            accv_r_r_r_1 = __fmaf_rn(lrow_r_r_r_1[9], a_vals[9], accv_r_r_r_1);
                            float negv_r_r_r_1 = -accv_r_r_r_1;
                            a_vals[10] = ((jj < 10) ? negv_r_r_r_1 : a_vals[10]);
                            if (jj < 10) {
                                {
                                    uint32_t _addr_11 = static_cast<uint32_t>(a_col + 2720);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(negv_r_r_r_1) : "memory");
                                }
                            }
                            float lrow_r_r_r_2[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_2[(0) + 3]))
                                : "r"(l_row0 + 2992));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_2[(4) + 3]))
                                : "r"(l_row0 + 2992 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_2[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_2[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_2[(8) + 3]))
                                : "r"(l_row0 + 2992 + 32));
                            float accv_r_r_r_2 = 0.0f;
                            accv_r_r_r_2 = __fmaf_rn(lrow_r_r_r_2[0], a_vals[0], accv_r_r_r_2);
                            accv_r_r_r_2 = __fmaf_rn(lrow_r_r_r_2[1], a_vals[1], accv_r_r_r_2);
                            accv_r_r_r_2 = __fmaf_rn(lrow_r_r_r_2[2], a_vals[2], accv_r_r_r_2);
                            accv_r_r_r_2 = __fmaf_rn(lrow_r_r_r_2[3], a_vals[3], accv_r_r_r_2);
                            accv_r_r_r_2 = __fmaf_rn(lrow_r_r_r_2[4], a_vals[4], accv_r_r_r_2);
                            accv_r_r_r_2 = __fmaf_rn(lrow_r_r_r_2[5], a_vals[5], accv_r_r_r_2);
                            accv_r_r_r_2 = __fmaf_rn(lrow_r_r_r_2[6], a_vals[6], accv_r_r_r_2);
                            accv_r_r_r_2 = __fmaf_rn(lrow_r_r_r_2[7], a_vals[7], accv_r_r_r_2);
                            accv_r_r_r_2 = __fmaf_rn(lrow_r_r_r_2[8], a_vals[8], accv_r_r_r_2);
                            accv_r_r_r_2 = __fmaf_rn(lrow_r_r_r_2[9], a_vals[9], accv_r_r_r_2);
                            accv_r_r_r_2 = __fmaf_rn(lrow_r_r_r_2[10], a_vals[10], accv_r_r_r_2);
                            float negv_r_r_r_2 = -accv_r_r_r_2;
                            a_vals[11] = ((jj < 11) ? negv_r_r_r_2 : a_vals[11]);
                            if (jj < 11) {
                                {
                                    uint32_t _addr_12 = static_cast<uint32_t>(a_col + 2992);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(negv_r_r_r_2) : "memory");
                                }
                            }
                            float lrow_r_r_r_3[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_3[(0) + 3]))
                                : "r"(l_row0 + 3264));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_3[(4) + 3]))
                                : "r"(l_row0 + 3264 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_3[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_3[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_3[(8) + 3]))
                                : "r"(l_row0 + 3264 + 32));
                            float accv_r_r_r_3 = 0.0f;
                            accv_r_r_r_3 = __fmaf_rn(lrow_r_r_r_3[0], a_vals[0], accv_r_r_r_3);
                            accv_r_r_r_3 = __fmaf_rn(lrow_r_r_r_3[1], a_vals[1], accv_r_r_r_3);
                            accv_r_r_r_3 = __fmaf_rn(lrow_r_r_r_3[2], a_vals[2], accv_r_r_r_3);
                            accv_r_r_r_3 = __fmaf_rn(lrow_r_r_r_3[3], a_vals[3], accv_r_r_r_3);
                            accv_r_r_r_3 = __fmaf_rn(lrow_r_r_r_3[4], a_vals[4], accv_r_r_r_3);
                            accv_r_r_r_3 = __fmaf_rn(lrow_r_r_r_3[5], a_vals[5], accv_r_r_r_3);
                            accv_r_r_r_3 = __fmaf_rn(lrow_r_r_r_3[6], a_vals[6], accv_r_r_r_3);
                            accv_r_r_r_3 = __fmaf_rn(lrow_r_r_r_3[7], a_vals[7], accv_r_r_r_3);
                            accv_r_r_r_3 = __fmaf_rn(lrow_r_r_r_3[8], a_vals[8], accv_r_r_r_3);
                            accv_r_r_r_3 = __fmaf_rn(lrow_r_r_r_3[9], a_vals[9], accv_r_r_r_3);
                            accv_r_r_r_3 = __fmaf_rn(lrow_r_r_r_3[10], a_vals[10], accv_r_r_r_3);
                            accv_r_r_r_3 = __fmaf_rn(lrow_r_r_r_3[11], a_vals[11], accv_r_r_r_3);
                            float negv_r_r_r_3 = -accv_r_r_r_3;
                            a_vals[12] = ((jj < 12) ? negv_r_r_r_3 : a_vals[12]);
                            if (jj < 12) {
                                {
                                    uint32_t _addr_13 = static_cast<uint32_t>(a_col + 3264);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(negv_r_r_r_3) : "memory");
                                }
                            }
                            float lrow_r_r_r_r_0[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[(0) + 3]))
                                : "r"(l_row0 + 3536));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[(4) + 3]))
                                : "r"(l_row0 + 3536 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[(8) + 3]))
                                : "r"(l_row0 + 3536 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_0[(12) + 3]))
                                : "r"(l_row0 + 3536 + 48));
                            float accv_r_r_r_r_0 = 0.0f;
                            accv_r_r_r_r_0 = __fmaf_rn(lrow_r_r_r_r_0[0], a_vals[0], accv_r_r_r_r_0);
                            accv_r_r_r_r_0 = __fmaf_rn(lrow_r_r_r_r_0[1], a_vals[1], accv_r_r_r_r_0);
                            accv_r_r_r_r_0 = __fmaf_rn(lrow_r_r_r_r_0[2], a_vals[2], accv_r_r_r_r_0);
                            accv_r_r_r_r_0 = __fmaf_rn(lrow_r_r_r_r_0[3], a_vals[3], accv_r_r_r_r_0);
                            accv_r_r_r_r_0 = __fmaf_rn(lrow_r_r_r_r_0[4], a_vals[4], accv_r_r_r_r_0);
                            accv_r_r_r_r_0 = __fmaf_rn(lrow_r_r_r_r_0[5], a_vals[5], accv_r_r_r_r_0);
                            accv_r_r_r_r_0 = __fmaf_rn(lrow_r_r_r_r_0[6], a_vals[6], accv_r_r_r_r_0);
                            accv_r_r_r_r_0 = __fmaf_rn(lrow_r_r_r_r_0[7], a_vals[7], accv_r_r_r_r_0);
                            accv_r_r_r_r_0 = __fmaf_rn(lrow_r_r_r_r_0[8], a_vals[8], accv_r_r_r_r_0);
                            accv_r_r_r_r_0 = __fmaf_rn(lrow_r_r_r_r_0[9], a_vals[9], accv_r_r_r_r_0);
                            accv_r_r_r_r_0 = __fmaf_rn(lrow_r_r_r_r_0[10], a_vals[10], accv_r_r_r_r_0);
                            accv_r_r_r_r_0 = __fmaf_rn(lrow_r_r_r_r_0[11], a_vals[11], accv_r_r_r_r_0);
                            accv_r_r_r_r_0 = __fmaf_rn(lrow_r_r_r_r_0[12], a_vals[12], accv_r_r_r_r_0);
                            float negv_r_r_r_r_0 = -accv_r_r_r_r_0;
                            a_vals[13] = ((jj < 13) ? negv_r_r_r_r_0 : a_vals[13]);
                            if (jj < 13) {
                                {
                                    uint32_t _addr_14 = static_cast<uint32_t>(a_col + 3536);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(negv_r_r_r_r_0) : "memory");
                                }
                            }
                            float lrow_r_r_r_r_1[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[(0) + 3]))
                                : "r"(l_row0 + 3808));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[(4) + 3]))
                                : "r"(l_row0 + 3808 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[(8) + 3]))
                                : "r"(l_row0 + 3808 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_1[(12) + 3]))
                                : "r"(l_row0 + 3808 + 48));
                            float accv_r_r_r_r_1 = 0.0f;
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[0], a_vals[0], accv_r_r_r_r_1);
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[1], a_vals[1], accv_r_r_r_r_1);
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[2], a_vals[2], accv_r_r_r_r_1);
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[3], a_vals[3], accv_r_r_r_r_1);
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[4], a_vals[4], accv_r_r_r_r_1);
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[5], a_vals[5], accv_r_r_r_r_1);
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[6], a_vals[6], accv_r_r_r_r_1);
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[7], a_vals[7], accv_r_r_r_r_1);
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[8], a_vals[8], accv_r_r_r_r_1);
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[9], a_vals[9], accv_r_r_r_r_1);
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[10], a_vals[10], accv_r_r_r_r_1);
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[11], a_vals[11], accv_r_r_r_r_1);
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[12], a_vals[12], accv_r_r_r_r_1);
                            accv_r_r_r_r_1 = __fmaf_rn(lrow_r_r_r_r_1[13], a_vals[13], accv_r_r_r_r_1);
                            float negv_r_r_r_r_1 = -accv_r_r_r_r_1;
                            a_vals[14] = ((jj < 14) ? negv_r_r_r_r_1 : a_vals[14]);
                            if (jj < 14) {
                                {
                                    uint32_t _addr_15 = static_cast<uint32_t>(a_col + 3808);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(negv_r_r_r_r_1) : "memory");
                                }
                            }
                            float lrow_r_r_r_r_2[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[(0) + 3]))
                                : "r"(l_row0 + 4080));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[(4) + 3]))
                                : "r"(l_row0 + 4080 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[(8) + 3]))
                                : "r"(l_row0 + 4080 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_r_r_r_r_2[(12) + 3]))
                                : "r"(l_row0 + 4080 + 48));
                            float accv_r_r_r_r_2 = 0.0f;
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[0], a_vals[0], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[1], a_vals[1], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[2], a_vals[2], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[3], a_vals[3], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[4], a_vals[4], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[5], a_vals[5], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[6], a_vals[6], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[7], a_vals[7], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[8], a_vals[8], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[9], a_vals[9], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[10], a_vals[10], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[11], a_vals[11], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[12], a_vals[12], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[13], a_vals[13], accv_r_r_r_r_2);
                            accv_r_r_r_r_2 = __fmaf_rn(lrow_r_r_r_r_2[14], a_vals[14], accv_r_r_r_r_2);
                            float negv_r_r_r_r_2 = -accv_r_r_r_r_2;
                            a_vals[15] = ((jj < 15) ? negv_r_r_r_r_2 : a_vals[15]);
                            if (jj < 15) {
                                {
                                    uint32_t _addr_16 = static_cast<uint32_t>(a_col + 4080);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(negv_r_r_r_r_2) : "memory");
                                }
                            }
                        }
                        if (tid_0 >= 64) {
                            int t2 = tid_0 - 64;
                            mbarrier_wait(v_full_addr + (vst_c) * 8, _phase_v_full);
                            int qi_1_0 = t2;
                            int srow_2_0 = qi_1_0 >> 4;
                            int d0_1_0 = (qi_1_0 & 15) * 8;
                            float bs_r_1_0[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_0[0])) : "r"(beta_base + srow_2_0 * 4));
                            float fvb_1_0 = bs_r_1_0[0];
                            unsigned int vaddr_1_0 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_0 / 64 * 8192 + srow_2_0 * 128 + d0_1_0 % 64 * 2 ^ (d0_1_0 / 64 * 8192 + srow_2_0 * 128 + d0_1_0 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w[0])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 3]))
                                : "r"(vaddr_1_0));
                            float unpack_bf16[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w[_pair]) << 16);
                                (&unpack_bf16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w[_pair]) & 0xffff0000u);
                            }
                            float s[8];
                            s[0] = unpack_bf16[0] * fvb_1_0;
                            s[1] = unpack_bf16[1] * fvb_1_0;
                            s[2] = unpack_bf16[2] * fvb_1_0;
                            s[3] = unpack_bf16[3] * fvb_1_0;
                            s[4] = unpack_bf16[4] * fvb_1_0;
                            s[5] = unpack_bf16[5] * fvb_1_0;
                            s[6] = unpack_bf16[6] * fvb_1_0;
                            s[7] = unpack_bf16[7] * fvb_1_0;
                            unsigned int s_1_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s[_lp*2 + 0], s[_lp*2+1 + 0]));
                                s_1_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_0), "r"(s_1_bf16[0]), "r"(s_1_bf16[1]), "r"(s_1_bf16[2]), "r"(s_1_bf16[3]) : "memory");
                            int qi_1_1 = t2 + 64;
                            int srow_2_1 = qi_1_1 >> 4;
                            int d0_1_1 = (qi_1_1 & 15) * 8;
                            float bs_r_1_1[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_1[0])) : "r"(beta_base + srow_2_1 * 4));
                            float fvb_1_1 = bs_r_1_1[0];
                            unsigned int vaddr_1_1 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_1 / 64 * 8192 + srow_2_1 * 128 + d0_1_1 % 64 * 2 ^ (d0_1_1 / 64 * 8192 + srow_2_1 * 128 + d0_1_1 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_0[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_0[(0) + 3]))
                                : "r"(vaddr_1_1));
                            float unpack_bf16_0[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_0[_pair]) << 16);
                                (&unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_0[_pair]) & 0xffff0000u);
                            }
                            float s_0[8];
                            s_0[0] = unpack_bf16_0[0] * fvb_1_1;
                            s_0[1] = unpack_bf16_0[1] * fvb_1_1;
                            s_0[2] = unpack_bf16_0[2] * fvb_1_1;
                            s_0[3] = unpack_bf16_0[3] * fvb_1_1;
                            s_0[4] = unpack_bf16_0[4] * fvb_1_1;
                            s_0[5] = unpack_bf16_0[5] * fvb_1_1;
                            s_0[6] = unpack_bf16_0[6] * fvb_1_1;
                            s_0[7] = unpack_bf16_0[7] * fvb_1_1;
                            unsigned int s_1_bf16_0[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_0[_lp*2 + 0], s_0[_lp*2+1 + 0]));
                                s_1_bf16_0[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_1), "r"(s_1_bf16_0[0]), "r"(s_1_bf16_0[1]), "r"(s_1_bf16_0[2]), "r"(s_1_bf16_0[3]) : "memory");
                            int qi_1_2 = t2 + 128;
                            int srow_2_2 = qi_1_2 >> 4;
                            int d0_1_2 = (qi_1_2 & 15) * 8;
                            float bs_r_1_2[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_2[0])) : "r"(beta_base + srow_2_2 * 4));
                            float fvb_1_2 = bs_r_1_2[0];
                            unsigned int vaddr_1_2 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_2 / 64 * 8192 + srow_2_2 * 128 + d0_1_2 % 64 * 2 ^ (d0_1_2 / 64 * 8192 + srow_2_2 * 128 + d0_1_2 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_1[(0) + 3]))
                                : "r"(vaddr_1_2));
                            float unpack_bf16_1[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_1[_pair]) << 16);
                                (&unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_1[_pair]) & 0xffff0000u);
                            }
                            float s_2[8];
                            s_2[0] = unpack_bf16_1[0] * fvb_1_2;
                            s_2[1] = unpack_bf16_1[1] * fvb_1_2;
                            s_2[2] = unpack_bf16_1[2] * fvb_1_2;
                            s_2[3] = unpack_bf16_1[3] * fvb_1_2;
                            s_2[4] = unpack_bf16_1[4] * fvb_1_2;
                            s_2[5] = unpack_bf16_1[5] * fvb_1_2;
                            s_2[6] = unpack_bf16_1[6] * fvb_1_2;
                            s_2[7] = unpack_bf16_1[7] * fvb_1_2;
                            unsigned int s_1_bf16_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_2[_lp*2 + 0], s_2[_lp*2+1 + 0]));
                                s_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_2), "r"(s_1_bf16_1[0]), "r"(s_1_bf16_1[1]), "r"(s_1_bf16_1[2]), "r"(s_1_bf16_1[3]) : "memory");
                            int qi_1_3 = t2 + 192;
                            int srow_2_3 = qi_1_3 >> 4;
                            int d0_1_3 = (qi_1_3 & 15) * 8;
                            float bs_r_1_3[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_3[0])) : "r"(beta_base + srow_2_3 * 4));
                            float fvb_1_3 = bs_r_1_3[0];
                            unsigned int vaddr_1_3 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_3 / 64 * 8192 + srow_2_3 * 128 + d0_1_3 % 64 * 2 ^ (d0_1_3 / 64 * 8192 + srow_2_3 * 128 + d0_1_3 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_2[(0) + 3]))
                                : "r"(vaddr_1_3));
                            float unpack_bf16_2[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_2[_pair]) << 16);
                                (&unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_2[_pair]) & 0xffff0000u);
                            }
                            float s_3[8];
                            s_3[0] = unpack_bf16_2[0] * fvb_1_3;
                            s_3[1] = unpack_bf16_2[1] * fvb_1_3;
                            s_3[2] = unpack_bf16_2[2] * fvb_1_3;
                            s_3[3] = unpack_bf16_2[3] * fvb_1_3;
                            s_3[4] = unpack_bf16_2[4] * fvb_1_3;
                            s_3[5] = unpack_bf16_2[5] * fvb_1_3;
                            s_3[6] = unpack_bf16_2[6] * fvb_1_3;
                            s_3[7] = unpack_bf16_2[7] * fvb_1_3;
                            unsigned int s_1_bf16_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_3[_lp*2 + 0], s_3[_lp*2+1 + 0]));
                                s_1_bf16_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_3), "r"(s_1_bf16_2[0]), "r"(s_1_bf16_2[1]), "r"(s_1_bf16_2[2]), "r"(s_1_bf16_2[3]) : "memory");
                            int qi_1_4 = t2 + 256;
                            int srow_2_4 = qi_1_4 >> 4;
                            int d0_1_4 = (qi_1_4 & 15) * 8;
                            float bs_r_1_4[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_4[0])) : "r"(beta_base + srow_2_4 * 4));
                            float fvb_1_4 = bs_r_1_4[0];
                            unsigned int vaddr_1_4 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_4 / 64 * 8192 + srow_2_4 * 128 + d0_1_4 % 64 * 2 ^ (d0_1_4 / 64 * 8192 + srow_2_4 * 128 + d0_1_4 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_3[(0) + 3]))
                                : "r"(vaddr_1_4));
                            float unpack_bf16_3[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_3[_pair]) << 16);
                                (&unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_3[_pair]) & 0xffff0000u);
                            }
                            float s_4[8];
                            s_4[0] = unpack_bf16_3[0] * fvb_1_4;
                            s_4[1] = unpack_bf16_3[1] * fvb_1_4;
                            s_4[2] = unpack_bf16_3[2] * fvb_1_4;
                            s_4[3] = unpack_bf16_3[3] * fvb_1_4;
                            s_4[4] = unpack_bf16_3[4] * fvb_1_4;
                            s_4[5] = unpack_bf16_3[5] * fvb_1_4;
                            s_4[6] = unpack_bf16_3[6] * fvb_1_4;
                            s_4[7] = unpack_bf16_3[7] * fvb_1_4;
                            unsigned int s_1_bf16_3[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_4[_lp*2 + 0], s_4[_lp*2+1 + 0]));
                                s_1_bf16_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_4), "r"(s_1_bf16_3[0]), "r"(s_1_bf16_3[1]), "r"(s_1_bf16_3[2]), "r"(s_1_bf16_3[3]) : "memory");
                            int qi_1_5 = t2 + 320;
                            int srow_2_5 = qi_1_5 >> 4;
                            int d0_1_5 = (qi_1_5 & 15) * 8;
                            float bs_r_1_5[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_5[0])) : "r"(beta_base + srow_2_5 * 4));
                            float fvb_1_5 = bs_r_1_5[0];
                            unsigned int vaddr_1_5 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_5 / 64 * 8192 + srow_2_5 * 128 + d0_1_5 % 64 * 2 ^ (d0_1_5 / 64 * 8192 + srow_2_5 * 128 + d0_1_5 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_4[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_4[(0) + 3]))
                                : "r"(vaddr_1_5));
                            float unpack_bf16_4[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_4[_pair]) << 16);
                                (&unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_4[_pair]) & 0xffff0000u);
                            }
                            float s_5[8];
                            s_5[0] = unpack_bf16_4[0] * fvb_1_5;
                            s_5[1] = unpack_bf16_4[1] * fvb_1_5;
                            s_5[2] = unpack_bf16_4[2] * fvb_1_5;
                            s_5[3] = unpack_bf16_4[3] * fvb_1_5;
                            s_5[4] = unpack_bf16_4[4] * fvb_1_5;
                            s_5[5] = unpack_bf16_4[5] * fvb_1_5;
                            s_5[6] = unpack_bf16_4[6] * fvb_1_5;
                            s_5[7] = unpack_bf16_4[7] * fvb_1_5;
                            unsigned int s_1_bf16_4[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_5[_lp*2 + 0], s_5[_lp*2+1 + 0]));
                                s_1_bf16_4[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_5), "r"(s_1_bf16_4[0]), "r"(s_1_bf16_4[1]), "r"(s_1_bf16_4[2]), "r"(s_1_bf16_4[3]) : "memory");
                            int qi_1_6 = t2 + 384;
                            int srow_2_6 = qi_1_6 >> 4;
                            int d0_1_6 = (qi_1_6 & 15) * 8;
                            float bs_r_1_6[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_6[0])) : "r"(beta_base + srow_2_6 * 4));
                            float fvb_1_6 = bs_r_1_6[0];
                            unsigned int vaddr_1_6 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_6 / 64 * 8192 + srow_2_6 * 128 + d0_1_6 % 64 * 2 ^ (d0_1_6 / 64 * 8192 + srow_2_6 * 128 + d0_1_6 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_5[(0) + 3]))
                                : "r"(vaddr_1_6));
                            float unpack_bf16_5[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_5[_pair]) << 16);
                                (&unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_5[_pair]) & 0xffff0000u);
                            }
                            float s_6[8];
                            s_6[0] = unpack_bf16_5[0] * fvb_1_6;
                            s_6[1] = unpack_bf16_5[1] * fvb_1_6;
                            s_6[2] = unpack_bf16_5[2] * fvb_1_6;
                            s_6[3] = unpack_bf16_5[3] * fvb_1_6;
                            s_6[4] = unpack_bf16_5[4] * fvb_1_6;
                            s_6[5] = unpack_bf16_5[5] * fvb_1_6;
                            s_6[6] = unpack_bf16_5[6] * fvb_1_6;
                            s_6[7] = unpack_bf16_5[7] * fvb_1_6;
                            unsigned int s_1_bf16_5[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_6[_lp*2 + 0], s_6[_lp*2+1 + 0]));
                                s_1_bf16_5[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_6), "r"(s_1_bf16_5[0]), "r"(s_1_bf16_5[1]), "r"(s_1_bf16_5[2]), "r"(s_1_bf16_5[3]) : "memory");
                            int qi_1_7 = t2 + 448;
                            int srow_2_7 = qi_1_7 >> 4;
                            int d0_1_7 = (qi_1_7 & 15) * 8;
                            float bs_r_1_7[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_7[0])) : "r"(beta_base + srow_2_7 * 4));
                            float fvb_1_7 = bs_r_1_7[0];
                            unsigned int vaddr_1_7 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_7 / 64 * 8192 + srow_2_7 * 128 + d0_1_7 % 64 * 2 ^ (d0_1_7 / 64 * 8192 + srow_2_7 * 128 + d0_1_7 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_6[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_6[(0) + 3]))
                                : "r"(vaddr_1_7));
                            float unpack_bf16_6[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_6[_pair]) << 16);
                                (&unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_6[_pair]) & 0xffff0000u);
                            }
                            float s_7[8];
                            s_7[0] = unpack_bf16_6[0] * fvb_1_7;
                            s_7[1] = unpack_bf16_6[1] * fvb_1_7;
                            s_7[2] = unpack_bf16_6[2] * fvb_1_7;
                            s_7[3] = unpack_bf16_6[3] * fvb_1_7;
                            s_7[4] = unpack_bf16_6[4] * fvb_1_7;
                            s_7[5] = unpack_bf16_6[5] * fvb_1_7;
                            s_7[6] = unpack_bf16_6[6] * fvb_1_7;
                            s_7[7] = unpack_bf16_6[7] * fvb_1_7;
                            unsigned int s_1_bf16_6[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_7[_lp*2 + 0], s_7[_lp*2+1 + 0]));
                                s_1_bf16_6[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_7), "r"(s_1_bf16_6[0]), "r"(s_1_bf16_6[1]), "r"(s_1_bf16_6[2]), "r"(s_1_bf16_6[3]) : "memory");
                            int qi_1_8 = t2 + 512;
                            int srow_2_8 = qi_1_8 >> 4;
                            int d0_1_8 = (qi_1_8 & 15) * 8;
                            float bs_r_1_8[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_8[0])) : "r"(beta_base + srow_2_8 * 4));
                            float fvb_1_8 = bs_r_1_8[0];
                            unsigned int vaddr_1_8 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_8 / 64 * 8192 + srow_2_8 * 128 + d0_1_8 % 64 * 2 ^ (d0_1_8 / 64 * 8192 + srow_2_8 * 128 + d0_1_8 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_7[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_7[(0) + 3]))
                                : "r"(vaddr_1_8));
                            float unpack_bf16_7[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_7[_pair]) << 16);
                                (&unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_7[_pair]) & 0xffff0000u);
                            }
                            float s_8[8];
                            s_8[0] = unpack_bf16_7[0] * fvb_1_8;
                            s_8[1] = unpack_bf16_7[1] * fvb_1_8;
                            s_8[2] = unpack_bf16_7[2] * fvb_1_8;
                            s_8[3] = unpack_bf16_7[3] * fvb_1_8;
                            s_8[4] = unpack_bf16_7[4] * fvb_1_8;
                            s_8[5] = unpack_bf16_7[5] * fvb_1_8;
                            s_8[6] = unpack_bf16_7[6] * fvb_1_8;
                            s_8[7] = unpack_bf16_7[7] * fvb_1_8;
                            unsigned int s_1_bf16_7[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_8[_lp*2 + 0], s_8[_lp*2+1 + 0]));
                                s_1_bf16_7[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_8), "r"(s_1_bf16_7[0]), "r"(s_1_bf16_7[1]), "r"(s_1_bf16_7[2]), "r"(s_1_bf16_7[3]) : "memory");
                            int qi_1_9 = t2 + 576;
                            int srow_2_9 = qi_1_9 >> 4;
                            int d0_1_9 = (qi_1_9 & 15) * 8;
                            float bs_r_1_9[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_9[0])) : "r"(beta_base + srow_2_9 * 4));
                            float fvb_1_9 = bs_r_1_9[0];
                            unsigned int vaddr_1_9 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_9 / 64 * 8192 + srow_2_9 * 128 + d0_1_9 % 64 * 2 ^ (d0_1_9 / 64 * 8192 + srow_2_9 * 128 + d0_1_9 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_8[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_8[(0) + 3]))
                                : "r"(vaddr_1_9));
                            float unpack_bf16_8[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_8[_pair]) << 16);
                                (&unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_8[_pair]) & 0xffff0000u);
                            }
                            float s_9[8];
                            s_9[0] = unpack_bf16_8[0] * fvb_1_9;
                            s_9[1] = unpack_bf16_8[1] * fvb_1_9;
                            s_9[2] = unpack_bf16_8[2] * fvb_1_9;
                            s_9[3] = unpack_bf16_8[3] * fvb_1_9;
                            s_9[4] = unpack_bf16_8[4] * fvb_1_9;
                            s_9[5] = unpack_bf16_8[5] * fvb_1_9;
                            s_9[6] = unpack_bf16_8[6] * fvb_1_9;
                            s_9[7] = unpack_bf16_8[7] * fvb_1_9;
                            unsigned int s_1_bf16_8[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_9[_lp*2 + 0], s_9[_lp*2+1 + 0]));
                                s_1_bf16_8[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_9), "r"(s_1_bf16_8[0]), "r"(s_1_bf16_8[1]), "r"(s_1_bf16_8[2]), "r"(s_1_bf16_8[3]) : "memory");
                            int qi_1_10 = t2 + 640;
                            int srow_2_10 = qi_1_10 >> 4;
                            int d0_1_10 = (qi_1_10 & 15) * 8;
                            float bs_r_1_10[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_10[0])) : "r"(beta_base + srow_2_10 * 4));
                            float fvb_1_10 = bs_r_1_10[0];
                            unsigned int vaddr_1_10 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_10 / 64 * 8192 + srow_2_10 * 128 + d0_1_10 % 64 * 2 ^ (d0_1_10 / 64 * 8192 + srow_2_10 * 128 + d0_1_10 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_9[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_9[(0) + 3]))
                                : "r"(vaddr_1_10));
                            float unpack_bf16_9[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_9[_pair]) << 16);
                                (&unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_9[_pair]) & 0xffff0000u);
                            }
                            float s_10[8];
                            s_10[0] = unpack_bf16_9[0] * fvb_1_10;
                            s_10[1] = unpack_bf16_9[1] * fvb_1_10;
                            s_10[2] = unpack_bf16_9[2] * fvb_1_10;
                            s_10[3] = unpack_bf16_9[3] * fvb_1_10;
                            s_10[4] = unpack_bf16_9[4] * fvb_1_10;
                            s_10[5] = unpack_bf16_9[5] * fvb_1_10;
                            s_10[6] = unpack_bf16_9[6] * fvb_1_10;
                            s_10[7] = unpack_bf16_9[7] * fvb_1_10;
                            unsigned int s_1_bf16_9[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_10[_lp*2 + 0], s_10[_lp*2+1 + 0]));
                                s_1_bf16_9[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_10), "r"(s_1_bf16_9[0]), "r"(s_1_bf16_9[1]), "r"(s_1_bf16_9[2]), "r"(s_1_bf16_9[3]) : "memory");
                            int qi_1_11 = t2 + 704;
                            int srow_2_11 = qi_1_11 >> 4;
                            int d0_1_11 = (qi_1_11 & 15) * 8;
                            float bs_r_1_11[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_11[0])) : "r"(beta_base + srow_2_11 * 4));
                            float fvb_1_11 = bs_r_1_11[0];
                            unsigned int vaddr_1_11 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_11 / 64 * 8192 + srow_2_11 * 128 + d0_1_11 % 64 * 2 ^ (d0_1_11 / 64 * 8192 + srow_2_11 * 128 + d0_1_11 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_10[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_10[(0) + 3]))
                                : "r"(vaddr_1_11));
                            float unpack_bf16_10[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_10[_pair]) << 16);
                                (&unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_10[_pair]) & 0xffff0000u);
                            }
                            float s_11[8];
                            s_11[0] = unpack_bf16_10[0] * fvb_1_11;
                            s_11[1] = unpack_bf16_10[1] * fvb_1_11;
                            s_11[2] = unpack_bf16_10[2] * fvb_1_11;
                            s_11[3] = unpack_bf16_10[3] * fvb_1_11;
                            s_11[4] = unpack_bf16_10[4] * fvb_1_11;
                            s_11[5] = unpack_bf16_10[5] * fvb_1_11;
                            s_11[6] = unpack_bf16_10[6] * fvb_1_11;
                            s_11[7] = unpack_bf16_10[7] * fvb_1_11;
                            unsigned int s_1_bf16_10[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_11[_lp*2 + 0], s_11[_lp*2+1 + 0]));
                                s_1_bf16_10[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_11), "r"(s_1_bf16_10[0]), "r"(s_1_bf16_10[1]), "r"(s_1_bf16_10[2]), "r"(s_1_bf16_10[3]) : "memory");
                            int qi_1_12 = t2 + 768;
                            int srow_2_12 = qi_1_12 >> 4;
                            int d0_1_12 = (qi_1_12 & 15) * 8;
                            float bs_r_1_12[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_12[0])) : "r"(beta_base + srow_2_12 * 4));
                            float fvb_1_12 = bs_r_1_12[0];
                            unsigned int vaddr_1_12 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_12 / 64 * 8192 + srow_2_12 * 128 + d0_1_12 % 64 * 2 ^ (d0_1_12 / 64 * 8192 + srow_2_12 * 128 + d0_1_12 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_11[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_11[(0) + 3]))
                                : "r"(vaddr_1_12));
                            float unpack_bf16_11[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_11[_pair]) << 16);
                                (&unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_11[_pair]) & 0xffff0000u);
                            }
                            float s_12[8];
                            s_12[0] = unpack_bf16_11[0] * fvb_1_12;
                            s_12[1] = unpack_bf16_11[1] * fvb_1_12;
                            s_12[2] = unpack_bf16_11[2] * fvb_1_12;
                            s_12[3] = unpack_bf16_11[3] * fvb_1_12;
                            s_12[4] = unpack_bf16_11[4] * fvb_1_12;
                            s_12[5] = unpack_bf16_11[5] * fvb_1_12;
                            s_12[6] = unpack_bf16_11[6] * fvb_1_12;
                            s_12[7] = unpack_bf16_11[7] * fvb_1_12;
                            unsigned int s_1_bf16_11[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_12[_lp*2 + 0], s_12[_lp*2+1 + 0]));
                                s_1_bf16_11[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_12), "r"(s_1_bf16_11[0]), "r"(s_1_bf16_11[1]), "r"(s_1_bf16_11[2]), "r"(s_1_bf16_11[3]) : "memory");
                            int qi_1_13 = t2 + 832;
                            int srow_2_13 = qi_1_13 >> 4;
                            int d0_1_13 = (qi_1_13 & 15) * 8;
                            float bs_r_1_13[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_13[0])) : "r"(beta_base + srow_2_13 * 4));
                            float fvb_1_13 = bs_r_1_13[0];
                            unsigned int vaddr_1_13 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_13 / 64 * 8192 + srow_2_13 * 128 + d0_1_13 % 64 * 2 ^ (d0_1_13 / 64 * 8192 + srow_2_13 * 128 + d0_1_13 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_12[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_12[(0) + 3]))
                                : "r"(vaddr_1_13));
                            float unpack_bf16_12[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_12[_pair]) << 16);
                                (&unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_12[_pair]) & 0xffff0000u);
                            }
                            float s_13[8];
                            s_13[0] = unpack_bf16_12[0] * fvb_1_13;
                            s_13[1] = unpack_bf16_12[1] * fvb_1_13;
                            s_13[2] = unpack_bf16_12[2] * fvb_1_13;
                            s_13[3] = unpack_bf16_12[3] * fvb_1_13;
                            s_13[4] = unpack_bf16_12[4] * fvb_1_13;
                            s_13[5] = unpack_bf16_12[5] * fvb_1_13;
                            s_13[6] = unpack_bf16_12[6] * fvb_1_13;
                            s_13[7] = unpack_bf16_12[7] * fvb_1_13;
                            unsigned int s_1_bf16_12[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_13[_lp*2 + 0], s_13[_lp*2+1 + 0]));
                                s_1_bf16_12[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_13), "r"(s_1_bf16_12[0]), "r"(s_1_bf16_12[1]), "r"(s_1_bf16_12[2]), "r"(s_1_bf16_12[3]) : "memory");
                            int qi_1_14 = t2 + 896;
                            int srow_2_14 = qi_1_14 >> 4;
                            int d0_1_14 = (qi_1_14 & 15) * 8;
                            float bs_r_1_14[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_14[0])) : "r"(beta_base + srow_2_14 * 4));
                            float fvb_1_14 = bs_r_1_14[0];
                            unsigned int vaddr_1_14 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_14 / 64 * 8192 + srow_2_14 * 128 + d0_1_14 % 64 * 2 ^ (d0_1_14 / 64 * 8192 + srow_2_14 * 128 + d0_1_14 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_13[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_13[(0) + 3]))
                                : "r"(vaddr_1_14));
                            float unpack_bf16_13[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_13[_pair]) << 16);
                                (&unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_13[_pair]) & 0xffff0000u);
                            }
                            float s_14[8];
                            s_14[0] = unpack_bf16_13[0] * fvb_1_14;
                            s_14[1] = unpack_bf16_13[1] * fvb_1_14;
                            s_14[2] = unpack_bf16_13[2] * fvb_1_14;
                            s_14[3] = unpack_bf16_13[3] * fvb_1_14;
                            s_14[4] = unpack_bf16_13[4] * fvb_1_14;
                            s_14[5] = unpack_bf16_13[5] * fvb_1_14;
                            s_14[6] = unpack_bf16_13[6] * fvb_1_14;
                            s_14[7] = unpack_bf16_13[7] * fvb_1_14;
                            unsigned int s_1_bf16_13[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_14[_lp*2 + 0], s_14[_lp*2+1 + 0]));
                                s_1_bf16_13[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_14), "r"(s_1_bf16_13[0]), "r"(s_1_bf16_13[1]), "r"(s_1_bf16_13[2]), "r"(s_1_bf16_13[3]) : "memory");
                            int qi_1_15 = t2 + 960;
                            int srow_2_15 = qi_1_15 >> 4;
                            int d0_1_15 = (qi_1_15 & 15) * 8;
                            float bs_r_1_15[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_15[0])) : "r"(beta_base + srow_2_15 * 4));
                            float fvb_1_15 = bs_r_1_15[0];
                            unsigned int vaddr_1_15 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_15 / 64 * 8192 + srow_2_15 * 128 + d0_1_15 % 64 * 2 ^ (d0_1_15 / 64 * 8192 + srow_2_15 * 128 + d0_1_15 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_14[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_14[(0) + 3]))
                                : "r"(vaddr_1_15));
                            float unpack_bf16_14[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_14[_pair]) << 16);
                                (&unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_14[_pair]) & 0xffff0000u);
                            }
                            float s_15[8];
                            s_15[0] = unpack_bf16_14[0] * fvb_1_15;
                            s_15[1] = unpack_bf16_14[1] * fvb_1_15;
                            s_15[2] = unpack_bf16_14[2] * fvb_1_15;
                            s_15[3] = unpack_bf16_14[3] * fvb_1_15;
                            s_15[4] = unpack_bf16_14[4] * fvb_1_15;
                            s_15[5] = unpack_bf16_14[5] * fvb_1_15;
                            s_15[6] = unpack_bf16_14[6] * fvb_1_15;
                            s_15[7] = unpack_bf16_14[7] * fvb_1_15;
                            unsigned int s_1_bf16_14[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_15[_lp*2 + 0], s_15[_lp*2+1 + 0]));
                                s_1_bf16_14[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_15), "r"(s_1_bf16_14[0]), "r"(s_1_bf16_14[1]), "r"(s_1_bf16_14[2]), "r"(s_1_bf16_14[3]) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int rp8 = lane_1 & 7;
                        int pc8 = (lane_1 >> 3 & 3) * 4;
                        int prow8 = rp8 * 272;
                        int acol8 = prow8 + pc8 * 4;
                        if (w_id < 3) {
                            int blk8_1 = w_id * 4416;
                            float xs8[8];
                            xs8[0] = 0.0f;
                            xs8[1] = 0.0f;
                            xs8[2] = 0.0f;
                            xs8[3] = 0.0f;
                            xs8[4] = 0.0f;
                            xs8[5] = 0.0f;
                            xs8[6] = 0.0f;
                            xs8[7] = 0.0f;
                            float pv8[16];
                            float pw8[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8[(0) + 3]))
                                : "r"(l_base + blk8_1 + 4352 + prow8));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8[0])), "=r"(*reinterpret_cast<uint32_t*>(&pw8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8[(0) + 3]))
                                : "r"(l_base + blk8_1 + 4352 + prow8 + 2176));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8[(4) + 3]))
                                : "r"(l_base + blk8_1 + 4352 + prow8 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8[4])), "=r"(*reinterpret_cast<uint32_t*>(&pw8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8[(4) + 3]))
                                : "r"(l_base + blk8_1 + 4352 + prow8 + 2192));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv8[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8[(8) + 3]))
                                : "r"(l_base + blk8_1 + 4352 + prow8 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8[8])), "=r"(*reinterpret_cast<uint32_t*>(&pw8[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8[(8) + 3]))
                                : "r"(l_base + blk8_1 + 4352 + prow8 + 2208));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv8[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8[(12) + 3]))
                                : "r"(l_base + blk8_1 + 4352 + prow8 + 48));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8[12])), "=r"(*reinterpret_cast<uint32_t*>(&pw8[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8[(12) + 3]))
                                : "r"(l_base + blk8_1 + 4352 + prow8 + 2224));
                            float s8_0 = 0.0f;
                            float s8_1 = 0.0f;
                            float s8_2 = 0.0f;
                            float s8_3 = 0.0f;
                            float s8_4 = 0.0f;
                            float s8_5 = 0.0f;
                            float s8_6 = 0.0f;
                            float s8_7 = 0.0f;
                            float q8_0[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_0[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 0));
                            s8_0 = __fmaf_rn(pv8[0], q8_0[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[0], q8_0[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[0], q8_0[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[0], q8_0[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[0], q8_0[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[0], q8_0[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[0], q8_0[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[0], q8_0[3], s8_7);
                            float q8_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_1[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 272));
                            s8_0 = __fmaf_rn(pv8[1], q8_1[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[1], q8_1[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[1], q8_1[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[1], q8_1[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[1], q8_1[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[1], q8_1[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[1], q8_1[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[1], q8_1[3], s8_7);
                            float q8_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_2[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 544));
                            s8_0 = __fmaf_rn(pv8[2], q8_2[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[2], q8_2[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[2], q8_2[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[2], q8_2[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[2], q8_2[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[2], q8_2[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[2], q8_2[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[2], q8_2[3], s8_7);
                            float q8_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_3[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 816));
                            s8_0 = __fmaf_rn(pv8[3], q8_3[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[3], q8_3[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[3], q8_3[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[3], q8_3[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[3], q8_3[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[3], q8_3[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[3], q8_3[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[3], q8_3[3], s8_7);
                            float q8_4[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_4[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 1088));
                            s8_0 = __fmaf_rn(pv8[4], q8_4[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[4], q8_4[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[4], q8_4[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[4], q8_4[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[4], q8_4[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[4], q8_4[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[4], q8_4[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[4], q8_4[3], s8_7);
                            float q8_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_5[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 1360));
                            s8_0 = __fmaf_rn(pv8[5], q8_5[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[5], q8_5[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[5], q8_5[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[5], q8_5[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[5], q8_5[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[5], q8_5[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[5], q8_5[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[5], q8_5[3], s8_7);
                            float q8_6[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_6[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 1632));
                            s8_0 = __fmaf_rn(pv8[6], q8_6[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[6], q8_6[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[6], q8_6[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[6], q8_6[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[6], q8_6[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[6], q8_6[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[6], q8_6[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[6], q8_6[3], s8_7);
                            float q8_7[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_7[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 1904));
                            s8_0 = __fmaf_rn(pv8[7], q8_7[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[7], q8_7[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[7], q8_7[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[7], q8_7[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[7], q8_7[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[7], q8_7[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[7], q8_7[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[7], q8_7[3], s8_7);
                            float q8_8[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_8[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 2176));
                            s8_0 = __fmaf_rn(pv8[8], q8_8[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[8], q8_8[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[8], q8_8[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[8], q8_8[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[8], q8_8[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[8], q8_8[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[8], q8_8[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[8], q8_8[3], s8_7);
                            float q8_9[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_9[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 2448));
                            s8_0 = __fmaf_rn(pv8[9], q8_9[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[9], q8_9[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[9], q8_9[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[9], q8_9[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[9], q8_9[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[9], q8_9[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[9], q8_9[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[9], q8_9[3], s8_7);
                            float q8_10[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_10[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 2720));
                            s8_0 = __fmaf_rn(pv8[10], q8_10[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[10], q8_10[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[10], q8_10[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[10], q8_10[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[10], q8_10[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[10], q8_10[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[10], q8_10[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[10], q8_10[3], s8_7);
                            float q8_11[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_11[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 2992));
                            s8_0 = __fmaf_rn(pv8[11], q8_11[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[11], q8_11[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[11], q8_11[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[11], q8_11[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[11], q8_11[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[11], q8_11[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[11], q8_11[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[11], q8_11[3], s8_7);
                            float q8_12[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_12[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 3264));
                            s8_0 = __fmaf_rn(pv8[12], q8_12[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[12], q8_12[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[12], q8_12[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[12], q8_12[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[12], q8_12[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[12], q8_12[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[12], q8_12[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[12], q8_12[3], s8_7);
                            float q8_13[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_13[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 3536));
                            s8_0 = __fmaf_rn(pv8[13], q8_13[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[13], q8_13[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[13], q8_13[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[13], q8_13[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[13], q8_13[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[13], q8_13[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[13], q8_13[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[13], q8_13[3], s8_7);
                            float q8_14[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_14[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 3808));
                            s8_0 = __fmaf_rn(pv8[14], q8_14[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[14], q8_14[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[14], q8_14[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[14], q8_14[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[14], q8_14[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[14], q8_14[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[14], q8_14[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[14], q8_14[3], s8_7);
                            float q8_15[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_15[(0) + 3]))
                                : "r"(a_base + blk8_1 + pc8 * 4 + 4080));
                            s8_0 = __fmaf_rn(pv8[15], q8_15[0], s8_0);
                            s8_1 = __fmaf_rn(pv8[15], q8_15[1], s8_1);
                            s8_2 = __fmaf_rn(pv8[15], q8_15[2], s8_2);
                            s8_3 = __fmaf_rn(pv8[15], q8_15[3], s8_3);
                            s8_4 = __fmaf_rn(pw8[15], q8_15[0], s8_4);
                            s8_5 = __fmaf_rn(pw8[15], q8_15[1], s8_5);
                            s8_6 = __fmaf_rn(pw8[15], q8_15[2], s8_6);
                            s8_7 = __fmaf_rn(pw8[15], q8_15[3], s8_7);
                            xs8[0] = xs8[0] + s8_0;
                            xs8[1] = xs8[1] + s8_1;
                            xs8[2] = xs8[2] + s8_2;
                            xs8[3] = xs8[3] + s8_3;
                            xs8[4] = xs8[4] + s8_4;
                            xs8[5] = xs8[5] + s8_5;
                            xs8[6] = xs8[6] + s8_6;
                            xs8[7] = xs8[7] + s8_7;
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + blk8_1 + acol8), "f"(xs8[0]), "f"(xs8[1]), "f"(xs8[2]), "f"(xs8[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + blk8_1 + acol8 + 2176), "f"(xs8[4]), "f"(xs8[5]), "f"(xs8[6]), "f"(xs8[7]) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (w_id < 3) {
                            int blk8_2 = w_id * 4416;
                            float pv8_1[16];
                            float pw8_1[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[(0) + 3]))
                                : "r"(a_base + blk8_2 + 4416 + prow8));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[(0) + 3]))
                                : "r"(a_base + blk8_2 + 4416 + prow8 + 2176));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[(4) + 3]))
                                : "r"(a_base + blk8_2 + 4416 + prow8 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[(4) + 3]))
                                : "r"(a_base + blk8_2 + 4416 + prow8 + 2192));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[(8) + 3]))
                                : "r"(a_base + blk8_2 + 4416 + prow8 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[(8) + 3]))
                                : "r"(a_base + blk8_2 + 4416 + prow8 + 2208));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1[(12) + 3]))
                                : "r"(a_base + blk8_2 + 4416 + prow8 + 48));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1[(12) + 3]))
                                : "r"(a_base + blk8_2 + 4416 + prow8 + 2224));
                            float s8_0_1 = 0.0f;
                            float s8_1_1 = 0.0f;
                            float s8_2_1 = 0.0f;
                            float s8_3_1 = 0.0f;
                            float s8_4_1 = 0.0f;
                            float s8_5_1 = 0.0f;
                            float s8_6_1 = 0.0f;
                            float s8_7_1 = 0.0f;
                            float q8_16[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_16[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 0));
                            s8_0_1 = __fmaf_rn(pv8_1[0], q8_16[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[0], q8_16[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[0], q8_16[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[0], q8_16[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[0], q8_16[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[0], q8_16[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[0], q8_16[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[0], q8_16[3], s8_7_1);
                            float q8_17[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_17[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 272));
                            s8_0_1 = __fmaf_rn(pv8_1[1], q8_17[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[1], q8_17[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[1], q8_17[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[1], q8_17[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[1], q8_17[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[1], q8_17[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[1], q8_17[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[1], q8_17[3], s8_7_1);
                            float q8_18[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_18[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 544));
                            s8_0_1 = __fmaf_rn(pv8_1[2], q8_18[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[2], q8_18[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[2], q8_18[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[2], q8_18[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[2], q8_18[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[2], q8_18[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[2], q8_18[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[2], q8_18[3], s8_7_1);
                            float q8_19[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_19[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 816));
                            s8_0_1 = __fmaf_rn(pv8_1[3], q8_19[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[3], q8_19[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[3], q8_19[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[3], q8_19[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[3], q8_19[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[3], q8_19[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[3], q8_19[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[3], q8_19[3], s8_7_1);
                            float q8_20[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_20[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 1088));
                            s8_0_1 = __fmaf_rn(pv8_1[4], q8_20[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[4], q8_20[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[4], q8_20[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[4], q8_20[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[4], q8_20[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[4], q8_20[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[4], q8_20[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[4], q8_20[3], s8_7_1);
                            float q8_21[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_21[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 1360));
                            s8_0_1 = __fmaf_rn(pv8_1[5], q8_21[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[5], q8_21[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[5], q8_21[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[5], q8_21[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[5], q8_21[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[5], q8_21[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[5], q8_21[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[5], q8_21[3], s8_7_1);
                            float q8_22[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_22[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 1632));
                            s8_0_1 = __fmaf_rn(pv8_1[6], q8_22[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[6], q8_22[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[6], q8_22[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[6], q8_22[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[6], q8_22[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[6], q8_22[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[6], q8_22[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[6], q8_22[3], s8_7_1);
                            float q8_23[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_23[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 1904));
                            s8_0_1 = __fmaf_rn(pv8_1[7], q8_23[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[7], q8_23[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[7], q8_23[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[7], q8_23[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[7], q8_23[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[7], q8_23[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[7], q8_23[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[7], q8_23[3], s8_7_1);
                            float q8_24[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_24[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 2176));
                            s8_0_1 = __fmaf_rn(pv8_1[8], q8_24[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[8], q8_24[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[8], q8_24[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[8], q8_24[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[8], q8_24[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[8], q8_24[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[8], q8_24[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[8], q8_24[3], s8_7_1);
                            float q8_25[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_25[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_25[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_25[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 2448));
                            s8_0_1 = __fmaf_rn(pv8_1[9], q8_25[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[9], q8_25[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[9], q8_25[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[9], q8_25[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[9], q8_25[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[9], q8_25[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[9], q8_25[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[9], q8_25[3], s8_7_1);
                            float q8_26[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_26[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 2720));
                            s8_0_1 = __fmaf_rn(pv8_1[10], q8_26[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[10], q8_26[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[10], q8_26[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[10], q8_26[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[10], q8_26[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[10], q8_26[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[10], q8_26[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[10], q8_26[3], s8_7_1);
                            float q8_27[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_27[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 2992));
                            s8_0_1 = __fmaf_rn(pv8_1[11], q8_27[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[11], q8_27[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[11], q8_27[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[11], q8_27[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[11], q8_27[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[11], q8_27[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[11], q8_27[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[11], q8_27[3], s8_7_1);
                            float q8_28[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_28[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_28[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_28[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 3264));
                            s8_0_1 = __fmaf_rn(pv8_1[12], q8_28[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[12], q8_28[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[12], q8_28[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[12], q8_28[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[12], q8_28[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[12], q8_28[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[12], q8_28[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[12], q8_28[3], s8_7_1);
                            float q8_29[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_29[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_29[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_29[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 3536));
                            s8_0_1 = __fmaf_rn(pv8_1[13], q8_29[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[13], q8_29[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[13], q8_29[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[13], q8_29[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[13], q8_29[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[13], q8_29[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[13], q8_29[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[13], q8_29[3], s8_7_1);
                            float q8_30[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_30[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 3808));
                            s8_0_1 = __fmaf_rn(pv8_1[14], q8_30[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[14], q8_30[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[14], q8_30[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[14], q8_30[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[14], q8_30[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[14], q8_30[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[14], q8_30[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[14], q8_30[3], s8_7_1);
                            float q8_31[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_31[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_31[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_31[(0) + 3]))
                                : "r"(l_base + blk8_2 + pc8 * 4 + 4080));
                            s8_0_1 = __fmaf_rn(pv8_1[15], q8_31[0], s8_0_1);
                            s8_1_1 = __fmaf_rn(pv8_1[15], q8_31[1], s8_1_1);
                            s8_2_1 = __fmaf_rn(pv8_1[15], q8_31[2], s8_2_1);
                            s8_3_1 = __fmaf_rn(pv8_1[15], q8_31[3], s8_3_1);
                            s8_4_1 = __fmaf_rn(pw8_1[15], q8_31[0], s8_4_1);
                            s8_5_1 = __fmaf_rn(pw8_1[15], q8_31[1], s8_5_1);
                            s8_6_1 = __fmaf_rn(pw8_1[15], q8_31[2], s8_6_1);
                            s8_7_1 = __fmaf_rn(pw8_1[15], q8_31[3], s8_7_1);
                            float ns8[8];
                            ns8[0] = -s8_0_1;
                            ns8[1] = -s8_1_1;
                            ns8[2] = -s8_2_1;
                            ns8[3] = -s8_3_1;
                            ns8[4] = -s8_4_1;
                            ns8[5] = -s8_5_1;
                            ns8[6] = -s8_6_1;
                            ns8[7] = -s8_7_1;
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(a_base + blk8_2 + 4352 + acol8), "f"(ns8[0]), "f"(ns8[1]), "f"(ns8[2]), "f"(ns8[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(a_base + blk8_2 + 4352 + acol8 + 2176), "f"(ns8[4]), "f"(ns8[5]), "f"(ns8[6]), "f"(ns8[7]) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int g8 = w_id >> 1;
                        int t8 = w_id & 1;
                        float pv8_2[16];
                        float pw8_2[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[(0) + 3]))
                            : "r"(l_base + (g8 + 2) * 4352 + (g8 + t8) * 64 + prow8));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[(0) + 3]))
                            : "r"(l_base + (g8 + 2) * 4352 + (g8 + t8) * 64 + prow8 + 2176));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[(4) + 3]))
                            : "r"(l_base + (g8 + 2) * 4352 + (g8 + t8) * 64 + prow8 + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[(4) + 3]))
                            : "r"(l_base + (g8 + 2) * 4352 + (g8 + t8) * 64 + prow8 + 2192));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[(8) + 3]))
                            : "r"(l_base + (g8 + 2) * 4352 + (g8 + t8) * 64 + prow8 + 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[(8) + 3]))
                            : "r"(l_base + (g8 + 2) * 4352 + (g8 + t8) * 64 + prow8 + 2208));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2[(12) + 3]))
                            : "r"(l_base + (g8 + 2) * 4352 + (g8 + t8) * 64 + prow8 + 48));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2[(12) + 3]))
                            : "r"(l_base + (g8 + 2) * 4352 + (g8 + t8) * 64 + prow8 + 2224));
                        float s8_0_2 = 0.0f;
                        float s8_1_2 = 0.0f;
                        float s8_2_2 = 0.0f;
                        float s8_3_2 = 0.0f;
                        float s8_4_2 = 0.0f;
                        float s8_5_2 = 0.0f;
                        float s8_6_2 = 0.0f;
                        float s8_7_2 = 0.0f;
                        float q8_32[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_32[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_32[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_32[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 0));
                        s8_0_2 = __fmaf_rn(pv8_2[0], q8_32[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[0], q8_32[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[0], q8_32[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[0], q8_32[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[0], q8_32[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[0], q8_32[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[0], q8_32[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[0], q8_32[3], s8_7_2);
                        float q8_33[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_33[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_33[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_33[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_33[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 272));
                        s8_0_2 = __fmaf_rn(pv8_2[1], q8_33[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[1], q8_33[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[1], q8_33[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[1], q8_33[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[1], q8_33[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[1], q8_33[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[1], q8_33[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[1], q8_33[3], s8_7_2);
                        float q8_34[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_34[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_34[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_34[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_34[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 544));
                        s8_0_2 = __fmaf_rn(pv8_2[2], q8_34[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[2], q8_34[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[2], q8_34[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[2], q8_34[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[2], q8_34[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[2], q8_34[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[2], q8_34[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[2], q8_34[3], s8_7_2);
                        float q8_35[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_35[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_35[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_35[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_35[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 816));
                        s8_0_2 = __fmaf_rn(pv8_2[3], q8_35[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[3], q8_35[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[3], q8_35[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[3], q8_35[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[3], q8_35[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[3], q8_35[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[3], q8_35[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[3], q8_35[3], s8_7_2);
                        float q8_36[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_36[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_36[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_36[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 1088));
                        s8_0_2 = __fmaf_rn(pv8_2[4], q8_36[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[4], q8_36[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[4], q8_36[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[4], q8_36[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[4], q8_36[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[4], q8_36[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[4], q8_36[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[4], q8_36[3], s8_7_2);
                        float q8_37[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_37[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_37[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_37[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 1360));
                        s8_0_2 = __fmaf_rn(pv8_2[5], q8_37[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[5], q8_37[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[5], q8_37[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[5], q8_37[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[5], q8_37[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[5], q8_37[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[5], q8_37[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[5], q8_37[3], s8_7_2);
                        float q8_38[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_38[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_38[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_38[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_38[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 1632));
                        s8_0_2 = __fmaf_rn(pv8_2[6], q8_38[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[6], q8_38[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[6], q8_38[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[6], q8_38[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[6], q8_38[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[6], q8_38[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[6], q8_38[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[6], q8_38[3], s8_7_2);
                        float q8_39[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_39[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 1904));
                        s8_0_2 = __fmaf_rn(pv8_2[7], q8_39[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[7], q8_39[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[7], q8_39[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[7], q8_39[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[7], q8_39[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[7], q8_39[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[7], q8_39[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[7], q8_39[3], s8_7_2);
                        float q8_40[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_40[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_40[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_40[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_40[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 2176));
                        s8_0_2 = __fmaf_rn(pv8_2[8], q8_40[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[8], q8_40[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[8], q8_40[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[8], q8_40[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[8], q8_40[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[8], q8_40[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[8], q8_40[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[8], q8_40[3], s8_7_2);
                        float q8_41[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_41[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_41[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_41[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_41[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 2448));
                        s8_0_2 = __fmaf_rn(pv8_2[9], q8_41[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[9], q8_41[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[9], q8_41[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[9], q8_41[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[9], q8_41[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[9], q8_41[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[9], q8_41[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[9], q8_41[3], s8_7_2);
                        float q8_42[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_42[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_42[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_42[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_42[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 2720));
                        s8_0_2 = __fmaf_rn(pv8_2[10], q8_42[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[10], q8_42[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[10], q8_42[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[10], q8_42[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[10], q8_42[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[10], q8_42[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[10], q8_42[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[10], q8_42[3], s8_7_2);
                        float q8_43[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_43[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_43[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_43[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_43[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 2992));
                        s8_0_2 = __fmaf_rn(pv8_2[11], q8_43[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[11], q8_43[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[11], q8_43[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[11], q8_43[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[11], q8_43[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[11], q8_43[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[11], q8_43[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[11], q8_43[3], s8_7_2);
                        float q8_44[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_44[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_44[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_44[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_44[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 3264));
                        s8_0_2 = __fmaf_rn(pv8_2[12], q8_44[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[12], q8_44[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[12], q8_44[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[12], q8_44[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[12], q8_44[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[12], q8_44[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[12], q8_44[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[12], q8_44[3], s8_7_2);
                        float q8_45[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_45[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_45[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_45[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_45[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 3536));
                        s8_0_2 = __fmaf_rn(pv8_2[13], q8_45[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[13], q8_45[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[13], q8_45[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[13], q8_45[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[13], q8_45[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[13], q8_45[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[13], q8_45[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[13], q8_45[3], s8_7_2);
                        float q8_46[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_46[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_46[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_46[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_46[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 3808));
                        s8_0_2 = __fmaf_rn(pv8_2[14], q8_46[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[14], q8_46[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[14], q8_46[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[14], q8_46[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[14], q8_46[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[14], q8_46[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[14], q8_46[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[14], q8_46[3], s8_7_2);
                        float q8_47[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q8_47[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_47[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_47[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_47[(0) + 3]))
                            : "r"(a_base + (g8 + t8) * 4352 + g8 * 64 + pc8 * 4 + 4080));
                        s8_0_2 = __fmaf_rn(pv8_2[15], q8_47[0], s8_0_2);
                        s8_1_2 = __fmaf_rn(pv8_2[15], q8_47[1], s8_1_2);
                        s8_2_2 = __fmaf_rn(pv8_2[15], q8_47[2], s8_2_2);
                        s8_3_2 = __fmaf_rn(pv8_2[15], q8_47[3], s8_3_2);
                        s8_4_2 = __fmaf_rn(pw8_2[15], q8_47[0], s8_4_2);
                        s8_5_2 = __fmaf_rn(pw8_2[15], q8_47[1], s8_5_2);
                        s8_6_2 = __fmaf_rn(pw8_2[15], q8_47[2], s8_6_2);
                        s8_7_2 = __fmaf_rn(pw8_2[15], q8_47[3], s8_7_2);
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + w_id * 4416 + acol8), "f"(s8_0_2), "f"(s8_1_2), "f"(s8_2_2), "f"(s8_3_2) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + w_id * 4416 + acol8 + 2176), "f"(s8_4_2), "f"(s8_5_2), "f"(s8_6_2), "f"(s8_7_2) : "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (t8 == 0) {
                            int blk8_4 = w_id * 4416;
                            float ld8a[8];
                            float ld8b[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ld8a[0])), "=r"(*reinterpret_cast<uint32_t*>(&ld8a[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ld8a[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ld8a[(0) + 3]))
                                : "r"(l_base + blk8_4 + acol8));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ld8a[4])), "=r"(*reinterpret_cast<uint32_t*>(&ld8a[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ld8a[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ld8a[(4) + 3]))
                                : "r"(l_base + blk8_4 + acol8 + 2176));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ld8b[0])), "=r"(*reinterpret_cast<uint32_t*>(&ld8b[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ld8b[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ld8b[(0) + 3]))
                                : "r"(l_base + blk8_4 + 4416 + acol8));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ld8b[4])), "=r"(*reinterpret_cast<uint32_t*>(&ld8b[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ld8b[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ld8b[(4) + 3]))
                                : "r"(l_base + blk8_4 + 4416 + acol8 + 2176));
                            float xs8s[8];
                            xs8s[0] = 0.0f;
                            xs8s[1] = 0.0f;
                            xs8s[2] = 0.0f;
                            xs8s[3] = 0.0f;
                            xs8s[4] = 0.0f;
                            xs8s[5] = 0.0f;
                            xs8s[6] = 0.0f;
                            xs8s[7] = 0.0f;
                            xs8s[0] = xs8s[0] + ld8a[0];
                            xs8s[1] = xs8s[1] + ld8a[1];
                            xs8s[2] = xs8s[2] + ld8a[2];
                            xs8s[3] = xs8s[3] + ld8a[3];
                            xs8s[4] = xs8s[4] + ld8a[4];
                            xs8s[5] = xs8s[5] + ld8a[5];
                            xs8s[6] = xs8s[6] + ld8a[6];
                            xs8s[7] = xs8s[7] + ld8a[7];
                            xs8s[0] = xs8s[0] + ld8b[0];
                            xs8s[1] = xs8s[1] + ld8b[1];
                            xs8s[2] = xs8s[2] + ld8b[2];
                            xs8s[3] = xs8s[3] + ld8b[3];
                            xs8s[4] = xs8s[4] + ld8b[4];
                            xs8s[5] = xs8s[5] + ld8b[5];
                            xs8s[6] = xs8s[6] + ld8b[6];
                            xs8s[7] = xs8s[7] + ld8b[7];
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + blk8_4 + acol8), "f"(xs8s[0]), "f"(xs8s[1]), "f"(xs8s[2]), "f"(xs8s[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + blk8_4 + acol8 + 2176), "f"(xs8s[4]), "f"(xs8s[5]), "f"(xs8s[6]), "f"(xs8s[7]) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (t8 == g8) {
                            int blk8_5 = (g8 + 2) * 4416;
                            float pv8_0[16];
                            float pw8_0[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[(0) + 3]))
                                : "r"(a_base + blk8_5 + prow8));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[(0) + 3]))
                                : "r"(a_base + blk8_5 + prow8 + 2176));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[(4) + 3]))
                                : "r"(a_base + blk8_5 + prow8 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[(4) + 3]))
                                : "r"(a_base + blk8_5 + prow8 + 2192));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[(8) + 3]))
                                : "r"(a_base + blk8_5 + prow8 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[(8) + 3]))
                                : "r"(a_base + blk8_5 + prow8 + 2208));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_0[(12) + 3]))
                                : "r"(a_base + blk8_5 + prow8 + 48));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_0[(12) + 3]))
                                : "r"(a_base + blk8_5 + prow8 + 2224));
                            float s8_0_0 = 0.0f;
                            float s8_1_0 = 0.0f;
                            float s8_2_0 = 0.0f;
                            float s8_3_0 = 0.0f;
                            float s8_4_0 = 0.0f;
                            float s8_5_0 = 0.0f;
                            float s8_6_0 = 0.0f;
                            float s8_7_0 = 0.0f;
                            float q8_48[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_48[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_48[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_48[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_48[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 0));
                            s8_0_0 = __fmaf_rn(pv8_0[0], q8_48[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[0], q8_48[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[0], q8_48[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[0], q8_48[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[0], q8_48[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[0], q8_48[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[0], q8_48[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[0], q8_48[3], s8_7_0);
                            float q8_49[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_49[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_49[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_49[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_49[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 272));
                            s8_0_0 = __fmaf_rn(pv8_0[1], q8_49[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[1], q8_49[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[1], q8_49[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[1], q8_49[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[1], q8_49[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[1], q8_49[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[1], q8_49[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[1], q8_49[3], s8_7_0);
                            float q8_50[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_50[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_50[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_50[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_50[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 544));
                            s8_0_0 = __fmaf_rn(pv8_0[2], q8_50[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[2], q8_50[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[2], q8_50[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[2], q8_50[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[2], q8_50[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[2], q8_50[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[2], q8_50[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[2], q8_50[3], s8_7_0);
                            float q8_51[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_51[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_51[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_51[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_51[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 816));
                            s8_0_0 = __fmaf_rn(pv8_0[3], q8_51[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[3], q8_51[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[3], q8_51[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[3], q8_51[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[3], q8_51[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[3], q8_51[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[3], q8_51[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[3], q8_51[3], s8_7_0);
                            float q8_52[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_52[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_52[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_52[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_52[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 1088));
                            s8_0_0 = __fmaf_rn(pv8_0[4], q8_52[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[4], q8_52[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[4], q8_52[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[4], q8_52[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[4], q8_52[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[4], q8_52[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[4], q8_52[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[4], q8_52[3], s8_7_0);
                            float q8_53[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_53[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_53[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_53[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_53[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 1360));
                            s8_0_0 = __fmaf_rn(pv8_0[5], q8_53[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[5], q8_53[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[5], q8_53[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[5], q8_53[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[5], q8_53[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[5], q8_53[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[5], q8_53[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[5], q8_53[3], s8_7_0);
                            float q8_54[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_54[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_54[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_54[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_54[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 1632));
                            s8_0_0 = __fmaf_rn(pv8_0[6], q8_54[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[6], q8_54[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[6], q8_54[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[6], q8_54[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[6], q8_54[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[6], q8_54[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[6], q8_54[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[6], q8_54[3], s8_7_0);
                            float q8_55[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_55[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_55[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_55[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_55[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 1904));
                            s8_0_0 = __fmaf_rn(pv8_0[7], q8_55[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[7], q8_55[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[7], q8_55[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[7], q8_55[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[7], q8_55[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[7], q8_55[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[7], q8_55[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[7], q8_55[3], s8_7_0);
                            float q8_56[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_56[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_56[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_56[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_56[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 2176));
                            s8_0_0 = __fmaf_rn(pv8_0[8], q8_56[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[8], q8_56[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[8], q8_56[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[8], q8_56[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[8], q8_56[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[8], q8_56[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[8], q8_56[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[8], q8_56[3], s8_7_0);
                            float q8_57[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_57[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_57[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_57[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_57[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 2448));
                            s8_0_0 = __fmaf_rn(pv8_0[9], q8_57[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[9], q8_57[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[9], q8_57[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[9], q8_57[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[9], q8_57[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[9], q8_57[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[9], q8_57[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[9], q8_57[3], s8_7_0);
                            float q8_58[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_58[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_58[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_58[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_58[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 2720));
                            s8_0_0 = __fmaf_rn(pv8_0[10], q8_58[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[10], q8_58[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[10], q8_58[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[10], q8_58[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[10], q8_58[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[10], q8_58[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[10], q8_58[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[10], q8_58[3], s8_7_0);
                            float q8_59[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_59[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_59[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_59[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_59[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 2992));
                            s8_0_0 = __fmaf_rn(pv8_0[11], q8_59[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[11], q8_59[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[11], q8_59[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[11], q8_59[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[11], q8_59[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[11], q8_59[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[11], q8_59[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[11], q8_59[3], s8_7_0);
                            float q8_60[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_60[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_60[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_60[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_60[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 3264));
                            s8_0_0 = __fmaf_rn(pv8_0[12], q8_60[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[12], q8_60[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[12], q8_60[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[12], q8_60[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[12], q8_60[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[12], q8_60[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[12], q8_60[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[12], q8_60[3], s8_7_0);
                            float q8_61[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_61[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_61[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_61[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_61[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 3536));
                            s8_0_0 = __fmaf_rn(pv8_0[13], q8_61[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[13], q8_61[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[13], q8_61[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[13], q8_61[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[13], q8_61[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[13], q8_61[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[13], q8_61[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[13], q8_61[3], s8_7_0);
                            float q8_62[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_62[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_62[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_62[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_62[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 3808));
                            s8_0_0 = __fmaf_rn(pv8_0[14], q8_62[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[14], q8_62[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[14], q8_62[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[14], q8_62[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[14], q8_62[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[14], q8_62[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[14], q8_62[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[14], q8_62[3], s8_7_0);
                            float q8_63[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_63[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_63[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_63[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_63[(0) + 3]))
                                : "r"(l_base + g8 * 8832 + pc8 * 4 + 4080));
                            s8_0_0 = __fmaf_rn(pv8_0[15], q8_63[0], s8_0_0);
                            s8_1_0 = __fmaf_rn(pv8_0[15], q8_63[1], s8_1_0);
                            s8_2_0 = __fmaf_rn(pv8_0[15], q8_63[2], s8_2_0);
                            s8_3_0 = __fmaf_rn(pv8_0[15], q8_63[3], s8_3_0);
                            s8_4_0 = __fmaf_rn(pw8_0[15], q8_63[0], s8_4_0);
                            s8_5_0 = __fmaf_rn(pw8_0[15], q8_63[1], s8_5_0);
                            s8_6_0 = __fmaf_rn(pw8_0[15], q8_63[2], s8_6_0);
                            s8_7_0 = __fmaf_rn(pw8_0[15], q8_63[3], s8_7_0);
                            float ns8_1[8];
                            ns8_1[0] = -s8_0_0;
                            ns8_1[1] = -s8_1_0;
                            ns8_1[2] = -s8_2_0;
                            ns8_1[3] = -s8_3_0;
                            ns8_1[4] = -s8_4_0;
                            ns8_1[5] = -s8_5_0;
                            ns8_1[6] = -s8_6_0;
                            ns8_1[7] = -s8_7_0;
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(a_base + (g8 + 2) * 4352 + g8 * 64 + acol8), "f"(ns8_1[0]), "f"(ns8_1[1]), "f"(ns8_1[2]), "f"(ns8_1[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(a_base + (g8 + 2) * 4352 + g8 * 64 + acol8 + 2176), "f"(ns8_1[4]), "f"(ns8_1[5]), "f"(ns8_1[6]), "f"(ns8_1[7]) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (w_id > 0) {
                            int m8 = 3 - w_id;
                            float pv8_1_1[16];
                            float pw8_1_1[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[(0) + 3]))
                                : "r"(l_base + 13056 + m8 * 64 + prow8));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[(0) + 3]))
                                : "r"(l_base + 13056 + m8 * 64 + prow8 + 2176));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[(4) + 3]))
                                : "r"(l_base + 13056 + m8 * 64 + prow8 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[(4) + 3]))
                                : "r"(l_base + 13056 + m8 * 64 + prow8 + 2192));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[(8) + 3]))
                                : "r"(l_base + 13056 + m8 * 64 + prow8 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[(8) + 3]))
                                : "r"(l_base + 13056 + m8 * 64 + prow8 + 2208));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_1_1[(12) + 3]))
                                : "r"(l_base + 13056 + m8 * 64 + prow8 + 48));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_1_1[(12) + 3]))
                                : "r"(l_base + 13056 + m8 * 64 + prow8 + 2224));
                            float s8_0_1_1 = 0.0f;
                            float s8_1_1_1 = 0.0f;
                            float s8_2_1_1 = 0.0f;
                            float s8_3_1_1 = 0.0f;
                            float s8_4_1_1 = 0.0f;
                            float s8_5_1_1 = 0.0f;
                            float s8_6_1_1 = 0.0f;
                            float s8_7_1_1 = 0.0f;
                            float q8_64[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_64[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_64[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_64[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_64[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 0));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[0], q8_64[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[0], q8_64[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[0], q8_64[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[0], q8_64[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[0], q8_64[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[0], q8_64[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[0], q8_64[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[0], q8_64[3], s8_7_1_1);
                            float q8_65[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_65[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_65[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_65[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_65[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 272));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[1], q8_65[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[1], q8_65[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[1], q8_65[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[1], q8_65[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[1], q8_65[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[1], q8_65[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[1], q8_65[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[1], q8_65[3], s8_7_1_1);
                            float q8_66[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_66[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_66[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_66[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_66[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 544));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[2], q8_66[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[2], q8_66[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[2], q8_66[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[2], q8_66[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[2], q8_66[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[2], q8_66[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[2], q8_66[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[2], q8_66[3], s8_7_1_1);
                            float q8_67[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_67[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_67[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_67[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_67[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 816));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[3], q8_67[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[3], q8_67[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[3], q8_67[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[3], q8_67[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[3], q8_67[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[3], q8_67[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[3], q8_67[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[3], q8_67[3], s8_7_1_1);
                            float q8_68[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_68[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_68[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_68[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_68[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 1088));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[4], q8_68[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[4], q8_68[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[4], q8_68[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[4], q8_68[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[4], q8_68[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[4], q8_68[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[4], q8_68[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[4], q8_68[3], s8_7_1_1);
                            float q8_69[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_69[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_69[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_69[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_69[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 1360));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[5], q8_69[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[5], q8_69[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[5], q8_69[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[5], q8_69[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[5], q8_69[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[5], q8_69[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[5], q8_69[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[5], q8_69[3], s8_7_1_1);
                            float q8_70[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_70[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_70[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_70[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_70[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 1632));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[6], q8_70[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[6], q8_70[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[6], q8_70[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[6], q8_70[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[6], q8_70[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[6], q8_70[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[6], q8_70[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[6], q8_70[3], s8_7_1_1);
                            float q8_71[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_71[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_71[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_71[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_71[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 1904));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[7], q8_71[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[7], q8_71[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[7], q8_71[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[7], q8_71[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[7], q8_71[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[7], q8_71[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[7], q8_71[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[7], q8_71[3], s8_7_1_1);
                            float q8_72[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_72[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_72[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_72[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_72[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 2176));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[8], q8_72[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[8], q8_72[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[8], q8_72[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[8], q8_72[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[8], q8_72[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[8], q8_72[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[8], q8_72[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[8], q8_72[3], s8_7_1_1);
                            float q8_73[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_73[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_73[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_73[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_73[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 2448));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[9], q8_73[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[9], q8_73[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[9], q8_73[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[9], q8_73[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[9], q8_73[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[9], q8_73[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[9], q8_73[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[9], q8_73[3], s8_7_1_1);
                            float q8_74[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_74[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_74[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_74[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_74[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 2720));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[10], q8_74[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[10], q8_74[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[10], q8_74[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[10], q8_74[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[10], q8_74[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[10], q8_74[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[10], q8_74[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[10], q8_74[3], s8_7_1_1);
                            float q8_75[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_75[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_75[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_75[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_75[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 2992));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[11], q8_75[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[11], q8_75[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[11], q8_75[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[11], q8_75[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[11], q8_75[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[11], q8_75[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[11], q8_75[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[11], q8_75[3], s8_7_1_1);
                            float q8_76[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_76[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_76[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_76[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_76[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 3264));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[12], q8_76[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[12], q8_76[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[12], q8_76[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[12], q8_76[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[12], q8_76[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[12], q8_76[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[12], q8_76[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[12], q8_76[3], s8_7_1_1);
                            float q8_77[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_77[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_77[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_77[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_77[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 3536));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[13], q8_77[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[13], q8_77[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[13], q8_77[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[13], q8_77[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[13], q8_77[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[13], q8_77[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[13], q8_77[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[13], q8_77[3], s8_7_1_1);
                            float q8_78[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_78[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_78[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_78[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_78[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 3808));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[14], q8_78[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[14], q8_78[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[14], q8_78[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[14], q8_78[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[14], q8_78[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[14], q8_78[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[14], q8_78[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[14], q8_78[3], s8_7_1_1);
                            float q8_79[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_79[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_79[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_79[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_79[(0) + 3]))
                                : "r"(a_base + m8 * 4352 + pc8 * 4 + 4080));
                            s8_0_1_1 = __fmaf_rn(pv8_1_1[15], q8_79[0], s8_0_1_1);
                            s8_1_1_1 = __fmaf_rn(pv8_1_1[15], q8_79[1], s8_1_1_1);
                            s8_2_1_1 = __fmaf_rn(pv8_1_1[15], q8_79[2], s8_2_1_1);
                            s8_3_1_1 = __fmaf_rn(pv8_1_1[15], q8_79[3], s8_3_1_1);
                            s8_4_1_1 = __fmaf_rn(pw8_1_1[15], q8_79[0], s8_4_1_1);
                            s8_5_1_1 = __fmaf_rn(pw8_1_1[15], q8_79[1], s8_5_1_1);
                            s8_6_1_1 = __fmaf_rn(pw8_1_1[15], q8_79[2], s8_6_1_1);
                            s8_7_1_1 = __fmaf_rn(pw8_1_1[15], q8_79[3], s8_7_1_1);
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + m8 * 4416 + acol8), "f"(s8_0_1_1), "f"(s8_1_1_1), "f"(s8_2_1_1), "f"(s8_3_1_1) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + m8 * 4416 + acol8 + 2176), "f"(s8_4_1_1), "f"(s8_5_1_1), "f"(s8_6_1_1), "f"(s8_7_1_1) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (w_id == 3) {
                            float ld8a_1[8];
                            float ld8b_1[8];
                            float ld8c[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ld8a_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ld8a_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ld8a_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ld8a_1[(0) + 3]))
                                : "r"(l_base + acol8));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ld8a_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ld8a_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ld8a_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ld8a_1[(4) + 3]))
                                : "r"(l_base + acol8 + 2176));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ld8b_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ld8b_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ld8b_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ld8b_1[(0) + 3]))
                                : "r"(l_base + 4416 + acol8));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ld8b_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ld8b_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ld8b_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ld8b_1[(4) + 3]))
                                : "r"(l_base + 4416 + acol8 + 2176));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ld8c[0])), "=r"(*reinterpret_cast<uint32_t*>(&ld8c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ld8c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ld8c[(0) + 3]))
                                : "r"(l_base + 8832 + acol8));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ld8c[4])), "=r"(*reinterpret_cast<uint32_t*>(&ld8c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ld8c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ld8c[(4) + 3]))
                                : "r"(l_base + 8832 + acol8 + 2176));
                            float xs8t[8];
                            xs8t[0] = 0.0f;
                            xs8t[1] = 0.0f;
                            xs8t[2] = 0.0f;
                            xs8t[3] = 0.0f;
                            xs8t[4] = 0.0f;
                            xs8t[5] = 0.0f;
                            xs8t[6] = 0.0f;
                            xs8t[7] = 0.0f;
                            xs8t[0] = xs8t[0] + ld8a_1[0];
                            xs8t[1] = xs8t[1] + ld8a_1[1];
                            xs8t[2] = xs8t[2] + ld8a_1[2];
                            xs8t[3] = xs8t[3] + ld8a_1[3];
                            xs8t[4] = xs8t[4] + ld8a_1[4];
                            xs8t[5] = xs8t[5] + ld8a_1[5];
                            xs8t[6] = xs8t[6] + ld8a_1[6];
                            xs8t[7] = xs8t[7] + ld8a_1[7];
                            xs8t[0] = xs8t[0] + ld8b_1[0];
                            xs8t[1] = xs8t[1] + ld8b_1[1];
                            xs8t[2] = xs8t[2] + ld8b_1[2];
                            xs8t[3] = xs8t[3] + ld8b_1[3];
                            xs8t[4] = xs8t[4] + ld8b_1[4];
                            xs8t[5] = xs8t[5] + ld8b_1[5];
                            xs8t[6] = xs8t[6] + ld8b_1[6];
                            xs8t[7] = xs8t[7] + ld8b_1[7];
                            xs8t[0] = xs8t[0] + ld8c[0];
                            xs8t[1] = xs8t[1] + ld8c[1];
                            xs8t[2] = xs8t[2] + ld8c[2];
                            xs8t[3] = xs8t[3] + ld8c[3];
                            xs8t[4] = xs8t[4] + ld8c[4];
                            xs8t[5] = xs8t[5] + ld8c[5];
                            xs8t[6] = xs8t[6] + ld8c[6];
                            xs8t[7] = xs8t[7] + ld8c[7];
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + 13248 + acol8), "f"(xs8t[0]), "f"(xs8t[1]), "f"(xs8t[2]), "f"(xs8t[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + 13248 + acol8 + 2176), "f"(xs8t[4]), "f"(xs8t[5]), "f"(xs8t[6]), "f"(xs8t[7]) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (w_id == 3) {
                            float pv8_2_1[16];
                            float pw8_2_1[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[(0) + 3]))
                                : "r"(a_base + 13248 + prow8));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[(0) + 3]))
                                : "r"(a_base + 13248 + prow8 + 2176));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[(4) + 3]))
                                : "r"(a_base + 13248 + prow8 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[(4) + 3]))
                                : "r"(a_base + 13248 + prow8 + 2192));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[(8) + 3]))
                                : "r"(a_base + 13248 + prow8 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[(8) + 3]))
                                : "r"(a_base + 13248 + prow8 + 2208));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv8_2_1[(12) + 3]))
                                : "r"(a_base + 13248 + prow8 + 48));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pw8_2_1[(12) + 3]))
                                : "r"(a_base + 13248 + prow8 + 2224));
                            float s8_0_2_1 = 0.0f;
                            float s8_1_2_1 = 0.0f;
                            float s8_2_2_1 = 0.0f;
                            float s8_3_2_1 = 0.0f;
                            float s8_4_2_1 = 0.0f;
                            float s8_5_2_1 = 0.0f;
                            float s8_6_2_1 = 0.0f;
                            float s8_7_2_1 = 0.0f;
                            float q8_80[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_80[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_80[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_80[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_80[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 0));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[0], q8_80[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[0], q8_80[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[0], q8_80[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[0], q8_80[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[0], q8_80[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[0], q8_80[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[0], q8_80[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[0], q8_80[3], s8_7_2_1);
                            float q8_81[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_81[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_81[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_81[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_81[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 272));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[1], q8_81[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[1], q8_81[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[1], q8_81[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[1], q8_81[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[1], q8_81[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[1], q8_81[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[1], q8_81[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[1], q8_81[3], s8_7_2_1);
                            float q8_82[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_82[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_82[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_82[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_82[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 544));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[2], q8_82[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[2], q8_82[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[2], q8_82[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[2], q8_82[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[2], q8_82[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[2], q8_82[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[2], q8_82[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[2], q8_82[3], s8_7_2_1);
                            float q8_83[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_83[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_83[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_83[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_83[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 816));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[3], q8_83[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[3], q8_83[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[3], q8_83[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[3], q8_83[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[3], q8_83[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[3], q8_83[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[3], q8_83[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[3], q8_83[3], s8_7_2_1);
                            float q8_84[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_84[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_84[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_84[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_84[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 1088));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[4], q8_84[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[4], q8_84[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[4], q8_84[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[4], q8_84[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[4], q8_84[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[4], q8_84[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[4], q8_84[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[4], q8_84[3], s8_7_2_1);
                            float q8_85[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_85[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_85[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_85[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_85[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 1360));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[5], q8_85[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[5], q8_85[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[5], q8_85[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[5], q8_85[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[5], q8_85[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[5], q8_85[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[5], q8_85[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[5], q8_85[3], s8_7_2_1);
                            float q8_86[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_86[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_86[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_86[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_86[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 1632));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[6], q8_86[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[6], q8_86[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[6], q8_86[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[6], q8_86[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[6], q8_86[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[6], q8_86[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[6], q8_86[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[6], q8_86[3], s8_7_2_1);
                            float q8_87[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_87[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_87[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_87[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_87[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 1904));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[7], q8_87[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[7], q8_87[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[7], q8_87[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[7], q8_87[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[7], q8_87[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[7], q8_87[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[7], q8_87[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[7], q8_87[3], s8_7_2_1);
                            float q8_88[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_88[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_88[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_88[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_88[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 2176));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[8], q8_88[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[8], q8_88[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[8], q8_88[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[8], q8_88[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[8], q8_88[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[8], q8_88[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[8], q8_88[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[8], q8_88[3], s8_7_2_1);
                            float q8_89[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_89[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_89[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_89[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_89[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 2448));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[9], q8_89[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[9], q8_89[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[9], q8_89[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[9], q8_89[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[9], q8_89[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[9], q8_89[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[9], q8_89[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[9], q8_89[3], s8_7_2_1);
                            float q8_90[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_90[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_90[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_90[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_90[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 2720));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[10], q8_90[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[10], q8_90[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[10], q8_90[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[10], q8_90[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[10], q8_90[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[10], q8_90[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[10], q8_90[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[10], q8_90[3], s8_7_2_1);
                            float q8_91[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_91[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_91[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_91[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_91[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 2992));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[11], q8_91[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[11], q8_91[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[11], q8_91[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[11], q8_91[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[11], q8_91[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[11], q8_91[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[11], q8_91[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[11], q8_91[3], s8_7_2_1);
                            float q8_92[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_92[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_92[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_92[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_92[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 3264));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[12], q8_92[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[12], q8_92[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[12], q8_92[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[12], q8_92[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[12], q8_92[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[12], q8_92[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[12], q8_92[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[12], q8_92[3], s8_7_2_1);
                            float q8_93[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_93[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_93[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_93[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_93[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 3536));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[13], q8_93[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[13], q8_93[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[13], q8_93[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[13], q8_93[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[13], q8_93[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[13], q8_93[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[13], q8_93[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[13], q8_93[3], s8_7_2_1);
                            float q8_94[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_94[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_94[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_94[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_94[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 3808));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[14], q8_94[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[14], q8_94[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[14], q8_94[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[14], q8_94[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[14], q8_94[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[14], q8_94[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[14], q8_94[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[14], q8_94[3], s8_7_2_1);
                            float q8_95[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q8_95[0])), "=r"(*reinterpret_cast<uint32_t*>(&q8_95[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q8_95[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q8_95[(0) + 3]))
                                : "r"(l_base + 13248 + pc8 * 4 + 4080));
                            s8_0_2_1 = __fmaf_rn(pv8_2_1[15], q8_95[0], s8_0_2_1);
                            s8_1_2_1 = __fmaf_rn(pv8_2_1[15], q8_95[1], s8_1_2_1);
                            s8_2_2_1 = __fmaf_rn(pv8_2_1[15], q8_95[2], s8_2_2_1);
                            s8_3_2_1 = __fmaf_rn(pv8_2_1[15], q8_95[3], s8_3_2_1);
                            s8_4_2_1 = __fmaf_rn(pw8_2_1[15], q8_95[0], s8_4_2_1);
                            s8_5_2_1 = __fmaf_rn(pw8_2_1[15], q8_95[1], s8_5_2_1);
                            s8_6_2_1 = __fmaf_rn(pw8_2_1[15], q8_95[2], s8_6_2_1);
                            s8_7_2_1 = __fmaf_rn(pw8_2_1[15], q8_95[3], s8_7_2_1);
                            float ns8_2[8];
                            ns8_2[0] = -s8_0_2_1;
                            ns8_2[1] = -s8_1_2_1;
                            ns8_2[2] = -s8_2_2_1;
                            ns8_2[3] = -s8_3_2_1;
                            ns8_2[4] = -s8_4_2_1;
                            ns8_2[5] = -s8_5_2_1;
                            ns8_2[6] = -s8_6_2_1;
                            ns8_2[7] = -s8_7_2_1;
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(a_base + 13056 + acol8), "f"(ns8_2[0]), "f"(ns8_2[1]), "f"(ns8_2[2]), "f"(ns8_2[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(a_base + 13056 + acol8 + 2176), "f"(ns8_2[4]), "f"(ns8_2[5]), "f"(ns8_2[6]), "f"(ns8_2[7]) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                    }
                    if (j > 0) {
                        mbarrier_wait(wu_full_addr, _phase_wu_full_0);
                        _phase_wu_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    if (group > j) {
                        int qi_1_16 = tid_0;
                        int srow_2_16 = qi_1_16 >> 4;
                        int d0_1_16 = (qi_1_16 & 15) * 8;
                        float bs_r_1_16[1];
                        float es_r_1_0[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_16[0])) : "r"(beta_base + srow_2_16 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_0[0])) : "r"(eg_base + srow_2_16 * 4));
                        float fkb_1_0 = bs_r_1_16[0] * es_r_1_0[0];
                        unsigned int w_15[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_15[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_16 / 64 * 8192 + srow_2_16 * 128 + d0_1_16 % 64 * 2 ^ (d0_1_16 / 64 * 8192 + srow_2_16 * 128 + d0_1_16 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_15[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_15[_pair]) << 16);
                            (&unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_15[_pair]) & 0xffff0000u);
                        }
                        float s_1[8];
                        s_1[0] = unpack_bf16_15[0] * fkb_1_0;
                        s_1[1] = unpack_bf16_15[1] * fkb_1_0;
                        s_1[2] = unpack_bf16_15[2] * fkb_1_0;
                        s_1[3] = unpack_bf16_15[3] * fkb_1_0;
                        s_1[4] = unpack_bf16_15[4] * fkb_1_0;
                        s_1[5] = unpack_bf16_15[5] * fkb_1_0;
                        s_1[6] = unpack_bf16_15[6] * fkb_1_0;
                        s_1[7] = unpack_bf16_15[7] * fkb_1_0;
                        unsigned int s_1_bf16_15[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_1[_lp*2 + 0], s_1[_lp*2+1 + 0]));
                            s_1_bf16_15[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_16 / 64 * 8192 + srow_2_16 * 128 + d0_1_16 % 64 * 2 ^ (d0_1_16 / 64 * 8192 + srow_2_16 * 128 + d0_1_16 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_15[0]), "r"(s_1_bf16_15[1]), "r"(s_1_bf16_15[2]), "r"(s_1_bf16_15[3]) : "memory");
                        int qi_1_17 = tid_0 + 128;
                        int srow_2_17 = qi_1_17 >> 4;
                        int d0_1_17 = (qi_1_17 & 15) * 8;
                        float bs_r_1_17[1];
                        float es_r_1_1[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_17[0])) : "r"(beta_base + srow_2_17 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_1[0])) : "r"(eg_base + srow_2_17 * 4));
                        float fkb_1_1 = bs_r_1_17[0] * es_r_1_1[0];
                        unsigned int w_15_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_15_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_15_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_15_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_15_1[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_17 / 64 * 8192 + srow_2_17 * 128 + d0_1_17 % 64 * 2 ^ (d0_1_17 / 64 * 8192 + srow_2_17 * 128 + d0_1_17 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_15_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_15_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_15_1[_pair]) << 16);
                            (&unpack_bf16_15_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_15_1[_pair]) & 0xffff0000u);
                        }
                        float s_16[8];
                        s_16[0] = unpack_bf16_15_1[0] * fkb_1_1;
                        s_16[1] = unpack_bf16_15_1[1] * fkb_1_1;
                        s_16[2] = unpack_bf16_15_1[2] * fkb_1_1;
                        s_16[3] = unpack_bf16_15_1[3] * fkb_1_1;
                        s_16[4] = unpack_bf16_15_1[4] * fkb_1_1;
                        s_16[5] = unpack_bf16_15_1[5] * fkb_1_1;
                        s_16[6] = unpack_bf16_15_1[6] * fkb_1_1;
                        s_16[7] = unpack_bf16_15_1[7] * fkb_1_1;
                        unsigned int s_1_bf16_15_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_16[_lp*2 + 0], s_16[_lp*2+1 + 0]));
                            s_1_bf16_15_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_17 / 64 * 8192 + srow_2_17 * 128 + d0_1_17 % 64 * 2 ^ (d0_1_17 / 64 * 8192 + srow_2_17 * 128 + d0_1_17 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_15_1[0]), "r"(s_1_bf16_15_1[1]), "r"(s_1_bf16_15_1[2]), "r"(s_1_bf16_15_1[3]) : "memory");
                        int qi_1_18 = tid_0 + 256;
                        int srow_2_18 = qi_1_18 >> 4;
                        int d0_1_18 = (qi_1_18 & 15) * 8;
                        float bs_r_1_18[1];
                        float es_r_1_2[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_18[0])) : "r"(beta_base + srow_2_18 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_2[0])) : "r"(eg_base + srow_2_18 * 4));
                        float fkb_1_2 = bs_r_1_18[0] * es_r_1_2[0];
                        unsigned int w_16[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_16[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_18 / 64 * 8192 + srow_2_18 * 128 + d0_1_18 % 64 * 2 ^ (d0_1_18 / 64 * 8192 + srow_2_18 * 128 + d0_1_18 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_16[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_16[_pair]) << 16);
                            (&unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_16[_pair]) & 0xffff0000u);
                        }
                        float s_17[8];
                        s_17[0] = unpack_bf16_16[0] * fkb_1_2;
                        s_17[1] = unpack_bf16_16[1] * fkb_1_2;
                        s_17[2] = unpack_bf16_16[2] * fkb_1_2;
                        s_17[3] = unpack_bf16_16[3] * fkb_1_2;
                        s_17[4] = unpack_bf16_16[4] * fkb_1_2;
                        s_17[5] = unpack_bf16_16[5] * fkb_1_2;
                        s_17[6] = unpack_bf16_16[6] * fkb_1_2;
                        s_17[7] = unpack_bf16_16[7] * fkb_1_2;
                        unsigned int s_1_bf16_16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_17[_lp*2 + 0], s_17[_lp*2+1 + 0]));
                            s_1_bf16_16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_18 / 64 * 8192 + srow_2_18 * 128 + d0_1_18 % 64 * 2 ^ (d0_1_18 / 64 * 8192 + srow_2_18 * 128 + d0_1_18 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_16[0]), "r"(s_1_bf16_16[1]), "r"(s_1_bf16_16[2]), "r"(s_1_bf16_16[3]) : "memory");
                        int qi_1_19 = tid_0 + 384;
                        int srow_2_19 = qi_1_19 >> 4;
                        int d0_1_19 = (qi_1_19 & 15) * 8;
                        float bs_r_1_19[1];
                        float es_r_1_3[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_19[0])) : "r"(beta_base + srow_2_19 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_3[0])) : "r"(eg_base + srow_2_19 * 4));
                        float fkb_1_3 = bs_r_1_19[0] * es_r_1_3[0];
                        unsigned int w_17[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_17[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_19 / 64 * 8192 + srow_2_19 * 128 + d0_1_19 % 64 * 2 ^ (d0_1_19 / 64 * 8192 + srow_2_19 * 128 + d0_1_19 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_17[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_17[_pair]) << 16);
                            (&unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_17[_pair]) & 0xffff0000u);
                        }
                        float s_18[8];
                        s_18[0] = unpack_bf16_17[0] * fkb_1_3;
                        s_18[1] = unpack_bf16_17[1] * fkb_1_3;
                        s_18[2] = unpack_bf16_17[2] * fkb_1_3;
                        s_18[3] = unpack_bf16_17[3] * fkb_1_3;
                        s_18[4] = unpack_bf16_17[4] * fkb_1_3;
                        s_18[5] = unpack_bf16_17[5] * fkb_1_3;
                        s_18[6] = unpack_bf16_17[6] * fkb_1_3;
                        s_18[7] = unpack_bf16_17[7] * fkb_1_3;
                        unsigned int s_1_bf16_17[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_18[_lp*2 + 0], s_18[_lp*2+1 + 0]));
                            s_1_bf16_17[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_19 / 64 * 8192 + srow_2_19 * 128 + d0_1_19 % 64 * 2 ^ (d0_1_19 / 64 * 8192 + srow_2_19 * 128 + d0_1_19 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_17[0]), "r"(s_1_bf16_17[1]), "r"(s_1_bf16_17[2]), "r"(s_1_bf16_17[3]) : "memory");
                        int qi_1_20 = tid_0 + 512;
                        int srow_2_20 = qi_1_20 >> 4;
                        int d0_1_20 = (qi_1_20 & 15) * 8;
                        float bs_r_1_20[1];
                        float es_r_1_4[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_20[0])) : "r"(beta_base + srow_2_20 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_4[0])) : "r"(eg_base + srow_2_20 * 4));
                        float fkb_1_4 = bs_r_1_20[0] * es_r_1_4[0];
                        unsigned int w_18[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_18[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_20 / 64 * 8192 + srow_2_20 * 128 + d0_1_20 % 64 * 2 ^ (d0_1_20 / 64 * 8192 + srow_2_20 * 128 + d0_1_20 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_18[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_18[_pair]) << 16);
                            (&unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_18[_pair]) & 0xffff0000u);
                        }
                        float s_19[8];
                        s_19[0] = unpack_bf16_18[0] * fkb_1_4;
                        s_19[1] = unpack_bf16_18[1] * fkb_1_4;
                        s_19[2] = unpack_bf16_18[2] * fkb_1_4;
                        s_19[3] = unpack_bf16_18[3] * fkb_1_4;
                        s_19[4] = unpack_bf16_18[4] * fkb_1_4;
                        s_19[5] = unpack_bf16_18[5] * fkb_1_4;
                        s_19[6] = unpack_bf16_18[6] * fkb_1_4;
                        s_19[7] = unpack_bf16_18[7] * fkb_1_4;
                        unsigned int s_1_bf16_18[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_19[_lp*2 + 0], s_19[_lp*2+1 + 0]));
                            s_1_bf16_18[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_20 / 64 * 8192 + srow_2_20 * 128 + d0_1_20 % 64 * 2 ^ (d0_1_20 / 64 * 8192 + srow_2_20 * 128 + d0_1_20 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_18[0]), "r"(s_1_bf16_18[1]), "r"(s_1_bf16_18[2]), "r"(s_1_bf16_18[3]) : "memory");
                        int qi_1_21 = tid_0 + 640;
                        int srow_2_21 = qi_1_21 >> 4;
                        int d0_1_21 = (qi_1_21 & 15) * 8;
                        float bs_r_1_21[1];
                        float es_r_1_5[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_21[0])) : "r"(beta_base + srow_2_21 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_5[0])) : "r"(eg_base + srow_2_21 * 4));
                        float fkb_1_5 = bs_r_1_21[0] * es_r_1_5[0];
                        unsigned int w_19[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_19[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_21 / 64 * 8192 + srow_2_21 * 128 + d0_1_21 % 64 * 2 ^ (d0_1_21 / 64 * 8192 + srow_2_21 * 128 + d0_1_21 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_19[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_19[_pair]) << 16);
                            (&unpack_bf16_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_19[_pair]) & 0xffff0000u);
                        }
                        float s_20[8];
                        s_20[0] = unpack_bf16_19[0] * fkb_1_5;
                        s_20[1] = unpack_bf16_19[1] * fkb_1_5;
                        s_20[2] = unpack_bf16_19[2] * fkb_1_5;
                        s_20[3] = unpack_bf16_19[3] * fkb_1_5;
                        s_20[4] = unpack_bf16_19[4] * fkb_1_5;
                        s_20[5] = unpack_bf16_19[5] * fkb_1_5;
                        s_20[6] = unpack_bf16_19[6] * fkb_1_5;
                        s_20[7] = unpack_bf16_19[7] * fkb_1_5;
                        unsigned int s_1_bf16_19[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_20[_lp*2 + 0], s_20[_lp*2+1 + 0]));
                            s_1_bf16_19[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_21 / 64 * 8192 + srow_2_21 * 128 + d0_1_21 % 64 * 2 ^ (d0_1_21 / 64 * 8192 + srow_2_21 * 128 + d0_1_21 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_19[0]), "r"(s_1_bf16_19[1]), "r"(s_1_bf16_19[2]), "r"(s_1_bf16_19[3]) : "memory");
                        int qi_1_22 = tid_0 + 768;
                        int srow_2_22 = qi_1_22 >> 4;
                        int d0_1_22 = (qi_1_22 & 15) * 8;
                        float bs_r_1_22[1];
                        float es_r_1_6[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_22[0])) : "r"(beta_base + srow_2_22 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_6[0])) : "r"(eg_base + srow_2_22 * 4));
                        float fkb_1_6 = bs_r_1_22[0] * es_r_1_6[0];
                        unsigned int w_20[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_20[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_22 / 64 * 8192 + srow_2_22 * 128 + d0_1_22 % 64 * 2 ^ (d0_1_22 / 64 * 8192 + srow_2_22 * 128 + d0_1_22 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_20[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_20[_pair]) << 16);
                            (&unpack_bf16_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_20[_pair]) & 0xffff0000u);
                        }
                        float s_21[8];
                        s_21[0] = unpack_bf16_20[0] * fkb_1_6;
                        s_21[1] = unpack_bf16_20[1] * fkb_1_6;
                        s_21[2] = unpack_bf16_20[2] * fkb_1_6;
                        s_21[3] = unpack_bf16_20[3] * fkb_1_6;
                        s_21[4] = unpack_bf16_20[4] * fkb_1_6;
                        s_21[5] = unpack_bf16_20[5] * fkb_1_6;
                        s_21[6] = unpack_bf16_20[6] * fkb_1_6;
                        s_21[7] = unpack_bf16_20[7] * fkb_1_6;
                        unsigned int s_1_bf16_20[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_21[_lp*2 + 0], s_21[_lp*2+1 + 0]));
                            s_1_bf16_20[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_22 / 64 * 8192 + srow_2_22 * 128 + d0_1_22 % 64 * 2 ^ (d0_1_22 / 64 * 8192 + srow_2_22 * 128 + d0_1_22 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_20[0]), "r"(s_1_bf16_20[1]), "r"(s_1_bf16_20[2]), "r"(s_1_bf16_20[3]) : "memory");
                        int qi_1_23 = tid_0 + 896;
                        int srow_2_23 = qi_1_23 >> 4;
                        int d0_1_23 = (qi_1_23 & 15) * 8;
                        float bs_r_1_23[1];
                        float es_r_1_7[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_23[0])) : "r"(beta_base + srow_2_23 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_7[0])) : "r"(eg_base + srow_2_23 * 4));
                        float fkb_1_7 = bs_r_1_23[0] * es_r_1_7[0];
                        unsigned int w_21[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_21[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_23 / 64 * 8192 + srow_2_23 * 128 + d0_1_23 % 64 * 2 ^ (d0_1_23 / 64 * 8192 + srow_2_23 * 128 + d0_1_23 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_21[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_21[_pair]) << 16);
                            (&unpack_bf16_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_21[_pair]) & 0xffff0000u);
                        }
                        float s_22[8];
                        s_22[0] = unpack_bf16_21[0] * fkb_1_7;
                        s_22[1] = unpack_bf16_21[1] * fkb_1_7;
                        s_22[2] = unpack_bf16_21[2] * fkb_1_7;
                        s_22[3] = unpack_bf16_21[3] * fkb_1_7;
                        s_22[4] = unpack_bf16_21[4] * fkb_1_7;
                        s_22[5] = unpack_bf16_21[5] * fkb_1_7;
                        s_22[6] = unpack_bf16_21[6] * fkb_1_7;
                        s_22[7] = unpack_bf16_21[7] * fkb_1_7;
                        unsigned int s_1_bf16_21[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_22[_lp*2 + 0], s_22[_lp*2+1 + 0]));
                            s_1_bf16_21[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_23 / 64 * 8192 + srow_2_23 * 128 + d0_1_23 % 64 * 2 ^ (d0_1_23 / 64 * 8192 + srow_2_23 * 128 + d0_1_23 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_21[0]), "r"(s_1_bf16_21[1]), "r"(s_1_bf16_21[2]), "r"(s_1_bf16_21[3]) : "memory");
                        int arow = tid_0 >> 1;
                        int arow_valid = 0;
                        if (arow < clen) {
                            arow_valid = 1;
                        }
                        long long akk_base = ((long long)(irow0 + arow) * (long long)num_heads + (long long)hv) * 64;
                        int sa0_1_0 = (tid_0 & 1) * 4 * 8;
                        float av_1_0[8];
                        av_1_0[0] = 0.0f;
                        av_1_0[1] = 0.0f;
                        av_1_0[2] = 0.0f;
                        av_1_0[3] = 0.0f;
                        av_1_0[4] = 0.0f;
                        av_1_0[5] = 0.0f;
                        av_1_0[6] = 0.0f;
                        av_1_0[7] = 0.0f;
                        if (sa0_1_0 <= arow) {
                            if (arow_valid != 0) {
                                float ar_1_0[8];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_0[(0) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_1_0) * 4));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_1_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_0[(4) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_1_0 + 4) * 4));
                                av_1_0[0] = ((arow >= sa0_1_0) ? ar_1_0[0] : 0.0f);
                                av_1_0[1] = ((arow >= sa0_1_0 + 1) ? ar_1_0[1] : 0.0f);
                                av_1_0[2] = ((arow >= sa0_1_0 + 2) ? ar_1_0[2] : 0.0f);
                                av_1_0[3] = ((arow >= sa0_1_0 + 3) ? ar_1_0[3] : 0.0f);
                                av_1_0[4] = ((arow >= sa0_1_0 + 4) ? ar_1_0[4] : 0.0f);
                                av_1_0[5] = ((arow >= sa0_1_0 + 5) ? ar_1_0[5] : 0.0f);
                                av_1_0[6] = ((arow >= sa0_1_0 + 6) ? ar_1_0[6] : 0.0f);
                                av_1_0[7] = ((arow >= sa0_1_0 + 7) ? ar_1_0[7] : 0.0f);
                            }
                        }
                        unsigned int av_bf16_1_0[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_0[_lp*2 + 0], av_1_0[_lp*2+1 + 0]));
                            av_bf16_1_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_t_addr + (unsigned int)(sa0_1_0 / 64 * 8192 + arow * 128 + sa0_1_0 % 64 * 2 ^ (sa0_1_0 / 64 * 8192 + arow * 128 + sa0_1_0 % 64 * 2 >> 7 & 7) << 4))), "r"(av_bf16_1_0[0]), "r"(av_bf16_1_0[1]), "r"(av_bf16_1_0[2]), "r"(av_bf16_1_0[3]) : "memory");
                        unsigned int _packed_2[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_0[_lp*2 + 0], av_1_0[_lp*2+1 + 0]));
                            _packed_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_0[_lp*2 + 2], av_1_0[_lp*2+1 + 2]));
                            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_0[_lp*2 + 4], av_1_0[_lp*2+1 + 4]));
                            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_0[_lp*2 + 6], av_1_0[_lp*2+1 + 6]));
                            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(akk_out + (akk_base + (long long)sa0_1_0))[0] = reinterpret_cast<int4*>(_packed_2)[0];
                        int sa0_1_1 = ((tid_0 & 1) * 4 + 1) * 8;
                        float av_1_1[8];
                        av_1_1[0] = 0.0f;
                        av_1_1[1] = 0.0f;
                        av_1_1[2] = 0.0f;
                        av_1_1[3] = 0.0f;
                        av_1_1[4] = 0.0f;
                        av_1_1[5] = 0.0f;
                        av_1_1[6] = 0.0f;
                        av_1_1[7] = 0.0f;
                        if (sa0_1_1 <= arow) {
                            if (arow_valid != 0) {
                                float ar_1_1[8];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_1[(0) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_1_1) * 4));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_1[(4) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_1_1 + 4) * 4));
                                av_1_1[0] = ((arow >= sa0_1_1) ? ar_1_1[0] : 0.0f);
                                av_1_1[1] = ((arow >= sa0_1_1 + 1) ? ar_1_1[1] : 0.0f);
                                av_1_1[2] = ((arow >= sa0_1_1 + 2) ? ar_1_1[2] : 0.0f);
                                av_1_1[3] = ((arow >= sa0_1_1 + 3) ? ar_1_1[3] : 0.0f);
                                av_1_1[4] = ((arow >= sa0_1_1 + 4) ? ar_1_1[4] : 0.0f);
                                av_1_1[5] = ((arow >= sa0_1_1 + 5) ? ar_1_1[5] : 0.0f);
                                av_1_1[6] = ((arow >= sa0_1_1 + 6) ? ar_1_1[6] : 0.0f);
                                av_1_1[7] = ((arow >= sa0_1_1 + 7) ? ar_1_1[7] : 0.0f);
                            }
                        }
                        unsigned int av_bf16_1_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_1[_lp*2 + 0], av_1_1[_lp*2+1 + 0]));
                            av_bf16_1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_t_addr + (unsigned int)(sa0_1_1 / 64 * 8192 + arow * 128 + sa0_1_1 % 64 * 2 ^ (sa0_1_1 / 64 * 8192 + arow * 128 + sa0_1_1 % 64 * 2 >> 7 & 7) << 4))), "r"(av_bf16_1_1[0]), "r"(av_bf16_1_1[1]), "r"(av_bf16_1_1[2]), "r"(av_bf16_1_1[3]) : "memory");
                        unsigned int _packed_3[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_1[_lp*2 + 0], av_1_1[_lp*2+1 + 0]));
                            _packed_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_1[_lp*2 + 2], av_1_1[_lp*2+1 + 2]));
                            _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_1[_lp*2 + 4], av_1_1[_lp*2+1 + 4]));
                            _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_1[_lp*2 + 6], av_1_1[_lp*2+1 + 6]));
                            _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(akk_out + (akk_base + (long long)sa0_1_1))[0] = reinterpret_cast<int4*>(_packed_3)[0];
                        int sa0_1_2 = ((tid_0 & 1) * 4 + 2) * 8;
                        float av_1_2[8];
                        av_1_2[0] = 0.0f;
                        av_1_2[1] = 0.0f;
                        av_1_2[2] = 0.0f;
                        av_1_2[3] = 0.0f;
                        av_1_2[4] = 0.0f;
                        av_1_2[5] = 0.0f;
                        av_1_2[6] = 0.0f;
                        av_1_2[7] = 0.0f;
                        if (sa0_1_2 <= arow) {
                            if (arow_valid != 0) {
                                float ar_1_2[8];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_2[(0) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_1_2) * 4));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_1_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_2[(4) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_1_2 + 4) * 4));
                                av_1_2[0] = ((arow >= sa0_1_2) ? ar_1_2[0] : 0.0f);
                                av_1_2[1] = ((arow >= sa0_1_2 + 1) ? ar_1_2[1] : 0.0f);
                                av_1_2[2] = ((arow >= sa0_1_2 + 2) ? ar_1_2[2] : 0.0f);
                                av_1_2[3] = ((arow >= sa0_1_2 + 3) ? ar_1_2[3] : 0.0f);
                                av_1_2[4] = ((arow >= sa0_1_2 + 4) ? ar_1_2[4] : 0.0f);
                                av_1_2[5] = ((arow >= sa0_1_2 + 5) ? ar_1_2[5] : 0.0f);
                                av_1_2[6] = ((arow >= sa0_1_2 + 6) ? ar_1_2[6] : 0.0f);
                                av_1_2[7] = ((arow >= sa0_1_2 + 7) ? ar_1_2[7] : 0.0f);
                            }
                        }
                        unsigned int av_bf16_1_2[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_2[_lp*2 + 0], av_1_2[_lp*2+1 + 0]));
                            av_bf16_1_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_t_addr + (unsigned int)(sa0_1_2 / 64 * 8192 + arow * 128 + sa0_1_2 % 64 * 2 ^ (sa0_1_2 / 64 * 8192 + arow * 128 + sa0_1_2 % 64 * 2 >> 7 & 7) << 4))), "r"(av_bf16_1_2[0]), "r"(av_bf16_1_2[1]), "r"(av_bf16_1_2[2]), "r"(av_bf16_1_2[3]) : "memory");
                        unsigned int _packed_4[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_2[_lp*2 + 0], av_1_2[_lp*2+1 + 0]));
                            _packed_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_2[_lp*2 + 2], av_1_2[_lp*2+1 + 2]));
                            _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_2[_lp*2 + 4], av_1_2[_lp*2+1 + 4]));
                            _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_2[_lp*2 + 6], av_1_2[_lp*2+1 + 6]));
                            _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(akk_out + (akk_base + (long long)sa0_1_2))[0] = reinterpret_cast<int4*>(_packed_4)[0];
                        int sa0_1_3 = ((tid_0 & 1) * 4 + 3) * 8;
                        float av_1_3[8];
                        av_1_3[0] = 0.0f;
                        av_1_3[1] = 0.0f;
                        av_1_3[2] = 0.0f;
                        av_1_3[3] = 0.0f;
                        av_1_3[4] = 0.0f;
                        av_1_3[5] = 0.0f;
                        av_1_3[6] = 0.0f;
                        av_1_3[7] = 0.0f;
                        if (sa0_1_3 <= arow) {
                            if (arow_valid != 0) {
                                float ar_1_3[8];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_3[(0) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_1_3) * 4));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_1_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1_3[(4) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_1_3 + 4) * 4));
                                av_1_3[0] = ((arow >= sa0_1_3) ? ar_1_3[0] : 0.0f);
                                av_1_3[1] = ((arow >= sa0_1_3 + 1) ? ar_1_3[1] : 0.0f);
                                av_1_3[2] = ((arow >= sa0_1_3 + 2) ? ar_1_3[2] : 0.0f);
                                av_1_3[3] = ((arow >= sa0_1_3 + 3) ? ar_1_3[3] : 0.0f);
                                av_1_3[4] = ((arow >= sa0_1_3 + 4) ? ar_1_3[4] : 0.0f);
                                av_1_3[5] = ((arow >= sa0_1_3 + 5) ? ar_1_3[5] : 0.0f);
                                av_1_3[6] = ((arow >= sa0_1_3 + 6) ? ar_1_3[6] : 0.0f);
                                av_1_3[7] = ((arow >= sa0_1_3 + 7) ? ar_1_3[7] : 0.0f);
                            }
                        }
                        unsigned int av_bf16_1_3[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_3[_lp*2 + 0], av_1_3[_lp*2+1 + 0]));
                            av_bf16_1_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_t_addr + (unsigned int)(sa0_1_3 / 64 * 8192 + arow * 128 + sa0_1_3 % 64 * 2 ^ (sa0_1_3 / 64 * 8192 + arow * 128 + sa0_1_3 % 64 * 2 >> 7 & 7) << 4))), "r"(av_bf16_1_3[0]), "r"(av_bf16_1_3[1]), "r"(av_bf16_1_3[2]), "r"(av_bf16_1_3[3]) : "memory");
                        unsigned int _packed_5[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_3[_lp*2 + 0], av_1_3[_lp*2+1 + 0]));
                            _packed_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_3[_lp*2 + 2], av_1_3[_lp*2+1 + 2]));
                            _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_3[_lp*2 + 4], av_1_3[_lp*2+1 + 4]));
                            _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_1_3[_lp*2 + 6], av_1_3[_lp*2+1 + 6]));
                            _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(akk_out + (akk_base + (long long)sa0_1_3))[0] = reinterpret_cast<int4*>(_packed_5)[0];
                    }
                    if (j > 0) {
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int wrow = l_base + t_row * 272;
                        int urow = a_base + t_row * 272;
                        int col_1_0 = half * 16;
                        float _tmem_load_0[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                            : "r"(taddr + (unsigned int)row_base));
                        float _tmem_load_8_0[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[15]))
                            : "r"(taddr + (unsigned int)row_base + 128));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float wo_2_0[16];
                        float uo_1_0[16];
                        wo_2_0[0] = ((row_valid != 0) ? _tmem_load_0[0] : 0.0f);
                        uo_1_0[0] = ((row_valid != 0) ? _tmem_load_8_0[0] : 0.0f);
                        wo_2_0[1] = ((row_valid != 0) ? _tmem_load_0[1] : 0.0f);
                        uo_1_0[1] = ((row_valid != 0) ? _tmem_load_8_0[1] : 0.0f);
                        wo_2_0[2] = ((row_valid != 0) ? _tmem_load_0[2] : 0.0f);
                        uo_1_0[2] = ((row_valid != 0) ? _tmem_load_8_0[2] : 0.0f);
                        wo_2_0[3] = ((row_valid != 0) ? _tmem_load_0[3] : 0.0f);
                        uo_1_0[3] = ((row_valid != 0) ? _tmem_load_8_0[3] : 0.0f);
                        wo_2_0[4] = ((row_valid != 0) ? _tmem_load_0[4] : 0.0f);
                        uo_1_0[4] = ((row_valid != 0) ? _tmem_load_8_0[4] : 0.0f);
                        wo_2_0[5] = ((row_valid != 0) ? _tmem_load_0[5] : 0.0f);
                        uo_1_0[5] = ((row_valid != 0) ? _tmem_load_8_0[5] : 0.0f);
                        wo_2_0[6] = ((row_valid != 0) ? _tmem_load_0[6] : 0.0f);
                        uo_1_0[6] = ((row_valid != 0) ? _tmem_load_8_0[6] : 0.0f);
                        wo_2_0[7] = ((row_valid != 0) ? _tmem_load_0[7] : 0.0f);
                        uo_1_0[7] = ((row_valid != 0) ? _tmem_load_8_0[7] : 0.0f);
                        wo_2_0[8] = ((row_valid != 0) ? _tmem_load_0[8] : 0.0f);
                        uo_1_0[8] = ((row_valid != 0) ? _tmem_load_8_0[8] : 0.0f);
                        wo_2_0[9] = ((row_valid != 0) ? _tmem_load_0[9] : 0.0f);
                        uo_1_0[9] = ((row_valid != 0) ? _tmem_load_8_0[9] : 0.0f);
                        wo_2_0[10] = ((row_valid != 0) ? _tmem_load_0[10] : 0.0f);
                        uo_1_0[10] = ((row_valid != 0) ? _tmem_load_8_0[10] : 0.0f);
                        wo_2_0[11] = ((row_valid != 0) ? _tmem_load_0[11] : 0.0f);
                        uo_1_0[11] = ((row_valid != 0) ? _tmem_load_8_0[11] : 0.0f);
                        wo_2_0[12] = ((row_valid != 0) ? _tmem_load_0[12] : 0.0f);
                        uo_1_0[12] = ((row_valid != 0) ? _tmem_load_8_0[12] : 0.0f);
                        wo_2_0[13] = ((row_valid != 0) ? _tmem_load_0[13] : 0.0f);
                        uo_1_0[13] = ((row_valid != 0) ? _tmem_load_8_0[13] : 0.0f);
                        wo_2_0[14] = ((row_valid != 0) ? _tmem_load_0[14] : 0.0f);
                        uo_1_0[14] = ((row_valid != 0) ? _tmem_load_8_0[14] : 0.0f);
                        wo_2_0[15] = ((row_valid != 0) ? _tmem_load_0[15] : 0.0f);
                        uo_1_0[15] = ((row_valid != 0) ? _tmem_load_8_0[15] : 0.0f);
                        unsigned int pw_0[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 0], wo_2_0[_lp*2+1 + 0]));
                            pw_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 2], wo_2_0[_lp*2+1 + 2]));
                            pw_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 4], wo_2_0[_lp*2+1 + 4]));
                            pw_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 6], wo_2_0[_lp*2+1 + 6]));
                            pw_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 8], wo_2_0[_lp*2+1 + 8]));
                            pw_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 10], wo_2_0[_lp*2+1 + 10]));
                            pw_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 12], wo_2_0[_lp*2+1 + 12]));
                            pw_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 14], wo_2_0[_lp*2+1 + 14]));
                            pw_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_0 * 2), "r"(pw_0[0]), "r"(pw_0[1]), "r"(pw_0[2]), "r"(pw_0[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_0 * 2 + 16), "r"(pw_0[4]), "r"(pw_0[5]), "r"(pw_0[6]), "r"(pw_0[7]) : "memory");
                        unsigned int pu_0[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 0], uo_1_0[_lp*2+1 + 0]));
                            pu_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 2], uo_1_0[_lp*2+1 + 2]));
                            pu_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 4], uo_1_0[_lp*2+1 + 4]));
                            pu_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 6], uo_1_0[_lp*2+1 + 6]));
                            pu_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 8], uo_1_0[_lp*2+1 + 8]));
                            pu_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 10], uo_1_0[_lp*2+1 + 10]));
                            pu_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 12], uo_1_0[_lp*2+1 + 12]));
                            pu_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 14], uo_1_0[_lp*2+1 + 14]));
                            pu_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_0 * 2), "r"(pu_0[0]), "r"(pu_0[1]), "r"(pu_0[2]), "r"(pu_0[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_0 * 2 + 16), "r"(pu_0[4]), "r"(pu_0[5]), "r"(pu_0[6]), "r"(pu_0[7]) : "memory");
                        int col_1_1 = 32 + half * 16;
                        float _tmem_load_1[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                            : "r"(taddr + (unsigned int)row_base + 32));
                        float _tmem_load_8_1[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[15]))
                            : "r"(taddr + (unsigned int)row_base + 160));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float wo_2_1[16];
                        float uo_1_1[16];
                        wo_2_1[0] = ((row_valid != 0) ? _tmem_load_1[0] : 0.0f);
                        uo_1_1[0] = ((row_valid != 0) ? _tmem_load_8_1[0] : 0.0f);
                        wo_2_1[1] = ((row_valid != 0) ? _tmem_load_1[1] : 0.0f);
                        uo_1_1[1] = ((row_valid != 0) ? _tmem_load_8_1[1] : 0.0f);
                        wo_2_1[2] = ((row_valid != 0) ? _tmem_load_1[2] : 0.0f);
                        uo_1_1[2] = ((row_valid != 0) ? _tmem_load_8_1[2] : 0.0f);
                        wo_2_1[3] = ((row_valid != 0) ? _tmem_load_1[3] : 0.0f);
                        uo_1_1[3] = ((row_valid != 0) ? _tmem_load_8_1[3] : 0.0f);
                        wo_2_1[4] = ((row_valid != 0) ? _tmem_load_1[4] : 0.0f);
                        uo_1_1[4] = ((row_valid != 0) ? _tmem_load_8_1[4] : 0.0f);
                        wo_2_1[5] = ((row_valid != 0) ? _tmem_load_1[5] : 0.0f);
                        uo_1_1[5] = ((row_valid != 0) ? _tmem_load_8_1[5] : 0.0f);
                        wo_2_1[6] = ((row_valid != 0) ? _tmem_load_1[6] : 0.0f);
                        uo_1_1[6] = ((row_valid != 0) ? _tmem_load_8_1[6] : 0.0f);
                        wo_2_1[7] = ((row_valid != 0) ? _tmem_load_1[7] : 0.0f);
                        uo_1_1[7] = ((row_valid != 0) ? _tmem_load_8_1[7] : 0.0f);
                        wo_2_1[8] = ((row_valid != 0) ? _tmem_load_1[8] : 0.0f);
                        uo_1_1[8] = ((row_valid != 0) ? _tmem_load_8_1[8] : 0.0f);
                        wo_2_1[9] = ((row_valid != 0) ? _tmem_load_1[9] : 0.0f);
                        uo_1_1[9] = ((row_valid != 0) ? _tmem_load_8_1[9] : 0.0f);
                        wo_2_1[10] = ((row_valid != 0) ? _tmem_load_1[10] : 0.0f);
                        uo_1_1[10] = ((row_valid != 0) ? _tmem_load_8_1[10] : 0.0f);
                        wo_2_1[11] = ((row_valid != 0) ? _tmem_load_1[11] : 0.0f);
                        uo_1_1[11] = ((row_valid != 0) ? _tmem_load_8_1[11] : 0.0f);
                        wo_2_1[12] = ((row_valid != 0) ? _tmem_load_1[12] : 0.0f);
                        uo_1_1[12] = ((row_valid != 0) ? _tmem_load_8_1[12] : 0.0f);
                        wo_2_1[13] = ((row_valid != 0) ? _tmem_load_1[13] : 0.0f);
                        uo_1_1[13] = ((row_valid != 0) ? _tmem_load_8_1[13] : 0.0f);
                        wo_2_1[14] = ((row_valid != 0) ? _tmem_load_1[14] : 0.0f);
                        uo_1_1[14] = ((row_valid != 0) ? _tmem_load_8_1[14] : 0.0f);
                        wo_2_1[15] = ((row_valid != 0) ? _tmem_load_1[15] : 0.0f);
                        uo_1_1[15] = ((row_valid != 0) ? _tmem_load_8_1[15] : 0.0f);
                        unsigned int pw_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 0], wo_2_1[_lp*2+1 + 0]));
                            pw_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 2], wo_2_1[_lp*2+1 + 2]));
                            pw_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 4], wo_2_1[_lp*2+1 + 4]));
                            pw_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 6], wo_2_1[_lp*2+1 + 6]));
                            pw_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 8], wo_2_1[_lp*2+1 + 8]));
                            pw_1[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 10], wo_2_1[_lp*2+1 + 10]));
                            pw_1[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 12], wo_2_1[_lp*2+1 + 12]));
                            pw_1[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 14], wo_2_1[_lp*2+1 + 14]));
                            pw_1[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_1 * 2), "r"(pw_1[0]), "r"(pw_1[1]), "r"(pw_1[2]), "r"(pw_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_1 * 2 + 16), "r"(pw_1[4]), "r"(pw_1[5]), "r"(pw_1[6]), "r"(pw_1[7]) : "memory");
                        unsigned int pu_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 0], uo_1_1[_lp*2+1 + 0]));
                            pu_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 2], uo_1_1[_lp*2+1 + 2]));
                            pu_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 4], uo_1_1[_lp*2+1 + 4]));
                            pu_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 6], uo_1_1[_lp*2+1 + 6]));
                            pu_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 8], uo_1_1[_lp*2+1 + 8]));
                            pu_1[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 10], uo_1_1[_lp*2+1 + 10]));
                            pu_1[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 12], uo_1_1[_lp*2+1 + 12]));
                            pu_1[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 14], uo_1_1[_lp*2+1 + 14]));
                            pu_1[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_1 * 2), "r"(pu_1[0]), "r"(pu_1[1]), "r"(pu_1[2]), "r"(pu_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_1 * 2 + 16), "r"(pu_1[4]), "r"(pu_1[5]), "r"(pu_1[6]), "r"(pu_1[7]) : "memory");
                        int col_1_2 = 64 + half * 16;
                        float _tmem_load_2[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                            : "r"(taddr + (unsigned int)row_base + 64));
                        float _tmem_load_8_2[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[15]))
                            : "r"(taddr + (unsigned int)row_base + 192));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float wo_2_2[16];
                        float uo_1_2[16];
                        wo_2_2[0] = ((row_valid != 0) ? _tmem_load_2[0] : 0.0f);
                        uo_1_2[0] = ((row_valid != 0) ? _tmem_load_8_2[0] : 0.0f);
                        wo_2_2[1] = ((row_valid != 0) ? _tmem_load_2[1] : 0.0f);
                        uo_1_2[1] = ((row_valid != 0) ? _tmem_load_8_2[1] : 0.0f);
                        wo_2_2[2] = ((row_valid != 0) ? _tmem_load_2[2] : 0.0f);
                        uo_1_2[2] = ((row_valid != 0) ? _tmem_load_8_2[2] : 0.0f);
                        wo_2_2[3] = ((row_valid != 0) ? _tmem_load_2[3] : 0.0f);
                        uo_1_2[3] = ((row_valid != 0) ? _tmem_load_8_2[3] : 0.0f);
                        wo_2_2[4] = ((row_valid != 0) ? _tmem_load_2[4] : 0.0f);
                        uo_1_2[4] = ((row_valid != 0) ? _tmem_load_8_2[4] : 0.0f);
                        wo_2_2[5] = ((row_valid != 0) ? _tmem_load_2[5] : 0.0f);
                        uo_1_2[5] = ((row_valid != 0) ? _tmem_load_8_2[5] : 0.0f);
                        wo_2_2[6] = ((row_valid != 0) ? _tmem_load_2[6] : 0.0f);
                        uo_1_2[6] = ((row_valid != 0) ? _tmem_load_8_2[6] : 0.0f);
                        wo_2_2[7] = ((row_valid != 0) ? _tmem_load_2[7] : 0.0f);
                        uo_1_2[7] = ((row_valid != 0) ? _tmem_load_8_2[7] : 0.0f);
                        wo_2_2[8] = ((row_valid != 0) ? _tmem_load_2[8] : 0.0f);
                        uo_1_2[8] = ((row_valid != 0) ? _tmem_load_8_2[8] : 0.0f);
                        wo_2_2[9] = ((row_valid != 0) ? _tmem_load_2[9] : 0.0f);
                        uo_1_2[9] = ((row_valid != 0) ? _tmem_load_8_2[9] : 0.0f);
                        wo_2_2[10] = ((row_valid != 0) ? _tmem_load_2[10] : 0.0f);
                        uo_1_2[10] = ((row_valid != 0) ? _tmem_load_8_2[10] : 0.0f);
                        wo_2_2[11] = ((row_valid != 0) ? _tmem_load_2[11] : 0.0f);
                        uo_1_2[11] = ((row_valid != 0) ? _tmem_load_8_2[11] : 0.0f);
                        wo_2_2[12] = ((row_valid != 0) ? _tmem_load_2[12] : 0.0f);
                        uo_1_2[12] = ((row_valid != 0) ? _tmem_load_8_2[12] : 0.0f);
                        wo_2_2[13] = ((row_valid != 0) ? _tmem_load_2[13] : 0.0f);
                        uo_1_2[13] = ((row_valid != 0) ? _tmem_load_8_2[13] : 0.0f);
                        wo_2_2[14] = ((row_valid != 0) ? _tmem_load_2[14] : 0.0f);
                        uo_1_2[14] = ((row_valid != 0) ? _tmem_load_8_2[14] : 0.0f);
                        wo_2_2[15] = ((row_valid != 0) ? _tmem_load_2[15] : 0.0f);
                        uo_1_2[15] = ((row_valid != 0) ? _tmem_load_8_2[15] : 0.0f);
                        unsigned int pw_2[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 0], wo_2_2[_lp*2+1 + 0]));
                            pw_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 2], wo_2_2[_lp*2+1 + 2]));
                            pw_2[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 4], wo_2_2[_lp*2+1 + 4]));
                            pw_2[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 6], wo_2_2[_lp*2+1 + 6]));
                            pw_2[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 8], wo_2_2[_lp*2+1 + 8]));
                            pw_2[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 10], wo_2_2[_lp*2+1 + 10]));
                            pw_2[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 12], wo_2_2[_lp*2+1 + 12]));
                            pw_2[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 14], wo_2_2[_lp*2+1 + 14]));
                            pw_2[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_2 * 2), "r"(pw_2[0]), "r"(pw_2[1]), "r"(pw_2[2]), "r"(pw_2[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_2 * 2 + 16), "r"(pw_2[4]), "r"(pw_2[5]), "r"(pw_2[6]), "r"(pw_2[7]) : "memory");
                        unsigned int pu_2[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 0], uo_1_2[_lp*2+1 + 0]));
                            pu_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 2], uo_1_2[_lp*2+1 + 2]));
                            pu_2[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 4], uo_1_2[_lp*2+1 + 4]));
                            pu_2[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 6], uo_1_2[_lp*2+1 + 6]));
                            pu_2[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 8], uo_1_2[_lp*2+1 + 8]));
                            pu_2[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 10], uo_1_2[_lp*2+1 + 10]));
                            pu_2[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 12], uo_1_2[_lp*2+1 + 12]));
                            pu_2[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 14], uo_1_2[_lp*2+1 + 14]));
                            pu_2[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_2 * 2), "r"(pu_2[0]), "r"(pu_2[1]), "r"(pu_2[2]), "r"(pu_2[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_2 * 2 + 16), "r"(pu_2[4]), "r"(pu_2[5]), "r"(pu_2[6]), "r"(pu_2[7]) : "memory");
                        int col_1_3 = 96 + half * 16;
                        float _tmem_load_3[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                            : "r"(taddr + (unsigned int)row_base + 96));
                        float _tmem_load_8_3[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[15]))
                            : "r"(taddr + (unsigned int)row_base + 224));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float wo_2_3[16];
                        float uo_1_3[16];
                        wo_2_3[0] = ((row_valid != 0) ? _tmem_load_3[0] : 0.0f);
                        uo_1_3[0] = ((row_valid != 0) ? _tmem_load_8_3[0] : 0.0f);
                        wo_2_3[1] = ((row_valid != 0) ? _tmem_load_3[1] : 0.0f);
                        uo_1_3[1] = ((row_valid != 0) ? _tmem_load_8_3[1] : 0.0f);
                        wo_2_3[2] = ((row_valid != 0) ? _tmem_load_3[2] : 0.0f);
                        uo_1_3[2] = ((row_valid != 0) ? _tmem_load_8_3[2] : 0.0f);
                        wo_2_3[3] = ((row_valid != 0) ? _tmem_load_3[3] : 0.0f);
                        uo_1_3[3] = ((row_valid != 0) ? _tmem_load_8_3[3] : 0.0f);
                        wo_2_3[4] = ((row_valid != 0) ? _tmem_load_3[4] : 0.0f);
                        uo_1_3[4] = ((row_valid != 0) ? _tmem_load_8_3[4] : 0.0f);
                        wo_2_3[5] = ((row_valid != 0) ? _tmem_load_3[5] : 0.0f);
                        uo_1_3[5] = ((row_valid != 0) ? _tmem_load_8_3[5] : 0.0f);
                        wo_2_3[6] = ((row_valid != 0) ? _tmem_load_3[6] : 0.0f);
                        uo_1_3[6] = ((row_valid != 0) ? _tmem_load_8_3[6] : 0.0f);
                        wo_2_3[7] = ((row_valid != 0) ? _tmem_load_3[7] : 0.0f);
                        uo_1_3[7] = ((row_valid != 0) ? _tmem_load_8_3[7] : 0.0f);
                        wo_2_3[8] = ((row_valid != 0) ? _tmem_load_3[8] : 0.0f);
                        uo_1_3[8] = ((row_valid != 0) ? _tmem_load_8_3[8] : 0.0f);
                        wo_2_3[9] = ((row_valid != 0) ? _tmem_load_3[9] : 0.0f);
                        uo_1_3[9] = ((row_valid != 0) ? _tmem_load_8_3[9] : 0.0f);
                        wo_2_3[10] = ((row_valid != 0) ? _tmem_load_3[10] : 0.0f);
                        uo_1_3[10] = ((row_valid != 0) ? _tmem_load_8_3[10] : 0.0f);
                        wo_2_3[11] = ((row_valid != 0) ? _tmem_load_3[11] : 0.0f);
                        uo_1_3[11] = ((row_valid != 0) ? _tmem_load_8_3[11] : 0.0f);
                        wo_2_3[12] = ((row_valid != 0) ? _tmem_load_3[12] : 0.0f);
                        uo_1_3[12] = ((row_valid != 0) ? _tmem_load_8_3[12] : 0.0f);
                        wo_2_3[13] = ((row_valid != 0) ? _tmem_load_3[13] : 0.0f);
                        uo_1_3[13] = ((row_valid != 0) ? _tmem_load_8_3[13] : 0.0f);
                        wo_2_3[14] = ((row_valid != 0) ? _tmem_load_3[14] : 0.0f);
                        uo_1_3[14] = ((row_valid != 0) ? _tmem_load_8_3[14] : 0.0f);
                        wo_2_3[15] = ((row_valid != 0) ? _tmem_load_3[15] : 0.0f);
                        uo_1_3[15] = ((row_valid != 0) ? _tmem_load_8_3[15] : 0.0f);
                        unsigned int pw_3[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 0], wo_2_3[_lp*2+1 + 0]));
                            pw_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 2], wo_2_3[_lp*2+1 + 2]));
                            pw_3[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 4], wo_2_3[_lp*2+1 + 4]));
                            pw_3[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 6], wo_2_3[_lp*2+1 + 6]));
                            pw_3[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 8], wo_2_3[_lp*2+1 + 8]));
                            pw_3[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 10], wo_2_3[_lp*2+1 + 10]));
                            pw_3[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 12], wo_2_3[_lp*2+1 + 12]));
                            pw_3[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 14], wo_2_3[_lp*2+1 + 14]));
                            pw_3[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_3 * 2), "r"(pw_3[0]), "r"(pw_3[1]), "r"(pw_3[2]), "r"(pw_3[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_3 * 2 + 16), "r"(pw_3[4]), "r"(pw_3[5]), "r"(pw_3[6]), "r"(pw_3[7]) : "memory");
                        unsigned int pu_3[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 0], uo_1_3[_lp*2+1 + 0]));
                            pu_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 2], uo_1_3[_lp*2+1 + 2]));
                            pu_3[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 4], uo_1_3[_lp*2+1 + 4]));
                            pu_3[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 6], uo_1_3[_lp*2+1 + 6]));
                            pu_3[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 8], uo_1_3[_lp*2+1 + 8]));
                            pu_3[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 10], uo_1_3[_lp*2+1 + 10]));
                            pu_3[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 12], uo_1_3[_lp*2+1 + 12]));
                            pu_3[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 14], uo_1_3[_lp*2+1 + 14]));
                            pu_3[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_3 * 2), "r"(pu_3[0]), "r"(pu_3[1]), "r"(pu_3[2]), "r"(pu_3[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_3 * 2 + 16), "r"(pu_3[4]), "r"(pu_3[5]), "r"(pu_3[6]), "r"(pu_3[7]) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            long long grow0 = ((long long)(irow0 + w_id * 16) * (long long)num_heads + (long long)(hv - 1)) * 128;
                            long long gpitch = (long long)num_heads * 128;
                            {
                                void* _cpbulk_dst_17 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 0));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_17), "r"(l_s_addr + (unsigned int)((w_id * 16 + 0) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_18 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 1));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_18), "r"(l_s_addr + (unsigned int)((w_id * 16 + 1) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_19 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 2));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_19), "r"(l_s_addr + (unsigned int)((w_id * 16 + 2) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_20 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 3));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_20), "r"(l_s_addr + (unsigned int)((w_id * 16 + 3) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_21 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 4));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_21), "r"(l_s_addr + (unsigned int)((w_id * 16 + 4) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_22 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 5));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_22), "r"(l_s_addr + (unsigned int)((w_id * 16 + 5) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_23 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 6));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_23), "r"(l_s_addr + (unsigned int)((w_id * 16 + 6) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_24 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 7));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_24), "r"(l_s_addr + (unsigned int)((w_id * 16 + 7) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_25 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 8));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_25), "r"(l_s_addr + (unsigned int)((w_id * 16 + 8) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_26 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 9));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_26), "r"(l_s_addr + (unsigned int)((w_id * 16 + 9) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_27 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 10));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_27), "r"(l_s_addr + (unsigned int)((w_id * 16 + 10) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_28 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 11));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_28), "r"(l_s_addr + (unsigned int)((w_id * 16 + 11) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_29 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 12));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_29), "r"(l_s_addr + (unsigned int)((w_id * 16 + 12) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_30 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 13));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_30), "r"(l_s_addr + (unsigned int)((w_id * 16 + 13) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_31 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 14));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_31), "r"(l_s_addr + (unsigned int)((w_id * 16 + 14) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_32 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 15));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_32), "r"(l_s_addr + (unsigned int)((w_id * 16 + 15) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_33 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 0));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_33), "r"(a_s_addr + (unsigned int)((w_id * 16 + 0) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_34 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 1));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_34), "r"(a_s_addr + (unsigned int)((w_id * 16 + 1) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_35 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 2));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_35), "r"(a_s_addr + (unsigned int)((w_id * 16 + 2) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_36 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 3));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_36), "r"(a_s_addr + (unsigned int)((w_id * 16 + 3) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_37 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 4));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_37), "r"(a_s_addr + (unsigned int)((w_id * 16 + 4) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_38 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 5));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_38), "r"(a_s_addr + (unsigned int)((w_id * 16 + 5) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_39 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 6));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_39), "r"(a_s_addr + (unsigned int)((w_id * 16 + 6) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_40 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 7));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_40), "r"(a_s_addr + (unsigned int)((w_id * 16 + 7) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_41 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 8));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_41), "r"(a_s_addr + (unsigned int)((w_id * 16 + 8) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_42 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 9));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_42), "r"(a_s_addr + (unsigned int)((w_id * 16 + 9) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_43 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 10));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_43), "r"(a_s_addr + (unsigned int)((w_id * 16 + 10) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_44 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 11));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_44), "r"(a_s_addr + (unsigned int)((w_id * 16 + 11) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_45 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 12));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_45), "r"(a_s_addr + (unsigned int)((w_id * 16 + 12) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_46 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 13));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_46), "r"(a_s_addr + (unsigned int)((w_id * 16 + 13) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_47 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 14));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_47), "r"(a_s_addr + (unsigned int)((w_id * 16 + 14) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_48 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 15));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_48), "r"(a_s_addr + (unsigned int)((w_id * 16 + 15) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (group > j) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(op_ready_addr);
                        vst_c += 1;
                        if (vst_c == 2) { vst_c = 0; _phase_v_full ^= 1; }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                }
                asm volatile("cp.async.bulk.wait_group 0;");
            } else {
                #pragma unroll 1
                for (int j_1 = 0; j_1 < group; j_1++) {
                    int hv_p = hq * group + j_1;
                    if (tid_0 < 64) {
                        long long rbase = (long long)(irow0 + tid_0) * (long long)num_heads + (long long)hv_p;
                        unsigned int _packed_6[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_6[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_6[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_6[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_6[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_49 = reinterpret_cast<const unsigned*>(_packed_6);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128))), "r"(_raw_stv8_49[0]), "r"(_raw_stv8_49[1]), "r"(_raw_stv8_49[2]), "r"(_raw_stv8_49[3]), "r"(_raw_stv8_49[4]), "r"(_raw_stv8_49[5]), "r"(_raw_stv8_49[6]), "r"(_raw_stv8_49[7]) : "memory");
                        }
                        unsigned int _packed_38_0[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_38_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_38_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_38_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_38_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_38_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_38_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_38_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_38_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_50 = reinterpret_cast<const unsigned*>(_packed_38_0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128))), "r"(_raw_stv8_50[0]), "r"(_raw_stv8_50[1]), "r"(_raw_stv8_50[2]), "r"(_raw_stv8_50[3]), "r"(_raw_stv8_50[4]), "r"(_raw_stv8_50[5]), "r"(_raw_stv8_50[6]), "r"(_raw_stv8_50[7]) : "memory");
                        }
                        unsigned int _packed_7[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_7[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_7[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_7[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_7[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_51 = reinterpret_cast<const unsigned*>(_packed_7);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 16))), "r"(_raw_stv8_51[0]), "r"(_raw_stv8_51[1]), "r"(_raw_stv8_51[2]), "r"(_raw_stv8_51[3]), "r"(_raw_stv8_51[4]), "r"(_raw_stv8_51[5]), "r"(_raw_stv8_51[6]), "r"(_raw_stv8_51[7]) : "memory");
                        }
                        unsigned int _packed_38_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_38_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_38_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_38_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_38_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_38_1[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_38_1[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_38_1[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_38_1[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_52 = reinterpret_cast<const unsigned*>(_packed_38_1);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 16))), "r"(_raw_stv8_52[0]), "r"(_raw_stv8_52[1]), "r"(_raw_stv8_52[2]), "r"(_raw_stv8_52[3]), "r"(_raw_stv8_52[4]), "r"(_raw_stv8_52[5]), "r"(_raw_stv8_52[6]), "r"(_raw_stv8_52[7]) : "memory");
                        }
                        unsigned int _packed_8[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_8[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_8[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_8[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_8[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_8[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_8[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_8[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_53 = reinterpret_cast<const unsigned*>(_packed_8);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 32))), "r"(_raw_stv8_53[0]), "r"(_raw_stv8_53[1]), "r"(_raw_stv8_53[2]), "r"(_raw_stv8_53[3]), "r"(_raw_stv8_53[4]), "r"(_raw_stv8_53[5]), "r"(_raw_stv8_53[6]), "r"(_raw_stv8_53[7]) : "memory");
                        }
                        unsigned int _packed_38_2[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_38_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_38_2[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_38_2[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_38_2[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_38_2[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_38_2[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_38_2[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_38_2[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_54 = reinterpret_cast<const unsigned*>(_packed_38_2);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 32))), "r"(_raw_stv8_54[0]), "r"(_raw_stv8_54[1]), "r"(_raw_stv8_54[2]), "r"(_raw_stv8_54[3]), "r"(_raw_stv8_54[4]), "r"(_raw_stv8_54[5]), "r"(_raw_stv8_54[6]), "r"(_raw_stv8_54[7]) : "memory");
                        }
                        unsigned int _packed_9[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_9[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_9[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_9[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_9[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_9[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_9[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_9[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_9[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_55 = reinterpret_cast<const unsigned*>(_packed_9);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 48))), "r"(_raw_stv8_55[0]), "r"(_raw_stv8_55[1]), "r"(_raw_stv8_55[2]), "r"(_raw_stv8_55[3]), "r"(_raw_stv8_55[4]), "r"(_raw_stv8_55[5]), "r"(_raw_stv8_55[6]), "r"(_raw_stv8_55[7]) : "memory");
                        }
                        unsigned int _packed_38_3[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_38_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_38_3[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_38_3[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_38_3[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_38_3[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_38_3[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_38_3[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_38_3[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_56 = reinterpret_cast<const unsigned*>(_packed_38_3);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 48))), "r"(_raw_stv8_56[0]), "r"(_raw_stv8_56[1]), "r"(_raw_stv8_56[2]), "r"(_raw_stv8_56[3]), "r"(_raw_stv8_56[4]), "r"(_raw_stv8_56[5]), "r"(_raw_stv8_56[6]), "r"(_raw_stv8_56[7]) : "memory");
                        }
                        unsigned int _packed_10[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_10[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_10[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_10[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_10[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_10[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_10[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_10[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_10[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_57 = reinterpret_cast<const unsigned*>(_packed_10);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 64))), "r"(_raw_stv8_57[0]), "r"(_raw_stv8_57[1]), "r"(_raw_stv8_57[2]), "r"(_raw_stv8_57[3]), "r"(_raw_stv8_57[4]), "r"(_raw_stv8_57[5]), "r"(_raw_stv8_57[6]), "r"(_raw_stv8_57[7]) : "memory");
                        }
                        unsigned int _packed_38_4[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_38_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_38_4[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_38_4[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_38_4[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_38_4[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_38_4[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_38_4[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_38_4[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_58 = reinterpret_cast<const unsigned*>(_packed_38_4);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 64))), "r"(_raw_stv8_58[0]), "r"(_raw_stv8_58[1]), "r"(_raw_stv8_58[2]), "r"(_raw_stv8_58[3]), "r"(_raw_stv8_58[4]), "r"(_raw_stv8_58[5]), "r"(_raw_stv8_58[6]), "r"(_raw_stv8_58[7]) : "memory");
                        }
                        unsigned int _packed_11[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_11[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_11[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_11[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_11[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_11[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_11[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_11[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_11[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_59 = reinterpret_cast<const unsigned*>(_packed_11);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 80))), "r"(_raw_stv8_59[0]), "r"(_raw_stv8_59[1]), "r"(_raw_stv8_59[2]), "r"(_raw_stv8_59[3]), "r"(_raw_stv8_59[4]), "r"(_raw_stv8_59[5]), "r"(_raw_stv8_59[6]), "r"(_raw_stv8_59[7]) : "memory");
                        }
                        unsigned int _packed_38_5[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_38_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_38_5[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_38_5[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_38_5[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_38_5[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_38_5[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_38_5[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_38_5[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_60 = reinterpret_cast<const unsigned*>(_packed_38_5);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 80))), "r"(_raw_stv8_60[0]), "r"(_raw_stv8_60[1]), "r"(_raw_stv8_60[2]), "r"(_raw_stv8_60[3]), "r"(_raw_stv8_60[4]), "r"(_raw_stv8_60[5]), "r"(_raw_stv8_60[6]), "r"(_raw_stv8_60[7]) : "memory");
                        }
                        unsigned int _packed_12[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_12[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_12[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_12[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_12[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_12[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_12[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_12[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_12[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_61 = reinterpret_cast<const unsigned*>(_packed_12);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 96))), "r"(_raw_stv8_61[0]), "r"(_raw_stv8_61[1]), "r"(_raw_stv8_61[2]), "r"(_raw_stv8_61[3]), "r"(_raw_stv8_61[4]), "r"(_raw_stv8_61[5]), "r"(_raw_stv8_61[6]), "r"(_raw_stv8_61[7]) : "memory");
                        }
                        unsigned int _packed_38_6[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_38_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_38_6[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_38_6[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_38_6[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_38_6[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_38_6[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_38_6[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_38_6[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_62 = reinterpret_cast<const unsigned*>(_packed_38_6);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 96))), "r"(_raw_stv8_62[0]), "r"(_raw_stv8_62[1]), "r"(_raw_stv8_62[2]), "r"(_raw_stv8_62[3]), "r"(_raw_stv8_62[4]), "r"(_raw_stv8_62[5]), "r"(_raw_stv8_62[6]), "r"(_raw_stv8_62[7]) : "memory");
                        }
                        unsigned int _packed_13[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_13[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_13[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_13[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_13[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_13[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_13[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_13[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_13[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_63 = reinterpret_cast<const unsigned*>(_packed_13);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 112))), "r"(_raw_stv8_63[0]), "r"(_raw_stv8_63[1]), "r"(_raw_stv8_63[2]), "r"(_raw_stv8_63[3]), "r"(_raw_stv8_63[4]), "r"(_raw_stv8_63[5]), "r"(_raw_stv8_63[6]), "r"(_raw_stv8_63[7]) : "memory");
                        }
                        unsigned int _packed_38_7[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_38_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_38_7[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_38_7[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_38_7[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_38_7[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_38_7[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_38_7[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_38_7[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_64 = reinterpret_cast<const unsigned*>(_packed_38_7);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 112))), "r"(_raw_stv8_64[0]), "r"(_raw_stv8_64[1]), "r"(_raw_stv8_64[2]), "r"(_raw_stv8_64[3]), "r"(_raw_stv8_64[4]), "r"(_raw_stv8_64[5]), "r"(_raw_stv8_64[6]), "r"(_raw_stv8_64[7]) : "memory");
                        }
                        unsigned int _packed_14[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_14[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_14[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_14[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_14[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_14[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_14[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_14[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_14[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_65 = reinterpret_cast<const unsigned*>(_packed_14);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (rbase * 64))), "r"(_raw_stv8_65[0]), "r"(_raw_stv8_65[1]), "r"(_raw_stv8_65[2]), "r"(_raw_stv8_65[3]), "r"(_raw_stv8_65[4]), "r"(_raw_stv8_65[5]), "r"(_raw_stv8_65[6]), "r"(_raw_stv8_65[7]) : "memory");
                        }
                        unsigned int _packed_38_8[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_38_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_38_8[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_38_8[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_38_8[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_38_8[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_38_8[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_38_8[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_38_8[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_66 = reinterpret_cast<const unsigned*>(_packed_38_8);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(akk_out + (rbase * 64))), "r"(_raw_stv8_66[0]), "r"(_raw_stv8_66[1]), "r"(_raw_stv8_66[2]), "r"(_raw_stv8_66[3]), "r"(_raw_stv8_66[4]), "r"(_raw_stv8_66[5]), "r"(_raw_stv8_66[6]), "r"(_raw_stv8_66[7]) : "memory");
                        }
                        unsigned int _packed_15[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_15[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_15[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_15[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_15[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_15[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_15[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_15[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_15[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_67 = reinterpret_cast<const unsigned*>(_packed_15);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (rbase * 64 + 16))), "r"(_raw_stv8_67[0]), "r"(_raw_stv8_67[1]), "r"(_raw_stv8_67[2]), "r"(_raw_stv8_67[3]), "r"(_raw_stv8_67[4]), "r"(_raw_stv8_67[5]), "r"(_raw_stv8_67[6]), "r"(_raw_stv8_67[7]) : "memory");
                        }
                        unsigned int _packed_38_9[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_38_9[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_38_9[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_38_9[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_38_9[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_38_9[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_38_9[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_38_9[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_38_9[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_68 = reinterpret_cast<const unsigned*>(_packed_38_9);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(akk_out + (rbase * 64 + 16))), "r"(_raw_stv8_68[0]), "r"(_raw_stv8_68[1]), "r"(_raw_stv8_68[2]), "r"(_raw_stv8_68[3]), "r"(_raw_stv8_68[4]), "r"(_raw_stv8_68[5]), "r"(_raw_stv8_68[6]), "r"(_raw_stv8_68[7]) : "memory");
                        }
                        unsigned int _packed_16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_16[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_16[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_16[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_16[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_16[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_16[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_16[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_69 = reinterpret_cast<const unsigned*>(_packed_16);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (rbase * 64 + 32))), "r"(_raw_stv8_69[0]), "r"(_raw_stv8_69[1]), "r"(_raw_stv8_69[2]), "r"(_raw_stv8_69[3]), "r"(_raw_stv8_69[4]), "r"(_raw_stv8_69[5]), "r"(_raw_stv8_69[6]), "r"(_raw_stv8_69[7]) : "memory");
                        }
                        unsigned int _packed_38_10[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_38_10[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_38_10[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_38_10[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_38_10[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_38_10[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_38_10[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_38_10[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_38_10[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_70 = reinterpret_cast<const unsigned*>(_packed_38_10);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(akk_out + (rbase * 64 + 32))), "r"(_raw_stv8_70[0]), "r"(_raw_stv8_70[1]), "r"(_raw_stv8_70[2]), "r"(_raw_stv8_70[3]), "r"(_raw_stv8_70[4]), "r"(_raw_stv8_70[5]), "r"(_raw_stv8_70[6]), "r"(_raw_stv8_70[7]) : "memory");
                        }
                        unsigned int _packed_17[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_17[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_17[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_17[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_17[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_17[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_17[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_17[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_17[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_71 = reinterpret_cast<const unsigned*>(_packed_17);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (rbase * 64 + 48))), "r"(_raw_stv8_71[0]), "r"(_raw_stv8_71[1]), "r"(_raw_stv8_71[2]), "r"(_raw_stv8_71[3]), "r"(_raw_stv8_71[4]), "r"(_raw_stv8_71[5]), "r"(_raw_stv8_71[6]), "r"(_raw_stv8_71[7]) : "memory");
                        }
                        unsigned int _packed_38_11[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_38_11[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_38_11[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_38_11[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_38_11[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_38_11[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_38_11[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_38_11[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_38_11[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_72 = reinterpret_cast<const unsigned*>(_packed_38_11);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(akk_out + (rbase * 64 + 48))), "r"(_raw_stv8_72[0]), "r"(_raw_stv8_72[1]), "r"(_raw_stv8_72[2]), "r"(_raw_stv8_72[3]), "r"(_raw_stv8_72[4]), "r"(_raw_stv8_72[5]), "r"(_raw_stv8_72[6]), "r"(_raw_stv8_72[7]) : "memory");
                        }
                        *(reinterpret_cast<float*>(g_cum + rbase) + (0)) = 0.0f;
                        *(reinterpret_cast<float*>(rowscale + rbase) + (0)) = 0.0f;
                        *(reinterpret_cast<float*>(qscale + rbase) + (0)) = 0.0f;
                    }
                    if (tid_0 == 0) {
                        *(reinterpret_cast<float*>(gamma + ((long long)chunk * (long long)num_heads + (long long)hv_p)) + (0)) = 1.0f;
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (w_id == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            unsigned int vst_l = 0;
#if __CUDA_ARCH__ == 1070
#else
            int lane_l = lane;
            int lin_n = chunk + gridDim.x * hq + 152;
            int c_n = lin_n % gridDim.x;
            int hq_n = lin_n / gridDim.x;
            int ext0_n = chunk_bos[c_n];
            int clen_n = chunk_len[c_n];
            int nrow_n = ((hq_n < gridDim.y) ? clen_n : 0);
            long long tokp0 = (long long)(ext0_n + lane_l) * (long long)num_heads + (long long)(hq_n * group);
            long long tokp1 = (long long)(ext0_n + 32 + lane_l) * (long long)num_heads + (long long)(hq_n * group);
            if (lane_l < nrow_n) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(g + tokp0))); }
            if (lane_l < nrow_n) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(beta + tokp0))); }
            if (32 + lane_l < nrow_n) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(g + tokp1))); }
            if (32 + lane_l < nrow_n) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(beta + tokp1))); }
#endif
            unsigned int _phase_v_empty = 1;
            if (clen > 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(qk_full_addr, 32768);
                    int c0_1_0 = 0;
                    tma_3d_gmem2smem(q_p_addr, (&q_tma), c0_1_0, hq, ext0, qk_full_addr);
                    tma_3d_gmem2smem(k_p_addr, (&k_tma), c0_1_0, hq, ext0, qk_full_addr);
                    int c0_1_1 = 64;
                    tma_3d_gmem2smem(q_p_addr + 8192, (&q_tma), c0_1_1, hq, ext0, qk_full_addr);
                    tma_3d_gmem2smem(k_p_addr + 8192, (&k_tma), c0_1_1, hq, ext0, qk_full_addr);
                    #pragma unroll 1
                    for (int j_2 = 0; j_2 < group; j_2++) {
                        int hv_l = hq * group + j_2;
                        mbarrier_wait(v_empty_addr + (vst_l) * 8, _phase_v_empty);
                        mbarrier_arrive_expect_tx(v_full_addr + (vst_l) * 8, 16384);
                        int c0v_1_0 = 0;
                        tma_3d_gmem2smem(v_p_addr + vst_l * 16384, (&v_tma), c0v_1_0, hv_l, ext0, v_full_addr + (vst_l) * 8);
                        int c0v_1_1 = 64;
                        tma_3d_gmem2smem(v_p_addr + vst_l * 16384 + 8192, (&v_tma), c0v_1_1, hv_l, ext0, v_full_addr + (vst_l) * 8);
                        vst_l += 1;
                        if (vst_l == 2) { vst_l = 0; _phase_v_empty ^= 1; }
                    }
#if __CUDA_ARCH__ == 1070
#else
                    if (nrow_n > 0) {
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&q_tma))), "r"((int)(0)), "r"((int)(hq_n)), "r"((int)(ext0_n)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&k_tma))), "r"((int)(0)), "r"((int)(hq_n)), "r"((int)(ext0_n)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&q_tma))), "r"((int)(64)), "r"((int)(hq_n)), "r"((int)(ext0_n)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&k_tma))), "r"((int)(64)), "r"((int)(hq_n)), "r"((int)(ext0_n)) : "memory");
                        #pragma unroll 1
                        for (int jp = 0; jp < group; jp++) {
                            int hv_n = hq_n * group + jp;
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&v_tma))), "r"((int)(0)), "r"((int)(hv_n)), "r"((int)(ext0_n)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&v_tma))), "r"((int)(64)), "r"((int)(hv_n)), "r"((int)(ext0_n)) : "memory");
                        }
                    }
#endif
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            unsigned int vst_m = 0;
            unsigned int _phase_qk_full_0_1 = 0;
            unsigned int _phase_op_ready_0 = 0;
            if (clen > 0) {
                if (elect_sync()) {
                    mbarrier_wait(qk_full_addr, _phase_qk_full_0_1);
                    _phase_qk_full_0_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_0 = ((q_k_addr) >> 4) & 0x7FFF;
                    int _mma_b_lo_0 = ((k_k_addr) >> 4) & 0x7FFF;
#else
                    int _mma_a_lo_0 = ((q_k_addr) >> 4) & 0x3FFF;
                    int _mma_b_lo_0 = ((k_k_addr) >> 4) & 0x3FFF;
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc), "r"(0) : "memory");
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
                    :: "r"(_mma_b_lo_0), "r"(_mma_b_lo_0), "r"((tmem_acc + (64))), "r"(0) : "memory");
                    tcgen05_commit(gram_full_addr);
                    #pragma unroll 1
                    for (int _j = 0; _j < group; _j++) {
                        mbarrier_wait(op_ready_addr, _phase_op_ready_0);
                        _phase_op_ready_0 ^= 1;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_1 = ((akk_t_addr) >> 4) & 0x7FFF;
                        int _smem_desc_lo_0 = (((kb_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _mma_a_lo_1 = ((akk_t_addr) >> 4) & 0x3FFF;
                        int _smem_desc_lo_0 = (((kb_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_smem_desc_lo_0), "r"(tmem_acc), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_1 = ((((v_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (vst_m) * 1024;
#else
                        int _smem_desc_lo_1 = ((((v_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (vst_m) * 1024;
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
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_smem_desc_lo_1), "r"((tmem_acc + (128))), "r"(0) : "memory");
                        tcgen05_commit(wu_full_addr);
                        tcgen05_commit(v_empty_addr + (vst_m) * 8);
                        vst_m += 1;
                        if (vst_m == 2) { vst_m = 0; }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
