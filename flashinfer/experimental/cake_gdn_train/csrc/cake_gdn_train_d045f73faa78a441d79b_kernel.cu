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
                            float y0 = __shfl_up_sync(0xFFFFFFFF, x0, 1, 32);
                            float y1 = __shfl_up_sync(0xFFFFFFFF, x1, 1, 32);
                            x0 = x0 + ((lane_1 >= 1) ? y0 : 0.0f);
                            x1 = x1 + ((lane_1 >= 1) ? y1 : 0.0f);
                            float y0_0 = __shfl_up_sync(0xFFFFFFFF, x0, 2, 32);
                            float y1_1 = __shfl_up_sync(0xFFFFFFFF, x1, 2, 32);
                            x0 = x0 + ((lane_1 >= 2) ? y0_0 : 0.0f);
                            x1 = x1 + ((lane_1 >= 2) ? y1_1 : 0.0f);
                            float y0_2 = __shfl_up_sync(0xFFFFFFFF, x0, 4, 32);
                            float y1_3 = __shfl_up_sync(0xFFFFFFFF, x1, 4, 32);
                            x0 = x0 + ((lane_1 >= 4) ? y0_2 : 0.0f);
                            x1 = x1 + ((lane_1 >= 4) ? y1_3 : 0.0f);
                            float y0_4 = __shfl_up_sync(0xFFFFFFFF, x0, 8, 32);
                            float y1_5 = __shfl_up_sync(0xFFFFFFFF, x1, 8, 32);
                            x0 = x0 + ((lane_1 >= 8) ? y0_4 : 0.0f);
                            x1 = x1 + ((lane_1 >= 8) ? y1_5 : 0.0f);
                            float y0_6 = __shfl_up_sync(0xFFFFFFFF, x0, 16, 32);
                            float y1_7 = __shfl_up_sync(0xFFFFFFFF, x1, 16, 32);
                            x0 = x0 + ((lane_1 >= 16) ? y0_6 : 0.0f);
                            x1 = x1 + ((lane_1 >= 16) ? y1_7 : 0.0f);
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
                        int s0 = half * 16;
                        float gsv[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv[0])), "=r"(*reinterpret_cast<uint32_t*>(&gsv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv[(0) + 3]))
                            : "r"(gc_base + s0 * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv[4])), "=r"(*reinterpret_cast<uint32_t*>(&gsv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv[(4) + 3]))
                            : "r"(gc_base + (s0 + 4) * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv[8])), "=r"(*reinterpret_cast<uint32_t*>(&gsv[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv[(8) + 3]))
                            : "r"(gc_base + (s0 + 8) * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv[12])), "=r"(*reinterpret_cast<uint32_t*>(&gsv[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv[(12) + 3]))
                            : "r"(gc_base + (s0 + 12) * 4));
                        float aq[16];
                        float lvs[16];
                        int s = s0;
                        float fac = approx_exp2(fminf(gtr - gsv[0], 0.0f) * 1.442695f);
                        int on_aqk = 0;
                        int on_l = 0;
                        if (s <= t_row) {
                            on_aqk = row_valid;
                        }
                        if (s < t_row) {
                            on_l = row_valid;
                        }
                        aq[0] = ((on_aqk != 0) ? sqk[0] * scale * fac : 0.0f);
                        lvs[0] = ((on_l != 0) ? bt * skk[0] * fac : 0.0f);
                        int s_0 = s0 + 1;
                        float fac_1 = approx_exp2(fminf(gtr - gsv[1], 0.0f) * 1.442695f);
                        int on_aqk_2 = 0;
                        int on_l_3 = 0;
                        if (s_0 <= t_row) {
                            on_aqk_2 = row_valid;
                        }
                        if (s_0 < t_row) {
                            on_l_3 = row_valid;
                        }
                        aq[1] = ((on_aqk_2 != 0) ? sqk[1] * scale * fac_1 : 0.0f);
                        lvs[1] = ((on_l_3 != 0) ? bt * skk[1] * fac_1 : 0.0f);
                        int s_4 = s0 + 2;
                        float fac_5 = approx_exp2(fminf(gtr - gsv[2], 0.0f) * 1.442695f);
                        int on_aqk_6 = 0;
                        int on_l_7 = 0;
                        if (s_4 <= t_row) {
                            on_aqk_6 = row_valid;
                        }
                        if (s_4 < t_row) {
                            on_l_7 = row_valid;
                        }
                        aq[2] = ((on_aqk_6 != 0) ? sqk[2] * scale * fac_5 : 0.0f);
                        lvs[2] = ((on_l_7 != 0) ? bt * skk[2] * fac_5 : 0.0f);
                        int s_8 = s0 + 3;
                        float fac_9 = approx_exp2(fminf(gtr - gsv[3], 0.0f) * 1.442695f);
                        int on_aqk_10 = 0;
                        int on_l_11 = 0;
                        if (s_8 <= t_row) {
                            on_aqk_10 = row_valid;
                        }
                        if (s_8 < t_row) {
                            on_l_11 = row_valid;
                        }
                        aq[3] = ((on_aqk_10 != 0) ? sqk[3] * scale * fac_9 : 0.0f);
                        lvs[3] = ((on_l_11 != 0) ? bt * skk[3] * fac_9 : 0.0f);
                        int s_12 = s0 + 4;
                        float fac_13 = approx_exp2(fminf(gtr - gsv[4], 0.0f) * 1.442695f);
                        int on_aqk_14 = 0;
                        int on_l_15 = 0;
                        if (s_12 <= t_row) {
                            on_aqk_14 = row_valid;
                        }
                        if (s_12 < t_row) {
                            on_l_15 = row_valid;
                        }
                        aq[4] = ((on_aqk_14 != 0) ? sqk[4] * scale * fac_13 : 0.0f);
                        lvs[4] = ((on_l_15 != 0) ? bt * skk[4] * fac_13 : 0.0f);
                        int s_16 = s0 + 5;
                        float fac_17 = approx_exp2(fminf(gtr - gsv[5], 0.0f) * 1.442695f);
                        int on_aqk_18 = 0;
                        int on_l_19 = 0;
                        if (s_16 <= t_row) {
                            on_aqk_18 = row_valid;
                        }
                        if (s_16 < t_row) {
                            on_l_19 = row_valid;
                        }
                        aq[5] = ((on_aqk_18 != 0) ? sqk[5] * scale * fac_17 : 0.0f);
                        lvs[5] = ((on_l_19 != 0) ? bt * skk[5] * fac_17 : 0.0f);
                        int s_20 = s0 + 6;
                        float fac_21 = approx_exp2(fminf(gtr - gsv[6], 0.0f) * 1.442695f);
                        int on_aqk_22 = 0;
                        int on_l_23 = 0;
                        if (s_20 <= t_row) {
                            on_aqk_22 = row_valid;
                        }
                        if (s_20 < t_row) {
                            on_l_23 = row_valid;
                        }
                        aq[6] = ((on_aqk_22 != 0) ? sqk[6] * scale * fac_21 : 0.0f);
                        lvs[6] = ((on_l_23 != 0) ? bt * skk[6] * fac_21 : 0.0f);
                        int s_24 = s0 + 7;
                        float fac_25 = approx_exp2(fminf(gtr - gsv[7], 0.0f) * 1.442695f);
                        int on_aqk_26 = 0;
                        int on_l_27 = 0;
                        if (s_24 <= t_row) {
                            on_aqk_26 = row_valid;
                        }
                        if (s_24 < t_row) {
                            on_l_27 = row_valid;
                        }
                        aq[7] = ((on_aqk_26 != 0) ? sqk[7] * scale * fac_25 : 0.0f);
                        lvs[7] = ((on_l_27 != 0) ? bt * skk[7] * fac_25 : 0.0f);
                        int s_28 = s0 + 8;
                        float fac_29 = approx_exp2(fminf(gtr - gsv[8], 0.0f) * 1.442695f);
                        int on_aqk_30 = 0;
                        int on_l_31 = 0;
                        if (s_28 <= t_row) {
                            on_aqk_30 = row_valid;
                        }
                        if (s_28 < t_row) {
                            on_l_31 = row_valid;
                        }
                        aq[8] = ((on_aqk_30 != 0) ? sqk[8] * scale * fac_29 : 0.0f);
                        lvs[8] = ((on_l_31 != 0) ? bt * skk[8] * fac_29 : 0.0f);
                        int s_32 = s0 + 9;
                        float fac_33 = approx_exp2(fminf(gtr - gsv[9], 0.0f) * 1.442695f);
                        int on_aqk_34 = 0;
                        int on_l_35 = 0;
                        if (s_32 <= t_row) {
                            on_aqk_34 = row_valid;
                        }
                        if (s_32 < t_row) {
                            on_l_35 = row_valid;
                        }
                        aq[9] = ((on_aqk_34 != 0) ? sqk[9] * scale * fac_33 : 0.0f);
                        lvs[9] = ((on_l_35 != 0) ? bt * skk[9] * fac_33 : 0.0f);
                        int s_36 = s0 + 10;
                        float fac_37 = approx_exp2(fminf(gtr - gsv[10], 0.0f) * 1.442695f);
                        int on_aqk_38 = 0;
                        int on_l_39 = 0;
                        if (s_36 <= t_row) {
                            on_aqk_38 = row_valid;
                        }
                        if (s_36 < t_row) {
                            on_l_39 = row_valid;
                        }
                        aq[10] = ((on_aqk_38 != 0) ? sqk[10] * scale * fac_37 : 0.0f);
                        lvs[10] = ((on_l_39 != 0) ? bt * skk[10] * fac_37 : 0.0f);
                        int s_40 = s0 + 11;
                        float fac_41 = approx_exp2(fminf(gtr - gsv[11], 0.0f) * 1.442695f);
                        int on_aqk_42 = 0;
                        int on_l_43 = 0;
                        if (s_40 <= t_row) {
                            on_aqk_42 = row_valid;
                        }
                        if (s_40 < t_row) {
                            on_l_43 = row_valid;
                        }
                        aq[11] = ((on_aqk_42 != 0) ? sqk[11] * scale * fac_41 : 0.0f);
                        lvs[11] = ((on_l_43 != 0) ? bt * skk[11] * fac_41 : 0.0f);
                        int s_44 = s0 + 12;
                        float fac_45 = approx_exp2(fminf(gtr - gsv[12], 0.0f) * 1.442695f);
                        int on_aqk_46 = 0;
                        int on_l_47 = 0;
                        if (s_44 <= t_row) {
                            on_aqk_46 = row_valid;
                        }
                        if (s_44 < t_row) {
                            on_l_47 = row_valid;
                        }
                        aq[12] = ((on_aqk_46 != 0) ? sqk[12] * scale * fac_45 : 0.0f);
                        lvs[12] = ((on_l_47 != 0) ? bt * skk[12] * fac_45 : 0.0f);
                        int s_48 = s0 + 13;
                        float fac_49 = approx_exp2(fminf(gtr - gsv[13], 0.0f) * 1.442695f);
                        int on_aqk_50 = 0;
                        int on_l_51 = 0;
                        if (s_48 <= t_row) {
                            on_aqk_50 = row_valid;
                        }
                        if (s_48 < t_row) {
                            on_l_51 = row_valid;
                        }
                        aq[13] = ((on_aqk_50 != 0) ? sqk[13] * scale * fac_49 : 0.0f);
                        lvs[13] = ((on_l_51 != 0) ? bt * skk[13] * fac_49 : 0.0f);
                        int s_52 = s0 + 14;
                        float fac_53 = approx_exp2(fminf(gtr - gsv[14], 0.0f) * 1.442695f);
                        int on_aqk_54 = 0;
                        int on_l_55 = 0;
                        if (s_52 <= t_row) {
                            on_aqk_54 = row_valid;
                        }
                        if (s_52 < t_row) {
                            on_l_55 = row_valid;
                        }
                        aq[14] = ((on_aqk_54 != 0) ? sqk[14] * scale * fac_53 : 0.0f);
                        lvs[14] = ((on_l_55 != 0) ? bt * skk[14] * fac_53 : 0.0f);
                        int s_56 = s0 + 15;
                        float fac_57 = approx_exp2(fminf(gtr - gsv[15], 0.0f) * 1.442695f);
                        int on_aqk_58 = 0;
                        int on_l_59 = 0;
                        if (s_56 <= t_row) {
                            on_aqk_58 = row_valid;
                        }
                        if (s_56 < t_row) {
                            on_l_59 = row_valid;
                        }
                        aq[15] = ((on_aqk_58 != 0) ? sqk[15] * scale * fac_57 : 0.0f);
                        lvs[15] = ((on_l_59 != 0) ? bt * skk[15] * fac_57 : 0.0f);
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0) * 4), "f"(lvs[0]), "f"(lvs[1]), "f"(lvs[2]), "f"(lvs[3]) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0 + 4) * 4), "f"(lvs[4]), "f"(lvs[5]), "f"(lvs[6]), "f"(lvs[7]) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0 + 8) * 4), "f"(lvs[8]), "f"(lvs[9]), "f"(lvs[10]), "f"(lvs[11]) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0 + 12) * 4), "f"(lvs[12]), "f"(lvs[13]), "f"(lvs[14]), "f"(lvs[15]) : "memory");
                        unsigned int _packed_0[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq[_lp*2 + 0], aq[_lp*2+1 + 0]));
                            _packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq[_lp*2 + 2], aq[_lp*2+1 + 2]));
                            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq[_lp*2 + 4], aq[_lp*2+1 + 4]));
                            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq[_lp*2 + 6], aq[_lp*2+1 + 6]));
                            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq[_lp*2 + 8], aq[_lp*2+1 + 8]));
                            _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq[_lp*2 + 10], aq[_lp*2+1 + 10]));
                            _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq[_lp*2 + 12], aq[_lp*2+1 + 12]));
                            _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq[_lp*2 + 14], aq[_lp*2+1 + 14]));
                            _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(_packed_0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (aqk_base + (long long)s0))), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
                        }
                        int s0_60 = 32 + half * 16;
                        float gsv_61[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[0])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[(0) + 3]))
                            : "r"(gc_base + s0_60 * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[4])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[(4) + 3]))
                            : "r"(gc_base + (s0_60 + 4) * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[8])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[(8) + 3]))
                            : "r"(gc_base + (s0_60 + 8) * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[12])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_61[(12) + 3]))
                            : "r"(gc_base + (s0_60 + 12) * 4));
                        float aq_62[16];
                        float lvs_63[16];
                        int s_64 = s0_60;
                        float fac_65 = approx_exp2(fminf(gtr - gsv_61[0], 0.0f) * 1.442695f);
                        int on_aqk_66 = 0;
                        int on_l_67 = 0;
                        if (s_64 <= t_row) {
                            on_aqk_66 = row_valid;
                        }
                        if (s_64 < t_row) {
                            on_l_67 = row_valid;
                        }
                        aq_62[0] = ((on_aqk_66 != 0) ? sqk[16] * scale * fac_65 : 0.0f);
                        lvs_63[0] = ((on_l_67 != 0) ? bt * skk[16] * fac_65 : 0.0f);
                        int s_68 = s0_60 + 1;
                        float fac_69 = approx_exp2(fminf(gtr - gsv_61[1], 0.0f) * 1.442695f);
                        int on_aqk_70 = 0;
                        int on_l_71 = 0;
                        if (s_68 <= t_row) {
                            on_aqk_70 = row_valid;
                        }
                        if (s_68 < t_row) {
                            on_l_71 = row_valid;
                        }
                        aq_62[1] = ((on_aqk_70 != 0) ? sqk[17] * scale * fac_69 : 0.0f);
                        lvs_63[1] = ((on_l_71 != 0) ? bt * skk[17] * fac_69 : 0.0f);
                        int s_72 = s0_60 + 2;
                        float fac_73 = approx_exp2(fminf(gtr - gsv_61[2], 0.0f) * 1.442695f);
                        int on_aqk_74 = 0;
                        int on_l_75 = 0;
                        if (s_72 <= t_row) {
                            on_aqk_74 = row_valid;
                        }
                        if (s_72 < t_row) {
                            on_l_75 = row_valid;
                        }
                        aq_62[2] = ((on_aqk_74 != 0) ? sqk[18] * scale * fac_73 : 0.0f);
                        lvs_63[2] = ((on_l_75 != 0) ? bt * skk[18] * fac_73 : 0.0f);
                        int s_76 = s0_60 + 3;
                        float fac_77 = approx_exp2(fminf(gtr - gsv_61[3], 0.0f) * 1.442695f);
                        int on_aqk_78 = 0;
                        int on_l_79 = 0;
                        if (s_76 <= t_row) {
                            on_aqk_78 = row_valid;
                        }
                        if (s_76 < t_row) {
                            on_l_79 = row_valid;
                        }
                        aq_62[3] = ((on_aqk_78 != 0) ? sqk[19] * scale * fac_77 : 0.0f);
                        lvs_63[3] = ((on_l_79 != 0) ? bt * skk[19] * fac_77 : 0.0f);
                        int s_80 = s0_60 + 4;
                        float fac_81 = approx_exp2(fminf(gtr - gsv_61[4], 0.0f) * 1.442695f);
                        int on_aqk_82 = 0;
                        int on_l_83 = 0;
                        if (s_80 <= t_row) {
                            on_aqk_82 = row_valid;
                        }
                        if (s_80 < t_row) {
                            on_l_83 = row_valid;
                        }
                        aq_62[4] = ((on_aqk_82 != 0) ? sqk[20] * scale * fac_81 : 0.0f);
                        lvs_63[4] = ((on_l_83 != 0) ? bt * skk[20] * fac_81 : 0.0f);
                        int s_84 = s0_60 + 5;
                        float fac_85 = approx_exp2(fminf(gtr - gsv_61[5], 0.0f) * 1.442695f);
                        int on_aqk_86 = 0;
                        int on_l_87 = 0;
                        if (s_84 <= t_row) {
                            on_aqk_86 = row_valid;
                        }
                        if (s_84 < t_row) {
                            on_l_87 = row_valid;
                        }
                        aq_62[5] = ((on_aqk_86 != 0) ? sqk[21] * scale * fac_85 : 0.0f);
                        lvs_63[5] = ((on_l_87 != 0) ? bt * skk[21] * fac_85 : 0.0f);
                        int s_88 = s0_60 + 6;
                        float fac_89 = approx_exp2(fminf(gtr - gsv_61[6], 0.0f) * 1.442695f);
                        int on_aqk_90 = 0;
                        int on_l_91 = 0;
                        if (s_88 <= t_row) {
                            on_aqk_90 = row_valid;
                        }
                        if (s_88 < t_row) {
                            on_l_91 = row_valid;
                        }
                        aq_62[6] = ((on_aqk_90 != 0) ? sqk[22] * scale * fac_89 : 0.0f);
                        lvs_63[6] = ((on_l_91 != 0) ? bt * skk[22] * fac_89 : 0.0f);
                        int s_92 = s0_60 + 7;
                        float fac_93 = approx_exp2(fminf(gtr - gsv_61[7], 0.0f) * 1.442695f);
                        int on_aqk_94 = 0;
                        int on_l_95 = 0;
                        if (s_92 <= t_row) {
                            on_aqk_94 = row_valid;
                        }
                        if (s_92 < t_row) {
                            on_l_95 = row_valid;
                        }
                        aq_62[7] = ((on_aqk_94 != 0) ? sqk[23] * scale * fac_93 : 0.0f);
                        lvs_63[7] = ((on_l_95 != 0) ? bt * skk[23] * fac_93 : 0.0f);
                        int s_96 = s0_60 + 8;
                        float fac_97 = approx_exp2(fminf(gtr - gsv_61[8], 0.0f) * 1.442695f);
                        int on_aqk_98 = 0;
                        int on_l_99 = 0;
                        if (s_96 <= t_row) {
                            on_aqk_98 = row_valid;
                        }
                        if (s_96 < t_row) {
                            on_l_99 = row_valid;
                        }
                        aq_62[8] = ((on_aqk_98 != 0) ? sqk[24] * scale * fac_97 : 0.0f);
                        lvs_63[8] = ((on_l_99 != 0) ? bt * skk[24] * fac_97 : 0.0f);
                        int s_100 = s0_60 + 9;
                        float fac_101 = approx_exp2(fminf(gtr - gsv_61[9], 0.0f) * 1.442695f);
                        int on_aqk_102 = 0;
                        int on_l_103 = 0;
                        if (s_100 <= t_row) {
                            on_aqk_102 = row_valid;
                        }
                        if (s_100 < t_row) {
                            on_l_103 = row_valid;
                        }
                        aq_62[9] = ((on_aqk_102 != 0) ? sqk[25] * scale * fac_101 : 0.0f);
                        lvs_63[9] = ((on_l_103 != 0) ? bt * skk[25] * fac_101 : 0.0f);
                        int s_104 = s0_60 + 10;
                        float fac_105 = approx_exp2(fminf(gtr - gsv_61[10], 0.0f) * 1.442695f);
                        int on_aqk_106 = 0;
                        int on_l_107 = 0;
                        if (s_104 <= t_row) {
                            on_aqk_106 = row_valid;
                        }
                        if (s_104 < t_row) {
                            on_l_107 = row_valid;
                        }
                        aq_62[10] = ((on_aqk_106 != 0) ? sqk[26] * scale * fac_105 : 0.0f);
                        lvs_63[10] = ((on_l_107 != 0) ? bt * skk[26] * fac_105 : 0.0f);
                        int s_108 = s0_60 + 11;
                        float fac_109 = approx_exp2(fminf(gtr - gsv_61[11], 0.0f) * 1.442695f);
                        int on_aqk_110 = 0;
                        int on_l_111 = 0;
                        if (s_108 <= t_row) {
                            on_aqk_110 = row_valid;
                        }
                        if (s_108 < t_row) {
                            on_l_111 = row_valid;
                        }
                        aq_62[11] = ((on_aqk_110 != 0) ? sqk[27] * scale * fac_109 : 0.0f);
                        lvs_63[11] = ((on_l_111 != 0) ? bt * skk[27] * fac_109 : 0.0f);
                        int s_112 = s0_60 + 12;
                        float fac_113 = approx_exp2(fminf(gtr - gsv_61[12], 0.0f) * 1.442695f);
                        int on_aqk_114 = 0;
                        int on_l_115 = 0;
                        if (s_112 <= t_row) {
                            on_aqk_114 = row_valid;
                        }
                        if (s_112 < t_row) {
                            on_l_115 = row_valid;
                        }
                        aq_62[12] = ((on_aqk_114 != 0) ? sqk[28] * scale * fac_113 : 0.0f);
                        lvs_63[12] = ((on_l_115 != 0) ? bt * skk[28] * fac_113 : 0.0f);
                        int s_116 = s0_60 + 13;
                        float fac_117 = approx_exp2(fminf(gtr - gsv_61[13], 0.0f) * 1.442695f);
                        int on_aqk_118 = 0;
                        int on_l_119 = 0;
                        if (s_116 <= t_row) {
                            on_aqk_118 = row_valid;
                        }
                        if (s_116 < t_row) {
                            on_l_119 = row_valid;
                        }
                        aq_62[13] = ((on_aqk_118 != 0) ? sqk[29] * scale * fac_117 : 0.0f);
                        lvs_63[13] = ((on_l_119 != 0) ? bt * skk[29] * fac_117 : 0.0f);
                        int s_120 = s0_60 + 14;
                        float fac_121 = approx_exp2(fminf(gtr - gsv_61[14], 0.0f) * 1.442695f);
                        int on_aqk_122 = 0;
                        int on_l_123 = 0;
                        if (s_120 <= t_row) {
                            on_aqk_122 = row_valid;
                        }
                        if (s_120 < t_row) {
                            on_l_123 = row_valid;
                        }
                        aq_62[14] = ((on_aqk_122 != 0) ? sqk[30] * scale * fac_121 : 0.0f);
                        lvs_63[14] = ((on_l_123 != 0) ? bt * skk[30] * fac_121 : 0.0f);
                        int s_124 = s0_60 + 15;
                        float fac_125 = approx_exp2(fminf(gtr - gsv_61[15], 0.0f) * 1.442695f);
                        int on_aqk_126 = 0;
                        int on_l_127 = 0;
                        if (s_124 <= t_row) {
                            on_aqk_126 = row_valid;
                        }
                        if (s_124 < t_row) {
                            on_l_127 = row_valid;
                        }
                        aq_62[15] = ((on_aqk_126 != 0) ? sqk[31] * scale * fac_125 : 0.0f);
                        lvs_63[15] = ((on_l_127 != 0) ? bt * skk[31] * fac_125 : 0.0f);
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0_60) * 4), "f"(lvs_63[0]), "f"(lvs_63[1]), "f"(lvs_63[2]), "f"(lvs_63[3]) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0_60 + 4) * 4), "f"(lvs_63[4]), "f"(lvs_63[5]), "f"(lvs_63[6]), "f"(lvs_63[7]) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0_60 + 8) * 4), "f"(lvs_63[8]), "f"(lvs_63[9]), "f"(lvs_63[10]), "f"(lvs_63[11]) : "memory");
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(l_base + (t_row * 68 + s0_60 + 12) * 4), "f"(lvs_63[12]), "f"(lvs_63[13]), "f"(lvs_63[14]), "f"(lvs_63[15]) : "memory");
                        unsigned int _packed_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_62[_lp*2 + 0], aq_62[_lp*2+1 + 0]));
                            _packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_62[_lp*2 + 2], aq_62[_lp*2+1 + 2]));
                            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_62[_lp*2 + 4], aq_62[_lp*2+1 + 4]));
                            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_62[_lp*2 + 6], aq_62[_lp*2+1 + 6]));
                            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_62[_lp*2 + 8], aq_62[_lp*2+1 + 8]));
                            _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_62[_lp*2 + 10], aq_62[_lp*2+1 + 10]));
                            _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_62[_lp*2 + 12], aq_62[_lp*2+1 + 12]));
                            _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_62[_lp*2 + 14], aq_62[_lp*2+1 + 14]));
                            _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(_packed_1);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (aqk_base + (long long)s0_60))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
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
                            float lrow_0[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_0[(0) + 3]))
                                : "r"(l_row0 + 544));
                            float accv_1 = 0.0f;
                            accv_1 = __fmaf_rn(lrow_0[0], a_vals[0], accv_1);
                            accv_1 = __fmaf_rn(lrow_0[1], a_vals[1], accv_1);
                            float negv_2 = -accv_1;
                            a_vals[2] = ((jj < 2) ? negv_2 : a_vals[2]);
                            if (jj < 2) {
                                {
                                    uint32_t _addr_3 = static_cast<uint32_t>(a_col + 544);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3), "f"(negv_2) : "memory");
                                }
                            }
                            float lrow_3[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_3[(0) + 3]))
                                : "r"(l_row0 + 816));
                            float accv_4 = 0.0f;
                            accv_4 = __fmaf_rn(lrow_3[0], a_vals[0], accv_4);
                            accv_4 = __fmaf_rn(lrow_3[1], a_vals[1], accv_4);
                            accv_4 = __fmaf_rn(lrow_3[2], a_vals[2], accv_4);
                            float negv_5 = -accv_4;
                            a_vals[3] = ((jj < 3) ? negv_5 : a_vals[3]);
                            if (jj < 3) {
                                {
                                    uint32_t _addr_4 = static_cast<uint32_t>(a_col + 816);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_4), "f"(negv_5) : "memory");
                                }
                            }
                            float lrow_6[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_6[(0) + 3]))
                                : "r"(l_row0 + 1088));
                            float accv_7 = 0.0f;
                            accv_7 = __fmaf_rn(lrow_6[0], a_vals[0], accv_7);
                            accv_7 = __fmaf_rn(lrow_6[1], a_vals[1], accv_7);
                            accv_7 = __fmaf_rn(lrow_6[2], a_vals[2], accv_7);
                            accv_7 = __fmaf_rn(lrow_6[3], a_vals[3], accv_7);
                            float negv_8 = -accv_7;
                            a_vals[4] = ((jj < 4) ? negv_8 : a_vals[4]);
                            if (jj < 4) {
                                {
                                    uint32_t _addr_5 = static_cast<uint32_t>(a_col + 1088);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_5), "f"(negv_8) : "memory");
                                }
                            }
                            float lrow_9[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_9[(0) + 3]))
                                : "r"(l_row0 + 1360));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_9[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_9[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_9[(4) + 3]))
                                : "r"(l_row0 + 1360 + 16));
                            float accv_10 = 0.0f;
                            accv_10 = __fmaf_rn(lrow_9[0], a_vals[0], accv_10);
                            accv_10 = __fmaf_rn(lrow_9[1], a_vals[1], accv_10);
                            accv_10 = __fmaf_rn(lrow_9[2], a_vals[2], accv_10);
                            accv_10 = __fmaf_rn(lrow_9[3], a_vals[3], accv_10);
                            accv_10 = __fmaf_rn(lrow_9[4], a_vals[4], accv_10);
                            float negv_11 = -accv_10;
                            a_vals[5] = ((jj < 5) ? negv_11 : a_vals[5]);
                            if (jj < 5) {
                                {
                                    uint32_t _addr_6 = static_cast<uint32_t>(a_col + 1360);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_6), "f"(negv_11) : "memory");
                                }
                            }
                            float lrow_12[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_12[(0) + 3]))
                                : "r"(l_row0 + 1632));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_12[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_12[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_12[(4) + 3]))
                                : "r"(l_row0 + 1632 + 16));
                            float accv_13 = 0.0f;
                            accv_13 = __fmaf_rn(lrow_12[0], a_vals[0], accv_13);
                            accv_13 = __fmaf_rn(lrow_12[1], a_vals[1], accv_13);
                            accv_13 = __fmaf_rn(lrow_12[2], a_vals[2], accv_13);
                            accv_13 = __fmaf_rn(lrow_12[3], a_vals[3], accv_13);
                            accv_13 = __fmaf_rn(lrow_12[4], a_vals[4], accv_13);
                            accv_13 = __fmaf_rn(lrow_12[5], a_vals[5], accv_13);
                            float negv_14 = -accv_13;
                            a_vals[6] = ((jj < 6) ? negv_14 : a_vals[6]);
                            if (jj < 6) {
                                {
                                    uint32_t _addr_7 = static_cast<uint32_t>(a_col + 1632);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(negv_14) : "memory");
                                }
                            }
                            float lrow_15[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_15[(0) + 3]))
                                : "r"(l_row0 + 1904));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_15[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_15[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_15[(4) + 3]))
                                : "r"(l_row0 + 1904 + 16));
                            float accv_16 = 0.0f;
                            accv_16 = __fmaf_rn(lrow_15[0], a_vals[0], accv_16);
                            accv_16 = __fmaf_rn(lrow_15[1], a_vals[1], accv_16);
                            accv_16 = __fmaf_rn(lrow_15[2], a_vals[2], accv_16);
                            accv_16 = __fmaf_rn(lrow_15[3], a_vals[3], accv_16);
                            accv_16 = __fmaf_rn(lrow_15[4], a_vals[4], accv_16);
                            accv_16 = __fmaf_rn(lrow_15[5], a_vals[5], accv_16);
                            accv_16 = __fmaf_rn(lrow_15[6], a_vals[6], accv_16);
                            float negv_17 = -accv_16;
                            a_vals[7] = ((jj < 7) ? negv_17 : a_vals[7]);
                            if (jj < 7) {
                                {
                                    uint32_t _addr_8 = static_cast<uint32_t>(a_col + 1904);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_8), "f"(negv_17) : "memory");
                                }
                            }
                            float lrow_18[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_18[(0) + 3]))
                                : "r"(l_row0 + 2176));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_18[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_18[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_18[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_18[(4) + 3]))
                                : "r"(l_row0 + 2176 + 16));
                            float accv_19 = 0.0f;
                            accv_19 = __fmaf_rn(lrow_18[0], a_vals[0], accv_19);
                            accv_19 = __fmaf_rn(lrow_18[1], a_vals[1], accv_19);
                            accv_19 = __fmaf_rn(lrow_18[2], a_vals[2], accv_19);
                            accv_19 = __fmaf_rn(lrow_18[3], a_vals[3], accv_19);
                            accv_19 = __fmaf_rn(lrow_18[4], a_vals[4], accv_19);
                            accv_19 = __fmaf_rn(lrow_18[5], a_vals[5], accv_19);
                            accv_19 = __fmaf_rn(lrow_18[6], a_vals[6], accv_19);
                            accv_19 = __fmaf_rn(lrow_18[7], a_vals[7], accv_19);
                            float negv_20 = -accv_19;
                            a_vals[8] = ((jj < 8) ? negv_20 : a_vals[8]);
                            if (jj < 8) {
                                {
                                    uint32_t _addr_9 = static_cast<uint32_t>(a_col + 2176);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(negv_20) : "memory");
                                }
                            }
                            float lrow_21[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_21[(0) + 3]))
                                : "r"(l_row0 + 2448));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_21[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_21[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_21[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_21[(4) + 3]))
                                : "r"(l_row0 + 2448 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_21[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_21[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_21[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_21[(8) + 3]))
                                : "r"(l_row0 + 2448 + 32));
                            float accv_22 = 0.0f;
                            accv_22 = __fmaf_rn(lrow_21[0], a_vals[0], accv_22);
                            accv_22 = __fmaf_rn(lrow_21[1], a_vals[1], accv_22);
                            accv_22 = __fmaf_rn(lrow_21[2], a_vals[2], accv_22);
                            accv_22 = __fmaf_rn(lrow_21[3], a_vals[3], accv_22);
                            accv_22 = __fmaf_rn(lrow_21[4], a_vals[4], accv_22);
                            accv_22 = __fmaf_rn(lrow_21[5], a_vals[5], accv_22);
                            accv_22 = __fmaf_rn(lrow_21[6], a_vals[6], accv_22);
                            accv_22 = __fmaf_rn(lrow_21[7], a_vals[7], accv_22);
                            accv_22 = __fmaf_rn(lrow_21[8], a_vals[8], accv_22);
                            float negv_23 = -accv_22;
                            a_vals[9] = ((jj < 9) ? negv_23 : a_vals[9]);
                            if (jj < 9) {
                                {
                                    uint32_t _addr_10 = static_cast<uint32_t>(a_col + 2448);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(negv_23) : "memory");
                                }
                            }
                            float lrow_24[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_24[(0) + 3]))
                                : "r"(l_row0 + 2720));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_24[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_24[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_24[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_24[(4) + 3]))
                                : "r"(l_row0 + 2720 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_24[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_24[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_24[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_24[(8) + 3]))
                                : "r"(l_row0 + 2720 + 32));
                            float accv_25 = 0.0f;
                            accv_25 = __fmaf_rn(lrow_24[0], a_vals[0], accv_25);
                            accv_25 = __fmaf_rn(lrow_24[1], a_vals[1], accv_25);
                            accv_25 = __fmaf_rn(lrow_24[2], a_vals[2], accv_25);
                            accv_25 = __fmaf_rn(lrow_24[3], a_vals[3], accv_25);
                            accv_25 = __fmaf_rn(lrow_24[4], a_vals[4], accv_25);
                            accv_25 = __fmaf_rn(lrow_24[5], a_vals[5], accv_25);
                            accv_25 = __fmaf_rn(lrow_24[6], a_vals[6], accv_25);
                            accv_25 = __fmaf_rn(lrow_24[7], a_vals[7], accv_25);
                            accv_25 = __fmaf_rn(lrow_24[8], a_vals[8], accv_25);
                            accv_25 = __fmaf_rn(lrow_24[9], a_vals[9], accv_25);
                            float negv_26 = -accv_25;
                            a_vals[10] = ((jj < 10) ? negv_26 : a_vals[10]);
                            if (jj < 10) {
                                {
                                    uint32_t _addr_11 = static_cast<uint32_t>(a_col + 2720);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(negv_26) : "memory");
                                }
                            }
                            float lrow_27[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_27[(0) + 3]))
                                : "r"(l_row0 + 2992));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_27[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_27[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_27[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_27[(4) + 3]))
                                : "r"(l_row0 + 2992 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_27[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_27[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_27[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_27[(8) + 3]))
                                : "r"(l_row0 + 2992 + 32));
                            float accv_28 = 0.0f;
                            accv_28 = __fmaf_rn(lrow_27[0], a_vals[0], accv_28);
                            accv_28 = __fmaf_rn(lrow_27[1], a_vals[1], accv_28);
                            accv_28 = __fmaf_rn(lrow_27[2], a_vals[2], accv_28);
                            accv_28 = __fmaf_rn(lrow_27[3], a_vals[3], accv_28);
                            accv_28 = __fmaf_rn(lrow_27[4], a_vals[4], accv_28);
                            accv_28 = __fmaf_rn(lrow_27[5], a_vals[5], accv_28);
                            accv_28 = __fmaf_rn(lrow_27[6], a_vals[6], accv_28);
                            accv_28 = __fmaf_rn(lrow_27[7], a_vals[7], accv_28);
                            accv_28 = __fmaf_rn(lrow_27[8], a_vals[8], accv_28);
                            accv_28 = __fmaf_rn(lrow_27[9], a_vals[9], accv_28);
                            accv_28 = __fmaf_rn(lrow_27[10], a_vals[10], accv_28);
                            float negv_29 = -accv_28;
                            a_vals[11] = ((jj < 11) ? negv_29 : a_vals[11]);
                            if (jj < 11) {
                                {
                                    uint32_t _addr_12 = static_cast<uint32_t>(a_col + 2992);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(negv_29) : "memory");
                                }
                            }
                            float lrow_30[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_30[(0) + 3]))
                                : "r"(l_row0 + 3264));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_30[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_30[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_30[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_30[(4) + 3]))
                                : "r"(l_row0 + 3264 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_30[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_30[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_30[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_30[(8) + 3]))
                                : "r"(l_row0 + 3264 + 32));
                            float accv_31 = 0.0f;
                            accv_31 = __fmaf_rn(lrow_30[0], a_vals[0], accv_31);
                            accv_31 = __fmaf_rn(lrow_30[1], a_vals[1], accv_31);
                            accv_31 = __fmaf_rn(lrow_30[2], a_vals[2], accv_31);
                            accv_31 = __fmaf_rn(lrow_30[3], a_vals[3], accv_31);
                            accv_31 = __fmaf_rn(lrow_30[4], a_vals[4], accv_31);
                            accv_31 = __fmaf_rn(lrow_30[5], a_vals[5], accv_31);
                            accv_31 = __fmaf_rn(lrow_30[6], a_vals[6], accv_31);
                            accv_31 = __fmaf_rn(lrow_30[7], a_vals[7], accv_31);
                            accv_31 = __fmaf_rn(lrow_30[8], a_vals[8], accv_31);
                            accv_31 = __fmaf_rn(lrow_30[9], a_vals[9], accv_31);
                            accv_31 = __fmaf_rn(lrow_30[10], a_vals[10], accv_31);
                            accv_31 = __fmaf_rn(lrow_30[11], a_vals[11], accv_31);
                            float negv_32 = -accv_31;
                            a_vals[12] = ((jj < 12) ? negv_32 : a_vals[12]);
                            if (jj < 12) {
                                {
                                    uint32_t _addr_13 = static_cast<uint32_t>(a_col + 3264);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(negv_32) : "memory");
                                }
                            }
                            float lrow_33[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[(0) + 3]))
                                : "r"(l_row0 + 3536));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[(4) + 3]))
                                : "r"(l_row0 + 3536 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[(8) + 3]))
                                : "r"(l_row0 + 3536 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[12])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_33[(12) + 3]))
                                : "r"(l_row0 + 3536 + 48));
                            float accv_34 = 0.0f;
                            accv_34 = __fmaf_rn(lrow_33[0], a_vals[0], accv_34);
                            accv_34 = __fmaf_rn(lrow_33[1], a_vals[1], accv_34);
                            accv_34 = __fmaf_rn(lrow_33[2], a_vals[2], accv_34);
                            accv_34 = __fmaf_rn(lrow_33[3], a_vals[3], accv_34);
                            accv_34 = __fmaf_rn(lrow_33[4], a_vals[4], accv_34);
                            accv_34 = __fmaf_rn(lrow_33[5], a_vals[5], accv_34);
                            accv_34 = __fmaf_rn(lrow_33[6], a_vals[6], accv_34);
                            accv_34 = __fmaf_rn(lrow_33[7], a_vals[7], accv_34);
                            accv_34 = __fmaf_rn(lrow_33[8], a_vals[8], accv_34);
                            accv_34 = __fmaf_rn(lrow_33[9], a_vals[9], accv_34);
                            accv_34 = __fmaf_rn(lrow_33[10], a_vals[10], accv_34);
                            accv_34 = __fmaf_rn(lrow_33[11], a_vals[11], accv_34);
                            accv_34 = __fmaf_rn(lrow_33[12], a_vals[12], accv_34);
                            float negv_35 = -accv_34;
                            a_vals[13] = ((jj < 13) ? negv_35 : a_vals[13]);
                            if (jj < 13) {
                                {
                                    uint32_t _addr_14 = static_cast<uint32_t>(a_col + 3536);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(negv_35) : "memory");
                                }
                            }
                            float lrow_36[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[(0) + 3]))
                                : "r"(l_row0 + 3808));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[(4) + 3]))
                                : "r"(l_row0 + 3808 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[(8) + 3]))
                                : "r"(l_row0 + 3808 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[12])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_36[(12) + 3]))
                                : "r"(l_row0 + 3808 + 48));
                            float accv_37 = 0.0f;
                            accv_37 = __fmaf_rn(lrow_36[0], a_vals[0], accv_37);
                            accv_37 = __fmaf_rn(lrow_36[1], a_vals[1], accv_37);
                            accv_37 = __fmaf_rn(lrow_36[2], a_vals[2], accv_37);
                            accv_37 = __fmaf_rn(lrow_36[3], a_vals[3], accv_37);
                            accv_37 = __fmaf_rn(lrow_36[4], a_vals[4], accv_37);
                            accv_37 = __fmaf_rn(lrow_36[5], a_vals[5], accv_37);
                            accv_37 = __fmaf_rn(lrow_36[6], a_vals[6], accv_37);
                            accv_37 = __fmaf_rn(lrow_36[7], a_vals[7], accv_37);
                            accv_37 = __fmaf_rn(lrow_36[8], a_vals[8], accv_37);
                            accv_37 = __fmaf_rn(lrow_36[9], a_vals[9], accv_37);
                            accv_37 = __fmaf_rn(lrow_36[10], a_vals[10], accv_37);
                            accv_37 = __fmaf_rn(lrow_36[11], a_vals[11], accv_37);
                            accv_37 = __fmaf_rn(lrow_36[12], a_vals[12], accv_37);
                            accv_37 = __fmaf_rn(lrow_36[13], a_vals[13], accv_37);
                            float negv_38 = -accv_37;
                            a_vals[14] = ((jj < 14) ? negv_38 : a_vals[14]);
                            if (jj < 14) {
                                {
                                    uint32_t _addr_15 = static_cast<uint32_t>(a_col + 3808);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(negv_38) : "memory");
                                }
                            }
                            float lrow_39[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[(0) + 3]))
                                : "r"(l_row0 + 4080));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[4])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[(4) + 3]))
                                : "r"(l_row0 + 4080 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[8])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[(8) + 3]))
                                : "r"(l_row0 + 4080 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[12])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lrow_39[(12) + 3]))
                                : "r"(l_row0 + 4080 + 48));
                            float accv_40 = 0.0f;
                            accv_40 = __fmaf_rn(lrow_39[0], a_vals[0], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[1], a_vals[1], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[2], a_vals[2], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[3], a_vals[3], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[4], a_vals[4], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[5], a_vals[5], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[6], a_vals[6], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[7], a_vals[7], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[8], a_vals[8], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[9], a_vals[9], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[10], a_vals[10], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[11], a_vals[11], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[12], a_vals[12], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[13], a_vals[13], accv_40);
                            accv_40 = __fmaf_rn(lrow_39[14], a_vals[14], accv_40);
                            float negv_41 = -accv_40;
                            a_vals[15] = ((jj < 15) ? negv_41 : a_vals[15]);
                            if (jj < 15) {
                                {
                                    uint32_t _addr_16 = static_cast<uint32_t>(a_col + 4080);
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(negv_41) : "memory");
                                }
                            }
                        }
                        if (tid_0 >= 64) {
                            int t2 = tid_0 - 64;
                            mbarrier_wait(v_full_addr + (vst_c) * 8, _phase_v_full);
                            int qi = t2;
                            int srow = qi >> 4;
                            int d0 = (qi & 15) * 8;
                            float bs_r[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r[0])) : "r"(beta_base + srow * 4));
                            float fvb = bs_r[0];
                            unsigned int vaddr = (v_t_addr + vst_c * 16384 + (unsigned int)(d0 / 64 * 8192 + srow * 128 + d0 % 64 * 2 ^ (d0 / 64 * 8192 + srow * 128 + d0 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w[0])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 3]))
                                : "r"(vaddr));
                            float _unpack_bf16_0[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w[_pair]) << 16);
                                (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w[_pair]) & 0xffff0000u);
                            }
                            float s_1[8];
                            s_1[0] = _unpack_bf16_0[0] * fvb;
                            s_1[1] = _unpack_bf16_0[1] * fvb;
                            s_1[2] = _unpack_bf16_0[2] * fvb;
                            s_1[3] = _unpack_bf16_0[3] * fvb;
                            s_1[4] = _unpack_bf16_0[4] * fvb;
                            s_1[5] = _unpack_bf16_0[5] * fvb;
                            s_1[6] = _unpack_bf16_0[6] * fvb;
                            s_1[7] = _unpack_bf16_0[7] * fvb;
                            unsigned int s_1_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_1[_lp*2 + 0], s_1[_lp*2+1 + 0]));
                                s_1_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr), "r"(s_1_bf16[0]), "r"(s_1_bf16[1]), "r"(s_1_bf16[2]), "r"(s_1_bf16[3]) : "memory");
                            int qi_2 = t2 + 64;
                            int srow_3 = qi_2 >> 4;
                            int d0_4 = (qi_2 & 15) * 8;
                            float bs_r_5[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_5[0])) : "r"(beta_base + srow_3 * 4));
                            float fvb_6 = bs_r_5[0];
                            unsigned int vaddr_7 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_4 / 64 * 8192 + srow_3 * 128 + d0_4 % 64 * 2 ^ (d0_4 / 64 * 8192 + srow_3 * 128 + d0_4 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_8[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_8[(0) + 3]))
                                : "r"(vaddr_7));
                            float _unpack_bf16_1[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_8[_pair]) << 16);
                                (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_8[_pair]) & 0xffff0000u);
                            }
                            float s_9[8];
                            s_9[0] = _unpack_bf16_1[0] * fvb_6;
                            s_9[1] = _unpack_bf16_1[1] * fvb_6;
                            s_9[2] = _unpack_bf16_1[2] * fvb_6;
                            s_9[3] = _unpack_bf16_1[3] * fvb_6;
                            s_9[4] = _unpack_bf16_1[4] * fvb_6;
                            s_9[5] = _unpack_bf16_1[5] * fvb_6;
                            s_9[6] = _unpack_bf16_1[6] * fvb_6;
                            s_9[7] = _unpack_bf16_1[7] * fvb_6;
                            unsigned int s_9_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_9[_lp*2 + 0], s_9[_lp*2+1 + 0]));
                                s_9_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_7), "r"(s_9_bf16[0]), "r"(s_9_bf16[1]), "r"(s_9_bf16[2]), "r"(s_9_bf16[3]) : "memory");
                            int qi_10 = t2 + 128;
                            int srow_11 = qi_10 >> 4;
                            int d0_12 = (qi_10 & 15) * 8;
                            float bs_r_13[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_13[0])) : "r"(beta_base + srow_11 * 4));
                            float fvb_14 = bs_r_13[0];
                            unsigned int vaddr_15 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_12 / 64 * 8192 + srow_11 * 128 + d0_12 % 64 * 2 ^ (d0_12 / 64 * 8192 + srow_11 * 128 + d0_12 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_16[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_16[(0) + 3]))
                                : "r"(vaddr_15));
                            float _unpack_bf16_2[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_16[_pair]) << 16);
                                (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_16[_pair]) & 0xffff0000u);
                            }
                            float s_17[8];
                            s_17[0] = _unpack_bf16_2[0] * fvb_14;
                            s_17[1] = _unpack_bf16_2[1] * fvb_14;
                            s_17[2] = _unpack_bf16_2[2] * fvb_14;
                            s_17[3] = _unpack_bf16_2[3] * fvb_14;
                            s_17[4] = _unpack_bf16_2[4] * fvb_14;
                            s_17[5] = _unpack_bf16_2[5] * fvb_14;
                            s_17[6] = _unpack_bf16_2[6] * fvb_14;
                            s_17[7] = _unpack_bf16_2[7] * fvb_14;
                            unsigned int s_17_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_17[_lp*2 + 0], s_17[_lp*2+1 + 0]));
                                s_17_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_15), "r"(s_17_bf16[0]), "r"(s_17_bf16[1]), "r"(s_17_bf16[2]), "r"(s_17_bf16[3]) : "memory");
                            int qi_18 = t2 + 192;
                            int srow_19 = qi_18 >> 4;
                            int d0_20 = (qi_18 & 15) * 8;
                            float bs_r_21[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_21[0])) : "r"(beta_base + srow_19 * 4));
                            float fvb_22 = bs_r_21[0];
                            unsigned int vaddr_23 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_20 / 64 * 8192 + srow_19 * 128 + d0_20 % 64 * 2 ^ (d0_20 / 64 * 8192 + srow_19 * 128 + d0_20 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_24[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_24[(0) + 3]))
                                : "r"(vaddr_23));
                            float _unpack_bf16_3[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_24[_pair]) << 16);
                                (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_24[_pair]) & 0xffff0000u);
                            }
                            float s_25[8];
                            s_25[0] = _unpack_bf16_3[0] * fvb_22;
                            s_25[1] = _unpack_bf16_3[1] * fvb_22;
                            s_25[2] = _unpack_bf16_3[2] * fvb_22;
                            s_25[3] = _unpack_bf16_3[3] * fvb_22;
                            s_25[4] = _unpack_bf16_3[4] * fvb_22;
                            s_25[5] = _unpack_bf16_3[5] * fvb_22;
                            s_25[6] = _unpack_bf16_3[6] * fvb_22;
                            s_25[7] = _unpack_bf16_3[7] * fvb_22;
                            unsigned int s_25_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_25[_lp*2 + 0], s_25[_lp*2+1 + 0]));
                                s_25_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_23), "r"(s_25_bf16[0]), "r"(s_25_bf16[1]), "r"(s_25_bf16[2]), "r"(s_25_bf16[3]) : "memory");
                            int qi_26 = t2 + 256;
                            int srow_27 = qi_26 >> 4;
                            int d0_28 = (qi_26 & 15) * 8;
                            float bs_r_29[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_29[0])) : "r"(beta_base + srow_27 * 4));
                            float fvb_30 = bs_r_29[0];
                            unsigned int vaddr_31 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_28 / 64 * 8192 + srow_27 * 128 + d0_28 % 64 * 2 ^ (d0_28 / 64 * 8192 + srow_27 * 128 + d0_28 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_32[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_32[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_32[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_32[(0) + 3]))
                                : "r"(vaddr_31));
                            float _unpack_bf16_4[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_32[_pair]) << 16);
                                (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_32[_pair]) & 0xffff0000u);
                            }
                            float s_33[8];
                            s_33[0] = _unpack_bf16_4[0] * fvb_30;
                            s_33[1] = _unpack_bf16_4[1] * fvb_30;
                            s_33[2] = _unpack_bf16_4[2] * fvb_30;
                            s_33[3] = _unpack_bf16_4[3] * fvb_30;
                            s_33[4] = _unpack_bf16_4[4] * fvb_30;
                            s_33[5] = _unpack_bf16_4[5] * fvb_30;
                            s_33[6] = _unpack_bf16_4[6] * fvb_30;
                            s_33[7] = _unpack_bf16_4[7] * fvb_30;
                            unsigned int s_33_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_33[_lp*2 + 0], s_33[_lp*2+1 + 0]));
                                s_33_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_31), "r"(s_33_bf16[0]), "r"(s_33_bf16[1]), "r"(s_33_bf16[2]), "r"(s_33_bf16[3]) : "memory");
                            int qi_34 = t2 + 320;
                            int srow_35 = qi_34 >> 4;
                            int d0_36 = (qi_34 & 15) * 8;
                            float bs_r_37[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_37[0])) : "r"(beta_base + srow_35 * 4));
                            float fvb_38 = bs_r_37[0];
                            unsigned int vaddr_39 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_36 / 64 * 8192 + srow_35 * 128 + d0_36 % 64 * 2 ^ (d0_36 / 64 * 8192 + srow_35 * 128 + d0_36 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_40[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_40[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_40[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_40[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_40[(0) + 3]))
                                : "r"(vaddr_39));
                            float _unpack_bf16_5[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_40[_pair]) << 16);
                                (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_40[_pair]) & 0xffff0000u);
                            }
                            float s_41[8];
                            s_41[0] = _unpack_bf16_5[0] * fvb_38;
                            s_41[1] = _unpack_bf16_5[1] * fvb_38;
                            s_41[2] = _unpack_bf16_5[2] * fvb_38;
                            s_41[3] = _unpack_bf16_5[3] * fvb_38;
                            s_41[4] = _unpack_bf16_5[4] * fvb_38;
                            s_41[5] = _unpack_bf16_5[5] * fvb_38;
                            s_41[6] = _unpack_bf16_5[6] * fvb_38;
                            s_41[7] = _unpack_bf16_5[7] * fvb_38;
                            unsigned int s_41_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_41[_lp*2 + 0], s_41[_lp*2+1 + 0]));
                                s_41_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_39), "r"(s_41_bf16[0]), "r"(s_41_bf16[1]), "r"(s_41_bf16[2]), "r"(s_41_bf16[3]) : "memory");
                            int qi_42 = t2 + 384;
                            int srow_43 = qi_42 >> 4;
                            int d0_44 = (qi_42 & 15) * 8;
                            float bs_r_45[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_45[0])) : "r"(beta_base + srow_43 * 4));
                            float fvb_46 = bs_r_45[0];
                            unsigned int vaddr_47 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_44 / 64 * 8192 + srow_43 * 128 + d0_44 % 64 * 2 ^ (d0_44 / 64 * 8192 + srow_43 * 128 + d0_44 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_48[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_48[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_48[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_48[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_48[(0) + 3]))
                                : "r"(vaddr_47));
                            float _unpack_bf16_6[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_48[_pair]) << 16);
                                (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_48[_pair]) & 0xffff0000u);
                            }
                            float s_49[8];
                            s_49[0] = _unpack_bf16_6[0] * fvb_46;
                            s_49[1] = _unpack_bf16_6[1] * fvb_46;
                            s_49[2] = _unpack_bf16_6[2] * fvb_46;
                            s_49[3] = _unpack_bf16_6[3] * fvb_46;
                            s_49[4] = _unpack_bf16_6[4] * fvb_46;
                            s_49[5] = _unpack_bf16_6[5] * fvb_46;
                            s_49[6] = _unpack_bf16_6[6] * fvb_46;
                            s_49[7] = _unpack_bf16_6[7] * fvb_46;
                            unsigned int s_49_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_49[_lp*2 + 0], s_49[_lp*2+1 + 0]));
                                s_49_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_47), "r"(s_49_bf16[0]), "r"(s_49_bf16[1]), "r"(s_49_bf16[2]), "r"(s_49_bf16[3]) : "memory");
                            int qi_50 = t2 + 448;
                            int srow_51 = qi_50 >> 4;
                            int d0_52 = (qi_50 & 15) * 8;
                            float bs_r_53[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_53[0])) : "r"(beta_base + srow_51 * 4));
                            float fvb_54 = bs_r_53[0];
                            unsigned int vaddr_55 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_52 / 64 * 8192 + srow_51 * 128 + d0_52 % 64 * 2 ^ (d0_52 / 64 * 8192 + srow_51 * 128 + d0_52 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_56[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_56[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_56[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_56[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_56[(0) + 3]))
                                : "r"(vaddr_55));
                            float _unpack_bf16_7[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_56[_pair]) << 16);
                                (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_56[_pair]) & 0xffff0000u);
                            }
                            float s_57[8];
                            s_57[0] = _unpack_bf16_7[0] * fvb_54;
                            s_57[1] = _unpack_bf16_7[1] * fvb_54;
                            s_57[2] = _unpack_bf16_7[2] * fvb_54;
                            s_57[3] = _unpack_bf16_7[3] * fvb_54;
                            s_57[4] = _unpack_bf16_7[4] * fvb_54;
                            s_57[5] = _unpack_bf16_7[5] * fvb_54;
                            s_57[6] = _unpack_bf16_7[6] * fvb_54;
                            s_57[7] = _unpack_bf16_7[7] * fvb_54;
                            unsigned int s_57_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_57[_lp*2 + 0], s_57[_lp*2+1 + 0]));
                                s_57_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_55), "r"(s_57_bf16[0]), "r"(s_57_bf16[1]), "r"(s_57_bf16[2]), "r"(s_57_bf16[3]) : "memory");
                            int qi_58 = t2 + 512;
                            int srow_59 = qi_58 >> 4;
                            int d0_60 = (qi_58 & 15) * 8;
                            float bs_r_61[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_61[0])) : "r"(beta_base + srow_59 * 4));
                            float fvb_62 = bs_r_61[0];
                            unsigned int vaddr_63 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_60 / 64 * 8192 + srow_59 * 128 + d0_60 % 64 * 2 ^ (d0_60 / 64 * 8192 + srow_59 * 128 + d0_60 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_64[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_64[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_64[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_64[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_64[(0) + 3]))
                                : "r"(vaddr_63));
                            float _unpack_bf16_8[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_64[_pair]) << 16);
                                (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_64[_pair]) & 0xffff0000u);
                            }
                            float s_65[8];
                            s_65[0] = _unpack_bf16_8[0] * fvb_62;
                            s_65[1] = _unpack_bf16_8[1] * fvb_62;
                            s_65[2] = _unpack_bf16_8[2] * fvb_62;
                            s_65[3] = _unpack_bf16_8[3] * fvb_62;
                            s_65[4] = _unpack_bf16_8[4] * fvb_62;
                            s_65[5] = _unpack_bf16_8[5] * fvb_62;
                            s_65[6] = _unpack_bf16_8[6] * fvb_62;
                            s_65[7] = _unpack_bf16_8[7] * fvb_62;
                            unsigned int s_65_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_65[_lp*2 + 0], s_65[_lp*2+1 + 0]));
                                s_65_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_63), "r"(s_65_bf16[0]), "r"(s_65_bf16[1]), "r"(s_65_bf16[2]), "r"(s_65_bf16[3]) : "memory");
                            int qi_66 = t2 + 576;
                            int srow_67 = qi_66 >> 4;
                            int d0_68 = (qi_66 & 15) * 8;
                            float bs_r_69[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_69[0])) : "r"(beta_base + srow_67 * 4));
                            float fvb_70 = bs_r_69[0];
                            unsigned int vaddr_71 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_68 / 64 * 8192 + srow_67 * 128 + d0_68 % 64 * 2 ^ (d0_68 / 64 * 8192 + srow_67 * 128 + d0_68 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_72[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_72[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_72[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_72[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_72[(0) + 3]))
                                : "r"(vaddr_71));
                            float _unpack_bf16_9[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_72[_pair]) << 16);
                                (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_72[_pair]) & 0xffff0000u);
                            }
                            float s_73[8];
                            s_73[0] = _unpack_bf16_9[0] * fvb_70;
                            s_73[1] = _unpack_bf16_9[1] * fvb_70;
                            s_73[2] = _unpack_bf16_9[2] * fvb_70;
                            s_73[3] = _unpack_bf16_9[3] * fvb_70;
                            s_73[4] = _unpack_bf16_9[4] * fvb_70;
                            s_73[5] = _unpack_bf16_9[5] * fvb_70;
                            s_73[6] = _unpack_bf16_9[6] * fvb_70;
                            s_73[7] = _unpack_bf16_9[7] * fvb_70;
                            unsigned int s_73_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_73[_lp*2 + 0], s_73[_lp*2+1 + 0]));
                                s_73_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_71), "r"(s_73_bf16[0]), "r"(s_73_bf16[1]), "r"(s_73_bf16[2]), "r"(s_73_bf16[3]) : "memory");
                            int qi_74 = t2 + 640;
                            int srow_75 = qi_74 >> 4;
                            int d0_76 = (qi_74 & 15) * 8;
                            float bs_r_77[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_77[0])) : "r"(beta_base + srow_75 * 4));
                            float fvb_78 = bs_r_77[0];
                            unsigned int vaddr_79 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_76 / 64 * 8192 + srow_75 * 128 + d0_76 % 64 * 2 ^ (d0_76 / 64 * 8192 + srow_75 * 128 + d0_76 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_80[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_80[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_80[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_80[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_80[(0) + 3]))
                                : "r"(vaddr_79));
                            float _unpack_bf16_10[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_80[_pair]) << 16);
                                (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_80[_pair]) & 0xffff0000u);
                            }
                            float s_81[8];
                            s_81[0] = _unpack_bf16_10[0] * fvb_78;
                            s_81[1] = _unpack_bf16_10[1] * fvb_78;
                            s_81[2] = _unpack_bf16_10[2] * fvb_78;
                            s_81[3] = _unpack_bf16_10[3] * fvb_78;
                            s_81[4] = _unpack_bf16_10[4] * fvb_78;
                            s_81[5] = _unpack_bf16_10[5] * fvb_78;
                            s_81[6] = _unpack_bf16_10[6] * fvb_78;
                            s_81[7] = _unpack_bf16_10[7] * fvb_78;
                            unsigned int s_81_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_81[_lp*2 + 0], s_81[_lp*2+1 + 0]));
                                s_81_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_79), "r"(s_81_bf16[0]), "r"(s_81_bf16[1]), "r"(s_81_bf16[2]), "r"(s_81_bf16[3]) : "memory");
                            int qi_82 = t2 + 704;
                            int srow_83 = qi_82 >> 4;
                            int d0_84 = (qi_82 & 15) * 8;
                            float bs_r_85[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_85[0])) : "r"(beta_base + srow_83 * 4));
                            float fvb_86 = bs_r_85[0];
                            unsigned int vaddr_87 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_84 / 64 * 8192 + srow_83 * 128 + d0_84 % 64 * 2 ^ (d0_84 / 64 * 8192 + srow_83 * 128 + d0_84 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_88[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_88[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_88[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_88[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_88[(0) + 3]))
                                : "r"(vaddr_87));
                            float _unpack_bf16_11[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_88[_pair]) << 16);
                                (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_88[_pair]) & 0xffff0000u);
                            }
                            float s_89[8];
                            s_89[0] = _unpack_bf16_11[0] * fvb_86;
                            s_89[1] = _unpack_bf16_11[1] * fvb_86;
                            s_89[2] = _unpack_bf16_11[2] * fvb_86;
                            s_89[3] = _unpack_bf16_11[3] * fvb_86;
                            s_89[4] = _unpack_bf16_11[4] * fvb_86;
                            s_89[5] = _unpack_bf16_11[5] * fvb_86;
                            s_89[6] = _unpack_bf16_11[6] * fvb_86;
                            s_89[7] = _unpack_bf16_11[7] * fvb_86;
                            unsigned int s_89_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_89[_lp*2 + 0], s_89[_lp*2+1 + 0]));
                                s_89_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_87), "r"(s_89_bf16[0]), "r"(s_89_bf16[1]), "r"(s_89_bf16[2]), "r"(s_89_bf16[3]) : "memory");
                            int qi_90 = t2 + 768;
                            int srow_91 = qi_90 >> 4;
                            int d0_92 = (qi_90 & 15) * 8;
                            float bs_r_93[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_93[0])) : "r"(beta_base + srow_91 * 4));
                            float fvb_94 = bs_r_93[0];
                            unsigned int vaddr_95 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_92 / 64 * 8192 + srow_91 * 128 + d0_92 % 64 * 2 ^ (d0_92 / 64 * 8192 + srow_91 * 128 + d0_92 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_96[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_96[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_96[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_96[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_96[(0) + 3]))
                                : "r"(vaddr_95));
                            float _unpack_bf16_12[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_96[_pair]) << 16);
                                (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_96[_pair]) & 0xffff0000u);
                            }
                            float s_97[8];
                            s_97[0] = _unpack_bf16_12[0] * fvb_94;
                            s_97[1] = _unpack_bf16_12[1] * fvb_94;
                            s_97[2] = _unpack_bf16_12[2] * fvb_94;
                            s_97[3] = _unpack_bf16_12[3] * fvb_94;
                            s_97[4] = _unpack_bf16_12[4] * fvb_94;
                            s_97[5] = _unpack_bf16_12[5] * fvb_94;
                            s_97[6] = _unpack_bf16_12[6] * fvb_94;
                            s_97[7] = _unpack_bf16_12[7] * fvb_94;
                            unsigned int s_97_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_97[_lp*2 + 0], s_97[_lp*2+1 + 0]));
                                s_97_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_95), "r"(s_97_bf16[0]), "r"(s_97_bf16[1]), "r"(s_97_bf16[2]), "r"(s_97_bf16[3]) : "memory");
                            int qi_98 = t2 + 832;
                            int srow_99 = qi_98 >> 4;
                            int d0_100 = (qi_98 & 15) * 8;
                            float bs_r_101[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_101[0])) : "r"(beta_base + srow_99 * 4));
                            float fvb_102 = bs_r_101[0];
                            unsigned int vaddr_103 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_100 / 64 * 8192 + srow_99 * 128 + d0_100 % 64 * 2 ^ (d0_100 / 64 * 8192 + srow_99 * 128 + d0_100 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_104[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_104[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_104[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_104[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_104[(0) + 3]))
                                : "r"(vaddr_103));
                            float _unpack_bf16_13[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_104[_pair]) << 16);
                                (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_104[_pair]) & 0xffff0000u);
                            }
                            float s_105[8];
                            s_105[0] = _unpack_bf16_13[0] * fvb_102;
                            s_105[1] = _unpack_bf16_13[1] * fvb_102;
                            s_105[2] = _unpack_bf16_13[2] * fvb_102;
                            s_105[3] = _unpack_bf16_13[3] * fvb_102;
                            s_105[4] = _unpack_bf16_13[4] * fvb_102;
                            s_105[5] = _unpack_bf16_13[5] * fvb_102;
                            s_105[6] = _unpack_bf16_13[6] * fvb_102;
                            s_105[7] = _unpack_bf16_13[7] * fvb_102;
                            unsigned int s_105_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_105[_lp*2 + 0], s_105[_lp*2+1 + 0]));
                                s_105_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_103), "r"(s_105_bf16[0]), "r"(s_105_bf16[1]), "r"(s_105_bf16[2]), "r"(s_105_bf16[3]) : "memory");
                            int qi_106 = t2 + 896;
                            int srow_107 = qi_106 >> 4;
                            int d0_108 = (qi_106 & 15) * 8;
                            float bs_r_109[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_109[0])) : "r"(beta_base + srow_107 * 4));
                            float fvb_110 = bs_r_109[0];
                            unsigned int vaddr_111 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_108 / 64 * 8192 + srow_107 * 128 + d0_108 % 64 * 2 ^ (d0_108 / 64 * 8192 + srow_107 * 128 + d0_108 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_112[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_112[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_112[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_112[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_112[(0) + 3]))
                                : "r"(vaddr_111));
                            float _unpack_bf16_14[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_112[_pair]) << 16);
                                (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_112[_pair]) & 0xffff0000u);
                            }
                            float s_113[8];
                            s_113[0] = _unpack_bf16_14[0] * fvb_110;
                            s_113[1] = _unpack_bf16_14[1] * fvb_110;
                            s_113[2] = _unpack_bf16_14[2] * fvb_110;
                            s_113[3] = _unpack_bf16_14[3] * fvb_110;
                            s_113[4] = _unpack_bf16_14[4] * fvb_110;
                            s_113[5] = _unpack_bf16_14[5] * fvb_110;
                            s_113[6] = _unpack_bf16_14[6] * fvb_110;
                            s_113[7] = _unpack_bf16_14[7] * fvb_110;
                            unsigned int s_113_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_113[_lp*2 + 0], s_113[_lp*2+1 + 0]));
                                s_113_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_111), "r"(s_113_bf16[0]), "r"(s_113_bf16[1]), "r"(s_113_bf16[2]), "r"(s_113_bf16[3]) : "memory");
                            int qi_114 = t2 + 960;
                            int srow_115 = qi_114 >> 4;
                            int d0_116 = (qi_114 & 15) * 8;
                            float bs_r_117[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_117[0])) : "r"(beta_base + srow_115 * 4));
                            float fvb_118 = bs_r_117[0];
                            unsigned int vaddr_119 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_116 / 64 * 8192 + srow_115 * 128 + d0_116 % 64 * 2 ^ (d0_116 / 64 * 8192 + srow_115 * 128 + d0_116 % 64 * 2 >> 7 & 7) << 4));
                            unsigned int w_120[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w_120[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_120[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_120[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_120[(0) + 3]))
                                : "r"(vaddr_119));
                            float _unpack_bf16_15[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_120[_pair]) << 16);
                                (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_120[_pair]) & 0xffff0000u);
                            }
                            float s_121[8];
                            s_121[0] = _unpack_bf16_15[0] * fvb_118;
                            s_121[1] = _unpack_bf16_15[1] * fvb_118;
                            s_121[2] = _unpack_bf16_15[2] * fvb_118;
                            s_121[3] = _unpack_bf16_15[3] * fvb_118;
                            s_121[4] = _unpack_bf16_15[4] * fvb_118;
                            s_121[5] = _unpack_bf16_15[5] * fvb_118;
                            s_121[6] = _unpack_bf16_15[6] * fvb_118;
                            s_121[7] = _unpack_bf16_15[7] * fvb_118;
                            unsigned int s_121_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_121[_lp*2 + 0], s_121[_lp*2+1 + 0]));
                                s_121_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_119), "r"(s_121_bf16[0]), "r"(s_121_bf16[1]), "r"(s_121_bf16[2]), "r"(s_121_bf16[3]) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int pair = tid_0 >> 6;
                        int t64 = tid_0 & 63;
                        int pr = (t64 & 7) + (t64 >> 5 << 3);
                        int pc4 = (t64 >> 3 & 3) * 4;
                        int xc = pc4 >> 2;
                        int xo0 = xc * 16;
                        int xo1 = (xc ^ 1) * 16;
                        int xo2 = (xc ^ 2) * 16;
                        int xo3 = (xc ^ 3) * 16;
                        int xst = (xc ^ pr >> 1 & 3) * 16;
                        float xs[4];
                        xs[0] = 0.0f;
                        xs[1] = 0.0f;
                        xs[2] = 0.0f;
                        xs[3] = 0.0f;
                        float pv[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv[(0) + 3]))
                            : "r"(l_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pr * 272));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv[(4) + 3]))
                            : "r"(l_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pr * 272 + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv[(8) + 3]))
                            : "r"(l_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pr * 272 + 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv[(12) + 3]))
                            : "r"(l_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pr * 272 + 48));
                        float s0_128 = 0.0f;
                        float s1 = 0.0f;
                        float s2 = 0.0f;
                        float s3 = 0.0f;
                        float q[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q[0])), "=r"(*reinterpret_cast<uint32_t*>(&q[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4));
                        s0_128 = __fmaf_rn(pv[0], q[0], s0_128);
                        s1 = __fmaf_rn(pv[0], q[1], s1);
                        s2 = __fmaf_rn(pv[0], q[2], s2);
                        s3 = __fmaf_rn(pv[0], q[3], s3);
                        float q_129[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_129[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_129[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_129[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_129[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 272));
                        s0_128 = __fmaf_rn(pv[1], q_129[0], s0_128);
                        s1 = __fmaf_rn(pv[1], q_129[1], s1);
                        s2 = __fmaf_rn(pv[1], q_129[2], s2);
                        s3 = __fmaf_rn(pv[1], q_129[3], s3);
                        float q_130[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_130[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_130[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_130[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_130[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 544));
                        s0_128 = __fmaf_rn(pv[2], q_130[0], s0_128);
                        s1 = __fmaf_rn(pv[2], q_130[1], s1);
                        s2 = __fmaf_rn(pv[2], q_130[2], s2);
                        s3 = __fmaf_rn(pv[2], q_130[3], s3);
                        float q_131[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_131[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_131[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_131[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_131[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 816));
                        s0_128 = __fmaf_rn(pv[3], q_131[0], s0_128);
                        s1 = __fmaf_rn(pv[3], q_131[1], s1);
                        s2 = __fmaf_rn(pv[3], q_131[2], s2);
                        s3 = __fmaf_rn(pv[3], q_131[3], s3);
                        float q_132[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_132[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_132[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_132[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_132[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 1088));
                        s0_128 = __fmaf_rn(pv[4], q_132[0], s0_128);
                        s1 = __fmaf_rn(pv[4], q_132[1], s1);
                        s2 = __fmaf_rn(pv[4], q_132[2], s2);
                        s3 = __fmaf_rn(pv[4], q_132[3], s3);
                        float q_133[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_133[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_133[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_133[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_133[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 1360));
                        s0_128 = __fmaf_rn(pv[5], q_133[0], s0_128);
                        s1 = __fmaf_rn(pv[5], q_133[1], s1);
                        s2 = __fmaf_rn(pv[5], q_133[2], s2);
                        s3 = __fmaf_rn(pv[5], q_133[3], s3);
                        float q_134[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_134[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_134[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_134[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_134[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 1632));
                        s0_128 = __fmaf_rn(pv[6], q_134[0], s0_128);
                        s1 = __fmaf_rn(pv[6], q_134[1], s1);
                        s2 = __fmaf_rn(pv[6], q_134[2], s2);
                        s3 = __fmaf_rn(pv[6], q_134[3], s3);
                        float q_135[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_135[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_135[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_135[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_135[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 1904));
                        s0_128 = __fmaf_rn(pv[7], q_135[0], s0_128);
                        s1 = __fmaf_rn(pv[7], q_135[1], s1);
                        s2 = __fmaf_rn(pv[7], q_135[2], s2);
                        s3 = __fmaf_rn(pv[7], q_135[3], s3);
                        float q_136[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_136[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_136[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_136[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_136[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 2176));
                        s0_128 = __fmaf_rn(pv[8], q_136[0], s0_128);
                        s1 = __fmaf_rn(pv[8], q_136[1], s1);
                        s2 = __fmaf_rn(pv[8], q_136[2], s2);
                        s3 = __fmaf_rn(pv[8], q_136[3], s3);
                        float q_137[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_137[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_137[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_137[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_137[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 2448));
                        s0_128 = __fmaf_rn(pv[9], q_137[0], s0_128);
                        s1 = __fmaf_rn(pv[9], q_137[1], s1);
                        s2 = __fmaf_rn(pv[9], q_137[2], s2);
                        s3 = __fmaf_rn(pv[9], q_137[3], s3);
                        float q_138[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_138[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_138[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_138[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_138[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 2720));
                        s0_128 = __fmaf_rn(pv[10], q_138[0], s0_128);
                        s1 = __fmaf_rn(pv[10], q_138[1], s1);
                        s2 = __fmaf_rn(pv[10], q_138[2], s2);
                        s3 = __fmaf_rn(pv[10], q_138[3], s3);
                        float q_139[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_139[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_139[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_139[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_139[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 2992));
                        s0_128 = __fmaf_rn(pv[11], q_139[0], s0_128);
                        s1 = __fmaf_rn(pv[11], q_139[1], s1);
                        s2 = __fmaf_rn(pv[11], q_139[2], s2);
                        s3 = __fmaf_rn(pv[11], q_139[3], s3);
                        float q_140[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_140[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_140[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_140[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_140[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 3264));
                        s0_128 = __fmaf_rn(pv[12], q_140[0], s0_128);
                        s1 = __fmaf_rn(pv[12], q_140[1], s1);
                        s2 = __fmaf_rn(pv[12], q_140[2], s2);
                        s3 = __fmaf_rn(pv[12], q_140[3], s3);
                        float q_141[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_141[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_141[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_141[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_141[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 3536));
                        s0_128 = __fmaf_rn(pv[13], q_141[0], s0_128);
                        s1 = __fmaf_rn(pv[13], q_141[1], s1);
                        s2 = __fmaf_rn(pv[13], q_141[2], s2);
                        s3 = __fmaf_rn(pv[13], q_141[3], s3);
                        float q_142[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_142[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_142[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_142[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_142[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 3808));
                        s0_128 = __fmaf_rn(pv[14], q_142[0], s0_128);
                        s1 = __fmaf_rn(pv[14], q_142[1], s1);
                        s2 = __fmaf_rn(pv[14], q_142[2], s2);
                        s3 = __fmaf_rn(pv[14], q_142[3], s3);
                        float q_143[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_143[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_143[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_143[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_143[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 4080));
                        s0_128 = __fmaf_rn(pv[15], q_143[0], s0_128);
                        s1 = __fmaf_rn(pv[15], q_143[1], s1);
                        s2 = __fmaf_rn(pv[15], q_143[2], s2);
                        s3 = __fmaf_rn(pv[15], q_143[3], s3);
                        xs[0] = xs[0] + s0_128;
                        xs[1] = xs[1] + s1;
                        xs[2] = xs[2] + s2;
                        xs[3] = xs[3] + s3;
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(x_base + pair * 1024 + pr * 64 + xst), "f"(xs[0]), "f"(xs[1]), "f"(xs[2]), "f"(xs[3]) : "memory");
                        if (pair == 0) {
                            float xs_0[4];
                            xs_0[0] = 0.0f;
                            xs_0[1] = 0.0f;
                            xs_0[2] = 0.0f;
                            xs_0[3] = 0.0f;
                            float pv_1[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1[(0) + 3]))
                                : "r"(l_base + 13184 + pr * 272));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1[(4) + 3]))
                                : "r"(l_base + 13184 + pr * 272 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1[(8) + 3]))
                                : "r"(l_base + 13184 + pr * 272 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1[(12) + 3]))
                                : "r"(l_base + 13184 + pr * 272 + 48));
                            float s0_2 = 0.0f;
                            float s1_3 = 0.0f;
                            float s2_4 = 0.0f;
                            float s3_5 = 0.0f;
                            float q_6[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_6[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4));
                            s0_2 = __fmaf_rn(pv_1[0], q_6[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[0], q_6[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[0], q_6[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[0], q_6[3], s3_5);
                            float q_7[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_7[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 272));
                            s0_2 = __fmaf_rn(pv_1[1], q_7[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[1], q_7[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[1], q_7[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[1], q_7[3], s3_5);
                            float q_8[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_8[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 544));
                            s0_2 = __fmaf_rn(pv_1[2], q_8[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[2], q_8[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[2], q_8[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[2], q_8[3], s3_5);
                            float q_9[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_9[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 816));
                            s0_2 = __fmaf_rn(pv_1[3], q_9[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[3], q_9[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[3], q_9[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[3], q_9[3], s3_5);
                            float q_10[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_10[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 1088));
                            s0_2 = __fmaf_rn(pv_1[4], q_10[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[4], q_10[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[4], q_10[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[4], q_10[3], s3_5);
                            float q_11[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_11[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 1360));
                            s0_2 = __fmaf_rn(pv_1[5], q_11[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[5], q_11[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[5], q_11[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[5], q_11[3], s3_5);
                            float q_12[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_12[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 1632));
                            s0_2 = __fmaf_rn(pv_1[6], q_12[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[6], q_12[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[6], q_12[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[6], q_12[3], s3_5);
                            float q_13[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_13[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 1904));
                            s0_2 = __fmaf_rn(pv_1[7], q_13[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[7], q_13[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[7], q_13[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[7], q_13[3], s3_5);
                            float q_14[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_14[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 2176));
                            s0_2 = __fmaf_rn(pv_1[8], q_14[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[8], q_14[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[8], q_14[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[8], q_14[3], s3_5);
                            float q_15[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_15[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 2448));
                            s0_2 = __fmaf_rn(pv_1[9], q_15[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[9], q_15[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[9], q_15[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[9], q_15[3], s3_5);
                            float q_16[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_16[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 2720));
                            s0_2 = __fmaf_rn(pv_1[10], q_16[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[10], q_16[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[10], q_16[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[10], q_16[3], s3_5);
                            float q_17[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_17[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 2992));
                            s0_2 = __fmaf_rn(pv_1[11], q_17[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[11], q_17[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[11], q_17[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[11], q_17[3], s3_5);
                            float q_18[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_18[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 3264));
                            s0_2 = __fmaf_rn(pv_1[12], q_18[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[12], q_18[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[12], q_18[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[12], q_18[3], s3_5);
                            float q_19[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_19[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 3536));
                            s0_2 = __fmaf_rn(pv_1[13], q_19[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[13], q_19[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[13], q_19[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[13], q_19[3], s3_5);
                            float q_20[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_20[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 3808));
                            s0_2 = __fmaf_rn(pv_1[14], q_20[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[14], q_20[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[14], q_20[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[14], q_20[3], s3_5);
                            float q_21[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_21[(0) + 3]))
                                : "r"(a_base + 8832 + pc4 * 4 + 4080));
                            s0_2 = __fmaf_rn(pv_1[15], q_21[0], s0_2);
                            s1_3 = __fmaf_rn(pv_1[15], q_21[1], s1_3);
                            s2_4 = __fmaf_rn(pv_1[15], q_21[2], s2_4);
                            s3_5 = __fmaf_rn(pv_1[15], q_21[3], s3_5);
                            xs_0[0] = xs_0[0] + s0_2;
                            xs_0[1] = xs_0[1] + s1_3;
                            xs_0[2] = xs_0[2] + s2_4;
                            xs_0[3] = xs_0[3] + s3_5;
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(x_base + 2048 + pr * 64 + xst), "f"(xs_0[0]), "f"(xs_0[1]), "f"(xs_0[2]), "f"(xs_0[3]) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        float pv_144[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_144[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv_144[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_144[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_144[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + (pair + 1) * 16) * 4 + pr * 272));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_144[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv_144[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_144[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_144[(4) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + (pair + 1) * 16) * 4 + pr * 272 + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_144[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv_144[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_144[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_144[(8) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + (pair + 1) * 16) * 4 + pr * 272 + 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_144[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv_144[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_144[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_144[(12) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + (pair + 1) * 16) * 4 + pr * 272 + 48));
                        float s0_145 = 0.0f;
                        float s1_146 = 0.0f;
                        float s2_147 = 0.0f;
                        float s3_148 = 0.0f;
                        float q_149[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_149[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_149[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_149[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_149[(0) + 3]))
                            : "r"(x_base + pair * 1024 + xo0));
                        s0_145 = __fmaf_rn(pv_144[0], q_149[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[0], q_149[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[0], q_149[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[0], q_149[3], s3_148);
                        float q_150[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_150[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_150[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_150[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_150[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 64 + xo0));
                        s0_145 = __fmaf_rn(pv_144[1], q_150[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[1], q_150[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[1], q_150[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[1], q_150[3], s3_148);
                        float q_151[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_151[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_151[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_151[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_151[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 128 + xo1));
                        s0_145 = __fmaf_rn(pv_144[2], q_151[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[2], q_151[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[2], q_151[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[2], q_151[3], s3_148);
                        float q_152[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_152[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_152[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_152[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_152[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 192 + xo1));
                        s0_145 = __fmaf_rn(pv_144[3], q_152[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[3], q_152[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[3], q_152[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[3], q_152[3], s3_148);
                        float q_153[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_153[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_153[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_153[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_153[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 256 + xo2));
                        s0_145 = __fmaf_rn(pv_144[4], q_153[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[4], q_153[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[4], q_153[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[4], q_153[3], s3_148);
                        float q_154[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_154[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_154[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_154[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_154[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 320 + xo2));
                        s0_145 = __fmaf_rn(pv_144[5], q_154[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[5], q_154[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[5], q_154[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[5], q_154[3], s3_148);
                        float q_155[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_155[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_155[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_155[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_155[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 384 + xo3));
                        s0_145 = __fmaf_rn(pv_144[6], q_155[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[6], q_155[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[6], q_155[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[6], q_155[3], s3_148);
                        float q_156[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_156[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_156[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_156[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_156[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 448 + xo3));
                        s0_145 = __fmaf_rn(pv_144[7], q_156[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[7], q_156[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[7], q_156[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[7], q_156[3], s3_148);
                        float q_157[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_157[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_157[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_157[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_157[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 512 + xo0));
                        s0_145 = __fmaf_rn(pv_144[8], q_157[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[8], q_157[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[8], q_157[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[8], q_157[3], s3_148);
                        float q_158[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_158[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_158[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_158[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_158[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 576 + xo0));
                        s0_145 = __fmaf_rn(pv_144[9], q_158[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[9], q_158[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[9], q_158[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[9], q_158[3], s3_148);
                        float q_159[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_159[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_159[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_159[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_159[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 640 + xo1));
                        s0_145 = __fmaf_rn(pv_144[10], q_159[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[10], q_159[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[10], q_159[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[10], q_159[3], s3_148);
                        float q_160[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_160[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_160[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_160[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_160[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 704 + xo1));
                        s0_145 = __fmaf_rn(pv_144[11], q_160[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[11], q_160[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[11], q_160[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[11], q_160[3], s3_148);
                        float q_161[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_161[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_161[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_161[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_161[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 768 + xo2));
                        s0_145 = __fmaf_rn(pv_144[12], q_161[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[12], q_161[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[12], q_161[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[12], q_161[3], s3_148);
                        float q_162[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_162[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_162[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_162[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_162[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 832 + xo2));
                        s0_145 = __fmaf_rn(pv_144[13], q_162[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[13], q_162[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[13], q_162[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[13], q_162[3], s3_148);
                        float q_163[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_163[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_163[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_163[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_163[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 896 + xo3));
                        s0_145 = __fmaf_rn(pv_144[14], q_163[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[14], q_163[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[14], q_163[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[14], q_163[3], s3_148);
                        float q_164[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_164[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_164[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_164[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_164[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 960 + xo3));
                        s0_145 = __fmaf_rn(pv_144[15], q_164[0], s0_145);
                        s1_146 = __fmaf_rn(pv_144[15], q_164[1], s1_146);
                        s2_147 = __fmaf_rn(pv_144[15], q_164[2], s2_147);
                        s3_148 = __fmaf_rn(pv_144[15], q_164[3], s3_148);
                        float ns[4];
                        ns[0] = -s0_145;
                        ns[1] = -s1_146;
                        ns[2] = -s2_147;
                        ns[3] = -s3_148;
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + (pr * 68 + pc4) * 4), "f"(ns[0]), "f"(ns[1]), "f"(ns[2]), "f"(ns[3]) : "memory");
                        if (pair == 0) {
                            float pv_0[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0[(0) + 3]))
                                : "r"(a_base + 13248 + pr * 272));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0[(4) + 3]))
                                : "r"(a_base + 13248 + pr * 272 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0[(8) + 3]))
                                : "r"(a_base + 13248 + pr * 272 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0[(12) + 3]))
                                : "r"(a_base + 13248 + pr * 272 + 48));
                            float s0_1 = 0.0f;
                            float s1_2 = 0.0f;
                            float s2_3 = 0.0f;
                            float s3_4 = 0.0f;
                            float q_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_5[(0) + 3]))
                                : "r"(x_base + 2048 + xo0));
                            s0_1 = __fmaf_rn(pv_0[0], q_5[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[0], q_5[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[0], q_5[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[0], q_5[3], s3_4);
                            float q_6_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_6_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_6_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_6_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_6_1[(0) + 3]))
                                : "r"(x_base + 2048 + 64 + xo0));
                            s0_1 = __fmaf_rn(pv_0[1], q_6_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[1], q_6_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[1], q_6_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[1], q_6_1[3], s3_4);
                            float q_7_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_7_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_7_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_7_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_7_1[(0) + 3]))
                                : "r"(x_base + 2048 + 128 + xo1));
                            s0_1 = __fmaf_rn(pv_0[2], q_7_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[2], q_7_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[2], q_7_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[2], q_7_1[3], s3_4);
                            float q_8_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_8_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_8_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_8_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_8_1[(0) + 3]))
                                : "r"(x_base + 2048 + 192 + xo1));
                            s0_1 = __fmaf_rn(pv_0[3], q_8_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[3], q_8_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[3], q_8_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[3], q_8_1[3], s3_4);
                            float q_9_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_9_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_9_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_9_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_9_1[(0) + 3]))
                                : "r"(x_base + 2048 + 256 + xo2));
                            s0_1 = __fmaf_rn(pv_0[4], q_9_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[4], q_9_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[4], q_9_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[4], q_9_1[3], s3_4);
                            float q_10_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_10_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_10_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_10_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_10_1[(0) + 3]))
                                : "r"(x_base + 2048 + 320 + xo2));
                            s0_1 = __fmaf_rn(pv_0[5], q_10_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[5], q_10_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[5], q_10_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[5], q_10_1[3], s3_4);
                            float q_11_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_11_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_11_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_11_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_11_1[(0) + 3]))
                                : "r"(x_base + 2048 + 384 + xo3));
                            s0_1 = __fmaf_rn(pv_0[6], q_11_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[6], q_11_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[6], q_11_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[6], q_11_1[3], s3_4);
                            float q_12_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_12_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_12_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_12_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_12_1[(0) + 3]))
                                : "r"(x_base + 2048 + 448 + xo3));
                            s0_1 = __fmaf_rn(pv_0[7], q_12_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[7], q_12_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[7], q_12_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[7], q_12_1[3], s3_4);
                            float q_13_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_13_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_13_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_13_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_13_1[(0) + 3]))
                                : "r"(x_base + 2048 + 512 + xo0));
                            s0_1 = __fmaf_rn(pv_0[8], q_13_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[8], q_13_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[8], q_13_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[8], q_13_1[3], s3_4);
                            float q_14_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_14_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_14_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_14_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_14_1[(0) + 3]))
                                : "r"(x_base + 2048 + 576 + xo0));
                            s0_1 = __fmaf_rn(pv_0[9], q_14_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[9], q_14_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[9], q_14_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[9], q_14_1[3], s3_4);
                            float q_15_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_15_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_15_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_15_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_15_1[(0) + 3]))
                                : "r"(x_base + 2048 + 640 + xo1));
                            s0_1 = __fmaf_rn(pv_0[10], q_15_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[10], q_15_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[10], q_15_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[10], q_15_1[3], s3_4);
                            float q_16_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_16_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_16_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_16_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_16_1[(0) + 3]))
                                : "r"(x_base + 2048 + 704 + xo1));
                            s0_1 = __fmaf_rn(pv_0[11], q_16_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[11], q_16_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[11], q_16_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[11], q_16_1[3], s3_4);
                            float q_17_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_17_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_17_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_17_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_17_1[(0) + 3]))
                                : "r"(x_base + 2048 + 768 + xo2));
                            s0_1 = __fmaf_rn(pv_0[12], q_17_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[12], q_17_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[12], q_17_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[12], q_17_1[3], s3_4);
                            float q_18_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_18_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_18_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_18_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_18_1[(0) + 3]))
                                : "r"(x_base + 2048 + 832 + xo2));
                            s0_1 = __fmaf_rn(pv_0[13], q_18_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[13], q_18_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[13], q_18_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[13], q_18_1[3], s3_4);
                            float q_19_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_19_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_19_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_19_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_19_1[(0) + 3]))
                                : "r"(x_base + 2048 + 896 + xo3));
                            s0_1 = __fmaf_rn(pv_0[14], q_19_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[14], q_19_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[14], q_19_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[14], q_19_1[3], s3_4);
                            float q_20_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_20_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_20_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_20_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_20_1[(0) + 3]))
                                : "r"(x_base + 2048 + 960 + xo3));
                            s0_1 = __fmaf_rn(pv_0[15], q_20_1[0], s0_1);
                            s1_2 = __fmaf_rn(pv_0[15], q_20_1[1], s1_2);
                            s2_3 = __fmaf_rn(pv_0[15], q_20_1[2], s2_3);
                            s3_4 = __fmaf_rn(pv_0[15], q_20_1[3], s3_4);
                            float ns_21[4];
                            ns_21[0] = -s0_1;
                            ns_21[1] = -s1_2;
                            ns_21[2] = -s2_3;
                            ns_21[3] = -s3_4;
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(a_base + 13184 + (pr * 68 + pc4) * 4), "f"(ns_21[0]), "f"(ns_21[1]), "f"(ns_21[2]), "f"(ns_21[3]) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        float xs_165[4];
                        xs_165[0] = 0.0f;
                        xs_165[1] = 0.0f;
                        xs_165[2] = 0.0f;
                        xs_165[3] = 0.0f;
                        float pv_166[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_166[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv_166[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_166[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_166[(0) + 3]))
                            : "r"(l_base + ((pair + 2) * 16 * 68 + pair * 16) * 4 + pr * 272));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_166[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv_166[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_166[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_166[(4) + 3]))
                            : "r"(l_base + ((pair + 2) * 16 * 68 + pair * 16) * 4 + pr * 272 + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_166[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv_166[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_166[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_166[(8) + 3]))
                            : "r"(l_base + ((pair + 2) * 16 * 68 + pair * 16) * 4 + pr * 272 + 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_166[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv_166[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_166[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_166[(12) + 3]))
                            : "r"(l_base + ((pair + 2) * 16 * 68 + pair * 16) * 4 + pr * 272 + 48));
                        float s0_167 = 0.0f;
                        float s1_168 = 0.0f;
                        float s2_169 = 0.0f;
                        float s3_170 = 0.0f;
                        float q_171[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_171[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_171[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_171[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_171[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4));
                        s0_167 = __fmaf_rn(pv_166[0], q_171[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[0], q_171[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[0], q_171[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[0], q_171[3], s3_170);
                        float q_172[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_172[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_172[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_172[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_172[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 272));
                        s0_167 = __fmaf_rn(pv_166[1], q_172[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[1], q_172[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[1], q_172[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[1], q_172[3], s3_170);
                        float q_173[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_173[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_173[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_173[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_173[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 544));
                        s0_167 = __fmaf_rn(pv_166[2], q_173[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[2], q_173[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[2], q_173[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[2], q_173[3], s3_170);
                        float q_174[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_174[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_174[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_174[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_174[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 816));
                        s0_167 = __fmaf_rn(pv_166[3], q_174[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[3], q_174[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[3], q_174[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[3], q_174[3], s3_170);
                        float q_175[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_175[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_175[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_175[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_175[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 1088));
                        s0_167 = __fmaf_rn(pv_166[4], q_175[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[4], q_175[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[4], q_175[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[4], q_175[3], s3_170);
                        float q_176[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_176[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_176[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_176[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_176[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 1360));
                        s0_167 = __fmaf_rn(pv_166[5], q_176[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[5], q_176[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[5], q_176[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[5], q_176[3], s3_170);
                        float q_177[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_177[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_177[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_177[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_177[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 1632));
                        s0_167 = __fmaf_rn(pv_166[6], q_177[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[6], q_177[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[6], q_177[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[6], q_177[3], s3_170);
                        float q_178[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_178[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_178[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_178[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_178[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 1904));
                        s0_167 = __fmaf_rn(pv_166[7], q_178[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[7], q_178[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[7], q_178[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[7], q_178[3], s3_170);
                        float q_179[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_179[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_179[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_179[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_179[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 2176));
                        s0_167 = __fmaf_rn(pv_166[8], q_179[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[8], q_179[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[8], q_179[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[8], q_179[3], s3_170);
                        float q_180[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_180[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_180[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_180[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_180[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 2448));
                        s0_167 = __fmaf_rn(pv_166[9], q_180[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[9], q_180[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[9], q_180[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[9], q_180[3], s3_170);
                        float q_181[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_181[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_181[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_181[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_181[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 2720));
                        s0_167 = __fmaf_rn(pv_166[10], q_181[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[10], q_181[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[10], q_181[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[10], q_181[3], s3_170);
                        float q_182[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_182[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_182[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_182[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_182[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 2992));
                        s0_167 = __fmaf_rn(pv_166[11], q_182[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[11], q_182[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[11], q_182[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[11], q_182[3], s3_170);
                        float q_183[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_183[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_183[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_183[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_183[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 3264));
                        s0_167 = __fmaf_rn(pv_166[12], q_183[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[12], q_183[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[12], q_183[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[12], q_183[3], s3_170);
                        float q_184[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_184[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_184[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_184[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_184[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 3536));
                        s0_167 = __fmaf_rn(pv_166[13], q_184[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[13], q_184[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[13], q_184[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[13], q_184[3], s3_170);
                        float q_185[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_185[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_185[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_185[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_185[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 3808));
                        s0_167 = __fmaf_rn(pv_166[14], q_185[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[14], q_185[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[14], q_185[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[14], q_185[3], s3_170);
                        float q_186[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_186[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_186[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_186[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_186[(0) + 3]))
                            : "r"(a_base + (pair * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 4080));
                        s0_167 = __fmaf_rn(pv_166[15], q_186[0], s0_167);
                        s1_168 = __fmaf_rn(pv_166[15], q_186[1], s1_168);
                        s2_169 = __fmaf_rn(pv_166[15], q_186[2], s2_169);
                        s3_170 = __fmaf_rn(pv_166[15], q_186[3], s3_170);
                        xs_165[0] = xs_165[0] + s0_167;
                        xs_165[1] = xs_165[1] + s1_168;
                        xs_165[2] = xs_165[2] + s2_169;
                        xs_165[3] = xs_165[3] + s3_170;
                        float pv_187[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_187[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv_187[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_187[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_187[(0) + 3]))
                            : "r"(l_base + ((pair + 2) * 16 * 68 + (pair + 1) * 16) * 4 + pr * 272));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_187[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv_187[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_187[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_187[(4) + 3]))
                            : "r"(l_base + ((pair + 2) * 16 * 68 + (pair + 1) * 16) * 4 + pr * 272 + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_187[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv_187[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_187[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_187[(8) + 3]))
                            : "r"(l_base + ((pair + 2) * 16 * 68 + (pair + 1) * 16) * 4 + pr * 272 + 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_187[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv_187[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_187[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_187[(12) + 3]))
                            : "r"(l_base + ((pair + 2) * 16 * 68 + (pair + 1) * 16) * 4 + pr * 272 + 48));
                        float s0_188 = 0.0f;
                        float s1_189 = 0.0f;
                        float s2_190 = 0.0f;
                        float s3_191 = 0.0f;
                        float q_192[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_192[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_192[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_192[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_192[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4));
                        s0_188 = __fmaf_rn(pv_187[0], q_192[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[0], q_192[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[0], q_192[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[0], q_192[3], s3_191);
                        float q_193[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_193[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_193[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_193[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_193[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 272));
                        s0_188 = __fmaf_rn(pv_187[1], q_193[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[1], q_193[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[1], q_193[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[1], q_193[3], s3_191);
                        float q_194[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_194[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_194[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_194[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_194[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 544));
                        s0_188 = __fmaf_rn(pv_187[2], q_194[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[2], q_194[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[2], q_194[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[2], q_194[3], s3_191);
                        float q_195[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_195[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_195[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_195[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_195[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 816));
                        s0_188 = __fmaf_rn(pv_187[3], q_195[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[3], q_195[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[3], q_195[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[3], q_195[3], s3_191);
                        float q_196[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_196[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_196[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_196[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_196[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 1088));
                        s0_188 = __fmaf_rn(pv_187[4], q_196[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[4], q_196[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[4], q_196[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[4], q_196[3], s3_191);
                        float q_197[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_197[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_197[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_197[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_197[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 1360));
                        s0_188 = __fmaf_rn(pv_187[5], q_197[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[5], q_197[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[5], q_197[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[5], q_197[3], s3_191);
                        float q_198[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_198[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_198[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_198[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_198[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 1632));
                        s0_188 = __fmaf_rn(pv_187[6], q_198[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[6], q_198[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[6], q_198[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[6], q_198[3], s3_191);
                        float q_199[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_199[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_199[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_199[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_199[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 1904));
                        s0_188 = __fmaf_rn(pv_187[7], q_199[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[7], q_199[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[7], q_199[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[7], q_199[3], s3_191);
                        float q_200[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_200[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_200[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_200[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_200[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 2176));
                        s0_188 = __fmaf_rn(pv_187[8], q_200[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[8], q_200[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[8], q_200[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[8], q_200[3], s3_191);
                        float q_201[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_201[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_201[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_201[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_201[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 2448));
                        s0_188 = __fmaf_rn(pv_187[9], q_201[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[9], q_201[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[9], q_201[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[9], q_201[3], s3_191);
                        float q_202[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_202[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_202[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_202[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_202[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 2720));
                        s0_188 = __fmaf_rn(pv_187[10], q_202[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[10], q_202[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[10], q_202[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[10], q_202[3], s3_191);
                        float q_203[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_203[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_203[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_203[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_203[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 2992));
                        s0_188 = __fmaf_rn(pv_187[11], q_203[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[11], q_203[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[11], q_203[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[11], q_203[3], s3_191);
                        float q_204[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_204[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_204[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_204[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_204[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 3264));
                        s0_188 = __fmaf_rn(pv_187[12], q_204[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[12], q_204[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[12], q_204[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[12], q_204[3], s3_191);
                        float q_205[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_205[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_205[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_205[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_205[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 3536));
                        s0_188 = __fmaf_rn(pv_187[13], q_205[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[13], q_205[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[13], q_205[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[13], q_205[3], s3_191);
                        float q_206[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_206[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_206[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_206[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_206[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 3808));
                        s0_188 = __fmaf_rn(pv_187[14], q_206[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[14], q_206[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[14], q_206[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[14], q_206[3], s3_191);
                        float q_207[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_207[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_207[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_207[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_207[(0) + 3]))
                            : "r"(a_base + ((pair + 1) * 16 * 68 + pair * 16) * 4 + pc4 * 4 + 4080));
                        s0_188 = __fmaf_rn(pv_187[15], q_207[0], s0_188);
                        s1_189 = __fmaf_rn(pv_187[15], q_207[1], s1_189);
                        s2_190 = __fmaf_rn(pv_187[15], q_207[2], s2_190);
                        s3_191 = __fmaf_rn(pv_187[15], q_207[3], s3_191);
                        xs_165[0] = xs_165[0] + s0_188;
                        xs_165[1] = xs_165[1] + s1_189;
                        xs_165[2] = xs_165[2] + s2_190;
                        xs_165[3] = xs_165[3] + s3_191;
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(x_base + pair * 1024 + pr * 64 + xst), "f"(xs_165[0]), "f"(xs_165[1]), "f"(xs_165[2]), "f"(xs_165[3]) : "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        float pv_208[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_208[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv_208[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_208[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_208[(0) + 3]))
                            : "r"(a_base + ((pair + 2) * 16 * 68 + (pair + 2) * 16) * 4 + pr * 272));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_208[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv_208[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_208[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_208[(4) + 3]))
                            : "r"(a_base + ((pair + 2) * 16 * 68 + (pair + 2) * 16) * 4 + pr * 272 + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_208[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv_208[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_208[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_208[(8) + 3]))
                            : "r"(a_base + ((pair + 2) * 16 * 68 + (pair + 2) * 16) * 4 + pr * 272 + 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pv_208[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv_208[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_208[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_208[(12) + 3]))
                            : "r"(a_base + ((pair + 2) * 16 * 68 + (pair + 2) * 16) * 4 + pr * 272 + 48));
                        float s0_209 = 0.0f;
                        float s1_210 = 0.0f;
                        float s2_211 = 0.0f;
                        float s3_212 = 0.0f;
                        float q_213[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_213[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_213[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_213[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_213[(0) + 3]))
                            : "r"(x_base + pair * 1024 + xo0));
                        s0_209 = __fmaf_rn(pv_208[0], q_213[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[0], q_213[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[0], q_213[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[0], q_213[3], s3_212);
                        float q_214[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_214[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_214[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_214[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_214[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 64 + xo0));
                        s0_209 = __fmaf_rn(pv_208[1], q_214[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[1], q_214[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[1], q_214[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[1], q_214[3], s3_212);
                        float q_215[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_215[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_215[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_215[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_215[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 128 + xo1));
                        s0_209 = __fmaf_rn(pv_208[2], q_215[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[2], q_215[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[2], q_215[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[2], q_215[3], s3_212);
                        float q_216[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_216[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_216[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_216[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_216[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 192 + xo1));
                        s0_209 = __fmaf_rn(pv_208[3], q_216[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[3], q_216[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[3], q_216[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[3], q_216[3], s3_212);
                        float q_217[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_217[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_217[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_217[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_217[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 256 + xo2));
                        s0_209 = __fmaf_rn(pv_208[4], q_217[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[4], q_217[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[4], q_217[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[4], q_217[3], s3_212);
                        float q_218[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_218[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_218[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_218[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_218[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 320 + xo2));
                        s0_209 = __fmaf_rn(pv_208[5], q_218[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[5], q_218[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[5], q_218[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[5], q_218[3], s3_212);
                        float q_219[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_219[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_219[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_219[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_219[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 384 + xo3));
                        s0_209 = __fmaf_rn(pv_208[6], q_219[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[6], q_219[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[6], q_219[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[6], q_219[3], s3_212);
                        float q_220[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_220[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_220[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_220[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_220[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 448 + xo3));
                        s0_209 = __fmaf_rn(pv_208[7], q_220[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[7], q_220[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[7], q_220[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[7], q_220[3], s3_212);
                        float q_221[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_221[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_221[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_221[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_221[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 512 + xo0));
                        s0_209 = __fmaf_rn(pv_208[8], q_221[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[8], q_221[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[8], q_221[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[8], q_221[3], s3_212);
                        float q_222[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_222[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_222[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_222[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_222[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 576 + xo0));
                        s0_209 = __fmaf_rn(pv_208[9], q_222[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[9], q_222[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[9], q_222[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[9], q_222[3], s3_212);
                        float q_223[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_223[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_223[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_223[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_223[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 640 + xo1));
                        s0_209 = __fmaf_rn(pv_208[10], q_223[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[10], q_223[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[10], q_223[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[10], q_223[3], s3_212);
                        float q_224[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_224[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_224[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_224[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_224[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 704 + xo1));
                        s0_209 = __fmaf_rn(pv_208[11], q_224[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[11], q_224[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[11], q_224[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[11], q_224[3], s3_212);
                        float q_225[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_225[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_225[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_225[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_225[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 768 + xo2));
                        s0_209 = __fmaf_rn(pv_208[12], q_225[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[12], q_225[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[12], q_225[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[12], q_225[3], s3_212);
                        float q_226[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_226[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_226[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_226[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_226[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 832 + xo2));
                        s0_209 = __fmaf_rn(pv_208[13], q_226[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[13], q_226[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[13], q_226[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[13], q_226[3], s3_212);
                        float q_227[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_227[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_227[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_227[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_227[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 896 + xo3));
                        s0_209 = __fmaf_rn(pv_208[14], q_227[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[14], q_227[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[14], q_227[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[14], q_227[3], s3_212);
                        float q_228[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_228[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_228[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_228[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_228[(0) + 3]))
                            : "r"(x_base + pair * 1024 + 960 + xo3));
                        s0_209 = __fmaf_rn(pv_208[15], q_228[0], s0_209);
                        s1_210 = __fmaf_rn(pv_208[15], q_228[1], s1_210);
                        s2_211 = __fmaf_rn(pv_208[15], q_228[2], s2_211);
                        s3_212 = __fmaf_rn(pv_208[15], q_228[3], s3_212);
                        float ns_229[4];
                        ns_229[0] = -s0_209;
                        ns_229[1] = -s1_210;
                        ns_229[2] = -s2_211;
                        ns_229[3] = -s3_212;
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(a_base + ((pair + 2) * 16 * 68 + pair * 16) * 4 + (pr * 68 + pc4) * 4), "f"(ns_229[0]), "f"(ns_229[1]), "f"(ns_229[2]), "f"(ns_229[3]) : "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (pair == 0) {
                            float xs_0_1[4];
                            xs_0_1[0] = 0.0f;
                            xs_0_1[1] = 0.0f;
                            xs_0_1[2] = 0.0f;
                            xs_0_1[3] = 0.0f;
                            float pv_1_1[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[(0) + 3]))
                                : "r"(l_base + 13056 + pr * 272));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[(4) + 3]))
                                : "r"(l_base + 13056 + pr * 272 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[(8) + 3]))
                                : "r"(l_base + 13056 + pr * 272 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_1_1[(12) + 3]))
                                : "r"(l_base + 13056 + pr * 272 + 48));
                            float s0_2_1 = 0.0f;
                            float s1_3_1 = 0.0f;
                            float s2_4_1 = 0.0f;
                            float s3_5_1 = 0.0f;
                            float q_6_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_6_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_6_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_6_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_6_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4));
                            s0_2_1 = __fmaf_rn(pv_1_1[0], q_6_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[0], q_6_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[0], q_6_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[0], q_6_2[3], s3_5_1);
                            float q_7_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_7_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_7_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_7_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_7_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 272));
                            s0_2_1 = __fmaf_rn(pv_1_1[1], q_7_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[1], q_7_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[1], q_7_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[1], q_7_2[3], s3_5_1);
                            float q_8_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_8_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_8_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_8_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_8_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 544));
                            s0_2_1 = __fmaf_rn(pv_1_1[2], q_8_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[2], q_8_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[2], q_8_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[2], q_8_2[3], s3_5_1);
                            float q_9_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_9_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_9_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_9_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_9_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 816));
                            s0_2_1 = __fmaf_rn(pv_1_1[3], q_9_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[3], q_9_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[3], q_9_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[3], q_9_2[3], s3_5_1);
                            float q_10_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_10_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_10_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_10_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_10_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 1088));
                            s0_2_1 = __fmaf_rn(pv_1_1[4], q_10_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[4], q_10_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[4], q_10_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[4], q_10_2[3], s3_5_1);
                            float q_11_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_11_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_11_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_11_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_11_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 1360));
                            s0_2_1 = __fmaf_rn(pv_1_1[5], q_11_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[5], q_11_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[5], q_11_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[5], q_11_2[3], s3_5_1);
                            float q_12_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_12_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_12_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_12_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_12_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 1632));
                            s0_2_1 = __fmaf_rn(pv_1_1[6], q_12_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[6], q_12_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[6], q_12_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[6], q_12_2[3], s3_5_1);
                            float q_13_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_13_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_13_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_13_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_13_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 1904));
                            s0_2_1 = __fmaf_rn(pv_1_1[7], q_13_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[7], q_13_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[7], q_13_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[7], q_13_2[3], s3_5_1);
                            float q_14_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_14_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_14_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_14_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_14_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 2176));
                            s0_2_1 = __fmaf_rn(pv_1_1[8], q_14_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[8], q_14_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[8], q_14_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[8], q_14_2[3], s3_5_1);
                            float q_15_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_15_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_15_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_15_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_15_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 2448));
                            s0_2_1 = __fmaf_rn(pv_1_1[9], q_15_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[9], q_15_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[9], q_15_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[9], q_15_2[3], s3_5_1);
                            float q_16_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_16_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_16_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_16_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_16_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 2720));
                            s0_2_1 = __fmaf_rn(pv_1_1[10], q_16_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[10], q_16_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[10], q_16_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[10], q_16_2[3], s3_5_1);
                            float q_17_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_17_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_17_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_17_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_17_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 2992));
                            s0_2_1 = __fmaf_rn(pv_1_1[11], q_17_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[11], q_17_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[11], q_17_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[11], q_17_2[3], s3_5_1);
                            float q_18_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_18_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_18_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_18_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_18_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 3264));
                            s0_2_1 = __fmaf_rn(pv_1_1[12], q_18_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[12], q_18_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[12], q_18_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[12], q_18_2[3], s3_5_1);
                            float q_19_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_19_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_19_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_19_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_19_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 3536));
                            s0_2_1 = __fmaf_rn(pv_1_1[13], q_19_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[13], q_19_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[13], q_19_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[13], q_19_2[3], s3_5_1);
                            float q_20_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_20_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_20_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_20_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_20_2[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 3808));
                            s0_2_1 = __fmaf_rn(pv_1_1[14], q_20_2[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[14], q_20_2[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[14], q_20_2[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[14], q_20_2[3], s3_5_1);
                            float q_21_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_21_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_21_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_21_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_21_1[(0) + 3]))
                                : "r"(a_base + pc4 * 4 + 4080));
                            s0_2_1 = __fmaf_rn(pv_1_1[15], q_21_1[0], s0_2_1);
                            s1_3_1 = __fmaf_rn(pv_1_1[15], q_21_1[1], s1_3_1);
                            s2_4_1 = __fmaf_rn(pv_1_1[15], q_21_1[2], s2_4_1);
                            s3_5_1 = __fmaf_rn(pv_1_1[15], q_21_1[3], s3_5_1);
                            xs_0_1[0] = xs_0_1[0] + s0_2_1;
                            xs_0_1[1] = xs_0_1[1] + s1_3_1;
                            xs_0_1[2] = xs_0_1[2] + s2_4_1;
                            xs_0_1[3] = xs_0_1[3] + s3_5_1;
                            float pv_22[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_22[(0) + 3]))
                                : "r"(l_base + 13120 + pr * 272));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_22[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv_22[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_22[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_22[(4) + 3]))
                                : "r"(l_base + 13120 + pr * 272 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_22[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv_22[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_22[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_22[(8) + 3]))
                                : "r"(l_base + 13120 + pr * 272 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_22[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv_22[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_22[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_22[(12) + 3]))
                                : "r"(l_base + 13120 + pr * 272 + 48));
                            float s0_23 = 0.0f;
                            float s1_24 = 0.0f;
                            float s2_25 = 0.0f;
                            float s3_26 = 0.0f;
                            float q_27[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_27[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4));
                            s0_23 = __fmaf_rn(pv_22[0], q_27[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[0], q_27[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[0], q_27[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[0], q_27[3], s3_26);
                            float q_28[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_28[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_28[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_28[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 272));
                            s0_23 = __fmaf_rn(pv_22[1], q_28[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[1], q_28[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[1], q_28[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[1], q_28[3], s3_26);
                            float q_29[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_29[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_29[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_29[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 544));
                            s0_23 = __fmaf_rn(pv_22[2], q_29[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[2], q_29[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[2], q_29[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[2], q_29[3], s3_26);
                            float q_30[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_30[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 816));
                            s0_23 = __fmaf_rn(pv_22[3], q_30[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[3], q_30[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[3], q_30[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[3], q_30[3], s3_26);
                            float q_31[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_31[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_31[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_31[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 1088));
                            s0_23 = __fmaf_rn(pv_22[4], q_31[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[4], q_31[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[4], q_31[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[4], q_31[3], s3_26);
                            float q_32[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_32[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_32[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_32[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 1360));
                            s0_23 = __fmaf_rn(pv_22[5], q_32[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[5], q_32[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[5], q_32[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[5], q_32[3], s3_26);
                            float q_33[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_33[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_33[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_33[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_33[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 1632));
                            s0_23 = __fmaf_rn(pv_22[6], q_33[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[6], q_33[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[6], q_33[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[6], q_33[3], s3_26);
                            float q_34[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_34[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_34[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_34[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_34[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 1904));
                            s0_23 = __fmaf_rn(pv_22[7], q_34[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[7], q_34[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[7], q_34[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[7], q_34[3], s3_26);
                            float q_35[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_35[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_35[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_35[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_35[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 2176));
                            s0_23 = __fmaf_rn(pv_22[8], q_35[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[8], q_35[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[8], q_35[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[8], q_35[3], s3_26);
                            float q_36[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_36[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_36[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_36[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 2448));
                            s0_23 = __fmaf_rn(pv_22[9], q_36[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[9], q_36[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[9], q_36[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[9], q_36[3], s3_26);
                            float q_37[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_37[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_37[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_37[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 2720));
                            s0_23 = __fmaf_rn(pv_22[10], q_37[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[10], q_37[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[10], q_37[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[10], q_37[3], s3_26);
                            float q_38[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_38[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_38[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_38[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_38[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 2992));
                            s0_23 = __fmaf_rn(pv_22[11], q_38[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[11], q_38[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[11], q_38[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[11], q_38[3], s3_26);
                            float q_39[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_39[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 3264));
                            s0_23 = __fmaf_rn(pv_22[12], q_39[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[12], q_39[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[12], q_39[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[12], q_39[3], s3_26);
                            float q_40[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_40[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_40[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_40[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_40[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 3536));
                            s0_23 = __fmaf_rn(pv_22[13], q_40[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[13], q_40[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[13], q_40[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[13], q_40[3], s3_26);
                            float q_41[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_41[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_41[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_41[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_41[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 3808));
                            s0_23 = __fmaf_rn(pv_22[14], q_41[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[14], q_41[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[14], q_41[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[14], q_41[3], s3_26);
                            float q_42[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_42[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_42[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_42[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_42[(0) + 3]))
                                : "r"(a_base + 4352 + pc4 * 4 + 4080));
                            s0_23 = __fmaf_rn(pv_22[15], q_42[0], s0_23);
                            s1_24 = __fmaf_rn(pv_22[15], q_42[1], s1_24);
                            s2_25 = __fmaf_rn(pv_22[15], q_42[2], s2_25);
                            s3_26 = __fmaf_rn(pv_22[15], q_42[3], s3_26);
                            xs_0_1[0] = xs_0_1[0] + s0_23;
                            xs_0_1[1] = xs_0_1[1] + s1_24;
                            xs_0_1[2] = xs_0_1[2] + s2_25;
                            xs_0_1[3] = xs_0_1[3] + s3_26;
                            float pv_43[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_43[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv_43[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_43[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_43[(0) + 3]))
                                : "r"(l_base + 13184 + pr * 272));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_43[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv_43[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_43[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_43[(4) + 3]))
                                : "r"(l_base + 13184 + pr * 272 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_43[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv_43[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_43[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_43[(8) + 3]))
                                : "r"(l_base + 13184 + pr * 272 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_43[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv_43[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_43[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_43[(12) + 3]))
                                : "r"(l_base + 13184 + pr * 272 + 48));
                            float s0_44 = 0.0f;
                            float s1_45 = 0.0f;
                            float s2_46 = 0.0f;
                            float s3_47 = 0.0f;
                            float q_48[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_48[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_48[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_48[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_48[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4));
                            s0_44 = __fmaf_rn(pv_43[0], q_48[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[0], q_48[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[0], q_48[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[0], q_48[3], s3_47);
                            float q_49[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_49[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_49[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_49[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_49[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 272));
                            s0_44 = __fmaf_rn(pv_43[1], q_49[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[1], q_49[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[1], q_49[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[1], q_49[3], s3_47);
                            float q_50[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_50[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_50[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_50[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_50[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 544));
                            s0_44 = __fmaf_rn(pv_43[2], q_50[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[2], q_50[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[2], q_50[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[2], q_50[3], s3_47);
                            float q_51[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_51[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_51[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_51[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_51[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 816));
                            s0_44 = __fmaf_rn(pv_43[3], q_51[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[3], q_51[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[3], q_51[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[3], q_51[3], s3_47);
                            float q_52[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_52[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_52[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_52[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_52[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 1088));
                            s0_44 = __fmaf_rn(pv_43[4], q_52[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[4], q_52[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[4], q_52[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[4], q_52[3], s3_47);
                            float q_53[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_53[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_53[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_53[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_53[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 1360));
                            s0_44 = __fmaf_rn(pv_43[5], q_53[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[5], q_53[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[5], q_53[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[5], q_53[3], s3_47);
                            float q_54[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_54[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_54[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_54[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_54[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 1632));
                            s0_44 = __fmaf_rn(pv_43[6], q_54[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[6], q_54[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[6], q_54[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[6], q_54[3], s3_47);
                            float q_55[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_55[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_55[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_55[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_55[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 1904));
                            s0_44 = __fmaf_rn(pv_43[7], q_55[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[7], q_55[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[7], q_55[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[7], q_55[3], s3_47);
                            float q_56[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_56[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_56[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_56[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_56[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 2176));
                            s0_44 = __fmaf_rn(pv_43[8], q_56[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[8], q_56[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[8], q_56[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[8], q_56[3], s3_47);
                            float q_57[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_57[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_57[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_57[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_57[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 2448));
                            s0_44 = __fmaf_rn(pv_43[9], q_57[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[9], q_57[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[9], q_57[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[9], q_57[3], s3_47);
                            float q_58[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_58[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_58[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_58[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_58[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 2720));
                            s0_44 = __fmaf_rn(pv_43[10], q_58[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[10], q_58[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[10], q_58[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[10], q_58[3], s3_47);
                            float q_59[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_59[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_59[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_59[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_59[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 2992));
                            s0_44 = __fmaf_rn(pv_43[11], q_59[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[11], q_59[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[11], q_59[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[11], q_59[3], s3_47);
                            float q_60[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_60[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_60[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_60[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_60[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 3264));
                            s0_44 = __fmaf_rn(pv_43[12], q_60[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[12], q_60[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[12], q_60[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[12], q_60[3], s3_47);
                            float q_61[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_61[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_61[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_61[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_61[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 3536));
                            s0_44 = __fmaf_rn(pv_43[13], q_61[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[13], q_61[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[13], q_61[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[13], q_61[3], s3_47);
                            float q_62[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_62[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_62[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_62[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_62[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 3808));
                            s0_44 = __fmaf_rn(pv_43[14], q_62[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[14], q_62[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[14], q_62[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[14], q_62[3], s3_47);
                            float q_63[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_63[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_63[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_63[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_63[(0) + 3]))
                                : "r"(a_base + 8704 + pc4 * 4 + 4080));
                            s0_44 = __fmaf_rn(pv_43[15], q_63[0], s0_44);
                            s1_45 = __fmaf_rn(pv_43[15], q_63[1], s1_45);
                            s2_46 = __fmaf_rn(pv_43[15], q_63[2], s2_46);
                            s3_47 = __fmaf_rn(pv_43[15], q_63[3], s3_47);
                            xs_0_1[0] = xs_0_1[0] + s0_44;
                            xs_0_1[1] = xs_0_1[1] + s1_45;
                            xs_0_1[2] = xs_0_1[2] + s2_46;
                            xs_0_1[3] = xs_0_1[3] + s3_47;
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(x_base + 2048 + pr * 64 + xst), "f"(xs_0_1[0]), "f"(xs_0_1[1]), "f"(xs_0_1[2]), "f"(xs_0_1[3]) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (pair == 0) {
                            float pv_0_1[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[(0) + 3]))
                                : "r"(a_base + 13248 + pr * 272));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[(4) + 3]))
                                : "r"(a_base + 13248 + pr * 272 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[(8) + 3]))
                                : "r"(a_base + 13248 + pr * 272 + 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pv_0_1[(12) + 3]))
                                : "r"(a_base + 13248 + pr * 272 + 48));
                            float s0_1_1 = 0.0f;
                            float s1_2_1 = 0.0f;
                            float s2_3_1 = 0.0f;
                            float s3_4_1 = 0.0f;
                            float q_5_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_5_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_5_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_5_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_5_1[(0) + 3]))
                                : "r"(x_base + 2048 + xo0));
                            s0_1_1 = __fmaf_rn(pv_0_1[0], q_5_1[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[0], q_5_1[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[0], q_5_1[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[0], q_5_1[3], s3_4_1);
                            float q_6_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_6_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_6_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_6_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_6_3[(0) + 3]))
                                : "r"(x_base + 2048 + 64 + xo0));
                            s0_1_1 = __fmaf_rn(pv_0_1[1], q_6_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[1], q_6_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[1], q_6_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[1], q_6_3[3], s3_4_1);
                            float q_7_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_7_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_7_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_7_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_7_3[(0) + 3]))
                                : "r"(x_base + 2048 + 128 + xo1));
                            s0_1_1 = __fmaf_rn(pv_0_1[2], q_7_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[2], q_7_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[2], q_7_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[2], q_7_3[3], s3_4_1);
                            float q_8_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_8_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_8_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_8_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_8_3[(0) + 3]))
                                : "r"(x_base + 2048 + 192 + xo1));
                            s0_1_1 = __fmaf_rn(pv_0_1[3], q_8_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[3], q_8_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[3], q_8_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[3], q_8_3[3], s3_4_1);
                            float q_9_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_9_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_9_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_9_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_9_3[(0) + 3]))
                                : "r"(x_base + 2048 + 256 + xo2));
                            s0_1_1 = __fmaf_rn(pv_0_1[4], q_9_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[4], q_9_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[4], q_9_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[4], q_9_3[3], s3_4_1);
                            float q_10_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_10_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_10_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_10_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_10_3[(0) + 3]))
                                : "r"(x_base + 2048 + 320 + xo2));
                            s0_1_1 = __fmaf_rn(pv_0_1[5], q_10_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[5], q_10_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[5], q_10_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[5], q_10_3[3], s3_4_1);
                            float q_11_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_11_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_11_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_11_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_11_3[(0) + 3]))
                                : "r"(x_base + 2048 + 384 + xo3));
                            s0_1_1 = __fmaf_rn(pv_0_1[6], q_11_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[6], q_11_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[6], q_11_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[6], q_11_3[3], s3_4_1);
                            float q_12_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_12_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_12_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_12_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_12_3[(0) + 3]))
                                : "r"(x_base + 2048 + 448 + xo3));
                            s0_1_1 = __fmaf_rn(pv_0_1[7], q_12_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[7], q_12_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[7], q_12_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[7], q_12_3[3], s3_4_1);
                            float q_13_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_13_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_13_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_13_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_13_3[(0) + 3]))
                                : "r"(x_base + 2048 + 512 + xo0));
                            s0_1_1 = __fmaf_rn(pv_0_1[8], q_13_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[8], q_13_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[8], q_13_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[8], q_13_3[3], s3_4_1);
                            float q_14_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_14_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_14_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_14_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_14_3[(0) + 3]))
                                : "r"(x_base + 2048 + 576 + xo0));
                            s0_1_1 = __fmaf_rn(pv_0_1[9], q_14_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[9], q_14_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[9], q_14_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[9], q_14_3[3], s3_4_1);
                            float q_15_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_15_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_15_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_15_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_15_3[(0) + 3]))
                                : "r"(x_base + 2048 + 640 + xo1));
                            s0_1_1 = __fmaf_rn(pv_0_1[10], q_15_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[10], q_15_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[10], q_15_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[10], q_15_3[3], s3_4_1);
                            float q_16_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_16_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_16_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_16_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_16_3[(0) + 3]))
                                : "r"(x_base + 2048 + 704 + xo1));
                            s0_1_1 = __fmaf_rn(pv_0_1[11], q_16_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[11], q_16_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[11], q_16_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[11], q_16_3[3], s3_4_1);
                            float q_17_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_17_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_17_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_17_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_17_3[(0) + 3]))
                                : "r"(x_base + 2048 + 768 + xo2));
                            s0_1_1 = __fmaf_rn(pv_0_1[12], q_17_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[12], q_17_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[12], q_17_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[12], q_17_3[3], s3_4_1);
                            float q_18_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_18_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_18_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_18_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_18_3[(0) + 3]))
                                : "r"(x_base + 2048 + 832 + xo2));
                            s0_1_1 = __fmaf_rn(pv_0_1[13], q_18_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[13], q_18_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[13], q_18_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[13], q_18_3[3], s3_4_1);
                            float q_19_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_19_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_19_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_19_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_19_3[(0) + 3]))
                                : "r"(x_base + 2048 + 896 + xo3));
                            s0_1_1 = __fmaf_rn(pv_0_1[14], q_19_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[14], q_19_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[14], q_19_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[14], q_19_3[3], s3_4_1);
                            float q_20_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_20_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_20_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_20_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_20_3[(0) + 3]))
                                : "r"(x_base + 2048 + 960 + xo3));
                            s0_1_1 = __fmaf_rn(pv_0_1[15], q_20_3[0], s0_1_1);
                            s1_2_1 = __fmaf_rn(pv_0_1[15], q_20_3[1], s1_2_1);
                            s2_3_1 = __fmaf_rn(pv_0_1[15], q_20_3[2], s2_3_1);
                            s3_4_1 = __fmaf_rn(pv_0_1[15], q_20_3[3], s3_4_1);
                            float ns_21_1[4];
                            ns_21_1[0] = -s0_1_1;
                            ns_21_1[1] = -s1_2_1;
                            ns_21_1[2] = -s2_3_1;
                            ns_21_1[3] = -s3_4_1;
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(a_base + 13056 + (pr * 68 + pc4) * 4), "f"(ns_21_1[0]), "f"(ns_21_1[1]), "f"(ns_21_1[2]), "f"(ns_21_1[3]) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                    }
                    if (j > 0) {
                        mbarrier_wait(wu_full_addr, _phase_wu_full_0);
                        _phase_wu_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        long long wu_base = ((long long)(irow0 + t_row) * (long long)num_heads + (long long)(hv - 1)) * 128;
                        int col = half * 16;
                        float _tmem_load_0[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                            : "r"(taddr + (unsigned int)row_base));
                        float _tmem_load_1[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                            : "r"(taddr + (unsigned int)row_base + 128));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float wo[16];
                        float uo[16];
                        wo[0] = ((row_valid != 0) ? _tmem_load_0[0] : 0.0f);
                        uo[0] = ((row_valid != 0) ? _tmem_load_1[0] : 0.0f);
                        wo[1] = ((row_valid != 0) ? _tmem_load_0[1] : 0.0f);
                        uo[1] = ((row_valid != 0) ? _tmem_load_1[1] : 0.0f);
                        wo[2] = ((row_valid != 0) ? _tmem_load_0[2] : 0.0f);
                        uo[2] = ((row_valid != 0) ? _tmem_load_1[2] : 0.0f);
                        wo[3] = ((row_valid != 0) ? _tmem_load_0[3] : 0.0f);
                        uo[3] = ((row_valid != 0) ? _tmem_load_1[3] : 0.0f);
                        wo[4] = ((row_valid != 0) ? _tmem_load_0[4] : 0.0f);
                        uo[4] = ((row_valid != 0) ? _tmem_load_1[4] : 0.0f);
                        wo[5] = ((row_valid != 0) ? _tmem_load_0[5] : 0.0f);
                        uo[5] = ((row_valid != 0) ? _tmem_load_1[5] : 0.0f);
                        wo[6] = ((row_valid != 0) ? _tmem_load_0[6] : 0.0f);
                        uo[6] = ((row_valid != 0) ? _tmem_load_1[6] : 0.0f);
                        wo[7] = ((row_valid != 0) ? _tmem_load_0[7] : 0.0f);
                        uo[7] = ((row_valid != 0) ? _tmem_load_1[7] : 0.0f);
                        wo[8] = ((row_valid != 0) ? _tmem_load_0[8] : 0.0f);
                        uo[8] = ((row_valid != 0) ? _tmem_load_1[8] : 0.0f);
                        wo[9] = ((row_valid != 0) ? _tmem_load_0[9] : 0.0f);
                        uo[9] = ((row_valid != 0) ? _tmem_load_1[9] : 0.0f);
                        wo[10] = ((row_valid != 0) ? _tmem_load_0[10] : 0.0f);
                        uo[10] = ((row_valid != 0) ? _tmem_load_1[10] : 0.0f);
                        wo[11] = ((row_valid != 0) ? _tmem_load_0[11] : 0.0f);
                        uo[11] = ((row_valid != 0) ? _tmem_load_1[11] : 0.0f);
                        wo[12] = ((row_valid != 0) ? _tmem_load_0[12] : 0.0f);
                        uo[12] = ((row_valid != 0) ? _tmem_load_1[12] : 0.0f);
                        wo[13] = ((row_valid != 0) ? _tmem_load_0[13] : 0.0f);
                        uo[13] = ((row_valid != 0) ? _tmem_load_1[13] : 0.0f);
                        wo[14] = ((row_valid != 0) ? _tmem_load_0[14] : 0.0f);
                        uo[14] = ((row_valid != 0) ? _tmem_load_1[14] : 0.0f);
                        wo[15] = ((row_valid != 0) ? _tmem_load_0[15] : 0.0f);
                        uo[15] = ((row_valid != 0) ? _tmem_load_1[15] : 0.0f);
                        unsigned int _packed_2[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo[_lp*2 + 0], wo[_lp*2+1 + 0]));
                            _packed_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo[_lp*2 + 2], wo[_lp*2+1 + 2]));
                            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo[_lp*2 + 4], wo[_lp*2+1 + 4]));
                            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo[_lp*2 + 6], wo[_lp*2+1 + 6]));
                            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo[_lp*2 + 8], wo[_lp*2+1 + 8]));
                            _packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo[_lp*2 + 10], wo[_lp*2+1 + 10]));
                            _packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo[_lp*2 + 12], wo[_lp*2+1 + 12]));
                            _packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo[_lp*2 + 14], wo[_lp*2+1 + 14]));
                            _packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_17 = reinterpret_cast<const unsigned*>(_packed_2);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (wu_base + (long long)col))), "r"(_raw_stv8_17[0]), "r"(_raw_stv8_17[1]), "r"(_raw_stv8_17[2]), "r"(_raw_stv8_17[3]), "r"(_raw_stv8_17[4]), "r"(_raw_stv8_17[5]), "r"(_raw_stv8_17[6]), "r"(_raw_stv8_17[7]) : "memory");
                        }
                        unsigned int _packed_3[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo[_lp*2 + 0], uo[_lp*2+1 + 0]));
                            _packed_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo[_lp*2 + 2], uo[_lp*2+1 + 2]));
                            _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo[_lp*2 + 4], uo[_lp*2+1 + 4]));
                            _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo[_lp*2 + 6], uo[_lp*2+1 + 6]));
                            _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo[_lp*2 + 8], uo[_lp*2+1 + 8]));
                            _packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo[_lp*2 + 10], uo[_lp*2+1 + 10]));
                            _packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo[_lp*2 + 12], uo[_lp*2+1 + 12]));
                            _packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo[_lp*2 + 14], uo[_lp*2+1 + 14]));
                            _packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_18 = reinterpret_cast<const unsigned*>(_packed_3);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (wu_base + (long long)col))), "r"(_raw_stv8_18[0]), "r"(_raw_stv8_18[1]), "r"(_raw_stv8_18[2]), "r"(_raw_stv8_18[3]), "r"(_raw_stv8_18[4]), "r"(_raw_stv8_18[5]), "r"(_raw_stv8_18[6]), "r"(_raw_stv8_18[7]) : "memory");
                        }
                        int col_0 = 32 + half * 16;
                        float _tmem_load_2[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                            : "r"(taddr + (unsigned int)row_base + 32));
                        float _tmem_load_3[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                            : "r"(taddr + (unsigned int)row_base + 160));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float wo_1[16];
                        float uo_2[16];
                        wo_1[0] = ((row_valid != 0) ? _tmem_load_2[0] : 0.0f);
                        uo_2[0] = ((row_valid != 0) ? _tmem_load_3[0] : 0.0f);
                        wo_1[1] = ((row_valid != 0) ? _tmem_load_2[1] : 0.0f);
                        uo_2[1] = ((row_valid != 0) ? _tmem_load_3[1] : 0.0f);
                        wo_1[2] = ((row_valid != 0) ? _tmem_load_2[2] : 0.0f);
                        uo_2[2] = ((row_valid != 0) ? _tmem_load_3[2] : 0.0f);
                        wo_1[3] = ((row_valid != 0) ? _tmem_load_2[3] : 0.0f);
                        uo_2[3] = ((row_valid != 0) ? _tmem_load_3[3] : 0.0f);
                        wo_1[4] = ((row_valid != 0) ? _tmem_load_2[4] : 0.0f);
                        uo_2[4] = ((row_valid != 0) ? _tmem_load_3[4] : 0.0f);
                        wo_1[5] = ((row_valid != 0) ? _tmem_load_2[5] : 0.0f);
                        uo_2[5] = ((row_valid != 0) ? _tmem_load_3[5] : 0.0f);
                        wo_1[6] = ((row_valid != 0) ? _tmem_load_2[6] : 0.0f);
                        uo_2[6] = ((row_valid != 0) ? _tmem_load_3[6] : 0.0f);
                        wo_1[7] = ((row_valid != 0) ? _tmem_load_2[7] : 0.0f);
                        uo_2[7] = ((row_valid != 0) ? _tmem_load_3[7] : 0.0f);
                        wo_1[8] = ((row_valid != 0) ? _tmem_load_2[8] : 0.0f);
                        uo_2[8] = ((row_valid != 0) ? _tmem_load_3[8] : 0.0f);
                        wo_1[9] = ((row_valid != 0) ? _tmem_load_2[9] : 0.0f);
                        uo_2[9] = ((row_valid != 0) ? _tmem_load_3[9] : 0.0f);
                        wo_1[10] = ((row_valid != 0) ? _tmem_load_2[10] : 0.0f);
                        uo_2[10] = ((row_valid != 0) ? _tmem_load_3[10] : 0.0f);
                        wo_1[11] = ((row_valid != 0) ? _tmem_load_2[11] : 0.0f);
                        uo_2[11] = ((row_valid != 0) ? _tmem_load_3[11] : 0.0f);
                        wo_1[12] = ((row_valid != 0) ? _tmem_load_2[12] : 0.0f);
                        uo_2[12] = ((row_valid != 0) ? _tmem_load_3[12] : 0.0f);
                        wo_1[13] = ((row_valid != 0) ? _tmem_load_2[13] : 0.0f);
                        uo_2[13] = ((row_valid != 0) ? _tmem_load_3[13] : 0.0f);
                        wo_1[14] = ((row_valid != 0) ? _tmem_load_2[14] : 0.0f);
                        uo_2[14] = ((row_valid != 0) ? _tmem_load_3[14] : 0.0f);
                        wo_1[15] = ((row_valid != 0) ? _tmem_load_2[15] : 0.0f);
                        uo_2[15] = ((row_valid != 0) ? _tmem_load_3[15] : 0.0f);
                        unsigned int _packed_4[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_1[_lp*2 + 0], wo_1[_lp*2+1 + 0]));
                            _packed_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_1[_lp*2 + 2], wo_1[_lp*2+1 + 2]));
                            _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_1[_lp*2 + 4], wo_1[_lp*2+1 + 4]));
                            _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_1[_lp*2 + 6], wo_1[_lp*2+1 + 6]));
                            _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_1[_lp*2 + 8], wo_1[_lp*2+1 + 8]));
                            _packed_4[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_1[_lp*2 + 10], wo_1[_lp*2+1 + 10]));
                            _packed_4[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_1[_lp*2 + 12], wo_1[_lp*2+1 + 12]));
                            _packed_4[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_1[_lp*2 + 14], wo_1[_lp*2+1 + 14]));
                            _packed_4[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_19 = reinterpret_cast<const unsigned*>(_packed_4);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (wu_base + (long long)col_0))), "r"(_raw_stv8_19[0]), "r"(_raw_stv8_19[1]), "r"(_raw_stv8_19[2]), "r"(_raw_stv8_19[3]), "r"(_raw_stv8_19[4]), "r"(_raw_stv8_19[5]), "r"(_raw_stv8_19[6]), "r"(_raw_stv8_19[7]) : "memory");
                        }
                        unsigned int _packed_5[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_2[_lp*2 + 0], uo_2[_lp*2+1 + 0]));
                            _packed_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_2[_lp*2 + 2], uo_2[_lp*2+1 + 2]));
                            _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_2[_lp*2 + 4], uo_2[_lp*2+1 + 4]));
                            _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_2[_lp*2 + 6], uo_2[_lp*2+1 + 6]));
                            _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_2[_lp*2 + 8], uo_2[_lp*2+1 + 8]));
                            _packed_5[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_2[_lp*2 + 10], uo_2[_lp*2+1 + 10]));
                            _packed_5[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_2[_lp*2 + 12], uo_2[_lp*2+1 + 12]));
                            _packed_5[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_2[_lp*2 + 14], uo_2[_lp*2+1 + 14]));
                            _packed_5[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_20 = reinterpret_cast<const unsigned*>(_packed_5);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (wu_base + (long long)col_0))), "r"(_raw_stv8_20[0]), "r"(_raw_stv8_20[1]), "r"(_raw_stv8_20[2]), "r"(_raw_stv8_20[3]), "r"(_raw_stv8_20[4]), "r"(_raw_stv8_20[5]), "r"(_raw_stv8_20[6]), "r"(_raw_stv8_20[7]) : "memory");
                        }
                        int col_3 = 64 + half * 16;
                        float _tmem_load_4[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                            : "r"(taddr + (unsigned int)row_base + 64));
                        float _tmem_load_5[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                            : "r"(taddr + (unsigned int)row_base + 192));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float wo_4[16];
                        float uo_5[16];
                        wo_4[0] = ((row_valid != 0) ? _tmem_load_4[0] : 0.0f);
                        uo_5[0] = ((row_valid != 0) ? _tmem_load_5[0] : 0.0f);
                        wo_4[1] = ((row_valid != 0) ? _tmem_load_4[1] : 0.0f);
                        uo_5[1] = ((row_valid != 0) ? _tmem_load_5[1] : 0.0f);
                        wo_4[2] = ((row_valid != 0) ? _tmem_load_4[2] : 0.0f);
                        uo_5[2] = ((row_valid != 0) ? _tmem_load_5[2] : 0.0f);
                        wo_4[3] = ((row_valid != 0) ? _tmem_load_4[3] : 0.0f);
                        uo_5[3] = ((row_valid != 0) ? _tmem_load_5[3] : 0.0f);
                        wo_4[4] = ((row_valid != 0) ? _tmem_load_4[4] : 0.0f);
                        uo_5[4] = ((row_valid != 0) ? _tmem_load_5[4] : 0.0f);
                        wo_4[5] = ((row_valid != 0) ? _tmem_load_4[5] : 0.0f);
                        uo_5[5] = ((row_valid != 0) ? _tmem_load_5[5] : 0.0f);
                        wo_4[6] = ((row_valid != 0) ? _tmem_load_4[6] : 0.0f);
                        uo_5[6] = ((row_valid != 0) ? _tmem_load_5[6] : 0.0f);
                        wo_4[7] = ((row_valid != 0) ? _tmem_load_4[7] : 0.0f);
                        uo_5[7] = ((row_valid != 0) ? _tmem_load_5[7] : 0.0f);
                        wo_4[8] = ((row_valid != 0) ? _tmem_load_4[8] : 0.0f);
                        uo_5[8] = ((row_valid != 0) ? _tmem_load_5[8] : 0.0f);
                        wo_4[9] = ((row_valid != 0) ? _tmem_load_4[9] : 0.0f);
                        uo_5[9] = ((row_valid != 0) ? _tmem_load_5[9] : 0.0f);
                        wo_4[10] = ((row_valid != 0) ? _tmem_load_4[10] : 0.0f);
                        uo_5[10] = ((row_valid != 0) ? _tmem_load_5[10] : 0.0f);
                        wo_4[11] = ((row_valid != 0) ? _tmem_load_4[11] : 0.0f);
                        uo_5[11] = ((row_valid != 0) ? _tmem_load_5[11] : 0.0f);
                        wo_4[12] = ((row_valid != 0) ? _tmem_load_4[12] : 0.0f);
                        uo_5[12] = ((row_valid != 0) ? _tmem_load_5[12] : 0.0f);
                        wo_4[13] = ((row_valid != 0) ? _tmem_load_4[13] : 0.0f);
                        uo_5[13] = ((row_valid != 0) ? _tmem_load_5[13] : 0.0f);
                        wo_4[14] = ((row_valid != 0) ? _tmem_load_4[14] : 0.0f);
                        uo_5[14] = ((row_valid != 0) ? _tmem_load_5[14] : 0.0f);
                        wo_4[15] = ((row_valid != 0) ? _tmem_load_4[15] : 0.0f);
                        uo_5[15] = ((row_valid != 0) ? _tmem_load_5[15] : 0.0f);
                        unsigned int _packed_6[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_4[_lp*2 + 0], wo_4[_lp*2+1 + 0]));
                            _packed_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_4[_lp*2 + 2], wo_4[_lp*2+1 + 2]));
                            _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_4[_lp*2 + 4], wo_4[_lp*2+1 + 4]));
                            _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_4[_lp*2 + 6], wo_4[_lp*2+1 + 6]));
                            _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_4[_lp*2 + 8], wo_4[_lp*2+1 + 8]));
                            _packed_6[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_4[_lp*2 + 10], wo_4[_lp*2+1 + 10]));
                            _packed_6[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_4[_lp*2 + 12], wo_4[_lp*2+1 + 12]));
                            _packed_6[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_4[_lp*2 + 14], wo_4[_lp*2+1 + 14]));
                            _packed_6[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_21 = reinterpret_cast<const unsigned*>(_packed_6);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (wu_base + (long long)col_3))), "r"(_raw_stv8_21[0]), "r"(_raw_stv8_21[1]), "r"(_raw_stv8_21[2]), "r"(_raw_stv8_21[3]), "r"(_raw_stv8_21[4]), "r"(_raw_stv8_21[5]), "r"(_raw_stv8_21[6]), "r"(_raw_stv8_21[7]) : "memory");
                        }
                        unsigned int _packed_7[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_5[_lp*2 + 0], uo_5[_lp*2+1 + 0]));
                            _packed_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_5[_lp*2 + 2], uo_5[_lp*2+1 + 2]));
                            _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_5[_lp*2 + 4], uo_5[_lp*2+1 + 4]));
                            _packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_5[_lp*2 + 6], uo_5[_lp*2+1 + 6]));
                            _packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_5[_lp*2 + 8], uo_5[_lp*2+1 + 8]));
                            _packed_7[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_5[_lp*2 + 10], uo_5[_lp*2+1 + 10]));
                            _packed_7[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_5[_lp*2 + 12], uo_5[_lp*2+1 + 12]));
                            _packed_7[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_5[_lp*2 + 14], uo_5[_lp*2+1 + 14]));
                            _packed_7[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_22 = reinterpret_cast<const unsigned*>(_packed_7);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (wu_base + (long long)col_3))), "r"(_raw_stv8_22[0]), "r"(_raw_stv8_22[1]), "r"(_raw_stv8_22[2]), "r"(_raw_stv8_22[3]), "r"(_raw_stv8_22[4]), "r"(_raw_stv8_22[5]), "r"(_raw_stv8_22[6]), "r"(_raw_stv8_22[7]) : "memory");
                        }
                        int col_6 = 96 + half * 16;
                        float _tmem_load_6[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                            : "r"(taddr + (unsigned int)row_base + 96));
                        float _tmem_load_7[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                            : "r"(taddr + (unsigned int)row_base + 224));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float wo_7[16];
                        float uo_8[16];
                        wo_7[0] = ((row_valid != 0) ? _tmem_load_6[0] : 0.0f);
                        uo_8[0] = ((row_valid != 0) ? _tmem_load_7[0] : 0.0f);
                        wo_7[1] = ((row_valid != 0) ? _tmem_load_6[1] : 0.0f);
                        uo_8[1] = ((row_valid != 0) ? _tmem_load_7[1] : 0.0f);
                        wo_7[2] = ((row_valid != 0) ? _tmem_load_6[2] : 0.0f);
                        uo_8[2] = ((row_valid != 0) ? _tmem_load_7[2] : 0.0f);
                        wo_7[3] = ((row_valid != 0) ? _tmem_load_6[3] : 0.0f);
                        uo_8[3] = ((row_valid != 0) ? _tmem_load_7[3] : 0.0f);
                        wo_7[4] = ((row_valid != 0) ? _tmem_load_6[4] : 0.0f);
                        uo_8[4] = ((row_valid != 0) ? _tmem_load_7[4] : 0.0f);
                        wo_7[5] = ((row_valid != 0) ? _tmem_load_6[5] : 0.0f);
                        uo_8[5] = ((row_valid != 0) ? _tmem_load_7[5] : 0.0f);
                        wo_7[6] = ((row_valid != 0) ? _tmem_load_6[6] : 0.0f);
                        uo_8[6] = ((row_valid != 0) ? _tmem_load_7[6] : 0.0f);
                        wo_7[7] = ((row_valid != 0) ? _tmem_load_6[7] : 0.0f);
                        uo_8[7] = ((row_valid != 0) ? _tmem_load_7[7] : 0.0f);
                        wo_7[8] = ((row_valid != 0) ? _tmem_load_6[8] : 0.0f);
                        uo_8[8] = ((row_valid != 0) ? _tmem_load_7[8] : 0.0f);
                        wo_7[9] = ((row_valid != 0) ? _tmem_load_6[9] : 0.0f);
                        uo_8[9] = ((row_valid != 0) ? _tmem_load_7[9] : 0.0f);
                        wo_7[10] = ((row_valid != 0) ? _tmem_load_6[10] : 0.0f);
                        uo_8[10] = ((row_valid != 0) ? _tmem_load_7[10] : 0.0f);
                        wo_7[11] = ((row_valid != 0) ? _tmem_load_6[11] : 0.0f);
                        uo_8[11] = ((row_valid != 0) ? _tmem_load_7[11] : 0.0f);
                        wo_7[12] = ((row_valid != 0) ? _tmem_load_6[12] : 0.0f);
                        uo_8[12] = ((row_valid != 0) ? _tmem_load_7[12] : 0.0f);
                        wo_7[13] = ((row_valid != 0) ? _tmem_load_6[13] : 0.0f);
                        uo_8[13] = ((row_valid != 0) ? _tmem_load_7[13] : 0.0f);
                        wo_7[14] = ((row_valid != 0) ? _tmem_load_6[14] : 0.0f);
                        uo_8[14] = ((row_valid != 0) ? _tmem_load_7[14] : 0.0f);
                        wo_7[15] = ((row_valid != 0) ? _tmem_load_6[15] : 0.0f);
                        uo_8[15] = ((row_valid != 0) ? _tmem_load_7[15] : 0.0f);
                        unsigned int _packed_8[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_7[_lp*2 + 0], wo_7[_lp*2+1 + 0]));
                            _packed_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_7[_lp*2 + 2], wo_7[_lp*2+1 + 2]));
                            _packed_8[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_7[_lp*2 + 4], wo_7[_lp*2+1 + 4]));
                            _packed_8[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_7[_lp*2 + 6], wo_7[_lp*2+1 + 6]));
                            _packed_8[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_7[_lp*2 + 8], wo_7[_lp*2+1 + 8]));
                            _packed_8[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_7[_lp*2 + 10], wo_7[_lp*2+1 + 10]));
                            _packed_8[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_7[_lp*2 + 12], wo_7[_lp*2+1 + 12]));
                            _packed_8[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_7[_lp*2 + 14], wo_7[_lp*2+1 + 14]));
                            _packed_8[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_23 = reinterpret_cast<const unsigned*>(_packed_8);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (wu_base + (long long)col_6))), "r"(_raw_stv8_23[0]), "r"(_raw_stv8_23[1]), "r"(_raw_stv8_23[2]), "r"(_raw_stv8_23[3]), "r"(_raw_stv8_23[4]), "r"(_raw_stv8_23[5]), "r"(_raw_stv8_23[6]), "r"(_raw_stv8_23[7]) : "memory");
                        }
                        unsigned int _packed_9[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_8[_lp*2 + 0], uo_8[_lp*2+1 + 0]));
                            _packed_9[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_8[_lp*2 + 2], uo_8[_lp*2+1 + 2]));
                            _packed_9[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_8[_lp*2 + 4], uo_8[_lp*2+1 + 4]));
                            _packed_9[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_8[_lp*2 + 6], uo_8[_lp*2+1 + 6]));
                            _packed_9[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_8[_lp*2 + 8], uo_8[_lp*2+1 + 8]));
                            _packed_9[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_8[_lp*2 + 10], uo_8[_lp*2+1 + 10]));
                            _packed_9[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_8[_lp*2 + 12], uo_8[_lp*2+1 + 12]));
                            _packed_9[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_8[_lp*2 + 14], uo_8[_lp*2+1 + 14]));
                            _packed_9[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_24 = reinterpret_cast<const unsigned*>(_packed_9);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (wu_base + (long long)col_6))), "r"(_raw_stv8_24[0]), "r"(_raw_stv8_24[1]), "r"(_raw_stv8_24[2]), "r"(_raw_stv8_24[3]), "r"(_raw_stv8_24[4]), "r"(_raw_stv8_24[5]), "r"(_raw_stv8_24[6]), "r"(_raw_stv8_24[7]) : "memory");
                        }
                    }
                    if (group > j) {
                        int qi_1 = tid_0;
                        int srow_1 = qi_1 >> 4;
                        int d0_1 = (qi_1 & 15) * 8;
                        float bs_r_1[1];
                        float es_r[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1[0])) : "r"(beta_base + srow_1 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r[0])) : "r"(eg_base + srow_1 * 4));
                        float fkb = bs_r_1[0] * es_r[0];
                        unsigned int w_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_1[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1 / 64 * 8192 + srow_1 * 128 + d0_1 % 64 * 2 ^ (d0_1 / 64 * 8192 + srow_1 * 128 + d0_1 % 64 * 2 >> 7 & 7) << 4))));
                        float _unpack_bf16_16[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_1[_pair]) << 16);
                            (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_1[_pair]) & 0xffff0000u);
                        }
                        float s_2[8];
                        s_2[0] = _unpack_bf16_16[0] * fkb;
                        s_2[1] = _unpack_bf16_16[1] * fkb;
                        s_2[2] = _unpack_bf16_16[2] * fkb;
                        s_2[3] = _unpack_bf16_16[3] * fkb;
                        s_2[4] = _unpack_bf16_16[4] * fkb;
                        s_2[5] = _unpack_bf16_16[5] * fkb;
                        s_2[6] = _unpack_bf16_16[6] * fkb;
                        s_2[7] = _unpack_bf16_16[7] * fkb;
                        unsigned int s_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_2[_lp*2 + 0], s_2[_lp*2+1 + 0]));
                            s_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1 / 64 * 8192 + srow_1 * 128 + d0_1 % 64 * 2 ^ (d0_1 / 64 * 8192 + srow_1 * 128 + d0_1 % 64 * 2 >> 7 & 7) << 4))), "r"(s_bf16[0]), "r"(s_bf16[1]), "r"(s_bf16[2]), "r"(s_bf16[3]) : "memory");
                        int qi_0 = tid_0 + 128;
                        int srow_1_1 = qi_0 >> 4;
                        int d0_2 = (qi_0 & 15) * 8;
                        float bs_r_3[1];
                        float es_r_4[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_3[0])) : "r"(beta_base + srow_1_1 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_4[0])) : "r"(eg_base + srow_1_1 * 4));
                        float fkb_5 = bs_r_3[0] * es_r_4[0];
                        unsigned int w_6[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_6[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_2 / 64 * 8192 + srow_1_1 * 128 + d0_2 % 64 * 2 ^ (d0_2 / 64 * 8192 + srow_1_1 * 128 + d0_2 % 64 * 2 >> 7 & 7) << 4))));
                        float _unpack_bf16_17[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_6[_pair]) << 16);
                            (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_6[_pair]) & 0xffff0000u);
                        }
                        float s_7[8];
                        s_7[0] = _unpack_bf16_17[0] * fkb_5;
                        s_7[1] = _unpack_bf16_17[1] * fkb_5;
                        s_7[2] = _unpack_bf16_17[2] * fkb_5;
                        s_7[3] = _unpack_bf16_17[3] * fkb_5;
                        s_7[4] = _unpack_bf16_17[4] * fkb_5;
                        s_7[5] = _unpack_bf16_17[5] * fkb_5;
                        s_7[6] = _unpack_bf16_17[6] * fkb_5;
                        s_7[7] = _unpack_bf16_17[7] * fkb_5;
                        unsigned int s_7_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_7[_lp*2 + 0], s_7[_lp*2+1 + 0]));
                            s_7_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_2 / 64 * 8192 + srow_1_1 * 128 + d0_2 % 64 * 2 ^ (d0_2 / 64 * 8192 + srow_1_1 * 128 + d0_2 % 64 * 2 >> 7 & 7) << 4))), "r"(s_7_bf16[0]), "r"(s_7_bf16[1]), "r"(s_7_bf16[2]), "r"(s_7_bf16[3]) : "memory");
                        int qi_8 = tid_0 + 256;
                        int srow_9 = qi_8 >> 4;
                        int d0_10 = (qi_8 & 15) * 8;
                        float bs_r_11[1];
                        float es_r_12[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_11[0])) : "r"(beta_base + srow_9 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_12[0])) : "r"(eg_base + srow_9 * 4));
                        float fkb_13 = bs_r_11[0] * es_r_12[0];
                        unsigned int w_14[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_14[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_10 / 64 * 8192 + srow_9 * 128 + d0_10 % 64 * 2 ^ (d0_10 / 64 * 8192 + srow_9 * 128 + d0_10 % 64 * 2 >> 7 & 7) << 4))));
                        float _unpack_bf16_18[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_14[_pair]) << 16);
                            (&_unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_14[_pair]) & 0xffff0000u);
                        }
                        float s_15[8];
                        s_15[0] = _unpack_bf16_18[0] * fkb_13;
                        s_15[1] = _unpack_bf16_18[1] * fkb_13;
                        s_15[2] = _unpack_bf16_18[2] * fkb_13;
                        s_15[3] = _unpack_bf16_18[3] * fkb_13;
                        s_15[4] = _unpack_bf16_18[4] * fkb_13;
                        s_15[5] = _unpack_bf16_18[5] * fkb_13;
                        s_15[6] = _unpack_bf16_18[6] * fkb_13;
                        s_15[7] = _unpack_bf16_18[7] * fkb_13;
                        unsigned int s_15_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_15[_lp*2 + 0], s_15[_lp*2+1 + 0]));
                            s_15_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_10 / 64 * 8192 + srow_9 * 128 + d0_10 % 64 * 2 ^ (d0_10 / 64 * 8192 + srow_9 * 128 + d0_10 % 64 * 2 >> 7 & 7) << 4))), "r"(s_15_bf16[0]), "r"(s_15_bf16[1]), "r"(s_15_bf16[2]), "r"(s_15_bf16[3]) : "memory");
                        int qi_16 = tid_0 + 384;
                        int srow_17 = qi_16 >> 4;
                        int d0_18 = (qi_16 & 15) * 8;
                        float bs_r_19[1];
                        float es_r_20[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_19[0])) : "r"(beta_base + srow_17 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_20[0])) : "r"(eg_base + srow_17 * 4));
                        float fkb_21 = bs_r_19[0] * es_r_20[0];
                        unsigned int w_22[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_22[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_18 / 64 * 8192 + srow_17 * 128 + d0_18 % 64 * 2 ^ (d0_18 / 64 * 8192 + srow_17 * 128 + d0_18 % 64 * 2 >> 7 & 7) << 4))));
                        float _unpack_bf16_19[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_unpack_bf16_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_22[_pair]) << 16);
                            (&_unpack_bf16_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_22[_pair]) & 0xffff0000u);
                        }
                        float s_23[8];
                        s_23[0] = _unpack_bf16_19[0] * fkb_21;
                        s_23[1] = _unpack_bf16_19[1] * fkb_21;
                        s_23[2] = _unpack_bf16_19[2] * fkb_21;
                        s_23[3] = _unpack_bf16_19[3] * fkb_21;
                        s_23[4] = _unpack_bf16_19[4] * fkb_21;
                        s_23[5] = _unpack_bf16_19[5] * fkb_21;
                        s_23[6] = _unpack_bf16_19[6] * fkb_21;
                        s_23[7] = _unpack_bf16_19[7] * fkb_21;
                        unsigned int s_23_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_23[_lp*2 + 0], s_23[_lp*2+1 + 0]));
                            s_23_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_18 / 64 * 8192 + srow_17 * 128 + d0_18 % 64 * 2 ^ (d0_18 / 64 * 8192 + srow_17 * 128 + d0_18 % 64 * 2 >> 7 & 7) << 4))), "r"(s_23_bf16[0]), "r"(s_23_bf16[1]), "r"(s_23_bf16[2]), "r"(s_23_bf16[3]) : "memory");
                        int qi_24 = tid_0 + 512;
                        int srow_25 = qi_24 >> 4;
                        int d0_26 = (qi_24 & 15) * 8;
                        float bs_r_27[1];
                        float es_r_28[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_27[0])) : "r"(beta_base + srow_25 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_28[0])) : "r"(eg_base + srow_25 * 4));
                        float fkb_29 = bs_r_27[0] * es_r_28[0];
                        unsigned int w_30[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_30[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_26 / 64 * 8192 + srow_25 * 128 + d0_26 % 64 * 2 ^ (d0_26 / 64 * 8192 + srow_25 * 128 + d0_26 % 64 * 2 >> 7 & 7) << 4))));
                        float _unpack_bf16_20[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_unpack_bf16_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_30[_pair]) << 16);
                            (&_unpack_bf16_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_30[_pair]) & 0xffff0000u);
                        }
                        float s_31[8];
                        s_31[0] = _unpack_bf16_20[0] * fkb_29;
                        s_31[1] = _unpack_bf16_20[1] * fkb_29;
                        s_31[2] = _unpack_bf16_20[2] * fkb_29;
                        s_31[3] = _unpack_bf16_20[3] * fkb_29;
                        s_31[4] = _unpack_bf16_20[4] * fkb_29;
                        s_31[5] = _unpack_bf16_20[5] * fkb_29;
                        s_31[6] = _unpack_bf16_20[6] * fkb_29;
                        s_31[7] = _unpack_bf16_20[7] * fkb_29;
                        unsigned int s_31_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_31[_lp*2 + 0], s_31[_lp*2+1 + 0]));
                            s_31_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_26 / 64 * 8192 + srow_25 * 128 + d0_26 % 64 * 2 ^ (d0_26 / 64 * 8192 + srow_25 * 128 + d0_26 % 64 * 2 >> 7 & 7) << 4))), "r"(s_31_bf16[0]), "r"(s_31_bf16[1]), "r"(s_31_bf16[2]), "r"(s_31_bf16[3]) : "memory");
                        int qi_32 = tid_0 + 640;
                        int srow_33 = qi_32 >> 4;
                        int d0_34 = (qi_32 & 15) * 8;
                        float bs_r_35[1];
                        float es_r_36[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_35[0])) : "r"(beta_base + srow_33 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_36[0])) : "r"(eg_base + srow_33 * 4));
                        float fkb_37 = bs_r_35[0] * es_r_36[0];
                        unsigned int w_38[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_38[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_38[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_38[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_38[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_34 / 64 * 8192 + srow_33 * 128 + d0_34 % 64 * 2 ^ (d0_34 / 64 * 8192 + srow_33 * 128 + d0_34 % 64 * 2 >> 7 & 7) << 4))));
                        float _unpack_bf16_21[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_unpack_bf16_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_38[_pair]) << 16);
                            (&_unpack_bf16_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_38[_pair]) & 0xffff0000u);
                        }
                        float s_39[8];
                        s_39[0] = _unpack_bf16_21[0] * fkb_37;
                        s_39[1] = _unpack_bf16_21[1] * fkb_37;
                        s_39[2] = _unpack_bf16_21[2] * fkb_37;
                        s_39[3] = _unpack_bf16_21[3] * fkb_37;
                        s_39[4] = _unpack_bf16_21[4] * fkb_37;
                        s_39[5] = _unpack_bf16_21[5] * fkb_37;
                        s_39[6] = _unpack_bf16_21[6] * fkb_37;
                        s_39[7] = _unpack_bf16_21[7] * fkb_37;
                        unsigned int s_39_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_39[_lp*2 + 0], s_39[_lp*2+1 + 0]));
                            s_39_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_34 / 64 * 8192 + srow_33 * 128 + d0_34 % 64 * 2 ^ (d0_34 / 64 * 8192 + srow_33 * 128 + d0_34 % 64 * 2 >> 7 & 7) << 4))), "r"(s_39_bf16[0]), "r"(s_39_bf16[1]), "r"(s_39_bf16[2]), "r"(s_39_bf16[3]) : "memory");
                        int qi_40 = tid_0 + 768;
                        int srow_41 = qi_40 >> 4;
                        int d0_42 = (qi_40 & 15) * 8;
                        float bs_r_43[1];
                        float es_r_44[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_43[0])) : "r"(beta_base + srow_41 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_44[0])) : "r"(eg_base + srow_41 * 4));
                        float fkb_45 = bs_r_43[0] * es_r_44[0];
                        unsigned int w_46[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_46[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_46[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_46[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_46[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_42 / 64 * 8192 + srow_41 * 128 + d0_42 % 64 * 2 ^ (d0_42 / 64 * 8192 + srow_41 * 128 + d0_42 % 64 * 2 >> 7 & 7) << 4))));
                        float _unpack_bf16_22[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_unpack_bf16_22[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_46[_pair]) << 16);
                            (&_unpack_bf16_22[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_46[_pair]) & 0xffff0000u);
                        }
                        float s_47[8];
                        s_47[0] = _unpack_bf16_22[0] * fkb_45;
                        s_47[1] = _unpack_bf16_22[1] * fkb_45;
                        s_47[2] = _unpack_bf16_22[2] * fkb_45;
                        s_47[3] = _unpack_bf16_22[3] * fkb_45;
                        s_47[4] = _unpack_bf16_22[4] * fkb_45;
                        s_47[5] = _unpack_bf16_22[5] * fkb_45;
                        s_47[6] = _unpack_bf16_22[6] * fkb_45;
                        s_47[7] = _unpack_bf16_22[7] * fkb_45;
                        unsigned int s_47_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_47[_lp*2 + 0], s_47[_lp*2+1 + 0]));
                            s_47_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_42 / 64 * 8192 + srow_41 * 128 + d0_42 % 64 * 2 ^ (d0_42 / 64 * 8192 + srow_41 * 128 + d0_42 % 64 * 2 >> 7 & 7) << 4))), "r"(s_47_bf16[0]), "r"(s_47_bf16[1]), "r"(s_47_bf16[2]), "r"(s_47_bf16[3]) : "memory");
                        int qi_48 = tid_0 + 896;
                        int srow_49 = qi_48 >> 4;
                        int d0_50 = (qi_48 & 15) * 8;
                        float bs_r_51[1];
                        float es_r_52[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_51[0])) : "r"(beta_base + srow_49 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_52[0])) : "r"(eg_base + srow_49 * 4));
                        float fkb_53 = bs_r_51[0] * es_r_52[0];
                        unsigned int w_54[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_54[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_54[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_54[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_54[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_50 / 64 * 8192 + srow_49 * 128 + d0_50 % 64 * 2 ^ (d0_50 / 64 * 8192 + srow_49 * 128 + d0_50 % 64 * 2 >> 7 & 7) << 4))));
                        float _unpack_bf16_23[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_unpack_bf16_23[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_54[_pair]) << 16);
                            (&_unpack_bf16_23[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_54[_pair]) & 0xffff0000u);
                        }
                        float s_55[8];
                        s_55[0] = _unpack_bf16_23[0] * fkb_53;
                        s_55[1] = _unpack_bf16_23[1] * fkb_53;
                        s_55[2] = _unpack_bf16_23[2] * fkb_53;
                        s_55[3] = _unpack_bf16_23[3] * fkb_53;
                        s_55[4] = _unpack_bf16_23[4] * fkb_53;
                        s_55[5] = _unpack_bf16_23[5] * fkb_53;
                        s_55[6] = _unpack_bf16_23[6] * fkb_53;
                        s_55[7] = _unpack_bf16_23[7] * fkb_53;
                        unsigned int s_55_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_55[_lp*2 + 0], s_55[_lp*2+1 + 0]));
                            s_55_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_50 / 64 * 8192 + srow_49 * 128 + d0_50 % 64 * 2 ^ (d0_50 / 64 * 8192 + srow_49 * 128 + d0_50 % 64 * 2 >> 7 & 7) << 4))), "r"(s_55_bf16[0]), "r"(s_55_bf16[1]), "r"(s_55_bf16[2]), "r"(s_55_bf16[3]) : "memory");
                        int arow = tid_0 >> 1;
                        int arow_valid = 0;
                        if (arow < clen) {
                            arow_valid = 1;
                        }
                        long long akk_base = ((long long)(irow0 + arow) * (long long)num_heads + (long long)hv) * 64;
                        int sa0 = (tid_0 & 1) * 4 * 8;
                        float av[8];
                        av[0] = 0.0f;
                        av[1] = 0.0f;
                        av[2] = 0.0f;
                        av[3] = 0.0f;
                        av[4] = 0.0f;
                        av[5] = 0.0f;
                        av[6] = 0.0f;
                        av[7] = 0.0f;
                        if (sa0 <= arow) {
                            if (arow_valid != 0) {
                                float ar[8];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar[0])), "=r"(*reinterpret_cast<uint32_t*>(&ar[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar[(0) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0) * 4));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar[4])), "=r"(*reinterpret_cast<uint32_t*>(&ar[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar[(4) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0 + 4) * 4));
                                av[0] = ((arow >= sa0) ? ar[0] : 0.0f);
                                av[1] = ((arow >= sa0 + 1) ? ar[1] : 0.0f);
                                av[2] = ((arow >= sa0 + 2) ? ar[2] : 0.0f);
                                av[3] = ((arow >= sa0 + 3) ? ar[3] : 0.0f);
                                av[4] = ((arow >= sa0 + 4) ? ar[4] : 0.0f);
                                av[5] = ((arow >= sa0 + 5) ? ar[5] : 0.0f);
                                av[6] = ((arow >= sa0 + 6) ? ar[6] : 0.0f);
                                av[7] = ((arow >= sa0 + 7) ? ar[7] : 0.0f);
                            }
                        }
                        unsigned int av_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av[_lp*2 + 0], av[_lp*2+1 + 0]));
                            av_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_t_addr + (unsigned int)(sa0 / 64 * 8192 + arow * 128 + sa0 % 64 * 2 ^ (sa0 / 64 * 8192 + arow * 128 + sa0 % 64 * 2 >> 7 & 7) << 4))), "r"(av_bf16[0]), "r"(av_bf16[1]), "r"(av_bf16[2]), "r"(av_bf16[3]) : "memory");
                        unsigned int _packed_10[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av[_lp*2 + 0], av[_lp*2+1 + 0]));
                            _packed_10[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av[_lp*2 + 2], av[_lp*2+1 + 2]));
                            _packed_10[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av[_lp*2 + 4], av[_lp*2+1 + 4]));
                            _packed_10[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av[_lp*2 + 6], av[_lp*2+1 + 6]));
                            _packed_10[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(akk_out + (akk_base + (long long)sa0))[0] = reinterpret_cast<int4*>(_packed_10)[0];
                        int sa0_56 = ((tid_0 & 1) * 4 + 1) * 8;
                        float av_57[8];
                        av_57[0] = 0.0f;
                        av_57[1] = 0.0f;
                        av_57[2] = 0.0f;
                        av_57[3] = 0.0f;
                        av_57[4] = 0.0f;
                        av_57[5] = 0.0f;
                        av_57[6] = 0.0f;
                        av_57[7] = 0.0f;
                        if (sa0_56 <= arow) {
                            if (arow_valid != 0) {
                                float ar_1[8];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1[(0) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_56) * 4));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_1[(4) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_56 + 4) * 4));
                                av_57[0] = ((arow >= sa0_56) ? ar_1[0] : 0.0f);
                                av_57[1] = ((arow >= sa0_56 + 1) ? ar_1[1] : 0.0f);
                                av_57[2] = ((arow >= sa0_56 + 2) ? ar_1[2] : 0.0f);
                                av_57[3] = ((arow >= sa0_56 + 3) ? ar_1[3] : 0.0f);
                                av_57[4] = ((arow >= sa0_56 + 4) ? ar_1[4] : 0.0f);
                                av_57[5] = ((arow >= sa0_56 + 5) ? ar_1[5] : 0.0f);
                                av_57[6] = ((arow >= sa0_56 + 6) ? ar_1[6] : 0.0f);
                                av_57[7] = ((arow >= sa0_56 + 7) ? ar_1[7] : 0.0f);
                            }
                        }
                        unsigned int av_57_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_57[_lp*2 + 0], av_57[_lp*2+1 + 0]));
                            av_57_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_t_addr + (unsigned int)(sa0_56 / 64 * 8192 + arow * 128 + sa0_56 % 64 * 2 ^ (sa0_56 / 64 * 8192 + arow * 128 + sa0_56 % 64 * 2 >> 7 & 7) << 4))), "r"(av_57_bf16[0]), "r"(av_57_bf16[1]), "r"(av_57_bf16[2]), "r"(av_57_bf16[3]) : "memory");
                        unsigned int _packed_11[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_57[_lp*2 + 0], av_57[_lp*2+1 + 0]));
                            _packed_11[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_57[_lp*2 + 2], av_57[_lp*2+1 + 2]));
                            _packed_11[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_57[_lp*2 + 4], av_57[_lp*2+1 + 4]));
                            _packed_11[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_57[_lp*2 + 6], av_57[_lp*2+1 + 6]));
                            _packed_11[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(akk_out + (akk_base + (long long)sa0_56))[0] = reinterpret_cast<int4*>(_packed_11)[0];
                        int sa0_58 = ((tid_0 & 1) * 4 + 2) * 8;
                        float av_59[8];
                        av_59[0] = 0.0f;
                        av_59[1] = 0.0f;
                        av_59[2] = 0.0f;
                        av_59[3] = 0.0f;
                        av_59[4] = 0.0f;
                        av_59[5] = 0.0f;
                        av_59[6] = 0.0f;
                        av_59[7] = 0.0f;
                        if (sa0_58 <= arow) {
                            if (arow_valid != 0) {
                                float ar_2[8];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ar_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_2[(0) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_58) * 4));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ar_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_2[(4) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_58 + 4) * 4));
                                av_59[0] = ((arow >= sa0_58) ? ar_2[0] : 0.0f);
                                av_59[1] = ((arow >= sa0_58 + 1) ? ar_2[1] : 0.0f);
                                av_59[2] = ((arow >= sa0_58 + 2) ? ar_2[2] : 0.0f);
                                av_59[3] = ((arow >= sa0_58 + 3) ? ar_2[3] : 0.0f);
                                av_59[4] = ((arow >= sa0_58 + 4) ? ar_2[4] : 0.0f);
                                av_59[5] = ((arow >= sa0_58 + 5) ? ar_2[5] : 0.0f);
                                av_59[6] = ((arow >= sa0_58 + 6) ? ar_2[6] : 0.0f);
                                av_59[7] = ((arow >= sa0_58 + 7) ? ar_2[7] : 0.0f);
                            }
                        }
                        unsigned int av_59_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_59[_lp*2 + 0], av_59[_lp*2+1 + 0]));
                            av_59_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_t_addr + (unsigned int)(sa0_58 / 64 * 8192 + arow * 128 + sa0_58 % 64 * 2 ^ (sa0_58 / 64 * 8192 + arow * 128 + sa0_58 % 64 * 2 >> 7 & 7) << 4))), "r"(av_59_bf16[0]), "r"(av_59_bf16[1]), "r"(av_59_bf16[2]), "r"(av_59_bf16[3]) : "memory");
                        unsigned int _packed_12[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_59[_lp*2 + 0], av_59[_lp*2+1 + 0]));
                            _packed_12[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_59[_lp*2 + 2], av_59[_lp*2+1 + 2]));
                            _packed_12[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_59[_lp*2 + 4], av_59[_lp*2+1 + 4]));
                            _packed_12[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_59[_lp*2 + 6], av_59[_lp*2+1 + 6]));
                            _packed_12[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(akk_out + (akk_base + (long long)sa0_58))[0] = reinterpret_cast<int4*>(_packed_12)[0];
                        int sa0_60 = ((tid_0 & 1) * 4 + 3) * 8;
                        float av_61[8];
                        av_61[0] = 0.0f;
                        av_61[1] = 0.0f;
                        av_61[2] = 0.0f;
                        av_61[3] = 0.0f;
                        av_61[4] = 0.0f;
                        av_61[5] = 0.0f;
                        av_61[6] = 0.0f;
                        av_61[7] = 0.0f;
                        if (sa0_60 <= arow) {
                            if (arow_valid != 0) {
                                float ar_3[8];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ar_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_3[(0) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_60) * 4));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ar_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ar_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ar_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ar_3[(4) + 3]))
                                    : "r"(a_base + (arow * 68 + sa0_60 + 4) * 4));
                                av_61[0] = ((arow >= sa0_60) ? ar_3[0] : 0.0f);
                                av_61[1] = ((arow >= sa0_60 + 1) ? ar_3[1] : 0.0f);
                                av_61[2] = ((arow >= sa0_60 + 2) ? ar_3[2] : 0.0f);
                                av_61[3] = ((arow >= sa0_60 + 3) ? ar_3[3] : 0.0f);
                                av_61[4] = ((arow >= sa0_60 + 4) ? ar_3[4] : 0.0f);
                                av_61[5] = ((arow >= sa0_60 + 5) ? ar_3[5] : 0.0f);
                                av_61[6] = ((arow >= sa0_60 + 6) ? ar_3[6] : 0.0f);
                                av_61[7] = ((arow >= sa0_60 + 7) ? ar_3[7] : 0.0f);
                            }
                        }
                        unsigned int av_61_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_61[_lp*2 + 0], av_61[_lp*2+1 + 0]));
                            av_61_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_t_addr + (unsigned int)(sa0_60 / 64 * 8192 + arow * 128 + sa0_60 % 64 * 2 ^ (sa0_60 / 64 * 8192 + arow * 128 + sa0_60 % 64 * 2 >> 7 & 7) << 4))), "r"(av_61_bf16[0]), "r"(av_61_bf16[1]), "r"(av_61_bf16[2]), "r"(av_61_bf16[3]) : "memory");
                        unsigned int _packed_13[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_61[_lp*2 + 0], av_61[_lp*2+1 + 0]));
                            _packed_13[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_61[_lp*2 + 2], av_61[_lp*2+1 + 2]));
                            _packed_13[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_61[_lp*2 + 4], av_61[_lp*2+1 + 4]));
                            _packed_13[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(av_61[_lp*2 + 6], av_61[_lp*2+1 + 6]));
                            _packed_13[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(akk_out + (akk_base + (long long)sa0_60))[0] = reinterpret_cast<int4*>(_packed_13)[0];
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(op_ready_addr);
                        vst_c += 1;
                        if (vst_c == 2) { vst_c = 0; _phase_v_full ^= 1; }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                }
            } else {
                #pragma unroll 1
                for (int j_1 = 0; j_1 < group; j_1++) {
                    int hv_p = hq * group + j_1;
                    if (tid_0 < 64) {
                        long long rbase = (long long)(irow0 + tid_0) * (long long)num_heads + (long long)hv_p;
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
                            const unsigned* _raw_stv8_25 = reinterpret_cast<const unsigned*>(_packed_14);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128))), "r"(_raw_stv8_25[0]), "r"(_raw_stv8_25[1]), "r"(_raw_stv8_25[2]), "r"(_raw_stv8_25[3]), "r"(_raw_stv8_25[4]), "r"(_raw_stv8_25[5]), "r"(_raw_stv8_25[6]), "r"(_raw_stv8_25[7]) : "memory");
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
                            const unsigned* _raw_stv8_26 = reinterpret_cast<const unsigned*>(_packed_15);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128))), "r"(_raw_stv8_26[0]), "r"(_raw_stv8_26[1]), "r"(_raw_stv8_26[2]), "r"(_raw_stv8_26[3]), "r"(_raw_stv8_26[4]), "r"(_raw_stv8_26[5]), "r"(_raw_stv8_26[6]), "r"(_raw_stv8_26[7]) : "memory");
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
                            const unsigned* _raw_stv8_27 = reinterpret_cast<const unsigned*>(_packed_16);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 16))), "r"(_raw_stv8_27[0]), "r"(_raw_stv8_27[1]), "r"(_raw_stv8_27[2]), "r"(_raw_stv8_27[3]), "r"(_raw_stv8_27[4]), "r"(_raw_stv8_27[5]), "r"(_raw_stv8_27[6]), "r"(_raw_stv8_27[7]) : "memory");
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
                            const unsigned* _raw_stv8_28 = reinterpret_cast<const unsigned*>(_packed_17);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 16))), "r"(_raw_stv8_28[0]), "r"(_raw_stv8_28[1]), "r"(_raw_stv8_28[2]), "r"(_raw_stv8_28[3]), "r"(_raw_stv8_28[4]), "r"(_raw_stv8_28[5]), "r"(_raw_stv8_28[6]), "r"(_raw_stv8_28[7]) : "memory");
                        }
                        unsigned int _packed_18[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_18[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_18[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_18[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_18[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_18[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_18[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_18[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_18[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_29 = reinterpret_cast<const unsigned*>(_packed_18);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 32))), "r"(_raw_stv8_29[0]), "r"(_raw_stv8_29[1]), "r"(_raw_stv8_29[2]), "r"(_raw_stv8_29[3]), "r"(_raw_stv8_29[4]), "r"(_raw_stv8_29[5]), "r"(_raw_stv8_29[6]), "r"(_raw_stv8_29[7]) : "memory");
                        }
                        unsigned int _packed_19[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_19[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_19[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_19[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_19[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_19[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_19[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_19[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_19[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_30 = reinterpret_cast<const unsigned*>(_packed_19);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 32))), "r"(_raw_stv8_30[0]), "r"(_raw_stv8_30[1]), "r"(_raw_stv8_30[2]), "r"(_raw_stv8_30[3]), "r"(_raw_stv8_30[4]), "r"(_raw_stv8_30[5]), "r"(_raw_stv8_30[6]), "r"(_raw_stv8_30[7]) : "memory");
                        }
                        unsigned int _packed_20[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_20[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_20[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_20[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_20[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_20[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_20[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_20[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_20[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_31 = reinterpret_cast<const unsigned*>(_packed_20);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 48))), "r"(_raw_stv8_31[0]), "r"(_raw_stv8_31[1]), "r"(_raw_stv8_31[2]), "r"(_raw_stv8_31[3]), "r"(_raw_stv8_31[4]), "r"(_raw_stv8_31[5]), "r"(_raw_stv8_31[6]), "r"(_raw_stv8_31[7]) : "memory");
                        }
                        unsigned int _packed_21[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_21[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_21[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_21[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_21[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_21[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_21[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_21[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_21[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_32 = reinterpret_cast<const unsigned*>(_packed_21);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 48))), "r"(_raw_stv8_32[0]), "r"(_raw_stv8_32[1]), "r"(_raw_stv8_32[2]), "r"(_raw_stv8_32[3]), "r"(_raw_stv8_32[4]), "r"(_raw_stv8_32[5]), "r"(_raw_stv8_32[6]), "r"(_raw_stv8_32[7]) : "memory");
                        }
                        unsigned int _packed_22[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_22[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_22[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_22[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_22[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_22[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_22[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_22[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_22[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_33 = reinterpret_cast<const unsigned*>(_packed_22);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 64))), "r"(_raw_stv8_33[0]), "r"(_raw_stv8_33[1]), "r"(_raw_stv8_33[2]), "r"(_raw_stv8_33[3]), "r"(_raw_stv8_33[4]), "r"(_raw_stv8_33[5]), "r"(_raw_stv8_33[6]), "r"(_raw_stv8_33[7]) : "memory");
                        }
                        unsigned int _packed_23[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_23[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_23[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_23[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_23[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_23[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_23[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_23[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_23[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_34 = reinterpret_cast<const unsigned*>(_packed_23);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 64))), "r"(_raw_stv8_34[0]), "r"(_raw_stv8_34[1]), "r"(_raw_stv8_34[2]), "r"(_raw_stv8_34[3]), "r"(_raw_stv8_34[4]), "r"(_raw_stv8_34[5]), "r"(_raw_stv8_34[6]), "r"(_raw_stv8_34[7]) : "memory");
                        }
                        unsigned int _packed_24[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_24[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_24[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_24[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_24[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_24[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_24[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_24[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_24[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_35 = reinterpret_cast<const unsigned*>(_packed_24);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 80))), "r"(_raw_stv8_35[0]), "r"(_raw_stv8_35[1]), "r"(_raw_stv8_35[2]), "r"(_raw_stv8_35[3]), "r"(_raw_stv8_35[4]), "r"(_raw_stv8_35[5]), "r"(_raw_stv8_35[6]), "r"(_raw_stv8_35[7]) : "memory");
                        }
                        unsigned int _packed_25[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_25[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_25[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_25[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_25[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_25[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_25[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_25[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_25[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_36 = reinterpret_cast<const unsigned*>(_packed_25);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 80))), "r"(_raw_stv8_36[0]), "r"(_raw_stv8_36[1]), "r"(_raw_stv8_36[2]), "r"(_raw_stv8_36[3]), "r"(_raw_stv8_36[4]), "r"(_raw_stv8_36[5]), "r"(_raw_stv8_36[6]), "r"(_raw_stv8_36[7]) : "memory");
                        }
                        unsigned int _packed_26[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_26[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_26[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_26[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_26[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_26[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_26[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_26[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_26[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_37 = reinterpret_cast<const unsigned*>(_packed_26);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 96))), "r"(_raw_stv8_37[0]), "r"(_raw_stv8_37[1]), "r"(_raw_stv8_37[2]), "r"(_raw_stv8_37[3]), "r"(_raw_stv8_37[4]), "r"(_raw_stv8_37[5]), "r"(_raw_stv8_37[6]), "r"(_raw_stv8_37[7]) : "memory");
                        }
                        unsigned int _packed_27[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_27[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_27[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_27[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_27[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_27[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_27[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_27[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_27[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_38 = reinterpret_cast<const unsigned*>(_packed_27);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 96))), "r"(_raw_stv8_38[0]), "r"(_raw_stv8_38[1]), "r"(_raw_stv8_38[2]), "r"(_raw_stv8_38[3]), "r"(_raw_stv8_38[4]), "r"(_raw_stv8_38[5]), "r"(_raw_stv8_38[6]), "r"(_raw_stv8_38[7]) : "memory");
                        }
                        unsigned int _packed_28[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_28[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_28[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_28[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_28[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_28[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_28[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_28[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_28[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_39 = reinterpret_cast<const unsigned*>(_packed_28);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128 + 112))), "r"(_raw_stv8_39[0]), "r"(_raw_stv8_39[1]), "r"(_raw_stv8_39[2]), "r"(_raw_stv8_39[3]), "r"(_raw_stv8_39[4]), "r"(_raw_stv8_39[5]), "r"(_raw_stv8_39[6]), "r"(_raw_stv8_39[7]) : "memory");
                        }
                        unsigned int _packed_29[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_29[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_29[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_29[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_29[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_29[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_29[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_29[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_29[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_40 = reinterpret_cast<const unsigned*>(_packed_29);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128 + 112))), "r"(_raw_stv8_40[0]), "r"(_raw_stv8_40[1]), "r"(_raw_stv8_40[2]), "r"(_raw_stv8_40[3]), "r"(_raw_stv8_40[4]), "r"(_raw_stv8_40[5]), "r"(_raw_stv8_40[6]), "r"(_raw_stv8_40[7]) : "memory");
                        }
                        unsigned int _packed_30[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_30[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_30[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_30[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_30[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_30[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_30[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_30[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_30[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_41 = reinterpret_cast<const unsigned*>(_packed_30);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (rbase * 64))), "r"(_raw_stv8_41[0]), "r"(_raw_stv8_41[1]), "r"(_raw_stv8_41[2]), "r"(_raw_stv8_41[3]), "r"(_raw_stv8_41[4]), "r"(_raw_stv8_41[5]), "r"(_raw_stv8_41[6]), "r"(_raw_stv8_41[7]) : "memory");
                        }
                        unsigned int _packed_31[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_31[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_31[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_31[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_31[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_31[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_31[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_31[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_31[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_42 = reinterpret_cast<const unsigned*>(_packed_31);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(akk_out + (rbase * 64))), "r"(_raw_stv8_42[0]), "r"(_raw_stv8_42[1]), "r"(_raw_stv8_42[2]), "r"(_raw_stv8_42[3]), "r"(_raw_stv8_42[4]), "r"(_raw_stv8_42[5]), "r"(_raw_stv8_42[6]), "r"(_raw_stv8_42[7]) : "memory");
                        }
                        unsigned int _packed_32[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_32[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_32[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_32[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_32[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_32[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_32[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_32[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_32[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_43 = reinterpret_cast<const unsigned*>(_packed_32);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (rbase * 64 + 16))), "r"(_raw_stv8_43[0]), "r"(_raw_stv8_43[1]), "r"(_raw_stv8_43[2]), "r"(_raw_stv8_43[3]), "r"(_raw_stv8_43[4]), "r"(_raw_stv8_43[5]), "r"(_raw_stv8_43[6]), "r"(_raw_stv8_43[7]) : "memory");
                        }
                        unsigned int _packed_33[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_33[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_33[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_33[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_33[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_33[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_33[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_33[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_33[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_44 = reinterpret_cast<const unsigned*>(_packed_33);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(akk_out + (rbase * 64 + 16))), "r"(_raw_stv8_44[0]), "r"(_raw_stv8_44[1]), "r"(_raw_stv8_44[2]), "r"(_raw_stv8_44[3]), "r"(_raw_stv8_44[4]), "r"(_raw_stv8_44[5]), "r"(_raw_stv8_44[6]), "r"(_raw_stv8_44[7]) : "memory");
                        }
                        unsigned int _packed_34[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_34[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_34[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_34[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_34[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_34[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_34[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_34[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_34[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_45 = reinterpret_cast<const unsigned*>(_packed_34);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (rbase * 64 + 32))), "r"(_raw_stv8_45[0]), "r"(_raw_stv8_45[1]), "r"(_raw_stv8_45[2]), "r"(_raw_stv8_45[3]), "r"(_raw_stv8_45[4]), "r"(_raw_stv8_45[5]), "r"(_raw_stv8_45[6]), "r"(_raw_stv8_45[7]) : "memory");
                        }
                        unsigned int _packed_35[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_35[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_35[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_35[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_35[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_35[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_35[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_35[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_35[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_46 = reinterpret_cast<const unsigned*>(_packed_35);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(akk_out + (rbase * 64 + 32))), "r"(_raw_stv8_46[0]), "r"(_raw_stv8_46[1]), "r"(_raw_stv8_46[2]), "r"(_raw_stv8_46[3]), "r"(_raw_stv8_46[4]), "r"(_raw_stv8_46[5]), "r"(_raw_stv8_46[6]), "r"(_raw_stv8_46[7]) : "memory");
                        }
                        unsigned int _packed_36[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_36[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_36[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_36[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_36[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_36[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_36[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_36[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_36[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_47 = reinterpret_cast<const unsigned*>(_packed_36);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (rbase * 64 + 48))), "r"(_raw_stv8_47[0]), "r"(_raw_stv8_47[1]), "r"(_raw_stv8_47[2]), "r"(_raw_stv8_47[3]), "r"(_raw_stv8_47[4]), "r"(_raw_stv8_47[5]), "r"(_raw_stv8_47[6]), "r"(_raw_stv8_47[7]) : "memory");
                        }
                        unsigned int _packed_37[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            _packed_37[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            _packed_37[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            _packed_37[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            _packed_37[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            _packed_37[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            _packed_37[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            _packed_37[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            _packed_37[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_48 = reinterpret_cast<const unsigned*>(_packed_37);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(akk_out + (rbase * 64 + 48))), "r"(_raw_stv8_48[0]), "r"(_raw_stv8_48[1]), "r"(_raw_stv8_48[2]), "r"(_raw_stv8_48[3]), "r"(_raw_stv8_48[4]), "r"(_raw_stv8_48[5]), "r"(_raw_stv8_48[6]), "r"(_raw_stv8_48[7]) : "memory");
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
                    int c0 = 0;
                    tma_3d_gmem2smem(q_p_addr, (&q_tma), c0, hq, ext0, qk_full_addr);
                    tma_3d_gmem2smem(k_p_addr, (&k_tma), c0, hq, ext0, qk_full_addr);
                    int c0_0 = 64;
                    tma_3d_gmem2smem(q_p_addr + 8192, (&q_tma), c0_0, hq, ext0, qk_full_addr);
                    tma_3d_gmem2smem(k_p_addr + 8192, (&k_tma), c0_0, hq, ext0, qk_full_addr);
                    #pragma unroll 1
                    for (int j_2 = 0; j_2 < group; j_2++) {
                        int hv_l = hq * group + j_2;
                        mbarrier_wait(v_empty_addr + (vst_l) * 8, _phase_v_empty);
                        mbarrier_arrive_expect_tx(v_full_addr + (vst_l) * 8, 16384);
                        int c0v = 0;
                        tma_3d_gmem2smem(v_p_addr + vst_l * 16384, (&v_tma), c0v, hv_l, ext0, v_full_addr + (vst_l) * 8);
                        int c0v_0 = 64;
                        tma_3d_gmem2smem(v_p_addr + vst_l * 16384 + 8192, (&v_tma), c0v_0, hv_l, ext0, v_full_addr + (vst_l) * 8);
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
