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
#define TMEM_NCOLS 128
#define TMEM_ACC_COLS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_S_KN_OFF 0
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 16384
#define SMEM_S_KNT_OFF 0
#define SMEM_S_KNT_STAGE_BYTES 16384
#define SMEM_S_KNT_STRIDE 16384
#define SMEM_S_LT_OFF 16384
#define SMEM_S_LT_STAGE_BYTES 16384
#define SMEM_S_LT_STRIDE 16384
#define SMEM_S_A_OFF 16384
#define SMEM_S_A_STAGE_BYTES 8192
#define SMEM_S_A_STRIDE 8192
#define SMEM_S_STAGE_OFF 24576
#define SMEM_S_STAGE_STAGE_BYTES 8192
#define SMEM_S_STAGE_STRIDE 8192
#define SMEM_S_X_OFF 32768
#define SMEM_S_X_STAGE_BYTES 16384
#define SMEM_S_X_STRIDE 16384
#define SMEM_S_V_OFF 32768
#define SMEM_S_V_STAGE_BYTES 16384
#define SMEM_S_V_STRIDE 16384
#define SMEM_S_VT_OFF 32768
#define SMEM_S_VT_STAGE_BYTES 16384
#define SMEM_S_VT_STRIDE 16384
#define SMEM_S_T_OFF 49152
#define SMEM_S_T_STAGE_BYTES 3840
#define SMEM_S_T_STRIDE 3840
#define SMEM_S_G_OFF 52992
#define SMEM_S_G_STAGE_BYTES 256
#define SMEM_S_G_STRIDE 256
#define SMEM_S_B_OFF 53248
#define SMEM_S_B_STAGE_BYTES 256
#define SMEM_S_B_STRIDE 256
#define SMEM_TOTAL 53632
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 4

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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_det_train_dabeef52c8d449bbee3f(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ v, float* __restrict__ g_cs, float* __restrict__ beta, __nv_bfloat16* __restrict__ A_out, __nv_bfloat16* __restrict__ w_out, __nv_bfloat16* __restrict__ u_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, int recompute)
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

    const int mbar_base = smem + 53504;
    #define m1_addr (mbar_base + 0)
    #define m2_addr (mbar_base + 8)
    #define m3_addr (mbar_base + 16)
    #define l1_addr (mbar_base + 24)
    #define lp_addr (mbar_base + 32)
    #define l2_addr (mbar_base + 40)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_kn_addr = smem + 0;
    __nv_bfloat16* s_knT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_knT_addr = smem + 0;
    float* s_lt = reinterpret_cast<float*>(smem_raw + 16384);
    const int s_lt_addr = smem + 16384;
    __nv_bfloat16* s_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_a_addr = smem + 16384;
    int* s_stage = reinterpret_cast<int*>(smem_raw + 24576);
    const int s_stage_addr = smem + 24576;
    float* s_x = reinterpret_cast<float*>(smem_raw + 32768);
    const int s_x_addr = smem + 32768;
    __nv_bfloat16* s_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_v_addr = smem + 32768;
    __nv_bfloat16* s_vT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_vT_addr = smem + 32768;
    float* s_t = reinterpret_cast<float*>(smem_raw + 49152);
    const int s_t_addr = smem + 49152;
    float* s_g = reinterpret_cast<float*>(smem_raw + 52992);
    const int s_g_addr = smem + 52992;
    float* s_b = reinterpret_cast<float*>(smem_raw + 53248);
    const int s_b_addr = smem + 53248;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 6 barriers)
    // Mbarriers at smem_raw[53504..53552)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // m1: 1 barriers, init_count=1
            mbarrier_init(smem + 53504, 1);
            // m2: 1 barriers, init_count=1
            mbarrier_init(smem + 53512, 1);
            // m3: 1 barriers, init_count=1
            mbarrier_init(smem + 53520, 1);
            // l1: 1 barriers, init_count=128
            mbarrier_init(smem + 53528, 128);
            // lp: 1 barriers, init_count=128
            mbarrier_init(smem + 53536, 128);
            // l2: 1 barriers, init_count=128
            mbarrier_init(smem + 53544, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 53552);
    if (warp == 0) {
        int _tmem_hold = smem + 53552;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
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
    int group = num_v_heads / num_heads;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int grid_x = gridDim.x;
    int linear = block_x + block_y * grid_x;
    int span = group * grid_x;
    int hq = linear / span;
    int chunk = linear % span / group;
    int hv = hq * group + linear % group;
    int tok0 = chunk_start[chunk];
    int n_valid = chunk_len[chunk];
    int frag_r = lane_0 / 4;
    int col_q = lane_0 % 4 * 2;
    int taddr_1 = tmem_acc_cols + (warp_1 * 32 << 16);
    int r0 = tid / 16;
    int c0 = tid % 16 * 8;
    long long row_stride = (long long)num_heads * 128;
    long long src_row = (long long)(tok0 + r0) * row_stride + (long long)hq * 128 + (long long)c0;
    unsigned int dst0 = s_kn_addr + (unsigned int)(c0 / 64 * 8192 + (r0 * 128 + c0 % 64 * 2 ^ (r0 * 128 + c0 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0), "l"(kn + src_row), "r"((n_valid > r0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 1024), "l"(kn + (src_row + 8 * row_stride)), "r"((n_valid > r0 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 2048), "l"(kn + (src_row + 16 * row_stride)), "r"((n_valid > r0 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 3072), "l"(kn + (src_row + 24 * row_stride)), "r"((n_valid > r0 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 4096), "l"(kn + (src_row + 32 * row_stride)), "r"((n_valid > r0 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 5120), "l"(kn + (src_row + 40 * row_stride)), "r"((n_valid > r0 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 6144), "l"(kn + (src_row + 48 * row_stride)), "r"((n_valid > r0 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 7168), "l"(kn + (src_row + 56 * row_stride)), "r"((n_valid > r0 + 56) ? 16 : 0));
    if (recompute != 0) {
        int ra = tid / 8;
        int ca = tid % 8 * 8;
        long long a_stride = (long long)num_v_heads * 64;
        long long a_src = (long long)(tok0 + ra) * a_stride + (long long)hv * 64 + (long long)ca;
        unsigned int a_dst = s_a_addr + (unsigned int)(ca / 64 * 8192 + (ra * 128 + ca % 64 * 2 ^ (ra * 128 + ca % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(a_dst), "l"(A_out + a_src), "r"((n_valid > ra) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(a_dst + 2048), "l"(A_out + (a_src + 16 * a_stride)), "r"((n_valid > ra + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(a_dst + 4096), "l"(A_out + (a_src + 32 * a_stride)), "r"((n_valid > ra + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(a_dst + 6144), "l"(A_out + (a_src + 48 * a_stride)), "r"((n_valid > ra + 48) ? 16 : 0));
        int r0_5 = tid / 16;
        int c0_6 = tid % 16 * 8;
        long long row_stride_7 = (long long)num_v_heads * 128;
        long long src_row_8 = (long long)(tok0 + r0_5) * row_stride_7 + (long long)hv * 128 + (long long)c0_6;
        unsigned int dst0_9 = s_v_addr + (unsigned int)(c0_6 / 64 * 8192 + (r0_5 * 128 + c0_6 % 64 * 2 ^ (r0_5 * 128 + c0_6 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9), "l"(v + src_row_8), "r"((n_valid > r0_5) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 1024), "l"(v + (src_row_8 + 8 * row_stride_7)), "r"((n_valid > r0_5 + 8) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 2048), "l"(v + (src_row_8 + 16 * row_stride_7)), "r"((n_valid > r0_5 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 3072), "l"(v + (src_row_8 + 24 * row_stride_7)), "r"((n_valid > r0_5 + 24) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 4096), "l"(v + (src_row_8 + 32 * row_stride_7)), "r"((n_valid > r0_5 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 5120), "l"(v + (src_row_8 + 40 * row_stride_7)), "r"((n_valid > r0_5 + 40) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 6144), "l"(v + (src_row_8 + 48 * row_stride_7)), "r"((n_valid > r0_5 + 48) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 7168), "l"(v + (src_row_8 + 56 * row_stride_7)), "r"((n_valid > r0_5 + 56) ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;");
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
    }
    float acc[32];
    unsigned int words[4];
    float vals8[8];
    unsigned int pk4[4];
    int words_o[4];
    asm volatile("cp.async.wait_group 0;");
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
    unsigned int _phase_m1_0 = 0;
    if (recompute == 0) {
        if (warp == 0) {
            uint32_t _elected_1 = elect_sync();
            asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_1));
            int _mma_kn_lo = ((s_kn_addr) >> 4) & 0x3FFF;
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
            :: "r"(_mma_kn_lo), "r"(_mma_kn_lo), "r"(tmem_acc_cols), "r"(0), "r"(_elected_1) : "memory");
            tcgen05_commit_leader(m1_addr, _elected_1);
        }
        mbarrier_wait(m1_addr, _phase_m1_0);
        _phase_m1_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (warp_1 == 0) {
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7]))
                : "r"(taddr_1 + 0));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float g_row_0[2];
            float b_row_0[2];
            g_row_0[0] = s_g[0 + frag_r];
            b_row_0[0] = s_b[0 + frag_r];
            g_row_0[1] = s_g[8 + frag_r];
            b_row_0[1] = s_b[8 + frag_r];
            int col_0_0_0_0 = 0 + col_q;
            float g_col_0_0_0_0 = s_g[col_0_0_0_0];
            int row_0_0_0_0_0 = 0 + frag_r;
            float val_0_0_0_0_0 = 0.0f;
            if (col_0_0_0_0 < row_0_0_0_0_0) {
                if (row_0_0_0_0_0 < n_valid) {
                    float _exp2_2 = approx_exp2(g_row_0[0] - g_col_0_0_0_0);
                    val_0_0_0_0_0 = b_row_0[0] * _exp2_2 * acc[0];
                }
            }
            s_lt[col_0_0_0_0 * 64 + (row_0_0_0_0_0 ^ (col_0_0_0_0 >> 1 & 3) << 3)] = val_0_0_0_0_0;
            int row_0_0_0_0_1 = 8 + frag_r;
            float val_0_0_0_0_1 = 0.0f;
            if (col_0_0_0_0 < row_0_0_0_0_1) {
                if (row_0_0_0_0_1 < n_valid) {
                    float _exp2_3 = approx_exp2(g_row_0[1] - g_col_0_0_0_0);
                    val_0_0_0_0_1 = b_row_0[1] * _exp2_3 * acc[2];
                }
            }
            s_lt[col_0_0_0_0 * 64 + (row_0_0_0_0_1 ^ (col_0_0_0_0 >> 1 & 3) << 3)] = val_0_0_0_0_1;
            int col_0_0_0_1 = 1 + col_q;
            float g_col_0_0_0_1 = s_g[col_0_0_0_1];
            int row_0_0_0_1_0 = 0 + frag_r;
            float val_0_0_0_1_0 = 0.0f;
            if (col_0_0_0_1 < row_0_0_0_1_0) {
                if (row_0_0_0_1_0 < n_valid) {
                    float _exp2_4 = approx_exp2(g_row_0[0] - g_col_0_0_0_1);
                    val_0_0_0_1_0 = b_row_0[0] * _exp2_4 * acc[1];
                }
            }
            s_lt[col_0_0_0_1 * 64 + (row_0_0_0_1_0 ^ (col_0_0_0_1 >> 1 & 3) << 3)] = val_0_0_0_1_0;
            int row_0_0_0_1_1 = 8 + frag_r;
            float val_0_0_0_1_1 = 0.0f;
            if (col_0_0_0_1 < row_0_0_0_1_1) {
                if (row_0_0_0_1_1 < n_valid) {
                    float _exp2_5 = approx_exp2(g_row_0[1] - g_col_0_0_0_1);
                    val_0_0_0_1_1 = b_row_0[1] * _exp2_5 * acc[3];
                }
            }
            s_lt[col_0_0_0_1 * 64 + (row_0_0_0_1_1 ^ (col_0_0_0_1 >> 1 & 3) << 3)] = val_0_0_0_1_1;
            int col_0_0_1_0 = 8 + col_q;
            float g_col_0_0_1_0 = s_g[col_0_0_1_0];
            int row_0_0_1_0_0 = 0 + frag_r;
            float val_0_0_1_0_0 = 0.0f;
            if (col_0_0_1_0 < row_0_0_1_0_0) {
                if (row_0_0_1_0_0 < n_valid) {
                    float _exp2_6 = approx_exp2(g_row_0[0] - g_col_0_0_1_0);
                    val_0_0_1_0_0 = b_row_0[0] * _exp2_6 * acc[4];
                }
            }
            s_lt[col_0_0_1_0 * 64 + (row_0_0_1_0_0 ^ (col_0_0_1_0 >> 1 & 3) << 3)] = val_0_0_1_0_0;
            int row_0_0_1_0_1 = 8 + frag_r;
            float val_0_0_1_0_1 = 0.0f;
            if (col_0_0_1_0 < row_0_0_1_0_1) {
                if (row_0_0_1_0_1 < n_valid) {
                    float _exp2_7 = approx_exp2(g_row_0[1] - g_col_0_0_1_0);
                    val_0_0_1_0_1 = b_row_0[1] * _exp2_7 * acc[6];
                }
            }
            s_lt[col_0_0_1_0 * 64 + (row_0_0_1_0_1 ^ (col_0_0_1_0 >> 1 & 3) << 3)] = val_0_0_1_0_1;
            int col_0_0_1_1 = 9 + col_q;
            float g_col_0_0_1_1 = s_g[col_0_0_1_1];
            int row_0_0_1_1_0 = 0 + frag_r;
            float val_0_0_1_1_0 = 0.0f;
            if (col_0_0_1_1 < row_0_0_1_1_0) {
                if (row_0_0_1_1_0 < n_valid) {
                    float _exp2_8 = approx_exp2(g_row_0[0] - g_col_0_0_1_1);
                    val_0_0_1_1_0 = b_row_0[0] * _exp2_8 * acc[5];
                }
            }
            s_lt[col_0_0_1_1 * 64 + (row_0_0_1_1_0 ^ (col_0_0_1_1 >> 1 & 3) << 3)] = val_0_0_1_1_0;
            int row_0_0_1_1_1 = 8 + frag_r;
            float val_0_0_1_1_1 = 0.0f;
            if (col_0_0_1_1 < row_0_0_1_1_1) {
                if (row_0_0_1_1_1 < n_valid) {
                    float _exp2_9 = approx_exp2(g_row_0[1] - g_col_0_0_1_1);
                    val_0_0_1_1_1 = b_row_0[1] * _exp2_9 * acc[7];
                }
            }
            s_lt[col_0_0_1_1 * 64 + (row_0_0_1_1_1 ^ (col_0_0_1_1 >> 1 & 3) << 3)] = val_0_0_1_1_1;
        }
        if (warp_1 == 1) {
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7]))
                : "r"(taddr_1 + 0));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                : "r"(taddr_1 + 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float g_row_1[2];
            float b_row_1[2];
            g_row_1[0] = s_g[16 + frag_r];
            b_row_1[0] = s_b[16 + frag_r];
            g_row_1[1] = s_g[24 + frag_r];
            b_row_1[1] = s_b[24 + frag_r];
            int col_1_0_0_0 = 0 + col_q;
            float g_col_1_0_0_0 = s_g[col_1_0_0_0];
            int row_1_0_0_0_0 = 16 + frag_r;
            float val_1_0_0_0_0 = 0.0f;
            if (col_1_0_0_0 < row_1_0_0_0_0) {
                if (row_1_0_0_0_0 < n_valid) {
                    float _exp2_10 = approx_exp2(g_row_1[0] - g_col_1_0_0_0);
                    val_1_0_0_0_0 = b_row_1[0] * _exp2_10 * acc[0];
                }
            }
            s_lt[col_1_0_0_0 * 64 + (row_1_0_0_0_0 ^ (col_1_0_0_0 >> 1 & 3) << 3)] = val_1_0_0_0_0;
            int row_1_0_0_0_1 = 24 + frag_r;
            float val_1_0_0_0_1 = 0.0f;
            if (col_1_0_0_0 < row_1_0_0_0_1) {
                if (row_1_0_0_0_1 < n_valid) {
                    float _exp2_11 = approx_exp2(g_row_1[1] - g_col_1_0_0_0);
                    val_1_0_0_0_1 = b_row_1[1] * _exp2_11 * acc[2];
                }
            }
            s_lt[col_1_0_0_0 * 64 + (row_1_0_0_0_1 ^ (col_1_0_0_0 >> 1 & 3) << 3)] = val_1_0_0_0_1;
            int col_1_0_0_1 = 1 + col_q;
            float g_col_1_0_0_1 = s_g[col_1_0_0_1];
            int row_1_0_0_1_0 = 16 + frag_r;
            float val_1_0_0_1_0 = 0.0f;
            if (col_1_0_0_1 < row_1_0_0_1_0) {
                if (row_1_0_0_1_0 < n_valid) {
                    float _exp2_12 = approx_exp2(g_row_1[0] - g_col_1_0_0_1);
                    val_1_0_0_1_0 = b_row_1[0] * _exp2_12 * acc[1];
                }
            }
            s_lt[col_1_0_0_1 * 64 + (row_1_0_0_1_0 ^ (col_1_0_0_1 >> 1 & 3) << 3)] = val_1_0_0_1_0;
            int row_1_0_0_1_1 = 24 + frag_r;
            float val_1_0_0_1_1 = 0.0f;
            if (col_1_0_0_1 < row_1_0_0_1_1) {
                if (row_1_0_0_1_1 < n_valid) {
                    float _exp2_13 = approx_exp2(g_row_1[1] - g_col_1_0_0_1);
                    val_1_0_0_1_1 = b_row_1[1] * _exp2_13 * acc[3];
                }
            }
            s_lt[col_1_0_0_1 * 64 + (row_1_0_0_1_1 ^ (col_1_0_0_1 >> 1 & 3) << 3)] = val_1_0_0_1_1;
            int col_1_0_1_0 = 8 + col_q;
            float g_col_1_0_1_0 = s_g[col_1_0_1_0];
            int row_1_0_1_0_0 = 16 + frag_r;
            float val_1_0_1_0_0 = 0.0f;
            if (col_1_0_1_0 < row_1_0_1_0_0) {
                if (row_1_0_1_0_0 < n_valid) {
                    float _exp2_14 = approx_exp2(g_row_1[0] - g_col_1_0_1_0);
                    val_1_0_1_0_0 = b_row_1[0] * _exp2_14 * acc[4];
                }
            }
            s_lt[col_1_0_1_0 * 64 + (row_1_0_1_0_0 ^ (col_1_0_1_0 >> 1 & 3) << 3)] = val_1_0_1_0_0;
            int row_1_0_1_0_1 = 24 + frag_r;
            float val_1_0_1_0_1 = 0.0f;
            if (col_1_0_1_0 < row_1_0_1_0_1) {
                if (row_1_0_1_0_1 < n_valid) {
                    float _exp2_15 = approx_exp2(g_row_1[1] - g_col_1_0_1_0);
                    val_1_0_1_0_1 = b_row_1[1] * _exp2_15 * acc[6];
                }
            }
            s_lt[col_1_0_1_0 * 64 + (row_1_0_1_0_1 ^ (col_1_0_1_0 >> 1 & 3) << 3)] = val_1_0_1_0_1;
            int col_1_0_1_1 = 9 + col_q;
            float g_col_1_0_1_1 = s_g[col_1_0_1_1];
            int row_1_0_1_1_0 = 16 + frag_r;
            float val_1_0_1_1_0 = 0.0f;
            if (col_1_0_1_1 < row_1_0_1_1_0) {
                if (row_1_0_1_1_0 < n_valid) {
                    float _exp2_16 = approx_exp2(g_row_1[0] - g_col_1_0_1_1);
                    val_1_0_1_1_0 = b_row_1[0] * _exp2_16 * acc[5];
                }
            }
            s_lt[col_1_0_1_1 * 64 + (row_1_0_1_1_0 ^ (col_1_0_1_1 >> 1 & 3) << 3)] = val_1_0_1_1_0;
            int row_1_0_1_1_1 = 24 + frag_r;
            float val_1_0_1_1_1 = 0.0f;
            if (col_1_0_1_1 < row_1_0_1_1_1) {
                if (row_1_0_1_1_1 < n_valid) {
                    float _exp2_17 = approx_exp2(g_row_1[1] - g_col_1_0_1_1);
                    val_1_0_1_1_1 = b_row_1[1] * _exp2_17 * acc[7];
                }
            }
            s_lt[col_1_0_1_1 * 64 + (row_1_0_1_1_1 ^ (col_1_0_1_1 >> 1 & 3) << 3)] = val_1_0_1_1_1;
            int col_1_1_0_0 = 16 + col_q;
            float g_col_1_1_0_0 = s_g[col_1_1_0_0];
            int row_1_1_0_0_0 = 16 + frag_r;
            float val_1_1_0_0_0 = 0.0f;
            if (col_1_1_0_0 < row_1_1_0_0_0) {
                if (row_1_1_0_0_0 < n_valid) {
                    float _exp2_18 = approx_exp2(g_row_1[0] - g_col_1_1_0_0);
                    val_1_1_0_0_0 = b_row_1[0] * _exp2_18 * acc[8];
                }
            }
            s_lt[col_1_1_0_0 * 64 + (row_1_1_0_0_0 ^ (col_1_1_0_0 >> 1 & 3) << 3)] = val_1_1_0_0_0;
            int row_1_1_0_0_1 = 24 + frag_r;
            float val_1_1_0_0_1 = 0.0f;
            if (col_1_1_0_0 < row_1_1_0_0_1) {
                if (row_1_1_0_0_1 < n_valid) {
                    float _exp2_19 = approx_exp2(g_row_1[1] - g_col_1_1_0_0);
                    val_1_1_0_0_1 = b_row_1[1] * _exp2_19 * acc[10];
                }
            }
            s_lt[col_1_1_0_0 * 64 + (row_1_1_0_0_1 ^ (col_1_1_0_0 >> 1 & 3) << 3)] = val_1_1_0_0_1;
            int col_1_1_0_1 = 17 + col_q;
            float g_col_1_1_0_1 = s_g[col_1_1_0_1];
            int row_1_1_0_1_0 = 16 + frag_r;
            float val_1_1_0_1_0 = 0.0f;
            if (col_1_1_0_1 < row_1_1_0_1_0) {
                if (row_1_1_0_1_0 < n_valid) {
                    float _exp2_20 = approx_exp2(g_row_1[0] - g_col_1_1_0_1);
                    val_1_1_0_1_0 = b_row_1[0] * _exp2_20 * acc[9];
                }
            }
            s_lt[col_1_1_0_1 * 64 + (row_1_1_0_1_0 ^ (col_1_1_0_1 >> 1 & 3) << 3)] = val_1_1_0_1_0;
            int row_1_1_0_1_1 = 24 + frag_r;
            float val_1_1_0_1_1 = 0.0f;
            if (col_1_1_0_1 < row_1_1_0_1_1) {
                if (row_1_1_0_1_1 < n_valid) {
                    float _exp2_21 = approx_exp2(g_row_1[1] - g_col_1_1_0_1);
                    val_1_1_0_1_1 = b_row_1[1] * _exp2_21 * acc[11];
                }
            }
            s_lt[col_1_1_0_1 * 64 + (row_1_1_0_1_1 ^ (col_1_1_0_1 >> 1 & 3) << 3)] = val_1_1_0_1_1;
            int col_1_1_1_0 = 24 + col_q;
            float g_col_1_1_1_0 = s_g[col_1_1_1_0];
            int row_1_1_1_0_0 = 16 + frag_r;
            float val_1_1_1_0_0 = 0.0f;
            if (col_1_1_1_0 < row_1_1_1_0_0) {
                if (row_1_1_1_0_0 < n_valid) {
                    float _exp2_22 = approx_exp2(g_row_1[0] - g_col_1_1_1_0);
                    val_1_1_1_0_0 = b_row_1[0] * _exp2_22 * acc[12];
                }
            }
            s_lt[col_1_1_1_0 * 64 + (row_1_1_1_0_0 ^ (col_1_1_1_0 >> 1 & 3) << 3)] = val_1_1_1_0_0;
            int row_1_1_1_0_1 = 24 + frag_r;
            float val_1_1_1_0_1 = 0.0f;
            if (col_1_1_1_0 < row_1_1_1_0_1) {
                if (row_1_1_1_0_1 < n_valid) {
                    float _exp2_23 = approx_exp2(g_row_1[1] - g_col_1_1_1_0);
                    val_1_1_1_0_1 = b_row_1[1] * _exp2_23 * acc[14];
                }
            }
            s_lt[col_1_1_1_0 * 64 + (row_1_1_1_0_1 ^ (col_1_1_1_0 >> 1 & 3) << 3)] = val_1_1_1_0_1;
            int col_1_1_1_1 = 25 + col_q;
            float g_col_1_1_1_1 = s_g[col_1_1_1_1];
            int row_1_1_1_1_0 = 16 + frag_r;
            float val_1_1_1_1_0 = 0.0f;
            if (col_1_1_1_1 < row_1_1_1_1_0) {
                if (row_1_1_1_1_0 < n_valid) {
                    float _exp2_24 = approx_exp2(g_row_1[0] - g_col_1_1_1_1);
                    val_1_1_1_1_0 = b_row_1[0] * _exp2_24 * acc[13];
                }
            }
            s_lt[col_1_1_1_1 * 64 + (row_1_1_1_1_0 ^ (col_1_1_1_1 >> 1 & 3) << 3)] = val_1_1_1_1_0;
            int row_1_1_1_1_1 = 24 + frag_r;
            float val_1_1_1_1_1 = 0.0f;
            if (col_1_1_1_1 < row_1_1_1_1_1) {
                if (row_1_1_1_1_1 < n_valid) {
                    float _exp2_25 = approx_exp2(g_row_1[1] - g_col_1_1_1_1);
                    val_1_1_1_1_1 = b_row_1[1] * _exp2_25 * acc[15];
                }
            }
            s_lt[col_1_1_1_1 * 64 + (row_1_1_1_1_1 ^ (col_1_1_1_1 >> 1 & 3) << 3)] = val_1_1_1_1_1;
        }
        if (warp_1 == 2) {
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7]))
                : "r"(taddr_1 + 0));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                : "r"(taddr_1 + 16));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23]))
                : "r"(taddr_1 + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float g_row_2[2];
            float b_row_2[2];
            g_row_2[0] = s_g[32 + frag_r];
            b_row_2[0] = s_b[32 + frag_r];
            g_row_2[1] = s_g[40 + frag_r];
            b_row_2[1] = s_b[40 + frag_r];
            int col_2_0_0_0 = 0 + col_q;
            float g_col_2_0_0_0 = s_g[col_2_0_0_0];
            int row_2_0_0_0_0 = 32 + frag_r;
            float val_2_0_0_0_0 = 0.0f;
            if (col_2_0_0_0 < row_2_0_0_0_0) {
                if (row_2_0_0_0_0 < n_valid) {
                    float _exp2_26 = approx_exp2(g_row_2[0] - g_col_2_0_0_0);
                    val_2_0_0_0_0 = b_row_2[0] * _exp2_26 * acc[0];
                }
            }
            s_lt[col_2_0_0_0 * 64 + (row_2_0_0_0_0 ^ (col_2_0_0_0 >> 1 & 3) << 3)] = val_2_0_0_0_0;
            int row_2_0_0_0_1 = 40 + frag_r;
            float val_2_0_0_0_1 = 0.0f;
            if (col_2_0_0_0 < row_2_0_0_0_1) {
                if (row_2_0_0_0_1 < n_valid) {
                    float _exp2_27 = approx_exp2(g_row_2[1] - g_col_2_0_0_0);
                    val_2_0_0_0_1 = b_row_2[1] * _exp2_27 * acc[2];
                }
            }
            s_lt[col_2_0_0_0 * 64 + (row_2_0_0_0_1 ^ (col_2_0_0_0 >> 1 & 3) << 3)] = val_2_0_0_0_1;
            int col_2_0_0_1 = 1 + col_q;
            float g_col_2_0_0_1 = s_g[col_2_0_0_1];
            int row_2_0_0_1_0 = 32 + frag_r;
            float val_2_0_0_1_0 = 0.0f;
            if (col_2_0_0_1 < row_2_0_0_1_0) {
                if (row_2_0_0_1_0 < n_valid) {
                    float _exp2_28 = approx_exp2(g_row_2[0] - g_col_2_0_0_1);
                    val_2_0_0_1_0 = b_row_2[0] * _exp2_28 * acc[1];
                }
            }
            s_lt[col_2_0_0_1 * 64 + (row_2_0_0_1_0 ^ (col_2_0_0_1 >> 1 & 3) << 3)] = val_2_0_0_1_0;
            int row_2_0_0_1_1 = 40 + frag_r;
            float val_2_0_0_1_1 = 0.0f;
            if (col_2_0_0_1 < row_2_0_0_1_1) {
                if (row_2_0_0_1_1 < n_valid) {
                    float _exp2_29 = approx_exp2(g_row_2[1] - g_col_2_0_0_1);
                    val_2_0_0_1_1 = b_row_2[1] * _exp2_29 * acc[3];
                }
            }
            s_lt[col_2_0_0_1 * 64 + (row_2_0_0_1_1 ^ (col_2_0_0_1 >> 1 & 3) << 3)] = val_2_0_0_1_1;
            int col_2_0_1_0 = 8 + col_q;
            float g_col_2_0_1_0 = s_g[col_2_0_1_0];
            int row_2_0_1_0_0 = 32 + frag_r;
            float val_2_0_1_0_0 = 0.0f;
            if (col_2_0_1_0 < row_2_0_1_0_0) {
                if (row_2_0_1_0_0 < n_valid) {
                    float _exp2_30 = approx_exp2(g_row_2[0] - g_col_2_0_1_0);
                    val_2_0_1_0_0 = b_row_2[0] * _exp2_30 * acc[4];
                }
            }
            s_lt[col_2_0_1_0 * 64 + (row_2_0_1_0_0 ^ (col_2_0_1_0 >> 1 & 3) << 3)] = val_2_0_1_0_0;
            int row_2_0_1_0_1 = 40 + frag_r;
            float val_2_0_1_0_1 = 0.0f;
            if (col_2_0_1_0 < row_2_0_1_0_1) {
                if (row_2_0_1_0_1 < n_valid) {
                    float _exp2_31 = approx_exp2(g_row_2[1] - g_col_2_0_1_0);
                    val_2_0_1_0_1 = b_row_2[1] * _exp2_31 * acc[6];
                }
            }
            s_lt[col_2_0_1_0 * 64 + (row_2_0_1_0_1 ^ (col_2_0_1_0 >> 1 & 3) << 3)] = val_2_0_1_0_1;
            int col_2_0_1_1 = 9 + col_q;
            float g_col_2_0_1_1 = s_g[col_2_0_1_1];
            int row_2_0_1_1_0 = 32 + frag_r;
            float val_2_0_1_1_0 = 0.0f;
            if (col_2_0_1_1 < row_2_0_1_1_0) {
                if (row_2_0_1_1_0 < n_valid) {
                    float _exp2_32 = approx_exp2(g_row_2[0] - g_col_2_0_1_1);
                    val_2_0_1_1_0 = b_row_2[0] * _exp2_32 * acc[5];
                }
            }
            s_lt[col_2_0_1_1 * 64 + (row_2_0_1_1_0 ^ (col_2_0_1_1 >> 1 & 3) << 3)] = val_2_0_1_1_0;
            int row_2_0_1_1_1 = 40 + frag_r;
            float val_2_0_1_1_1 = 0.0f;
            if (col_2_0_1_1 < row_2_0_1_1_1) {
                if (row_2_0_1_1_1 < n_valid) {
                    float _exp2_33 = approx_exp2(g_row_2[1] - g_col_2_0_1_1);
                    val_2_0_1_1_1 = b_row_2[1] * _exp2_33 * acc[7];
                }
            }
            s_lt[col_2_0_1_1 * 64 + (row_2_0_1_1_1 ^ (col_2_0_1_1 >> 1 & 3) << 3)] = val_2_0_1_1_1;
            int col_2_1_0_0 = 16 + col_q;
            float g_col_2_1_0_0 = s_g[col_2_1_0_0];
            int row_2_1_0_0_0 = 32 + frag_r;
            float val_2_1_0_0_0 = 0.0f;
            if (col_2_1_0_0 < row_2_1_0_0_0) {
                if (row_2_1_0_0_0 < n_valid) {
                    float _exp2_34 = approx_exp2(g_row_2[0] - g_col_2_1_0_0);
                    val_2_1_0_0_0 = b_row_2[0] * _exp2_34 * acc[8];
                }
            }
            s_lt[col_2_1_0_0 * 64 + (row_2_1_0_0_0 ^ (col_2_1_0_0 >> 1 & 3) << 3)] = val_2_1_0_0_0;
            int row_2_1_0_0_1 = 40 + frag_r;
            float val_2_1_0_0_1 = 0.0f;
            if (col_2_1_0_0 < row_2_1_0_0_1) {
                if (row_2_1_0_0_1 < n_valid) {
                    float _exp2_35 = approx_exp2(g_row_2[1] - g_col_2_1_0_0);
                    val_2_1_0_0_1 = b_row_2[1] * _exp2_35 * acc[10];
                }
            }
            s_lt[col_2_1_0_0 * 64 + (row_2_1_0_0_1 ^ (col_2_1_0_0 >> 1 & 3) << 3)] = val_2_1_0_0_1;
            int col_2_1_0_1 = 17 + col_q;
            float g_col_2_1_0_1 = s_g[col_2_1_0_1];
            int row_2_1_0_1_0 = 32 + frag_r;
            float val_2_1_0_1_0 = 0.0f;
            if (col_2_1_0_1 < row_2_1_0_1_0) {
                if (row_2_1_0_1_0 < n_valid) {
                    float _exp2_36 = approx_exp2(g_row_2[0] - g_col_2_1_0_1);
                    val_2_1_0_1_0 = b_row_2[0] * _exp2_36 * acc[9];
                }
            }
            s_lt[col_2_1_0_1 * 64 + (row_2_1_0_1_0 ^ (col_2_1_0_1 >> 1 & 3) << 3)] = val_2_1_0_1_0;
            int row_2_1_0_1_1 = 40 + frag_r;
            float val_2_1_0_1_1 = 0.0f;
            if (col_2_1_0_1 < row_2_1_0_1_1) {
                if (row_2_1_0_1_1 < n_valid) {
                    float _exp2_37 = approx_exp2(g_row_2[1] - g_col_2_1_0_1);
                    val_2_1_0_1_1 = b_row_2[1] * _exp2_37 * acc[11];
                }
            }
            s_lt[col_2_1_0_1 * 64 + (row_2_1_0_1_1 ^ (col_2_1_0_1 >> 1 & 3) << 3)] = val_2_1_0_1_1;
            int col_2_1_1_0 = 24 + col_q;
            float g_col_2_1_1_0 = s_g[col_2_1_1_0];
            int row_2_1_1_0_0 = 32 + frag_r;
            float val_2_1_1_0_0 = 0.0f;
            if (col_2_1_1_0 < row_2_1_1_0_0) {
                if (row_2_1_1_0_0 < n_valid) {
                    float _exp2_38 = approx_exp2(g_row_2[0] - g_col_2_1_1_0);
                    val_2_1_1_0_0 = b_row_2[0] * _exp2_38 * acc[12];
                }
            }
            s_lt[col_2_1_1_0 * 64 + (row_2_1_1_0_0 ^ (col_2_1_1_0 >> 1 & 3) << 3)] = val_2_1_1_0_0;
            int row_2_1_1_0_1 = 40 + frag_r;
            float val_2_1_1_0_1 = 0.0f;
            if (col_2_1_1_0 < row_2_1_1_0_1) {
                if (row_2_1_1_0_1 < n_valid) {
                    float _exp2_39 = approx_exp2(g_row_2[1] - g_col_2_1_1_0);
                    val_2_1_1_0_1 = b_row_2[1] * _exp2_39 * acc[14];
                }
            }
            s_lt[col_2_1_1_0 * 64 + (row_2_1_1_0_1 ^ (col_2_1_1_0 >> 1 & 3) << 3)] = val_2_1_1_0_1;
            int col_2_1_1_1 = 25 + col_q;
            float g_col_2_1_1_1 = s_g[col_2_1_1_1];
            int row_2_1_1_1_0 = 32 + frag_r;
            float val_2_1_1_1_0 = 0.0f;
            if (col_2_1_1_1 < row_2_1_1_1_0) {
                if (row_2_1_1_1_0 < n_valid) {
                    float _exp2_40 = approx_exp2(g_row_2[0] - g_col_2_1_1_1);
                    val_2_1_1_1_0 = b_row_2[0] * _exp2_40 * acc[13];
                }
            }
            s_lt[col_2_1_1_1 * 64 + (row_2_1_1_1_0 ^ (col_2_1_1_1 >> 1 & 3) << 3)] = val_2_1_1_1_0;
            int row_2_1_1_1_1 = 40 + frag_r;
            float val_2_1_1_1_1 = 0.0f;
            if (col_2_1_1_1 < row_2_1_1_1_1) {
                if (row_2_1_1_1_1 < n_valid) {
                    float _exp2_41 = approx_exp2(g_row_2[1] - g_col_2_1_1_1);
                    val_2_1_1_1_1 = b_row_2[1] * _exp2_41 * acc[15];
                }
            }
            s_lt[col_2_1_1_1 * 64 + (row_2_1_1_1_1 ^ (col_2_1_1_1 >> 1 & 3) << 3)] = val_2_1_1_1_1;
            int col_2_2_0_0 = 32 + col_q;
            float g_col_2_2_0_0 = s_g[col_2_2_0_0];
            int row_2_2_0_0_0 = 32 + frag_r;
            float val_2_2_0_0_0 = 0.0f;
            if (col_2_2_0_0 < row_2_2_0_0_0) {
                if (row_2_2_0_0_0 < n_valid) {
                    float _exp2_42 = approx_exp2(g_row_2[0] - g_col_2_2_0_0);
                    val_2_2_0_0_0 = b_row_2[0] * _exp2_42 * acc[16];
                }
            }
            s_lt[col_2_2_0_0 * 64 + (row_2_2_0_0_0 ^ (col_2_2_0_0 >> 1 & 3) << 3)] = val_2_2_0_0_0;
            int row_2_2_0_0_1 = 40 + frag_r;
            float val_2_2_0_0_1 = 0.0f;
            if (col_2_2_0_0 < row_2_2_0_0_1) {
                if (row_2_2_0_0_1 < n_valid) {
                    float _exp2_43 = approx_exp2(g_row_2[1] - g_col_2_2_0_0);
                    val_2_2_0_0_1 = b_row_2[1] * _exp2_43 * acc[18];
                }
            }
            s_lt[col_2_2_0_0 * 64 + (row_2_2_0_0_1 ^ (col_2_2_0_0 >> 1 & 3) << 3)] = val_2_2_0_0_1;
            int col_2_2_0_1 = 33 + col_q;
            float g_col_2_2_0_1 = s_g[col_2_2_0_1];
            int row_2_2_0_1_0 = 32 + frag_r;
            float val_2_2_0_1_0 = 0.0f;
            if (col_2_2_0_1 < row_2_2_0_1_0) {
                if (row_2_2_0_1_0 < n_valid) {
                    float _exp2_44 = approx_exp2(g_row_2[0] - g_col_2_2_0_1);
                    val_2_2_0_1_0 = b_row_2[0] * _exp2_44 * acc[17];
                }
            }
            s_lt[col_2_2_0_1 * 64 + (row_2_2_0_1_0 ^ (col_2_2_0_1 >> 1 & 3) << 3)] = val_2_2_0_1_0;
            int row_2_2_0_1_1 = 40 + frag_r;
            float val_2_2_0_1_1 = 0.0f;
            if (col_2_2_0_1 < row_2_2_0_1_1) {
                if (row_2_2_0_1_1 < n_valid) {
                    float _exp2_45 = approx_exp2(g_row_2[1] - g_col_2_2_0_1);
                    val_2_2_0_1_1 = b_row_2[1] * _exp2_45 * acc[19];
                }
            }
            s_lt[col_2_2_0_1 * 64 + (row_2_2_0_1_1 ^ (col_2_2_0_1 >> 1 & 3) << 3)] = val_2_2_0_1_1;
            int col_2_2_1_0 = 40 + col_q;
            float g_col_2_2_1_0 = s_g[col_2_2_1_0];
            int row_2_2_1_0_0 = 32 + frag_r;
            float val_2_2_1_0_0 = 0.0f;
            if (col_2_2_1_0 < row_2_2_1_0_0) {
                if (row_2_2_1_0_0 < n_valid) {
                    float _exp2_46 = approx_exp2(g_row_2[0] - g_col_2_2_1_0);
                    val_2_2_1_0_0 = b_row_2[0] * _exp2_46 * acc[20];
                }
            }
            s_lt[col_2_2_1_0 * 64 + (row_2_2_1_0_0 ^ (col_2_2_1_0 >> 1 & 3) << 3)] = val_2_2_1_0_0;
            int row_2_2_1_0_1 = 40 + frag_r;
            float val_2_2_1_0_1 = 0.0f;
            if (col_2_2_1_0 < row_2_2_1_0_1) {
                if (row_2_2_1_0_1 < n_valid) {
                    float _exp2_47 = approx_exp2(g_row_2[1] - g_col_2_2_1_0);
                    val_2_2_1_0_1 = b_row_2[1] * _exp2_47 * acc[22];
                }
            }
            s_lt[col_2_2_1_0 * 64 + (row_2_2_1_0_1 ^ (col_2_2_1_0 >> 1 & 3) << 3)] = val_2_2_1_0_1;
            int col_2_2_1_1 = 41 + col_q;
            float g_col_2_2_1_1 = s_g[col_2_2_1_1];
            int row_2_2_1_1_0 = 32 + frag_r;
            float val_2_2_1_1_0 = 0.0f;
            if (col_2_2_1_1 < row_2_2_1_1_0) {
                if (row_2_2_1_1_0 < n_valid) {
                    float _exp2_48 = approx_exp2(g_row_2[0] - g_col_2_2_1_1);
                    val_2_2_1_1_0 = b_row_2[0] * _exp2_48 * acc[21];
                }
            }
            s_lt[col_2_2_1_1 * 64 + (row_2_2_1_1_0 ^ (col_2_2_1_1 >> 1 & 3) << 3)] = val_2_2_1_1_0;
            int row_2_2_1_1_1 = 40 + frag_r;
            float val_2_2_1_1_1 = 0.0f;
            if (col_2_2_1_1 < row_2_2_1_1_1) {
                if (row_2_2_1_1_1 < n_valid) {
                    float _exp2_49 = approx_exp2(g_row_2[1] - g_col_2_2_1_1);
                    val_2_2_1_1_1 = b_row_2[1] * _exp2_49 * acc[23];
                }
            }
            s_lt[col_2_2_1_1 * 64 + (row_2_2_1_1_1 ^ (col_2_2_1_1 >> 1 & 3) << 3)] = val_2_2_1_1_1;
        }
        if (warp_1 == 3) {
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7]))
                : "r"(taddr_1 + 0));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                : "r"(taddr_1 + 16));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23]))
                : "r"(taddr_1 + 32));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                : "r"(taddr_1 + 48));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float g_row_3[2];
            float b_row_3[2];
            g_row_3[0] = s_g[48 + frag_r];
            b_row_3[0] = s_b[48 + frag_r];
            g_row_3[1] = s_g[56 + frag_r];
            b_row_3[1] = s_b[56 + frag_r];
            int col_3_0_0_0 = 0 + col_q;
            float g_col_3_0_0_0 = s_g[col_3_0_0_0];
            int row_3_0_0_0_0 = 48 + frag_r;
            float val_3_0_0_0_0 = 0.0f;
            if (col_3_0_0_0 < row_3_0_0_0_0) {
                if (row_3_0_0_0_0 < n_valid) {
                    float _exp2_50 = approx_exp2(g_row_3[0] - g_col_3_0_0_0);
                    val_3_0_0_0_0 = b_row_3[0] * _exp2_50 * acc[0];
                }
            }
            s_lt[col_3_0_0_0 * 64 + (row_3_0_0_0_0 ^ (col_3_0_0_0 >> 1 & 3) << 3)] = val_3_0_0_0_0;
            int row_3_0_0_0_1 = 56 + frag_r;
            float val_3_0_0_0_1 = 0.0f;
            if (col_3_0_0_0 < row_3_0_0_0_1) {
                if (row_3_0_0_0_1 < n_valid) {
                    float _exp2_51 = approx_exp2(g_row_3[1] - g_col_3_0_0_0);
                    val_3_0_0_0_1 = b_row_3[1] * _exp2_51 * acc[2];
                }
            }
            s_lt[col_3_0_0_0 * 64 + (row_3_0_0_0_1 ^ (col_3_0_0_0 >> 1 & 3) << 3)] = val_3_0_0_0_1;
            int col_3_0_0_1 = 1 + col_q;
            float g_col_3_0_0_1 = s_g[col_3_0_0_1];
            int row_3_0_0_1_0 = 48 + frag_r;
            float val_3_0_0_1_0 = 0.0f;
            if (col_3_0_0_1 < row_3_0_0_1_0) {
                if (row_3_0_0_1_0 < n_valid) {
                    float _exp2_52 = approx_exp2(g_row_3[0] - g_col_3_0_0_1);
                    val_3_0_0_1_0 = b_row_3[0] * _exp2_52 * acc[1];
                }
            }
            s_lt[col_3_0_0_1 * 64 + (row_3_0_0_1_0 ^ (col_3_0_0_1 >> 1 & 3) << 3)] = val_3_0_0_1_0;
            int row_3_0_0_1_1 = 56 + frag_r;
            float val_3_0_0_1_1 = 0.0f;
            if (col_3_0_0_1 < row_3_0_0_1_1) {
                if (row_3_0_0_1_1 < n_valid) {
                    float _exp2_53 = approx_exp2(g_row_3[1] - g_col_3_0_0_1);
                    val_3_0_0_1_1 = b_row_3[1] * _exp2_53 * acc[3];
                }
            }
            s_lt[col_3_0_0_1 * 64 + (row_3_0_0_1_1 ^ (col_3_0_0_1 >> 1 & 3) << 3)] = val_3_0_0_1_1;
            int col_3_0_1_0 = 8 + col_q;
            float g_col_3_0_1_0 = s_g[col_3_0_1_0];
            int row_3_0_1_0_0 = 48 + frag_r;
            float val_3_0_1_0_0 = 0.0f;
            if (col_3_0_1_0 < row_3_0_1_0_0) {
                if (row_3_0_1_0_0 < n_valid) {
                    float _exp2_54 = approx_exp2(g_row_3[0] - g_col_3_0_1_0);
                    val_3_0_1_0_0 = b_row_3[0] * _exp2_54 * acc[4];
                }
            }
            s_lt[col_3_0_1_0 * 64 + (row_3_0_1_0_0 ^ (col_3_0_1_0 >> 1 & 3) << 3)] = val_3_0_1_0_0;
            int row_3_0_1_0_1 = 56 + frag_r;
            float val_3_0_1_0_1 = 0.0f;
            if (col_3_0_1_0 < row_3_0_1_0_1) {
                if (row_3_0_1_0_1 < n_valid) {
                    float _exp2_55 = approx_exp2(g_row_3[1] - g_col_3_0_1_0);
                    val_3_0_1_0_1 = b_row_3[1] * _exp2_55 * acc[6];
                }
            }
            s_lt[col_3_0_1_0 * 64 + (row_3_0_1_0_1 ^ (col_3_0_1_0 >> 1 & 3) << 3)] = val_3_0_1_0_1;
            int col_3_0_1_1 = 9 + col_q;
            float g_col_3_0_1_1 = s_g[col_3_0_1_1];
            int row_3_0_1_1_0 = 48 + frag_r;
            float val_3_0_1_1_0 = 0.0f;
            if (col_3_0_1_1 < row_3_0_1_1_0) {
                if (row_3_0_1_1_0 < n_valid) {
                    float _exp2_56 = approx_exp2(g_row_3[0] - g_col_3_0_1_1);
                    val_3_0_1_1_0 = b_row_3[0] * _exp2_56 * acc[5];
                }
            }
            s_lt[col_3_0_1_1 * 64 + (row_3_0_1_1_0 ^ (col_3_0_1_1 >> 1 & 3) << 3)] = val_3_0_1_1_0;
            int row_3_0_1_1_1 = 56 + frag_r;
            float val_3_0_1_1_1 = 0.0f;
            if (col_3_0_1_1 < row_3_0_1_1_1) {
                if (row_3_0_1_1_1 < n_valid) {
                    float _exp2_57 = approx_exp2(g_row_3[1] - g_col_3_0_1_1);
                    val_3_0_1_1_1 = b_row_3[1] * _exp2_57 * acc[7];
                }
            }
            s_lt[col_3_0_1_1 * 64 + (row_3_0_1_1_1 ^ (col_3_0_1_1 >> 1 & 3) << 3)] = val_3_0_1_1_1;
            int col_3_1_0_0 = 16 + col_q;
            float g_col_3_1_0_0 = s_g[col_3_1_0_0];
            int row_3_1_0_0_0 = 48 + frag_r;
            float val_3_1_0_0_0 = 0.0f;
            if (col_3_1_0_0 < row_3_1_0_0_0) {
                if (row_3_1_0_0_0 < n_valid) {
                    float _exp2_58 = approx_exp2(g_row_3[0] - g_col_3_1_0_0);
                    val_3_1_0_0_0 = b_row_3[0] * _exp2_58 * acc[8];
                }
            }
            s_lt[col_3_1_0_0 * 64 + (row_3_1_0_0_0 ^ (col_3_1_0_0 >> 1 & 3) << 3)] = val_3_1_0_0_0;
            int row_3_1_0_0_1 = 56 + frag_r;
            float val_3_1_0_0_1 = 0.0f;
            if (col_3_1_0_0 < row_3_1_0_0_1) {
                if (row_3_1_0_0_1 < n_valid) {
                    float _exp2_59 = approx_exp2(g_row_3[1] - g_col_3_1_0_0);
                    val_3_1_0_0_1 = b_row_3[1] * _exp2_59 * acc[10];
                }
            }
            s_lt[col_3_1_0_0 * 64 + (row_3_1_0_0_1 ^ (col_3_1_0_0 >> 1 & 3) << 3)] = val_3_1_0_0_1;
            int col_3_1_0_1 = 17 + col_q;
            float g_col_3_1_0_1 = s_g[col_3_1_0_1];
            int row_3_1_0_1_0 = 48 + frag_r;
            float val_3_1_0_1_0 = 0.0f;
            if (col_3_1_0_1 < row_3_1_0_1_0) {
                if (row_3_1_0_1_0 < n_valid) {
                    float _exp2_60 = approx_exp2(g_row_3[0] - g_col_3_1_0_1);
                    val_3_1_0_1_0 = b_row_3[0] * _exp2_60 * acc[9];
                }
            }
            s_lt[col_3_1_0_1 * 64 + (row_3_1_0_1_0 ^ (col_3_1_0_1 >> 1 & 3) << 3)] = val_3_1_0_1_0;
            int row_3_1_0_1_1 = 56 + frag_r;
            float val_3_1_0_1_1 = 0.0f;
            if (col_3_1_0_1 < row_3_1_0_1_1) {
                if (row_3_1_0_1_1 < n_valid) {
                    float _exp2_61 = approx_exp2(g_row_3[1] - g_col_3_1_0_1);
                    val_3_1_0_1_1 = b_row_3[1] * _exp2_61 * acc[11];
                }
            }
            s_lt[col_3_1_0_1 * 64 + (row_3_1_0_1_1 ^ (col_3_1_0_1 >> 1 & 3) << 3)] = val_3_1_0_1_1;
            int col_3_1_1_0 = 24 + col_q;
            float g_col_3_1_1_0 = s_g[col_3_1_1_0];
            int row_3_1_1_0_0 = 48 + frag_r;
            float val_3_1_1_0_0 = 0.0f;
            if (col_3_1_1_0 < row_3_1_1_0_0) {
                if (row_3_1_1_0_0 < n_valid) {
                    float _exp2_62 = approx_exp2(g_row_3[0] - g_col_3_1_1_0);
                    val_3_1_1_0_0 = b_row_3[0] * _exp2_62 * acc[12];
                }
            }
            s_lt[col_3_1_1_0 * 64 + (row_3_1_1_0_0 ^ (col_3_1_1_0 >> 1 & 3) << 3)] = val_3_1_1_0_0;
            int row_3_1_1_0_1 = 56 + frag_r;
            float val_3_1_1_0_1 = 0.0f;
            if (col_3_1_1_0 < row_3_1_1_0_1) {
                if (row_3_1_1_0_1 < n_valid) {
                    float _exp2_63 = approx_exp2(g_row_3[1] - g_col_3_1_1_0);
                    val_3_1_1_0_1 = b_row_3[1] * _exp2_63 * acc[14];
                }
            }
            s_lt[col_3_1_1_0 * 64 + (row_3_1_1_0_1 ^ (col_3_1_1_0 >> 1 & 3) << 3)] = val_3_1_1_0_1;
            int col_3_1_1_1 = 25 + col_q;
            float g_col_3_1_1_1 = s_g[col_3_1_1_1];
            int row_3_1_1_1_0 = 48 + frag_r;
            float val_3_1_1_1_0 = 0.0f;
            if (col_3_1_1_1 < row_3_1_1_1_0) {
                if (row_3_1_1_1_0 < n_valid) {
                    float _exp2_64 = approx_exp2(g_row_3[0] - g_col_3_1_1_1);
                    val_3_1_1_1_0 = b_row_3[0] * _exp2_64 * acc[13];
                }
            }
            s_lt[col_3_1_1_1 * 64 + (row_3_1_1_1_0 ^ (col_3_1_1_1 >> 1 & 3) << 3)] = val_3_1_1_1_0;
            int row_3_1_1_1_1 = 56 + frag_r;
            float val_3_1_1_1_1 = 0.0f;
            if (col_3_1_1_1 < row_3_1_1_1_1) {
                if (row_3_1_1_1_1 < n_valid) {
                    float _exp2_65 = approx_exp2(g_row_3[1] - g_col_3_1_1_1);
                    val_3_1_1_1_1 = b_row_3[1] * _exp2_65 * acc[15];
                }
            }
            s_lt[col_3_1_1_1 * 64 + (row_3_1_1_1_1 ^ (col_3_1_1_1 >> 1 & 3) << 3)] = val_3_1_1_1_1;
            int col_3_2_0_0 = 32 + col_q;
            float g_col_3_2_0_0 = s_g[col_3_2_0_0];
            int row_3_2_0_0_0 = 48 + frag_r;
            float val_3_2_0_0_0 = 0.0f;
            if (col_3_2_0_0 < row_3_2_0_0_0) {
                if (row_3_2_0_0_0 < n_valid) {
                    float _exp2_66 = approx_exp2(g_row_3[0] - g_col_3_2_0_0);
                    val_3_2_0_0_0 = b_row_3[0] * _exp2_66 * acc[16];
                }
            }
            s_lt[col_3_2_0_0 * 64 + (row_3_2_0_0_0 ^ (col_3_2_0_0 >> 1 & 3) << 3)] = val_3_2_0_0_0;
            int row_3_2_0_0_1 = 56 + frag_r;
            float val_3_2_0_0_1 = 0.0f;
            if (col_3_2_0_0 < row_3_2_0_0_1) {
                if (row_3_2_0_0_1 < n_valid) {
                    float _exp2_67 = approx_exp2(g_row_3[1] - g_col_3_2_0_0);
                    val_3_2_0_0_1 = b_row_3[1] * _exp2_67 * acc[18];
                }
            }
            s_lt[col_3_2_0_0 * 64 + (row_3_2_0_0_1 ^ (col_3_2_0_0 >> 1 & 3) << 3)] = val_3_2_0_0_1;
            int col_3_2_0_1 = 33 + col_q;
            float g_col_3_2_0_1 = s_g[col_3_2_0_1];
            int row_3_2_0_1_0 = 48 + frag_r;
            float val_3_2_0_1_0 = 0.0f;
            if (col_3_2_0_1 < row_3_2_0_1_0) {
                if (row_3_2_0_1_0 < n_valid) {
                    float _exp2_68 = approx_exp2(g_row_3[0] - g_col_3_2_0_1);
                    val_3_2_0_1_0 = b_row_3[0] * _exp2_68 * acc[17];
                }
            }
            s_lt[col_3_2_0_1 * 64 + (row_3_2_0_1_0 ^ (col_3_2_0_1 >> 1 & 3) << 3)] = val_3_2_0_1_0;
            int row_3_2_0_1_1 = 56 + frag_r;
            float val_3_2_0_1_1 = 0.0f;
            if (col_3_2_0_1 < row_3_2_0_1_1) {
                if (row_3_2_0_1_1 < n_valid) {
                    float _exp2_69 = approx_exp2(g_row_3[1] - g_col_3_2_0_1);
                    val_3_2_0_1_1 = b_row_3[1] * _exp2_69 * acc[19];
                }
            }
            s_lt[col_3_2_0_1 * 64 + (row_3_2_0_1_1 ^ (col_3_2_0_1 >> 1 & 3) << 3)] = val_3_2_0_1_1;
            int col_3_2_1_0 = 40 + col_q;
            float g_col_3_2_1_0 = s_g[col_3_2_1_0];
            int row_3_2_1_0_0 = 48 + frag_r;
            float val_3_2_1_0_0 = 0.0f;
            if (col_3_2_1_0 < row_3_2_1_0_0) {
                if (row_3_2_1_0_0 < n_valid) {
                    float _exp2_70 = approx_exp2(g_row_3[0] - g_col_3_2_1_0);
                    val_3_2_1_0_0 = b_row_3[0] * _exp2_70 * acc[20];
                }
            }
            s_lt[col_3_2_1_0 * 64 + (row_3_2_1_0_0 ^ (col_3_2_1_0 >> 1 & 3) << 3)] = val_3_2_1_0_0;
            int row_3_2_1_0_1 = 56 + frag_r;
            float val_3_2_1_0_1 = 0.0f;
            if (col_3_2_1_0 < row_3_2_1_0_1) {
                if (row_3_2_1_0_1 < n_valid) {
                    float _exp2_71 = approx_exp2(g_row_3[1] - g_col_3_2_1_0);
                    val_3_2_1_0_1 = b_row_3[1] * _exp2_71 * acc[22];
                }
            }
            s_lt[col_3_2_1_0 * 64 + (row_3_2_1_0_1 ^ (col_3_2_1_0 >> 1 & 3) << 3)] = val_3_2_1_0_1;
            int col_3_2_1_1 = 41 + col_q;
            float g_col_3_2_1_1 = s_g[col_3_2_1_1];
            int row_3_2_1_1_0 = 48 + frag_r;
            float val_3_2_1_1_0 = 0.0f;
            if (col_3_2_1_1 < row_3_2_1_1_0) {
                if (row_3_2_1_1_0 < n_valid) {
                    float _exp2_72 = approx_exp2(g_row_3[0] - g_col_3_2_1_1);
                    val_3_2_1_1_0 = b_row_3[0] * _exp2_72 * acc[21];
                }
            }
            s_lt[col_3_2_1_1 * 64 + (row_3_2_1_1_0 ^ (col_3_2_1_1 >> 1 & 3) << 3)] = val_3_2_1_1_0;
            int row_3_2_1_1_1 = 56 + frag_r;
            float val_3_2_1_1_1 = 0.0f;
            if (col_3_2_1_1 < row_3_2_1_1_1) {
                if (row_3_2_1_1_1 < n_valid) {
                    float _exp2_73 = approx_exp2(g_row_3[1] - g_col_3_2_1_1);
                    val_3_2_1_1_1 = b_row_3[1] * _exp2_73 * acc[23];
                }
            }
            s_lt[col_3_2_1_1 * 64 + (row_3_2_1_1_1 ^ (col_3_2_1_1 >> 1 & 3) << 3)] = val_3_2_1_1_1;
            int col_3_3_0_0 = 48 + col_q;
            float g_col_3_3_0_0 = s_g[col_3_3_0_0];
            int row_3_3_0_0_0 = 48 + frag_r;
            float val_3_3_0_0_0 = 0.0f;
            if (col_3_3_0_0 < row_3_3_0_0_0) {
                if (row_3_3_0_0_0 < n_valid) {
                    float _exp2_74 = approx_exp2(g_row_3[0] - g_col_3_3_0_0);
                    val_3_3_0_0_0 = b_row_3[0] * _exp2_74 * acc[24];
                }
            }
            s_lt[col_3_3_0_0 * 64 + (row_3_3_0_0_0 ^ (col_3_3_0_0 >> 1 & 3) << 3)] = val_3_3_0_0_0;
            int row_3_3_0_0_1 = 56 + frag_r;
            float val_3_3_0_0_1 = 0.0f;
            if (col_3_3_0_0 < row_3_3_0_0_1) {
                if (row_3_3_0_0_1 < n_valid) {
                    float _exp2_75 = approx_exp2(g_row_3[1] - g_col_3_3_0_0);
                    val_3_3_0_0_1 = b_row_3[1] * _exp2_75 * acc[26];
                }
            }
            s_lt[col_3_3_0_0 * 64 + (row_3_3_0_0_1 ^ (col_3_3_0_0 >> 1 & 3) << 3)] = val_3_3_0_0_1;
            int col_3_3_0_1 = 49 + col_q;
            float g_col_3_3_0_1 = s_g[col_3_3_0_1];
            int row_3_3_0_1_0 = 48 + frag_r;
            float val_3_3_0_1_0 = 0.0f;
            if (col_3_3_0_1 < row_3_3_0_1_0) {
                if (row_3_3_0_1_0 < n_valid) {
                    float _exp2_76 = approx_exp2(g_row_3[0] - g_col_3_3_0_1);
                    val_3_3_0_1_0 = b_row_3[0] * _exp2_76 * acc[25];
                }
            }
            s_lt[col_3_3_0_1 * 64 + (row_3_3_0_1_0 ^ (col_3_3_0_1 >> 1 & 3) << 3)] = val_3_3_0_1_0;
            int row_3_3_0_1_1 = 56 + frag_r;
            float val_3_3_0_1_1 = 0.0f;
            if (col_3_3_0_1 < row_3_3_0_1_1) {
                if (row_3_3_0_1_1 < n_valid) {
                    float _exp2_77 = approx_exp2(g_row_3[1] - g_col_3_3_0_1);
                    val_3_3_0_1_1 = b_row_3[1] * _exp2_77 * acc[27];
                }
            }
            s_lt[col_3_3_0_1 * 64 + (row_3_3_0_1_1 ^ (col_3_3_0_1 >> 1 & 3) << 3)] = val_3_3_0_1_1;
            int col_3_3_1_0 = 56 + col_q;
            float g_col_3_3_1_0 = s_g[col_3_3_1_0];
            int row_3_3_1_0_0 = 48 + frag_r;
            float val_3_3_1_0_0 = 0.0f;
            if (col_3_3_1_0 < row_3_3_1_0_0) {
                if (row_3_3_1_0_0 < n_valid) {
                    float _exp2_78 = approx_exp2(g_row_3[0] - g_col_3_3_1_0);
                    val_3_3_1_0_0 = b_row_3[0] * _exp2_78 * acc[28];
                }
            }
            s_lt[col_3_3_1_0 * 64 + (row_3_3_1_0_0 ^ (col_3_3_1_0 >> 1 & 3) << 3)] = val_3_3_1_0_0;
            int row_3_3_1_0_1 = 56 + frag_r;
            float val_3_3_1_0_1 = 0.0f;
            if (col_3_3_1_0 < row_3_3_1_0_1) {
                if (row_3_3_1_0_1 < n_valid) {
                    float _exp2_79 = approx_exp2(g_row_3[1] - g_col_3_3_1_0);
                    val_3_3_1_0_1 = b_row_3[1] * _exp2_79 * acc[30];
                }
            }
            s_lt[col_3_3_1_0 * 64 + (row_3_3_1_0_1 ^ (col_3_3_1_0 >> 1 & 3) << 3)] = val_3_3_1_0_1;
            int col_3_3_1_1 = 57 + col_q;
            float g_col_3_3_1_1 = s_g[col_3_3_1_1];
            int row_3_3_1_1_0 = 48 + frag_r;
            float val_3_3_1_1_0 = 0.0f;
            if (col_3_3_1_1 < row_3_3_1_1_0) {
                if (row_3_3_1_1_0 < n_valid) {
                    float _exp2_80 = approx_exp2(g_row_3[0] - g_col_3_3_1_1);
                    val_3_3_1_1_0 = b_row_3[0] * _exp2_80 * acc[29];
                }
            }
            s_lt[col_3_3_1_1 * 64 + (row_3_3_1_1_0 ^ (col_3_3_1_1 >> 1 & 3) << 3)] = val_3_3_1_1_0;
            int row_3_3_1_1_1 = 56 + frag_r;
            float val_3_3_1_1_1 = 0.0f;
            if (col_3_3_1_1 < row_3_3_1_1_1) {
                if (row_3_3_1_1_1 < n_valid) {
                    float _exp2_81 = approx_exp2(g_row_3[1] - g_col_3_3_1_1);
                    val_3_3_1_1_1 = b_row_3[1] * _exp2_81 * acc[31];
                }
            }
            s_lt[col_3_3_1_1 * 64 + (row_3_3_1_1_1 ^ (col_3_3_1_1 >> 1 & 3) << 3)] = val_3_3_1_1_1;
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncthreads();
        float zero4[4];
        zero4[0] = 0.0f;
        zero4[1] = 0.0f;
        zero4[2] = 0.0f;
        zero4[3] = 0.0f;
        int z_half = tid_2 / 64;
        int zr = tid_2 % 64 / 4;
        int zc = tid_2 % 4 * 4;
        int zbi = 0;
        int zbj = 1 + z_half;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)(((zbi * 16 + zr) * 64 + (zbj * 16 + zc ^ (zbi * 16 + zr >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(zero4[0])), "r"(__float_as_uint(zero4[1])), "r"(__float_as_uint(zero4[2])), "r"(__float_as_uint(zero4[3])));
        int zbi_0 = z_half;
        int zbj_1 = 3 + -1 * z_half;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)(((zbi_0 * 16 + zr) * 64 + (zbj_1 * 16 + zc ^ (zbi_0 * 16 + zr >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(zero4[0])), "r"(__float_as_uint(zero4[1])), "r"(__float_as_uint(zero4[2])), "r"(__float_as_uint(zero4[3])));
        int zbi_2 = 1 + z_half;
        int zbj_3 = 3;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)(((zbi_2 * 16 + zr) * 64 + (zbj_3 * 16 + zc ^ (zbi_2 * 16 + zr >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(zero4[0])), "r"(__float_as_uint(zero4[1])), "r"(__float_as_uint(zero4[2])), "r"(__float_as_uint(zero4[3])));
        if (tid_2 < 64) {
            int dblk = tid_2 / 16;
            int dcol = tid_2 % 16;
            int drow0 = dblk * 16;
            int lt_d[2];
            lt_d[0] = drow0 * 65;
            lt_d[1] = drow0 * 64 + (drow0 ^ 16);
            int x_d[2];
            x_d[0] = drow0 * 65 + dcol;
            x_d[1] = drow0 * 64 + (drow0 ^ 16) + dcol;
            float xcol[16];
            #pragma unroll
            for (int di = 0; di < 16; di++) {
                float xv = 0.0f;
                if (di == dcol) {
                    xv = 1.0f;
                }
                #pragma unroll
                for (int dj = 0; dj < di; dj++) {
                    xv = __fmaf_rn(-s_lt[lt_d[(dj >> 1 & 3) << 3 >> 4 & 1] + (dj * 64 + (di ^ (dj >> 1 & 3) << 3 & 8))], xcol[dj], xv);
                }
                xcol[di] = xv;
            }
            #pragma unroll
            for (int di2 = 0; di2 < 16; di2++) {
                s_x[x_d[di2 >> 2 & 1] + di2 * 64] = xcol[di2];
            }
        } else {
            int s_row0 = (tid_2 - 64) / 16;
            int s_col = (tid_2 - 64) % 16 * 8;
            int base[2];
            base[0] = s_col / 64 * 8192 + s_row0 * 128 + (s_col % 64 * 2 ^ s_row0 << 4);
            base[1] = s_col / 64 * 8192 + s_row0 * 128 + (s_col % 64 * 2 ^ s_row0 + 4 << 4);
            float _exp2_0 = approx_exp2(s_g[s_row0]);
            int o_k0 = base[0] + 0;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k0));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0] * _exp2_0);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0] * _exp2_0);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0] * _exp2_0);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0] * _exp2_0);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0] * _exp2_0);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0] * _exp2_0);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0] * _exp2_0);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0] * _exp2_0);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k0), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_1 = approx_exp2(s_g[s_row0 + 4]);
            int o_k1 = base[1] + 512;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k1));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 4] * _exp2_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 4] * _exp2_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 4] * _exp2_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 4] * _exp2_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 4] * _exp2_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 4] * _exp2_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 4] * _exp2_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 4] * _exp2_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_2_1 = approx_exp2(s_g[s_row0 + 8]);
            int o_k2 = base[0] + 1024;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k2));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 8] * _exp2_2_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k2), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_3_1 = approx_exp2(s_g[s_row0 + 12]);
            int o_k3 = base[1] + 1536;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k3));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 12] * _exp2_3_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k3), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_4_1 = approx_exp2(s_g[s_row0 + 16]);
            int o_k4 = base[0] + 2048;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k4));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 16] * _exp2_4_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k4), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_5_1 = approx_exp2(s_g[s_row0 + 20]);
            int o_k5 = base[1] + 2560;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k5));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 20] * _exp2_5_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k5), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_6_1 = approx_exp2(s_g[s_row0 + 24]);
            int o_k6 = base[0] + 3072;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k6));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 24] * _exp2_6_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k6), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_7_1 = approx_exp2(s_g[s_row0 + 28]);
            int o_k7 = base[1] + 3584;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k7));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 28] * _exp2_7_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k7), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_8_1 = approx_exp2(s_g[s_row0 + 32]);
            int o_k8 = base[0] + 4096;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k8));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 32] * _exp2_8_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k8), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_9_1 = approx_exp2(s_g[s_row0 + 36]);
            int o_k9 = base[1] + 4608;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k9));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 36] * _exp2_9_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k9), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_10_1 = approx_exp2(s_g[s_row0 + 40]);
            int o_k10 = base[0] + 5120;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k10));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 40] * _exp2_10_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k10), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_11_1 = approx_exp2(s_g[s_row0 + 44]);
            int o_k11 = base[1] + 5632;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k11));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 44] * _exp2_11_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k11), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_12_1 = approx_exp2(s_g[s_row0 + 48]);
            int o_k12 = base[0] + 6144;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k12));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 48] * _exp2_12_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k12), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_13_1 = approx_exp2(s_g[s_row0 + 52]);
            int o_k13 = base[1] + 6656;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k13));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 52] * _exp2_13_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k13), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_14_1 = approx_exp2(s_g[s_row0 + 56]);
            int o_k14 = base[0] + 7168;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k14));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 56] * _exp2_14_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k14), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_15_1 = approx_exp2(s_g[s_row0 + 60]);
            int o_k15 = base[1] + 7680;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_k15));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 60] * _exp2_15_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_k15), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        }
        __syncthreads();
        float acc8[8];
        float lv[2];
        float xv4[4];
        float xi8[8];
        float tv16[16];
        int tr0 = lane_0 / 8 / 2 * 8 + lane_0 / 8 % 2 * 2 + lane_0 % 8 / 4 * 4;
        int tc0 = lane_0 % 4 * 4;
        int r3 = warp_1 * 4 + lane_0 / 8;
        int c3 = lane_0 % 8 * 2;
        if (warp_1 < 3) {
            acc8[0] = 0.0f;
            acc8[1] = 0.0f;
            acc8[2] = 0.0f;
            acc8[3] = 0.0f;
            acc8[4] = 0.0f;
            acc8[5] = 0.0f;
            acc8[6] = 0.0f;
            acc8[7] = 0.0f;
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((warp_1 * 16 * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((warp_1 * 16 * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 1 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 1 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 2 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 2 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 3 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 3 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 4 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 4 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 5 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 5 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 6 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 6 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 7 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 7 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 8 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 8 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 9 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 9 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 10 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 10 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 11 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 11 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 12 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 12 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 13 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 13 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 14 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 14 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 15 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 15 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_t_addr + (unsigned int)((warp_1 * 320 + tr0 * 20 + tc0) * 4)), "r"(__float_as_uint(acc8[0])), "r"(__float_as_uint(acc8[1])), "r"(__float_as_uint(acc8[2])), "r"(__float_as_uint(acc8[3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_t_addr + (unsigned int)((warp_1 * 320 + (tr0 + 1) * 20 + tc0) * 4)), "r"(__float_as_uint(acc8[4])), "r"(__float_as_uint(acc8[5])), "r"(__float_as_uint(acc8[6])), "r"(__float_as_uint(acc8[7])));
        }
        __syncthreads();
        if (warp_1 < 3) {
            acc8[0] = 0.0f;
            acc8[1] = 0.0f;
            acc8[2] = 0.0f;
            acc8[3] = 0.0f;
            acc8[4] = 0.0f;
            acc8[5] = 0.0f;
            acc8[6] = 0.0f;
            acc8[7] = 0.0f;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0) * 64 + ((warp_1 + 1) * 16 ^ ((warp_1 + 1) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0 + 1) * 64 + ((warp_1 + 1) * 16 ^ ((warp_1 + 1) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 20 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 40 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 60 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0) * 64 + ((warp_1 + 1) * 16 + 4 ^ ((warp_1 + 1) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0 + 1) * 64 + ((warp_1 + 1) * 16 + 4 ^ ((warp_1 + 1) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 80 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 100 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 120 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 140 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0) * 64 + ((warp_1 + 1) * 16 + 8 ^ ((warp_1 + 1) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0 + 1) * 64 + ((warp_1 + 1) * 16 + 8 ^ ((warp_1 + 1) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 160 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 180 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 200 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 220 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0) * 64 + ((warp_1 + 1) * 16 + 12 ^ ((warp_1 + 1) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0 + 1) * 64 + ((warp_1 + 1) * 16 + 12 ^ ((warp_1 + 1) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 240 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 260 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 280 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 300 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            acc8[0] = -acc8[0];
            acc8[1] = -acc8[1];
            acc8[2] = -acc8[2];
            acc8[3] = -acc8[3];
            acc8[4] = -acc8[4];
            acc8[5] = -acc8[5];
            acc8[6] = -acc8[6];
            acc8[7] = -acc8[7];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + tr0 >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(acc8[0])), "r"(__float_as_uint(acc8[1])), "r"(__float_as_uint(acc8[2])), "r"(__float_as_uint(acc8[3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0 + 1) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(acc8[4])), "r"(__float_as_uint(acc8[5])), "r"(__float_as_uint(acc8[6])), "r"(__float_as_uint(acc8[7])));
        }
        __syncthreads();
        if (warp_1 < 2) {
            acc8[0] = 0.0f;
            acc8[1] = 0.0f;
            acc8[2] = 0.0f;
            acc8[3] = 0.0f;
            acc8[4] = 0.0f;
            acc8[5] = 0.0f;
            acc8[6] = 0.0f;
            acc8[7] = 0.0f;
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((warp_1 * 16 * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((warp_1 * 16 * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 + 1) * 16 * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 + 1) * 16 * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 1 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 1 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 1) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 1 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 1) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 1 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 2 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 2 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 2) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 2 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 2) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 2 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 3 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 3 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 3) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 3 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 3) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 3 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 4 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 4 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 4) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 4 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 4) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 4 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 5 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 5 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 5) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 5 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 5) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 5 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 6 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 6 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 6) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 6 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 6) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 6 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 7 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 7 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 7) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 7 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 7) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 7 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 8 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 8 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 8) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 8 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 8) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 8 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 9 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 9 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 9) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 9 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 9) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 9 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 10 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 10 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 10) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 10 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 10) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 10 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 11 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 11 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 11) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 11 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 11) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 11 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 12 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 12 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 12) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 12 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 12) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 12 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 13 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 13 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 13) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 13 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 13) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 13 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 14 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 14 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 14) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 14 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 14) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 14 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 15 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 15 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 15) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 15 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 15) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 15 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_t_addr + (unsigned int)((warp_1 * 320 + tr0 * 20 + tc0) * 4)), "r"(__float_as_uint(acc8[0])), "r"(__float_as_uint(acc8[1])), "r"(__float_as_uint(acc8[2])), "r"(__float_as_uint(acc8[3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_t_addr + (unsigned int)((warp_1 * 320 + (tr0 + 1) * 20 + tc0) * 4)), "r"(__float_as_uint(acc8[4])), "r"(__float_as_uint(acc8[5])), "r"(__float_as_uint(acc8[6])), "r"(__float_as_uint(acc8[7])));
        }
        __syncthreads();
        if (warp_1 < 2) {
            acc8[0] = 0.0f;
            acc8[1] = 0.0f;
            acc8[2] = 0.0f;
            acc8[3] = 0.0f;
            acc8[4] = 0.0f;
            acc8[5] = 0.0f;
            acc8[6] = 0.0f;
            acc8[7] = 0.0f;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0) * 64 + ((warp_1 + 2) * 16 ^ ((warp_1 + 2) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0 + 1) * 64 + ((warp_1 + 2) * 16 ^ ((warp_1 + 2) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 20 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 40 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 60 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0) * 64 + ((warp_1 + 2) * 16 + 4 ^ ((warp_1 + 2) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0 + 1) * 64 + ((warp_1 + 2) * 16 + 4 ^ ((warp_1 + 2) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 80 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 100 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 120 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 140 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0) * 64 + ((warp_1 + 2) * 16 + 8 ^ ((warp_1 + 2) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0 + 1) * 64 + ((warp_1 + 2) * 16 + 8 ^ ((warp_1 + 2) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 160 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 180 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 200 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 220 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0) * 64 + ((warp_1 + 2) * 16 + 12 ^ ((warp_1 + 2) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0 + 1) * 64 + ((warp_1 + 2) * 16 + 12 ^ ((warp_1 + 2) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 240 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 260 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 280 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 300 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            acc8[0] = -acc8[0];
            acc8[1] = -acc8[1];
            acc8[2] = -acc8[2];
            acc8[3] = -acc8[3];
            acc8[4] = -acc8[4];
            acc8[5] = -acc8[5];
            acc8[6] = -acc8[6];
            acc8[7] = -acc8[7];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 2) * 16 + tr0 >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(acc8[0])), "r"(__float_as_uint(acc8[1])), "r"(__float_as_uint(acc8[2])), "r"(__float_as_uint(acc8[3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0 + 1) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 2) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(acc8[4])), "r"(__float_as_uint(acc8[5])), "r"(__float_as_uint(acc8[6])), "r"(__float_as_uint(acc8[7])));
        }
        __syncthreads();
        acc8[0] = 0.0f;
        acc8[1] = 0.0f;
        acc8[2] = 0.0f;
        acc8[3] = 0.0f;
        acc8[4] = 0.0f;
        acc8[5] = 0.0f;
        acc8[6] = 0.0f;
        acc8[7] = 0.0f;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((48 + r3 ^ 0) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((c3 ^ 0) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1024 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1024 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2048 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2048 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((64 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((64 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1088 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1088 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2112 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2112 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((128 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((128 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1152 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1152 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2176 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2176 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((192 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((192 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1216 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1216 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2240 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2240 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((256 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((256 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1280 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1280 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2304 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2304 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((320 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((320 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1344 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1344 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2368 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2368 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((384 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((384 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1408 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1408 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2432 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2432 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((448 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((448 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1472 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1472 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2496 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2496 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((512 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((512 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1536 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1536 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2560 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2560 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((576 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((576 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1600 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1600 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2624 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2624 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((640 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((640 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1664 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1664 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2688 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2688 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((704 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((704 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1728 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1728 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2752 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2752 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((768 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((768 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1792 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1792 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2816 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2816 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((832 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((832 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1856 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1856 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2880 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2880 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((896 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((896 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1920 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1920 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2944 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2944 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((960 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((960 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1984 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1984 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((3008 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((3008 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_t_addr + (unsigned int)((r3 * 20 + c3) * 4)), "r"(__float_as_uint(acc8[0])), "r"(__float_as_uint(acc8[1])));
        __syncthreads();
        acc8[0] = 0.0f;
        acc8[1] = 0.0f;
        acc8[2] = 0.0f;
        acc8[3] = 0.0f;
        acc8[4] = 0.0f;
        acc8[5] = 0.0f;
        acc8[6] = 0.0f;
        acc8[7] = 0.0f;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
            : "r"(s_x_addr + (unsigned int)(((48 + r3) * 64 + (48 ^ (48 + r3 >> 2 & 1) << 4)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1]))
            : "r"(s_t_addr + (unsigned int)(c3 * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(2) + 1]))
            : "r"(s_t_addr + (unsigned int)((20 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1]))
            : "r"(s_t_addr + (unsigned int)((40 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[6])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(6) + 1]))
            : "r"(s_t_addr + (unsigned int)((60 + c3) * 4)));
        acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[1], tv16[2], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[1], tv16[3], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[2], tv16[4], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[2], tv16[5], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[3], tv16[6], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[3], tv16[7], acc8[1]);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
            : "r"(s_x_addr + (unsigned int)(((48 + r3) * 64 + (52 ^ (48 + r3 >> 2 & 1) << 4)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1]))
            : "r"(s_t_addr + (unsigned int)((80 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(2) + 1]))
            : "r"(s_t_addr + (unsigned int)((100 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1]))
            : "r"(s_t_addr + (unsigned int)((120 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[6])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(6) + 1]))
            : "r"(s_t_addr + (unsigned int)((140 + c3) * 4)));
        acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[1], tv16[2], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[1], tv16[3], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[2], tv16[4], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[2], tv16[5], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[3], tv16[6], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[3], tv16[7], acc8[1]);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
            : "r"(s_x_addr + (unsigned int)(((48 + r3) * 64 + (56 ^ (48 + r3 >> 2 & 1) << 4)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1]))
            : "r"(s_t_addr + (unsigned int)((160 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(2) + 1]))
            : "r"(s_t_addr + (unsigned int)((180 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1]))
            : "r"(s_t_addr + (unsigned int)((200 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[6])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(6) + 1]))
            : "r"(s_t_addr + (unsigned int)((220 + c3) * 4)));
        acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[1], tv16[2], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[1], tv16[3], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[2], tv16[4], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[2], tv16[5], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[3], tv16[6], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[3], tv16[7], acc8[1]);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
            : "r"(s_x_addr + (unsigned int)(((48 + r3) * 64 + (60 ^ (48 + r3 >> 2 & 1) << 4)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1]))
            : "r"(s_t_addr + (unsigned int)((240 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(2) + 1]))
            : "r"(s_t_addr + (unsigned int)((260 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1]))
            : "r"(s_t_addr + (unsigned int)((280 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[6])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(6) + 1]))
            : "r"(s_t_addr + (unsigned int)((300 + c3) * 4)));
        acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[1], tv16[2], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[1], tv16[3], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[2], tv16[4], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[2], tv16[5], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[3], tv16[6], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[3], tv16[7], acc8[1]);
        acc8[0] = -acc8[0];
        acc8[1] = -acc8[1];
        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_x_addr + (unsigned int)(((48 + r3) * 64 + (c3 ^ (48 + r3 >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(acc8[0])), "r"(__float_as_uint(acc8[1])));
        __syncthreads();
        float quad[4];
        unsigned int pk2[2];
        int p_row = tid_2 / 16;
        int p_col = tid_2 % 16 * 4;
        unsigned int x_pub = s_x_addr + (unsigned int)((p_row * 64 + (p_col ^ (p_row >> 2 & 1) << 4)) * 4);
        int a_pub = p_row * 128 + (p_col * 2 ^ p_row << 4);
        long long a_idx0 = ((long long)(tok0 + p_row) * (long long)num_v_heads + (long long)hv) * 64 + (long long)p_col;
        long long a_rows8 = (long long)num_v_heads * 512;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&quad[0])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 3]))
                : "r"(x_pub + (unsigned int)(t * 8 * 64 * 4)));
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(quad[_lp*2 + 0], quad[_lp*2+1 + 0]));
                pk2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_a_addr + (unsigned int)(a_pub + t * 1024)), "r"(pk2[0]), "r"(pk2[1]));
            if (n_valid > p_row + t * 8) {
                unsigned int _packed_0[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(quad[_lp*2 + 0], quad[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(quad[_lp*2 + 2], quad[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(A_out + (a_idx0 + a_rows8 * (long long)t))[0] = reinterpret_cast<int2*>(_packed_0)[0];
            }
        }
    }
    if (recompute != 0) {
        int s_row0_1 = tid_2 / 16;
        int s_col_1 = tid_2 % 16 * 8;
        int base_1[1];
        base_1[0] = s_col_1 / 64 * 8192 + s_row0_1 * 128 + (s_col_1 % 64 * 2 ^ s_row0_1 << 4);
        float _exp2_16_1 = approx_exp2(s_g[s_row0_1]);
        int o_r0 = base_1[0] + 0;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_r0));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1] * _exp2_16_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_r0), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_17_1 = approx_exp2(s_g[s_row0_1 + 8]);
        int o_r1 = base_1[0] + 1024;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_r1));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_r1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_18_1 = approx_exp2(s_g[s_row0_1 + 16]);
        int o_r2 = base_1[0] + 2048;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_r2));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_r2), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_19_1 = approx_exp2(s_g[s_row0_1 + 24]);
        int o_r3 = base_1[0] + 3072;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_r3));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_r3), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_20_1 = approx_exp2(s_g[s_row0_1 + 32]);
        int o_r4 = base_1[0] + 4096;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_r4));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_r4), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_21_1 = approx_exp2(s_g[s_row0_1 + 40]);
        int o_r5 = base_1[0] + 5120;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_r5));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_r5), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_22_1 = approx_exp2(s_g[s_row0_1 + 48]);
        int o_r6 = base_1[0] + 6144;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_r6));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_r6), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_23_1 = approx_exp2(s_g[s_row0_1 + 56]);
        int o_r7 = base_1[0] + 7168;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_r7));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_r7), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned long long _mbarrier_arrival_token_82;
    asm volatile(
        "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
        : "=l"(_mbarrier_arrival_token_82)
        : "l"(smem_raw + ((lp_addr) - smem)), "r"((uint32_t)(1)) : "memory");
    __syncthreads();
    unsigned int _phase_lp_0 = 0;
    mbarrier_wait(lp_addr, _phase_lp_0);
    _phase_lp_0 ^= 1;
    if (recompute == 0) {
        int r0_4 = tid / 16;
        int c0_5 = tid % 16 * 8;
        long long row_stride_6 = (long long)num_v_heads * 128;
        long long src_row_7 = (long long)(tok0 + r0_4) * row_stride_6 + (long long)hv * 128 + (long long)c0_5;
        unsigned int dst0_8 = s_v_addr + (unsigned int)(c0_5 / 64 * 8192 + (r0_4 * 128 + c0_5 % 64 * 2 ^ (r0_4 * 128 + c0_5 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8), "l"(v + src_row_7), "r"((n_valid > r0_4) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 1024), "l"(v + (src_row_7 + 8 * row_stride_6)), "r"((n_valid > r0_4 + 8) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 2048), "l"(v + (src_row_7 + 16 * row_stride_6)), "r"((n_valid > r0_4 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 3072), "l"(v + (src_row_7 + 24 * row_stride_6)), "r"((n_valid > r0_4 + 24) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 4096), "l"(v + (src_row_7 + 32 * row_stride_6)), "r"((n_valid > r0_4 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 5120), "l"(v + (src_row_7 + 40 * row_stride_6)), "r"((n_valid > r0_4 + 40) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 6144), "l"(v + (src_row_7 + 48 * row_stride_6)), "r"((n_valid > r0_4 + 48) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 7168), "l"(v + (src_row_7 + 56 * row_stride_6)), "r"((n_valid > r0_4 + 56) ? 16 : 0));
        asm volatile("cp.async.commit_group;");
    }
    if (warp == 1) {
        uint32_t _elected_83 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_83));
        int _smem_desc_lo_84 = ((s_a_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_85 = (((s_knT_addr) >> 4) & 0x3FFF) | 0x2000000;
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
            :: "r"(_smem_desc_lo_84), "r"(_smem_desc_lo_85), "r"(tmem_acc_cols), "r"(0), "r"(_elected_83) : "memory");
        tcgen05_commit_leader(m2_addr, _elected_83);
    }
    unsigned int slot = s_stage_addr + (unsigned int)(warp_1 * 2048);
    int st_off = (lane_0 / 8 % 2 * 8 + lane_0 % 8) * 128;
    int sx = lane_0 / 16 ^ lane_0 % 8;
    int ld_row = lane_0 / 8;
    int ld_ch = lane_0 % 8;
    int row0 = warp_1 * 16;
    unsigned int _phase_m2_0 = 0;
    mbarrier_wait(m2_addr, _phase_m2_0);
    _phase_m2_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15])), "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
        : "r"(taddr_1 + 0));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_86 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 0) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_86), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 8], acc[_lp*2+1 + 8]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_87 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 2) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_87), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 16], acc[_lp*2+1 + 16]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_88 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 4) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_88), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 24], acc[_lp*2+1 + 24]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_89 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 6) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_89), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    __syncwarp();
    int row_o0_0_0 = ld_row + 0;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o0_0_0 * 128 + ((ld_ch ^ row_o0_0_0 & 7) << 4))));
    int orow0_0_0 = row0 + row_o0_0_0;
    if (orow0_0_0 < n_valid) {
        long long out_i0_0_0 = ((long long)(tok0 + orow0_0_0) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i0_0_0 / 2) = _iv4;
        }
    }
    int row_o0_0_1 = ld_row + 4;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o0_0_1 * 128 + ((ld_ch ^ row_o0_0_1 & 7) << 4))));
    int orow0_0_1 = row0 + row_o0_0_1;
    if (orow0_0_1 < n_valid) {
        long long out_i0_0_1 = ((long long)(tok0 + orow0_0_1) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i0_0_1 / 2) = _iv4;
        }
    }
    int row_o0_0_2 = ld_row + 8;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o0_0_2 * 128 + ((ld_ch ^ row_o0_0_2 & 7) << 4))));
    int orow0_0_2 = row0 + row_o0_0_2;
    if (orow0_0_2 < n_valid) {
        long long out_i0_0_2 = ((long long)(tok0 + orow0_0_2) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i0_0_2 / 2) = _iv4;
        }
    }
    int row_o0_0_3 = ld_row + 12;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o0_0_3 * 128 + ((ld_ch ^ row_o0_0_3 & 7) << 4))));
    int orow0_0_3 = row0 + row_o0_0_3;
    if (orow0_0_3 < n_valid) {
        long long out_i0_0_3 = ((long long)(tok0 + orow0_0_3) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i0_0_3 / 2) = _iv4;
        }
    }
    __syncwarp();
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15])), "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
        : "r"(taddr_1 + 64));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_90 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 0) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_90), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 8], acc[_lp*2+1 + 8]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_91 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 2) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_91), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 16], acc[_lp*2+1 + 16]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_92 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 4) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_92), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 24], acc[_lp*2+1 + 24]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_93 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 6) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_93), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    __syncwarp();
    int row_o0_1_0 = ld_row + 0;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o0_1_0 * 128 + ((ld_ch ^ row_o0_1_0 & 7) << 4))));
    int orow0_1_0 = row0 + row_o0_1_0;
    if (orow0_1_0 < n_valid) {
        long long out_i0_1_0 = ((long long)(tok0 + orow0_1_0) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i0_1_0 / 2) = _iv4;
        }
    }
    int row_o0_1_1 = ld_row + 4;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o0_1_1 * 128 + ((ld_ch ^ row_o0_1_1 & 7) << 4))));
    int orow0_1_1 = row0 + row_o0_1_1;
    if (orow0_1_1 < n_valid) {
        long long out_i0_1_1 = ((long long)(tok0 + orow0_1_1) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i0_1_1 / 2) = _iv4;
        }
    }
    int row_o0_1_2 = ld_row + 8;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o0_1_2 * 128 + ((ld_ch ^ row_o0_1_2 & 7) << 4))));
    int orow0_1_2 = row0 + row_o0_1_2;
    if (orow0_1_2 < n_valid) {
        long long out_i0_1_2 = ((long long)(tok0 + orow0_1_2) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i0_1_2 / 2) = _iv4;
        }
    }
    int row_o0_1_3 = ld_row + 12;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o0_1_3 * 128 + ((ld_ch ^ row_o0_1_3 & 7) << 4))));
    int orow0_1_3 = row0 + row_o0_1_3;
    if (orow0_1_3 < n_valid) {
        long long out_i0_1_3 = ((long long)(tok0 + orow0_1_3) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i0_1_3 / 2) = _iv4;
        }
    }
    __syncwarp();
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    int s_row0_v = tid_2 / 16;
    int s_col_v = tid_2 % 16 * 8;
    int base_v[1];
    base_v[0] = s_col_v / 64 * 8192 + s_row0_v * 128 + (s_col_v % 64 * 2 ^ s_row0_v << 4);
    int o_v0 = base_v[0] + 0;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_v0));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_v];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_v];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_v];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_v];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_v];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_v];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_v];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_v];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_v0), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_v1 = base_v[0] + 1024;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_v1));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_v + 8];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_v + 8];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_v + 8];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_v + 8];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_v + 8];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_v + 8];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_v + 8];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_v + 8];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_v1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_v2 = base_v[0] + 2048;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_v2));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_v + 16];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_v + 16];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_v + 16];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_v + 16];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_v + 16];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_v + 16];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_v + 16];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_v + 16];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_v2), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_v3 = base_v[0] + 3072;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_v3));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_v + 24];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_v + 24];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_v + 24];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_v + 24];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_v + 24];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_v + 24];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_v + 24];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_v + 24];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_v3), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_v4 = base_v[0] + 4096;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_v4));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_v + 32];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_v + 32];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_v + 32];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_v + 32];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_v + 32];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_v + 32];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_v + 32];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_v + 32];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_v4), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_v5 = base_v[0] + 5120;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_v5));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_v + 40];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_v + 40];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_v + 40];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_v + 40];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_v + 40];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_v + 40];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_v + 40];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_v + 40];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_v5), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_v6 = base_v[0] + 6144;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_v6));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_v + 48];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_v + 48];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_v + 48];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_v + 48];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_v + 48];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_v + 48];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_v + 48];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_v + 48];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_v6), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_v7 = base_v[0] + 7168;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_v7));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_v + 56];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_v + 56];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_v + 56];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_v + 56];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_v + 56];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_v + 56];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_v + 56];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_v + 56];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_v7), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    unsigned long long _mbarrier_arrival_token_94;
    asm volatile(
        "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
        : "=l"(_mbarrier_arrival_token_94)
        : "l"(smem_raw + ((l2_addr) - smem)), "r"((uint32_t)(1)) : "memory");
    __syncthreads();
    unsigned int _phase_l2_0 = 0;
    mbarrier_wait(l2_addr, _phase_l2_0);
    _phase_l2_0 ^= 1;
    if (warp == 2) {
        uint32_t _elected_95 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_95));
        int _smem_desc_lo_96 = ((s_a_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_97 = (((s_vT_addr) >> 4) & 0x3FFF) | 0x2000000;
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
            :: "r"(_smem_desc_lo_96), "r"(_smem_desc_lo_97), "r"((tmem_acc_cols + (1048576))), "r"(0), "r"(_elected_95) : "memory");
        tcgen05_commit_leader(m3_addr, _elected_95);
    }
    unsigned int _phase_m3_0 = 0;
    mbarrier_wait(m3_addr, _phase_m3_0);
    _phase_m3_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15])), "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
        : "r"(taddr_1 + 1048576 + 0));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_98 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 0) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_98), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 8], acc[_lp*2+1 + 8]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_99 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 2) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_99), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 16], acc[_lp*2+1 + 16]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_100 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 4) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_100), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 24], acc[_lp*2+1 + 24]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_101 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 6) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_101), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    __syncwarp();
    int row_o1_0_0 = ld_row + 0;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o1_0_0 * 128 + ((ld_ch ^ row_o1_0_0 & 7) << 4))));
    int orow1_0_0 = row0 + row_o1_0_0;
    if (orow1_0_0 < n_valid) {
        long long out_i1_0_0 = ((long long)(tok0 + orow1_0_0) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i1_0_0 / 2) = _iv4;
        }
    }
    int row_o1_0_1 = ld_row + 4;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o1_0_1 * 128 + ((ld_ch ^ row_o1_0_1 & 7) << 4))));
    int orow1_0_1 = row0 + row_o1_0_1;
    if (orow1_0_1 < n_valid) {
        long long out_i1_0_1 = ((long long)(tok0 + orow1_0_1) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i1_0_1 / 2) = _iv4;
        }
    }
    int row_o1_0_2 = ld_row + 8;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o1_0_2 * 128 + ((ld_ch ^ row_o1_0_2 & 7) << 4))));
    int orow1_0_2 = row0 + row_o1_0_2;
    if (orow1_0_2 < n_valid) {
        long long out_i1_0_2 = ((long long)(tok0 + orow1_0_2) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i1_0_2 / 2) = _iv4;
        }
    }
    int row_o1_0_3 = ld_row + 12;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o1_0_3 * 128 + ((ld_ch ^ row_o1_0_3 & 7) << 4))));
    int orow1_0_3 = row0 + row_o1_0_3;
    if (orow1_0_3 < n_valid) {
        long long out_i1_0_3 = ((long long)(tok0 + orow1_0_3) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i1_0_3 / 2) = _iv4;
        }
    }
    __syncwarp();
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15])), "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
        : "r"(taddr_1 + 1048576 + 64));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_102 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 0) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_102), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 8], acc[_lp*2+1 + 8]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_103 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 2) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_103), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 16], acc[_lp*2+1 + 16]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_104 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 4) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_104), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 24], acc[_lp*2+1 + 24]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    uint32_t _stmatrix_addr_105 = static_cast<uint32_t>(slot + (unsigned int)(st_off + ((sx ^ 6) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_105), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    __syncwarp();
    int row_o1_1_0 = ld_row + 0;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o1_1_0 * 128 + ((ld_ch ^ row_o1_1_0 & 7) << 4))));
    int orow1_1_0 = row0 + row_o1_1_0;
    if (orow1_1_0 < n_valid) {
        long long out_i1_1_0 = ((long long)(tok0 + orow1_1_0) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i1_1_0 / 2) = _iv4;
        }
    }
    int row_o1_1_1 = ld_row + 4;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o1_1_1 * 128 + ((ld_ch ^ row_o1_1_1 & 7) << 4))));
    int orow1_1_1 = row0 + row_o1_1_1;
    if (orow1_1_1 < n_valid) {
        long long out_i1_1_1 = ((long long)(tok0 + orow1_1_1) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i1_1_1 / 2) = _iv4;
        }
    }
    int row_o1_1_2 = ld_row + 8;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o1_1_2 * 128 + ((ld_ch ^ row_o1_1_2 & 7) << 4))));
    int orow1_1_2 = row0 + row_o1_1_2;
    if (orow1_1_2 < n_valid) {
        long long out_i1_1_2 = ((long long)(tok0 + orow1_1_2) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i1_1_2 / 2) = _iv4;
        }
    }
    int row_o1_1_3 = ld_row + 12;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o1_1_3 * 128 + ((ld_ch ^ row_o1_1_3 & 7) << 4))));
    int orow1_1_3 = row0 + row_o1_1_3;
    if (orow1_1_3 < n_valid) {
        long long out_i1_1_3 = ((long long)(tok0 + orow1_1_3) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i1_1_3 / 2) = _iv4;
        }
    }
    __syncwarp();
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (warp_1 == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
}

} // extern "C"
