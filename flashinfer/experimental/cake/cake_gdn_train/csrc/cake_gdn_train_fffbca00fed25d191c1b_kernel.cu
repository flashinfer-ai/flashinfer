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
kernel_cake_gdn_train_fffbca00fed25d191c1b(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ v, float* __restrict__ g_cs, float* __restrict__ beta, __nv_bfloat16* __restrict__ A_out, __nv_bfloat16* __restrict__ w_out, __nv_bfloat16* __restrict__ u_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, int recompute)
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
        int r0_0 = tid / 16;
        int c0_0 = tid % 16 * 8;
        long long row_stride_0 = (long long)num_v_heads * 128;
        long long src_row_0 = (long long)(tok0 + r0_0) * row_stride_0 + (long long)hv * 128 + (long long)c0_0;
        unsigned int dst0_0 = s_v_addr + (unsigned int)(c0_0 / 64 * 8192 + (r0_0 * 128 + c0_0 % 64 * 2 ^ (r0_0 * 128 + c0_0 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_0), "l"(v + src_row_0), "r"((n_valid > r0_0) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_0 + 1024), "l"(v + (src_row_0 + 8 * row_stride_0)), "r"((n_valid > r0_0 + 8) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_0 + 2048), "l"(v + (src_row_0 + 16 * row_stride_0)), "r"((n_valid > r0_0 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_0 + 3072), "l"(v + (src_row_0 + 24 * row_stride_0)), "r"((n_valid > r0_0 + 24) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_0 + 4096), "l"(v + (src_row_0 + 32 * row_stride_0)), "r"((n_valid > r0_0 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_0 + 5120), "l"(v + (src_row_0 + 40 * row_stride_0)), "r"((n_valid > r0_0 + 40) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_0 + 6144), "l"(v + (src_row_0 + 48 * row_stride_0)), "r"((n_valid > r0_0 + 48) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_0 + 7168), "l"(v + (src_row_0 + 56 * row_stride_0)), "r"((n_valid > r0_0 + 56) ? 16 : 0));
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
#if __CUDA_ARCH__ == 1070
            int _mma_kn_lo = ((s_kn_addr) >> 4) & 0x7FFF;
#else
            int _mma_kn_lo = ((s_kn_addr) >> 4) & 0x3FFF;
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
                : "r"(taddr_1));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float g_row[2];
            float b_row[2];
            g_row[0] = s_g[frag_r];
            b_row[0] = s_b[frag_r];
            g_row[1] = s_g[8 + frag_r];
            b_row[1] = s_b[8 + frag_r];
            int col_0 = col_q;
            float g_col_0 = s_g[col_0];
            int row_0 = frag_r;
            float val_0 = 0.0f;
            if (col_0 < row_0) {
                if (row_0 < n_valid) {
                    float _exp2_2 = approx_exp2(g_row[0] - g_col_0);
                    val_0 = b_row[0] * _exp2_2 * acc[0];
                }
            }
            s_lt[col_0 * 64 + (row_0 ^ (col_0 >> 1 & 3) << 3)] = val_0;
            int row_1 = 8 + frag_r;
            float val_1 = 0.0f;
            if (col_0 < row_1) {
                if (row_1 < n_valid) {
                    float _exp2_3 = approx_exp2(g_row[1] - g_col_0);
                    val_1 = b_row[1] * _exp2_3 * acc[2];
                }
            }
            s_lt[col_0 * 64 + (row_1 ^ (col_0 >> 1 & 3) << 3)] = val_1;
            int col_1 = 1 + col_q;
            float g_col_1 = s_g[col_1];
            int row_2 = frag_r;
            float val_2 = 0.0f;
            if (col_1 < row_2) {
                if (row_2 < n_valid) {
                    float _exp2_4 = approx_exp2(g_row[0] - g_col_1);
                    val_2 = b_row[0] * _exp2_4 * acc[1];
                }
            }
            s_lt[col_1 * 64 + (row_2 ^ (col_1 >> 1 & 3) << 3)] = val_2;
            int row_3 = 8 + frag_r;
            float val_3 = 0.0f;
            if (col_1 < row_3) {
                if (row_3 < n_valid) {
                    float _exp2_5 = approx_exp2(g_row[1] - g_col_1);
                    val_3 = b_row[1] * _exp2_5 * acc[3];
                }
            }
            s_lt[col_1 * 64 + (row_3 ^ (col_1 >> 1 & 3) << 3)] = val_3;
            int col_2 = 8 + col_q;
            float g_col_2 = s_g[col_2];
            int row_4 = frag_r;
            float val_4 = 0.0f;
            if (col_2 < row_4) {
                if (row_4 < n_valid) {
                    float _exp2_6 = approx_exp2(g_row[0] - g_col_2);
                    val_4 = b_row[0] * _exp2_6 * acc[4];
                }
            }
            s_lt[col_2 * 64 + (row_4 ^ (col_2 >> 1 & 3) << 3)] = val_4;
            int row_5 = 8 + frag_r;
            float val_5 = 0.0f;
            if (col_2 < row_5) {
                if (row_5 < n_valid) {
                    float _exp2_7 = approx_exp2(g_row[1] - g_col_2);
                    val_5 = b_row[1] * _exp2_7 * acc[6];
                }
            }
            s_lt[col_2 * 64 + (row_5 ^ (col_2 >> 1 & 3) << 3)] = val_5;
            int col_3 = 9 + col_q;
            float g_col_3 = s_g[col_3];
            int row_6 = frag_r;
            float val_6 = 0.0f;
            if (col_3 < row_6) {
                if (row_6 < n_valid) {
                    float _exp2_8 = approx_exp2(g_row[0] - g_col_3);
                    val_6 = b_row[0] * _exp2_8 * acc[5];
                }
            }
            s_lt[col_3 * 64 + (row_6 ^ (col_3 >> 1 & 3) << 3)] = val_6;
            int row_7 = 8 + frag_r;
            float val_7 = 0.0f;
            if (col_3 < row_7) {
                if (row_7 < n_valid) {
                    float _exp2_9 = approx_exp2(g_row[1] - g_col_3);
                    val_7 = b_row[1] * _exp2_9 * acc[7];
                }
            }
            s_lt[col_3 * 64 + (row_7 ^ (col_3 >> 1 & 3) << 3)] = val_7;
        }
        if (warp_1 == 1) {
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7]))
                : "r"(taddr_1));
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
            int col_4 = col_q;
            float g_col_4 = s_g[col_4];
            int row_8 = 16 + frag_r;
            float val_8 = 0.0f;
            if (col_4 < row_8) {
                if (row_8 < n_valid) {
                    float _exp2_10 = approx_exp2(g_row_1[0] - g_col_4);
                    val_8 = b_row_1[0] * _exp2_10 * acc[0];
                }
            }
            s_lt[col_4 * 64 + (row_8 ^ (col_4 >> 1 & 3) << 3)] = val_8;
            int row_9 = 24 + frag_r;
            float val_9 = 0.0f;
            if (col_4 < row_9) {
                if (row_9 < n_valid) {
                    float _exp2_11 = approx_exp2(g_row_1[1] - g_col_4);
                    val_9 = b_row_1[1] * _exp2_11 * acc[2];
                }
            }
            s_lt[col_4 * 64 + (row_9 ^ (col_4 >> 1 & 3) << 3)] = val_9;
            int col_5 = 1 + col_q;
            float g_col_5 = s_g[col_5];
            int row_10 = 16 + frag_r;
            float val_10 = 0.0f;
            if (col_5 < row_10) {
                if (row_10 < n_valid) {
                    float _exp2_12 = approx_exp2(g_row_1[0] - g_col_5);
                    val_10 = b_row_1[0] * _exp2_12 * acc[1];
                }
            }
            s_lt[col_5 * 64 + (row_10 ^ (col_5 >> 1 & 3) << 3)] = val_10;
            int row_11 = 24 + frag_r;
            float val_11 = 0.0f;
            if (col_5 < row_11) {
                if (row_11 < n_valid) {
                    float _exp2_13 = approx_exp2(g_row_1[1] - g_col_5);
                    val_11 = b_row_1[1] * _exp2_13 * acc[3];
                }
            }
            s_lt[col_5 * 64 + (row_11 ^ (col_5 >> 1 & 3) << 3)] = val_11;
            int col_6 = 8 + col_q;
            float g_col_6 = s_g[col_6];
            int row_12 = 16 + frag_r;
            float val_12 = 0.0f;
            if (col_6 < row_12) {
                if (row_12 < n_valid) {
                    float _exp2_14 = approx_exp2(g_row_1[0] - g_col_6);
                    val_12 = b_row_1[0] * _exp2_14 * acc[4];
                }
            }
            s_lt[col_6 * 64 + (row_12 ^ (col_6 >> 1 & 3) << 3)] = val_12;
            int row_13 = 24 + frag_r;
            float val_13 = 0.0f;
            if (col_6 < row_13) {
                if (row_13 < n_valid) {
                    float _exp2_15 = approx_exp2(g_row_1[1] - g_col_6);
                    val_13 = b_row_1[1] * _exp2_15 * acc[6];
                }
            }
            s_lt[col_6 * 64 + (row_13 ^ (col_6 >> 1 & 3) << 3)] = val_13;
            int col_7 = 9 + col_q;
            float g_col_7 = s_g[col_7];
            int row_14 = 16 + frag_r;
            float val_14 = 0.0f;
            if (col_7 < row_14) {
                if (row_14 < n_valid) {
                    float _exp2_16 = approx_exp2(g_row_1[0] - g_col_7);
                    val_14 = b_row_1[0] * _exp2_16 * acc[5];
                }
            }
            s_lt[col_7 * 64 + (row_14 ^ (col_7 >> 1 & 3) << 3)] = val_14;
            int row_15 = 24 + frag_r;
            float val_15 = 0.0f;
            if (col_7 < row_15) {
                if (row_15 < n_valid) {
                    float _exp2_17 = approx_exp2(g_row_1[1] - g_col_7);
                    val_15 = b_row_1[1] * _exp2_17 * acc[7];
                }
            }
            s_lt[col_7 * 64 + (row_15 ^ (col_7 >> 1 & 3) << 3)] = val_15;
            int col_8 = 16 + col_q;
            float g_col_8 = s_g[col_8];
            int row_16 = 16 + frag_r;
            float val_16 = 0.0f;
            if (col_8 < row_16) {
                if (row_16 < n_valid) {
                    float _exp2_18 = approx_exp2(g_row_1[0] - g_col_8);
                    val_16 = b_row_1[0] * _exp2_18 * acc[8];
                }
            }
            s_lt[col_8 * 64 + (row_16 ^ (col_8 >> 1 & 3) << 3)] = val_16;
            int row_17 = 24 + frag_r;
            float val_17 = 0.0f;
            if (col_8 < row_17) {
                if (row_17 < n_valid) {
                    float _exp2_19 = approx_exp2(g_row_1[1] - g_col_8);
                    val_17 = b_row_1[1] * _exp2_19 * acc[10];
                }
            }
            s_lt[col_8 * 64 + (row_17 ^ (col_8 >> 1 & 3) << 3)] = val_17;
            int col_9 = 17 + col_q;
            float g_col_9 = s_g[col_9];
            int row_18 = 16 + frag_r;
            float val_18 = 0.0f;
            if (col_9 < row_18) {
                if (row_18 < n_valid) {
                    float _exp2_20 = approx_exp2(g_row_1[0] - g_col_9);
                    val_18 = b_row_1[0] * _exp2_20 * acc[9];
                }
            }
            s_lt[col_9 * 64 + (row_18 ^ (col_9 >> 1 & 3) << 3)] = val_18;
            int row_19 = 24 + frag_r;
            float val_19 = 0.0f;
            if (col_9 < row_19) {
                if (row_19 < n_valid) {
                    float _exp2_21 = approx_exp2(g_row_1[1] - g_col_9);
                    val_19 = b_row_1[1] * _exp2_21 * acc[11];
                }
            }
            s_lt[col_9 * 64 + (row_19 ^ (col_9 >> 1 & 3) << 3)] = val_19;
            int col_10 = 24 + col_q;
            float g_col_10 = s_g[col_10];
            int row_20 = 16 + frag_r;
            float val_20 = 0.0f;
            if (col_10 < row_20) {
                if (row_20 < n_valid) {
                    float _exp2_22 = approx_exp2(g_row_1[0] - g_col_10);
                    val_20 = b_row_1[0] * _exp2_22 * acc[12];
                }
            }
            s_lt[col_10 * 64 + (row_20 ^ (col_10 >> 1 & 3) << 3)] = val_20;
            int row_21 = 24 + frag_r;
            float val_21 = 0.0f;
            if (col_10 < row_21) {
                if (row_21 < n_valid) {
                    float _exp2_23 = approx_exp2(g_row_1[1] - g_col_10);
                    val_21 = b_row_1[1] * _exp2_23 * acc[14];
                }
            }
            s_lt[col_10 * 64 + (row_21 ^ (col_10 >> 1 & 3) << 3)] = val_21;
            int col_11 = 25 + col_q;
            float g_col_11 = s_g[col_11];
            int row_22 = 16 + frag_r;
            float val_22 = 0.0f;
            if (col_11 < row_22) {
                if (row_22 < n_valid) {
                    float _exp2_24 = approx_exp2(g_row_1[0] - g_col_11);
                    val_22 = b_row_1[0] * _exp2_24 * acc[13];
                }
            }
            s_lt[col_11 * 64 + (row_22 ^ (col_11 >> 1 & 3) << 3)] = val_22;
            int row_23 = 24 + frag_r;
            float val_23 = 0.0f;
            if (col_11 < row_23) {
                if (row_23 < n_valid) {
                    float _exp2_25 = approx_exp2(g_row_1[1] - g_col_11);
                    val_23 = b_row_1[1] * _exp2_25 * acc[15];
                }
            }
            s_lt[col_11 * 64 + (row_23 ^ (col_11 >> 1 & 3) << 3)] = val_23;
        }
        if (warp_1 == 2) {
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7]))
                : "r"(taddr_1));
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
            int col_12 = col_q;
            float g_col_12 = s_g[col_12];
            int row_24 = 32 + frag_r;
            float val_24 = 0.0f;
            if (col_12 < row_24) {
                if (row_24 < n_valid) {
                    float _exp2_26 = approx_exp2(g_row_2[0] - g_col_12);
                    val_24 = b_row_2[0] * _exp2_26 * acc[0];
                }
            }
            s_lt[col_12 * 64 + (row_24 ^ (col_12 >> 1 & 3) << 3)] = val_24;
            int row_25 = 40 + frag_r;
            float val_25 = 0.0f;
            if (col_12 < row_25) {
                if (row_25 < n_valid) {
                    float _exp2_27 = approx_exp2(g_row_2[1] - g_col_12);
                    val_25 = b_row_2[1] * _exp2_27 * acc[2];
                }
            }
            s_lt[col_12 * 64 + (row_25 ^ (col_12 >> 1 & 3) << 3)] = val_25;
            int col_13 = 1 + col_q;
            float g_col_13 = s_g[col_13];
            int row_26 = 32 + frag_r;
            float val_26 = 0.0f;
            if (col_13 < row_26) {
                if (row_26 < n_valid) {
                    float _exp2_28 = approx_exp2(g_row_2[0] - g_col_13);
                    val_26 = b_row_2[0] * _exp2_28 * acc[1];
                }
            }
            s_lt[col_13 * 64 + (row_26 ^ (col_13 >> 1 & 3) << 3)] = val_26;
            int row_27 = 40 + frag_r;
            float val_27 = 0.0f;
            if (col_13 < row_27) {
                if (row_27 < n_valid) {
                    float _exp2_29 = approx_exp2(g_row_2[1] - g_col_13);
                    val_27 = b_row_2[1] * _exp2_29 * acc[3];
                }
            }
            s_lt[col_13 * 64 + (row_27 ^ (col_13 >> 1 & 3) << 3)] = val_27;
            int col_14 = 8 + col_q;
            float g_col_14 = s_g[col_14];
            int row_28 = 32 + frag_r;
            float val_28 = 0.0f;
            if (col_14 < row_28) {
                if (row_28 < n_valid) {
                    float _exp2_30 = approx_exp2(g_row_2[0] - g_col_14);
                    val_28 = b_row_2[0] * _exp2_30 * acc[4];
                }
            }
            s_lt[col_14 * 64 + (row_28 ^ (col_14 >> 1 & 3) << 3)] = val_28;
            int row_29 = 40 + frag_r;
            float val_29 = 0.0f;
            if (col_14 < row_29) {
                if (row_29 < n_valid) {
                    float _exp2_31 = approx_exp2(g_row_2[1] - g_col_14);
                    val_29 = b_row_2[1] * _exp2_31 * acc[6];
                }
            }
            s_lt[col_14 * 64 + (row_29 ^ (col_14 >> 1 & 3) << 3)] = val_29;
            int col_15 = 9 + col_q;
            float g_col_15 = s_g[col_15];
            int row_30 = 32 + frag_r;
            float val_30 = 0.0f;
            if (col_15 < row_30) {
                if (row_30 < n_valid) {
                    float _exp2_32 = approx_exp2(g_row_2[0] - g_col_15);
                    val_30 = b_row_2[0] * _exp2_32 * acc[5];
                }
            }
            s_lt[col_15 * 64 + (row_30 ^ (col_15 >> 1 & 3) << 3)] = val_30;
            int row_31 = 40 + frag_r;
            float val_31 = 0.0f;
            if (col_15 < row_31) {
                if (row_31 < n_valid) {
                    float _exp2_33 = approx_exp2(g_row_2[1] - g_col_15);
                    val_31 = b_row_2[1] * _exp2_33 * acc[7];
                }
            }
            s_lt[col_15 * 64 + (row_31 ^ (col_15 >> 1 & 3) << 3)] = val_31;
            int col_16 = 16 + col_q;
            float g_col_16 = s_g[col_16];
            int row_32 = 32 + frag_r;
            float val_32 = 0.0f;
            if (col_16 < row_32) {
                if (row_32 < n_valid) {
                    float _exp2_34 = approx_exp2(g_row_2[0] - g_col_16);
                    val_32 = b_row_2[0] * _exp2_34 * acc[8];
                }
            }
            s_lt[col_16 * 64 + (row_32 ^ (col_16 >> 1 & 3) << 3)] = val_32;
            int row_33 = 40 + frag_r;
            float val_33 = 0.0f;
            if (col_16 < row_33) {
                if (row_33 < n_valid) {
                    float _exp2_35 = approx_exp2(g_row_2[1] - g_col_16);
                    val_33 = b_row_2[1] * _exp2_35 * acc[10];
                }
            }
            s_lt[col_16 * 64 + (row_33 ^ (col_16 >> 1 & 3) << 3)] = val_33;
            int col_17 = 17 + col_q;
            float g_col_17 = s_g[col_17];
            int row_34 = 32 + frag_r;
            float val_34 = 0.0f;
            if (col_17 < row_34) {
                if (row_34 < n_valid) {
                    float _exp2_36 = approx_exp2(g_row_2[0] - g_col_17);
                    val_34 = b_row_2[0] * _exp2_36 * acc[9];
                }
            }
            s_lt[col_17 * 64 + (row_34 ^ (col_17 >> 1 & 3) << 3)] = val_34;
            int row_35 = 40 + frag_r;
            float val_35 = 0.0f;
            if (col_17 < row_35) {
                if (row_35 < n_valid) {
                    float _exp2_37 = approx_exp2(g_row_2[1] - g_col_17);
                    val_35 = b_row_2[1] * _exp2_37 * acc[11];
                }
            }
            s_lt[col_17 * 64 + (row_35 ^ (col_17 >> 1 & 3) << 3)] = val_35;
            int col_18 = 24 + col_q;
            float g_col_18 = s_g[col_18];
            int row_36 = 32 + frag_r;
            float val_36 = 0.0f;
            if (col_18 < row_36) {
                if (row_36 < n_valid) {
                    float _exp2_38 = approx_exp2(g_row_2[0] - g_col_18);
                    val_36 = b_row_2[0] * _exp2_38 * acc[12];
                }
            }
            s_lt[col_18 * 64 + (row_36 ^ (col_18 >> 1 & 3) << 3)] = val_36;
            int row_37 = 40 + frag_r;
            float val_37 = 0.0f;
            if (col_18 < row_37) {
                if (row_37 < n_valid) {
                    float _exp2_39 = approx_exp2(g_row_2[1] - g_col_18);
                    val_37 = b_row_2[1] * _exp2_39 * acc[14];
                }
            }
            s_lt[col_18 * 64 + (row_37 ^ (col_18 >> 1 & 3) << 3)] = val_37;
            int col_19 = 25 + col_q;
            float g_col_19 = s_g[col_19];
            int row_38 = 32 + frag_r;
            float val_38 = 0.0f;
            if (col_19 < row_38) {
                if (row_38 < n_valid) {
                    float _exp2_40 = approx_exp2(g_row_2[0] - g_col_19);
                    val_38 = b_row_2[0] * _exp2_40 * acc[13];
                }
            }
            s_lt[col_19 * 64 + (row_38 ^ (col_19 >> 1 & 3) << 3)] = val_38;
            int row_39 = 40 + frag_r;
            float val_39 = 0.0f;
            if (col_19 < row_39) {
                if (row_39 < n_valid) {
                    float _exp2_41 = approx_exp2(g_row_2[1] - g_col_19);
                    val_39 = b_row_2[1] * _exp2_41 * acc[15];
                }
            }
            s_lt[col_19 * 64 + (row_39 ^ (col_19 >> 1 & 3) << 3)] = val_39;
            int col_20 = 32 + col_q;
            float g_col_20 = s_g[col_20];
            int row_40 = 32 + frag_r;
            float val_40 = 0.0f;
            if (col_20 < row_40) {
                if (row_40 < n_valid) {
                    float _exp2_42 = approx_exp2(g_row_2[0] - g_col_20);
                    val_40 = b_row_2[0] * _exp2_42 * acc[16];
                }
            }
            s_lt[col_20 * 64 + (row_40 ^ (col_20 >> 1 & 3) << 3)] = val_40;
            int row_41 = 40 + frag_r;
            float val_41 = 0.0f;
            if (col_20 < row_41) {
                if (row_41 < n_valid) {
                    float _exp2_43 = approx_exp2(g_row_2[1] - g_col_20);
                    val_41 = b_row_2[1] * _exp2_43 * acc[18];
                }
            }
            s_lt[col_20 * 64 + (row_41 ^ (col_20 >> 1 & 3) << 3)] = val_41;
            int col_21 = 33 + col_q;
            float g_col_21 = s_g[col_21];
            int row_42 = 32 + frag_r;
            float val_42 = 0.0f;
            if (col_21 < row_42) {
                if (row_42 < n_valid) {
                    float _exp2_44 = approx_exp2(g_row_2[0] - g_col_21);
                    val_42 = b_row_2[0] * _exp2_44 * acc[17];
                }
            }
            s_lt[col_21 * 64 + (row_42 ^ (col_21 >> 1 & 3) << 3)] = val_42;
            int row_43 = 40 + frag_r;
            float val_43 = 0.0f;
            if (col_21 < row_43) {
                if (row_43 < n_valid) {
                    float _exp2_45 = approx_exp2(g_row_2[1] - g_col_21);
                    val_43 = b_row_2[1] * _exp2_45 * acc[19];
                }
            }
            s_lt[col_21 * 64 + (row_43 ^ (col_21 >> 1 & 3) << 3)] = val_43;
            int col_22 = 40 + col_q;
            float g_col_22 = s_g[col_22];
            int row_44 = 32 + frag_r;
            float val_44 = 0.0f;
            if (col_22 < row_44) {
                if (row_44 < n_valid) {
                    float _exp2_46 = approx_exp2(g_row_2[0] - g_col_22);
                    val_44 = b_row_2[0] * _exp2_46 * acc[20];
                }
            }
            s_lt[col_22 * 64 + (row_44 ^ (col_22 >> 1 & 3) << 3)] = val_44;
            int row_45 = 40 + frag_r;
            float val_45 = 0.0f;
            if (col_22 < row_45) {
                if (row_45 < n_valid) {
                    float _exp2_47 = approx_exp2(g_row_2[1] - g_col_22);
                    val_45 = b_row_2[1] * _exp2_47 * acc[22];
                }
            }
            s_lt[col_22 * 64 + (row_45 ^ (col_22 >> 1 & 3) << 3)] = val_45;
            int col_23 = 41 + col_q;
            float g_col_23 = s_g[col_23];
            int row_46 = 32 + frag_r;
            float val_46 = 0.0f;
            if (col_23 < row_46) {
                if (row_46 < n_valid) {
                    float _exp2_48 = approx_exp2(g_row_2[0] - g_col_23);
                    val_46 = b_row_2[0] * _exp2_48 * acc[21];
                }
            }
            s_lt[col_23 * 64 + (row_46 ^ (col_23 >> 1 & 3) << 3)] = val_46;
            int row_47 = 40 + frag_r;
            float val_47 = 0.0f;
            if (col_23 < row_47) {
                if (row_47 < n_valid) {
                    float _exp2_49 = approx_exp2(g_row_2[1] - g_col_23);
                    val_47 = b_row_2[1] * _exp2_49 * acc[23];
                }
            }
            s_lt[col_23 * 64 + (row_47 ^ (col_23 >> 1 & 3) << 3)] = val_47;
        }
        if (warp_1 == 3) {
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7]))
                : "r"(taddr_1));
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
            int col_24 = col_q;
            float g_col_24 = s_g[col_24];
            int row_48 = 48 + frag_r;
            float val_48 = 0.0f;
            if (col_24 < row_48) {
                if (row_48 < n_valid) {
                    float _exp2_50 = approx_exp2(g_row_3[0] - g_col_24);
                    val_48 = b_row_3[0] * _exp2_50 * acc[0];
                }
            }
            s_lt[col_24 * 64 + (row_48 ^ (col_24 >> 1 & 3) << 3)] = val_48;
            int row_49 = 56 + frag_r;
            float val_49 = 0.0f;
            if (col_24 < row_49) {
                if (row_49 < n_valid) {
                    float _exp2_51 = approx_exp2(g_row_3[1] - g_col_24);
                    val_49 = b_row_3[1] * _exp2_51 * acc[2];
                }
            }
            s_lt[col_24 * 64 + (row_49 ^ (col_24 >> 1 & 3) << 3)] = val_49;
            int col_25 = 1 + col_q;
            float g_col_25 = s_g[col_25];
            int row_50 = 48 + frag_r;
            float val_50 = 0.0f;
            if (col_25 < row_50) {
                if (row_50 < n_valid) {
                    float _exp2_52 = approx_exp2(g_row_3[0] - g_col_25);
                    val_50 = b_row_3[0] * _exp2_52 * acc[1];
                }
            }
            s_lt[col_25 * 64 + (row_50 ^ (col_25 >> 1 & 3) << 3)] = val_50;
            int row_51 = 56 + frag_r;
            float val_51 = 0.0f;
            if (col_25 < row_51) {
                if (row_51 < n_valid) {
                    float _exp2_53 = approx_exp2(g_row_3[1] - g_col_25);
                    val_51 = b_row_3[1] * _exp2_53 * acc[3];
                }
            }
            s_lt[col_25 * 64 + (row_51 ^ (col_25 >> 1 & 3) << 3)] = val_51;
            int col_26 = 8 + col_q;
            float g_col_26 = s_g[col_26];
            int row_52 = 48 + frag_r;
            float val_52 = 0.0f;
            if (col_26 < row_52) {
                if (row_52 < n_valid) {
                    float _exp2_54 = approx_exp2(g_row_3[0] - g_col_26);
                    val_52 = b_row_3[0] * _exp2_54 * acc[4];
                }
            }
            s_lt[col_26 * 64 + (row_52 ^ (col_26 >> 1 & 3) << 3)] = val_52;
            int row_53 = 56 + frag_r;
            float val_53 = 0.0f;
            if (col_26 < row_53) {
                if (row_53 < n_valid) {
                    float _exp2_55 = approx_exp2(g_row_3[1] - g_col_26);
                    val_53 = b_row_3[1] * _exp2_55 * acc[6];
                }
            }
            s_lt[col_26 * 64 + (row_53 ^ (col_26 >> 1 & 3) << 3)] = val_53;
            int col_27 = 9 + col_q;
            float g_col_27 = s_g[col_27];
            int row_54 = 48 + frag_r;
            float val_54 = 0.0f;
            if (col_27 < row_54) {
                if (row_54 < n_valid) {
                    float _exp2_56 = approx_exp2(g_row_3[0] - g_col_27);
                    val_54 = b_row_3[0] * _exp2_56 * acc[5];
                }
            }
            s_lt[col_27 * 64 + (row_54 ^ (col_27 >> 1 & 3) << 3)] = val_54;
            int row_55 = 56 + frag_r;
            float val_55 = 0.0f;
            if (col_27 < row_55) {
                if (row_55 < n_valid) {
                    float _exp2_57 = approx_exp2(g_row_3[1] - g_col_27);
                    val_55 = b_row_3[1] * _exp2_57 * acc[7];
                }
            }
            s_lt[col_27 * 64 + (row_55 ^ (col_27 >> 1 & 3) << 3)] = val_55;
            int col_28 = 16 + col_q;
            float g_col_28 = s_g[col_28];
            int row_56 = 48 + frag_r;
            float val_56 = 0.0f;
            if (col_28 < row_56) {
                if (row_56 < n_valid) {
                    float _exp2_58 = approx_exp2(g_row_3[0] - g_col_28);
                    val_56 = b_row_3[0] * _exp2_58 * acc[8];
                }
            }
            s_lt[col_28 * 64 + (row_56 ^ (col_28 >> 1 & 3) << 3)] = val_56;
            int row_57 = 56 + frag_r;
            float val_57 = 0.0f;
            if (col_28 < row_57) {
                if (row_57 < n_valid) {
                    float _exp2_59 = approx_exp2(g_row_3[1] - g_col_28);
                    val_57 = b_row_3[1] * _exp2_59 * acc[10];
                }
            }
            s_lt[col_28 * 64 + (row_57 ^ (col_28 >> 1 & 3) << 3)] = val_57;
            int col_29 = 17 + col_q;
            float g_col_29 = s_g[col_29];
            int row_58 = 48 + frag_r;
            float val_58 = 0.0f;
            if (col_29 < row_58) {
                if (row_58 < n_valid) {
                    float _exp2_60 = approx_exp2(g_row_3[0] - g_col_29);
                    val_58 = b_row_3[0] * _exp2_60 * acc[9];
                }
            }
            s_lt[col_29 * 64 + (row_58 ^ (col_29 >> 1 & 3) << 3)] = val_58;
            int row_59 = 56 + frag_r;
            float val_59 = 0.0f;
            if (col_29 < row_59) {
                if (row_59 < n_valid) {
                    float _exp2_61 = approx_exp2(g_row_3[1] - g_col_29);
                    val_59 = b_row_3[1] * _exp2_61 * acc[11];
                }
            }
            s_lt[col_29 * 64 + (row_59 ^ (col_29 >> 1 & 3) << 3)] = val_59;
            int col_30 = 24 + col_q;
            float g_col_30 = s_g[col_30];
            int row_60 = 48 + frag_r;
            float val_60 = 0.0f;
            if (col_30 < row_60) {
                if (row_60 < n_valid) {
                    float _exp2_62 = approx_exp2(g_row_3[0] - g_col_30);
                    val_60 = b_row_3[0] * _exp2_62 * acc[12];
                }
            }
            s_lt[col_30 * 64 + (row_60 ^ (col_30 >> 1 & 3) << 3)] = val_60;
            int row_61 = 56 + frag_r;
            float val_61 = 0.0f;
            if (col_30 < row_61) {
                if (row_61 < n_valid) {
                    float _exp2_63 = approx_exp2(g_row_3[1] - g_col_30);
                    val_61 = b_row_3[1] * _exp2_63 * acc[14];
                }
            }
            s_lt[col_30 * 64 + (row_61 ^ (col_30 >> 1 & 3) << 3)] = val_61;
            int col_31 = 25 + col_q;
            float g_col_31 = s_g[col_31];
            int row_62 = 48 + frag_r;
            float val_62 = 0.0f;
            if (col_31 < row_62) {
                if (row_62 < n_valid) {
                    float _exp2_64 = approx_exp2(g_row_3[0] - g_col_31);
                    val_62 = b_row_3[0] * _exp2_64 * acc[13];
                }
            }
            s_lt[col_31 * 64 + (row_62 ^ (col_31 >> 1 & 3) << 3)] = val_62;
            int row_63 = 56 + frag_r;
            float val_63 = 0.0f;
            if (col_31 < row_63) {
                if (row_63 < n_valid) {
                    float _exp2_65 = approx_exp2(g_row_3[1] - g_col_31);
                    val_63 = b_row_3[1] * _exp2_65 * acc[15];
                }
            }
            s_lt[col_31 * 64 + (row_63 ^ (col_31 >> 1 & 3) << 3)] = val_63;
            int col_32 = 32 + col_q;
            float g_col_32 = s_g[col_32];
            int row_64 = 48 + frag_r;
            float val_64 = 0.0f;
            if (col_32 < row_64) {
                if (row_64 < n_valid) {
                    float _exp2_66 = approx_exp2(g_row_3[0] - g_col_32);
                    val_64 = b_row_3[0] * _exp2_66 * acc[16];
                }
            }
            s_lt[col_32 * 64 + (row_64 ^ (col_32 >> 1 & 3) << 3)] = val_64;
            int row_65 = 56 + frag_r;
            float val_65 = 0.0f;
            if (col_32 < row_65) {
                if (row_65 < n_valid) {
                    float _exp2_67 = approx_exp2(g_row_3[1] - g_col_32);
                    val_65 = b_row_3[1] * _exp2_67 * acc[18];
                }
            }
            s_lt[col_32 * 64 + (row_65 ^ (col_32 >> 1 & 3) << 3)] = val_65;
            int col_33 = 33 + col_q;
            float g_col_33 = s_g[col_33];
            int row_66 = 48 + frag_r;
            float val_66 = 0.0f;
            if (col_33 < row_66) {
                if (row_66 < n_valid) {
                    float _exp2_68 = approx_exp2(g_row_3[0] - g_col_33);
                    val_66 = b_row_3[0] * _exp2_68 * acc[17];
                }
            }
            s_lt[col_33 * 64 + (row_66 ^ (col_33 >> 1 & 3) << 3)] = val_66;
            int row_67 = 56 + frag_r;
            float val_67 = 0.0f;
            if (col_33 < row_67) {
                if (row_67 < n_valid) {
                    float _exp2_69 = approx_exp2(g_row_3[1] - g_col_33);
                    val_67 = b_row_3[1] * _exp2_69 * acc[19];
                }
            }
            s_lt[col_33 * 64 + (row_67 ^ (col_33 >> 1 & 3) << 3)] = val_67;
            int col_34 = 40 + col_q;
            float g_col_34 = s_g[col_34];
            int row_68 = 48 + frag_r;
            float val_68 = 0.0f;
            if (col_34 < row_68) {
                if (row_68 < n_valid) {
                    float _exp2_70 = approx_exp2(g_row_3[0] - g_col_34);
                    val_68 = b_row_3[0] * _exp2_70 * acc[20];
                }
            }
            s_lt[col_34 * 64 + (row_68 ^ (col_34 >> 1 & 3) << 3)] = val_68;
            int row_69 = 56 + frag_r;
            float val_69 = 0.0f;
            if (col_34 < row_69) {
                if (row_69 < n_valid) {
                    float _exp2_71 = approx_exp2(g_row_3[1] - g_col_34);
                    val_69 = b_row_3[1] * _exp2_71 * acc[22];
                }
            }
            s_lt[col_34 * 64 + (row_69 ^ (col_34 >> 1 & 3) << 3)] = val_69;
            int col_35 = 41 + col_q;
            float g_col_35 = s_g[col_35];
            int row_70 = 48 + frag_r;
            float val_70 = 0.0f;
            if (col_35 < row_70) {
                if (row_70 < n_valid) {
                    float _exp2_72 = approx_exp2(g_row_3[0] - g_col_35);
                    val_70 = b_row_3[0] * _exp2_72 * acc[21];
                }
            }
            s_lt[col_35 * 64 + (row_70 ^ (col_35 >> 1 & 3) << 3)] = val_70;
            int row_71 = 56 + frag_r;
            float val_71 = 0.0f;
            if (col_35 < row_71) {
                if (row_71 < n_valid) {
                    float _exp2_73 = approx_exp2(g_row_3[1] - g_col_35);
                    val_71 = b_row_3[1] * _exp2_73 * acc[23];
                }
            }
            s_lt[col_35 * 64 + (row_71 ^ (col_35 >> 1 & 3) << 3)] = val_71;
            int col_36 = 48 + col_q;
            float g_col_36 = s_g[col_36];
            int row_72 = 48 + frag_r;
            float val_72 = 0.0f;
            if (col_36 < row_72) {
                if (row_72 < n_valid) {
                    float _exp2_74 = approx_exp2(g_row_3[0] - g_col_36);
                    val_72 = b_row_3[0] * _exp2_74 * acc[24];
                }
            }
            s_lt[col_36 * 64 + (row_72 ^ (col_36 >> 1 & 3) << 3)] = val_72;
            int row_73 = 56 + frag_r;
            float val_73 = 0.0f;
            if (col_36 < row_73) {
                if (row_73 < n_valid) {
                    float _exp2_75 = approx_exp2(g_row_3[1] - g_col_36);
                    val_73 = b_row_3[1] * _exp2_75 * acc[26];
                }
            }
            s_lt[col_36 * 64 + (row_73 ^ (col_36 >> 1 & 3) << 3)] = val_73;
            int col_37 = 49 + col_q;
            float g_col_37 = s_g[col_37];
            int row_74 = 48 + frag_r;
            float val_74 = 0.0f;
            if (col_37 < row_74) {
                if (row_74 < n_valid) {
                    float _exp2_76 = approx_exp2(g_row_3[0] - g_col_37);
                    val_74 = b_row_3[0] * _exp2_76 * acc[25];
                }
            }
            s_lt[col_37 * 64 + (row_74 ^ (col_37 >> 1 & 3) << 3)] = val_74;
            int row_75 = 56 + frag_r;
            float val_75 = 0.0f;
            if (col_37 < row_75) {
                if (row_75 < n_valid) {
                    float _exp2_77 = approx_exp2(g_row_3[1] - g_col_37);
                    val_75 = b_row_3[1] * _exp2_77 * acc[27];
                }
            }
            s_lt[col_37 * 64 + (row_75 ^ (col_37 >> 1 & 3) << 3)] = val_75;
            int col_38 = 56 + col_q;
            float g_col_38 = s_g[col_38];
            int row_76 = 48 + frag_r;
            float val_76 = 0.0f;
            if (col_38 < row_76) {
                if (row_76 < n_valid) {
                    float _exp2_78 = approx_exp2(g_row_3[0] - g_col_38);
                    val_76 = b_row_3[0] * _exp2_78 * acc[28];
                }
            }
            s_lt[col_38 * 64 + (row_76 ^ (col_38 >> 1 & 3) << 3)] = val_76;
            int row_77 = 56 + frag_r;
            float val_77 = 0.0f;
            if (col_38 < row_77) {
                if (row_77 < n_valid) {
                    float _exp2_79 = approx_exp2(g_row_3[1] - g_col_38);
                    val_77 = b_row_3[1] * _exp2_79 * acc[30];
                }
            }
            s_lt[col_38 * 64 + (row_77 ^ (col_38 >> 1 & 3) << 3)] = val_77;
            int col_39 = 57 + col_q;
            float g_col_39 = s_g[col_39];
            int row_78 = 48 + frag_r;
            float val_78 = 0.0f;
            if (col_39 < row_78) {
                if (row_78 < n_valid) {
                    float _exp2_80 = approx_exp2(g_row_3[0] - g_col_39);
                    val_78 = b_row_3[0] * _exp2_80 * acc[29];
                }
            }
            s_lt[col_39 * 64 + (row_78 ^ (col_39 >> 1 & 3) << 3)] = val_78;
            int row_79 = 56 + frag_r;
            float val_79 = 0.0f;
            if (col_39 < row_79) {
                if (row_79 < n_valid) {
                    float _exp2_81 = approx_exp2(g_row_3[1] - g_col_39);
                    val_79 = b_row_3[1] * _exp2_81 * acc[31];
                }
            }
            s_lt[col_39 * 64 + (row_79 ^ (col_39 >> 1 & 3) << 3)] = val_79;
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
            float exp2 = approx_exp2(s_g[s_row0]);
            int o_15_0 = base[0];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_0));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0] * exp2);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0] * exp2);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0] * exp2);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0] * exp2);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0] * exp2);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0] * exp2);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0] * exp2);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0] * exp2);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_0), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_0 = approx_exp2(s_g[s_row0 + 4]);
            int o_15_1 = base[1] + 512;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_1));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 4] * exp2_0);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 4] * exp2_0);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 4] * exp2_0);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 4] * exp2_0);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 4] * exp2_0);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 4] * exp2_0);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 4] * exp2_0);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 4] * exp2_0);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_1 = approx_exp2(s_g[s_row0 + 8]);
            int o_15_2 = base[0] + 1024;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_2));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 8] * exp2_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 8] * exp2_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 8] * exp2_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 8] * exp2_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 8] * exp2_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 8] * exp2_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 8] * exp2_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 8] * exp2_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_2), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_2 = approx_exp2(s_g[s_row0 + 12]);
            int o_15_3 = base[1] + 1536;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_3));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 12] * exp2_2);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 12] * exp2_2);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 12] * exp2_2);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 12] * exp2_2);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 12] * exp2_2);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 12] * exp2_2);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 12] * exp2_2);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 12] * exp2_2);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_3), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_3 = approx_exp2(s_g[s_row0 + 16]);
            int o_15_4 = base[0] + 2048;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_4));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 16] * exp2_3);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 16] * exp2_3);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 16] * exp2_3);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 16] * exp2_3);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 16] * exp2_3);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 16] * exp2_3);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 16] * exp2_3);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 16] * exp2_3);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_4), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_4 = approx_exp2(s_g[s_row0 + 20]);
            int o_15_5 = base[1] + 2560;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_5));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 20] * exp2_4);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 20] * exp2_4);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 20] * exp2_4);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 20] * exp2_4);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 20] * exp2_4);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 20] * exp2_4);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 20] * exp2_4);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 20] * exp2_4);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_5), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_5 = approx_exp2(s_g[s_row0 + 24]);
            int o_15_6 = base[0] + 3072;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_6));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 24] * exp2_5);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 24] * exp2_5);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 24] * exp2_5);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 24] * exp2_5);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 24] * exp2_5);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 24] * exp2_5);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 24] * exp2_5);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 24] * exp2_5);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_6), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_6 = approx_exp2(s_g[s_row0 + 28]);
            int o_15_7 = base[1] + 3584;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_7));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 28] * exp2_6);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 28] * exp2_6);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 28] * exp2_6);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 28] * exp2_6);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 28] * exp2_6);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 28] * exp2_6);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 28] * exp2_6);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 28] * exp2_6);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_7), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_7 = approx_exp2(s_g[s_row0 + 32]);
            int o_15_8 = base[0] + 4096;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_8));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 32] * exp2_7);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 32] * exp2_7);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 32] * exp2_7);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 32] * exp2_7);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 32] * exp2_7);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 32] * exp2_7);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 32] * exp2_7);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 32] * exp2_7);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_8), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_8 = approx_exp2(s_g[s_row0 + 36]);
            int o_15_9 = base[1] + 4608;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_9));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 36] * exp2_8);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 36] * exp2_8);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 36] * exp2_8);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 36] * exp2_8);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 36] * exp2_8);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 36] * exp2_8);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 36] * exp2_8);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 36] * exp2_8);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_9), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_9 = approx_exp2(s_g[s_row0 + 40]);
            int o_15_10 = base[0] + 5120;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_10));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 40] * exp2_9);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 40] * exp2_9);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 40] * exp2_9);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 40] * exp2_9);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 40] * exp2_9);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 40] * exp2_9);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 40] * exp2_9);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 40] * exp2_9);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_10), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_10 = approx_exp2(s_g[s_row0 + 44]);
            int o_15_11 = base[1] + 5632;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_11));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 44] * exp2_10);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 44] * exp2_10);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 44] * exp2_10);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 44] * exp2_10);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 44] * exp2_10);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 44] * exp2_10);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 44] * exp2_10);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 44] * exp2_10);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_11), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_11 = approx_exp2(s_g[s_row0 + 48]);
            int o_15_12 = base[0] + 6144;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_12));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 48] * exp2_11);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 48] * exp2_11);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 48] * exp2_11);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 48] * exp2_11);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 48] * exp2_11);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 48] * exp2_11);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 48] * exp2_11);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 48] * exp2_11);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_12), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_12 = approx_exp2(s_g[s_row0 + 52]);
            int o_15_13 = base[1] + 6656;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_13));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 52] * exp2_12);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 52] * exp2_12);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 52] * exp2_12);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 52] * exp2_12);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 52] * exp2_12);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 52] * exp2_12);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 52] * exp2_12);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 52] * exp2_12);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_13), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_13 = approx_exp2(s_g[s_row0 + 56]);
            int o_15_14 = base[0] + 7168;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_14));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 56] * exp2_13);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 56] * exp2_13);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 56] * exp2_13);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 56] * exp2_13);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 56] * exp2_13);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 56] * exp2_13);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 56] * exp2_13);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 56] * exp2_13);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_14), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float exp2_14 = approx_exp2(s_g[s_row0 + 60]);
            int o_15_15 = base[1] + 7680;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_15_15));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 60] * exp2_14);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 60] * exp2_14);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 60] * exp2_14);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 60] * exp2_14);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 60] * exp2_14);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 60] * exp2_14);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 60] * exp2_14);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 60] * exp2_14);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15_15), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
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
        float exp2_15 = approx_exp2(s_g[s_row0_1]);
        int o_7_0 = base_1[0];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_7_0));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1] * exp2_15);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1] * exp2_15);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1] * exp2_15);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1] * exp2_15);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1] * exp2_15);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1] * exp2_15);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1] * exp2_15);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1] * exp2_15);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_7_0), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float exp2_15_1 = approx_exp2(s_g[s_row0_1 + 8]);
        int o_7_1 = base_1[0] + 1024;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_7_1));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 8] * exp2_15_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 8] * exp2_15_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 8] * exp2_15_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 8] * exp2_15_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 8] * exp2_15_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 8] * exp2_15_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 8] * exp2_15_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 8] * exp2_15_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_7_1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float exp2_16 = approx_exp2(s_g[s_row0_1 + 16]);
        int o_7_2 = base_1[0] + 2048;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_7_2));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 16] * exp2_16);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 16] * exp2_16);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 16] * exp2_16);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 16] * exp2_16);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 16] * exp2_16);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 16] * exp2_16);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 16] * exp2_16);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 16] * exp2_16);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_7_2), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float exp2_17 = approx_exp2(s_g[s_row0_1 + 24]);
        int o_7_3 = base_1[0] + 3072;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_7_3));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 24] * exp2_17);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 24] * exp2_17);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 24] * exp2_17);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 24] * exp2_17);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 24] * exp2_17);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 24] * exp2_17);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 24] * exp2_17);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 24] * exp2_17);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_7_3), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float exp2_18 = approx_exp2(s_g[s_row0_1 + 32]);
        int o_7_4 = base_1[0] + 4096;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_7_4));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 32] * exp2_18);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 32] * exp2_18);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 32] * exp2_18);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 32] * exp2_18);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 32] * exp2_18);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 32] * exp2_18);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 32] * exp2_18);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 32] * exp2_18);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_7_4), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float exp2_19 = approx_exp2(s_g[s_row0_1 + 40]);
        int o_7_5 = base_1[0] + 5120;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_7_5));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 40] * exp2_19);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 40] * exp2_19);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 40] * exp2_19);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 40] * exp2_19);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 40] * exp2_19);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 40] * exp2_19);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 40] * exp2_19);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 40] * exp2_19);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_7_5), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float exp2_20 = approx_exp2(s_g[s_row0_1 + 48]);
        int o_7_6 = base_1[0] + 6144;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_7_6));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 48] * exp2_20);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 48] * exp2_20);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 48] * exp2_20);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 48] * exp2_20);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 48] * exp2_20);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 48] * exp2_20);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 48] * exp2_20);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 48] * exp2_20);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_7_6), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float exp2_21 = approx_exp2(s_g[s_row0_1 + 56]);
        int o_7_7 = base_1[0] + 7168;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_7_7));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 56] * exp2_21);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 56] * exp2_21);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 56] * exp2_21);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 56] * exp2_21);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 56] * exp2_21);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 56] * exp2_21);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 56] * exp2_21);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 56] * exp2_21);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_7_7), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
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
        int r0_1 = tid / 16;
        int c0_1 = tid % 16 * 8;
        long long row_stride_1 = (long long)num_v_heads * 128;
        long long src_row_1 = (long long)(tok0 + r0_1) * row_stride_1 + (long long)hv * 128 + (long long)c0_1;
        unsigned int dst0_1 = s_v_addr + (unsigned int)(c0_1 / 64 * 8192 + (r0_1 * 128 + c0_1 % 64 * 2 ^ (r0_1 * 128 + c0_1 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_1), "l"(v + src_row_1), "r"((n_valid > r0_1) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_1 + 1024), "l"(v + (src_row_1 + 8 * row_stride_1)), "r"((n_valid > r0_1 + 8) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_1 + 2048), "l"(v + (src_row_1 + 16 * row_stride_1)), "r"((n_valid > r0_1 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_1 + 3072), "l"(v + (src_row_1 + 24 * row_stride_1)), "r"((n_valid > r0_1 + 24) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_1 + 4096), "l"(v + (src_row_1 + 32 * row_stride_1)), "r"((n_valid > r0_1 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_1 + 5120), "l"(v + (src_row_1 + 40 * row_stride_1)), "r"((n_valid > r0_1 + 40) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_1 + 6144), "l"(v + (src_row_1 + 48 * row_stride_1)), "r"((n_valid > r0_1 + 48) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_1 + 7168), "l"(v + (src_row_1 + 56 * row_stride_1)), "r"((n_valid > r0_1 + 56) ? 16 : 0));
        asm volatile("cp.async.commit_group;");
    }
    if (warp == 1) {
        uint32_t _elected_83 = elect_sync();
        asm volatile("{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %0, 0;\n\t@leader tcgen05.fence::after_thread_sync;\n\t}\n" :: "r"(_elected_83));
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_84 = ((s_a_addr) >> 4) & 0x7FFF;
        int _smem_desc_lo_85 = (((s_knT_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
        int _smem_desc_lo_84 = ((s_a_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_85 = (((s_knT_addr) >> 4) & 0x3FFF) | 0x2000000;
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
        : "r"(taddr_1));
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
    int row_o_0 = ld_row;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_0 * 128 + ((ld_ch ^ row_o_0 & 7) << 4))));
    int orow_0 = row0 + row_o_0;
    if (orow_0 < n_valid) {
        long long out_i_0 = ((long long)(tok0 + orow_0) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i_0 / 2) = _iv4;
        }
    }
    int row_o_1 = ld_row + 4;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_1 * 128 + ((ld_ch ^ row_o_1 & 7) << 4))));
    int orow_1 = row0 + row_o_1;
    if (orow_1 < n_valid) {
        long long out_i_1 = ((long long)(tok0 + orow_1) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i_1 / 2) = _iv4;
        }
    }
    int row_o_2 = ld_row + 8;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_2 * 128 + ((ld_ch ^ row_o_2 & 7) << 4))));
    int orow_2 = row0 + row_o_2;
    if (orow_2 < n_valid) {
        long long out_i_2 = ((long long)(tok0 + orow_2) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i_2 / 2) = _iv4;
        }
    }
    int row_o_3 = ld_row + 12;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_3 * 128 + ((ld_ch ^ row_o_3 & 7) << 4))));
    int orow_3 = row0 + row_o_3;
    if (orow_3 < n_valid) {
        long long out_i_3 = ((long long)(tok0 + orow_3) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i_3 / 2) = _iv4;
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
    int row_o_4 = ld_row;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_4 * 128 + ((ld_ch ^ row_o_4 & 7) << 4))));
    int orow_4 = row0 + row_o_4;
    if (orow_4 < n_valid) {
        long long out_i_4 = ((long long)(tok0 + orow_4) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i_4 / 2) = _iv4;
        }
    }
    int row_o_5 = ld_row + 4;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_5 * 128 + ((ld_ch ^ row_o_5 & 7) << 4))));
    int orow_5 = row0 + row_o_5;
    if (orow_5 < n_valid) {
        long long out_i_5 = ((long long)(tok0 + orow_5) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i_5 / 2) = _iv4;
        }
    }
    int row_o_6 = ld_row + 8;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_6 * 128 + ((ld_ch ^ row_o_6 & 7) << 4))));
    int orow_6 = row0 + row_o_6;
    if (orow_6 < n_valid) {
        long long out_i_6 = ((long long)(tok0 + orow_6) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i_6 / 2) = _iv4;
        }
    }
    int row_o_7 = ld_row + 12;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_7 * 128 + ((ld_ch ^ row_o_7 & 7) << 4))));
    int orow_7 = row0 + row_o_7;
    if (orow_7 < n_valid) {
        long long out_i_7 = ((long long)(tok0 + orow_7) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_i_7 / 2) = _iv4;
        }
    }
    __syncwarp();
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    int s_row0_v = tid_2 / 16;
    int s_col_v = tid_2 % 16 * 8;
    int base_v[1];
    base_v[0] = s_col_v / 64 * 8192 + s_row0_v * 128 + (s_col_v % 64 * 2 ^ s_row0_v << 4);
    int o_1_0 = base_v[0];
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_1_0));
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
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_1_0), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_1_1 = base_v[0] + 1024;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_1_1));
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
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_1_1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_1_2 = base_v[0] + 2048;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_1_2));
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
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_1_2), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_1_3 = base_v[0] + 3072;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_1_3));
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
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_1_3), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_1_4 = base_v[0] + 4096;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_1_4));
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
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_1_4), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_1_5 = base_v[0] + 5120;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_1_5));
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
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_1_5), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_1_6 = base_v[0] + 6144;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_1_6));
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
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_1_6), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_1_7 = base_v[0] + 7168;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_1_7));
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
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_1_7), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
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
#if __CUDA_ARCH__ == 1070
        int _smem_desc_lo_96 = ((s_a_addr) >> 4) & 0x7FFF;
        int _smem_desc_lo_97 = (((s_vT_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
        int _smem_desc_lo_96 = ((s_a_addr) >> 4) & 0x3FFF;
        int _smem_desc_lo_97 = (((s_vT_addr) >> 4) & 0x3FFF) | 0x2000000;
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
        : "r"(taddr_1 + 1048576));
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
    int row_o_8 = ld_row;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_8 * 128 + ((ld_ch ^ row_o_8 & 7) << 4))));
    int orow_8 = row0 + row_o_8;
    if (orow_8 < n_valid) {
        long long out_i_8 = ((long long)(tok0 + orow_8) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i_8 / 2) = _iv4;
        }
    }
    int row_o_9 = ld_row + 4;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_9 * 128 + ((ld_ch ^ row_o_9 & 7) << 4))));
    int orow_9 = row0 + row_o_9;
    if (orow_9 < n_valid) {
        long long out_i_9 = ((long long)(tok0 + orow_9) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i_9 / 2) = _iv4;
        }
    }
    int row_o_10 = ld_row + 8;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_10 * 128 + ((ld_ch ^ row_o_10 & 7) << 4))));
    int orow_10 = row0 + row_o_10;
    if (orow_10 < n_valid) {
        long long out_i_10 = ((long long)(tok0 + orow_10) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i_10 / 2) = _iv4;
        }
    }
    int row_o_11 = ld_row + 12;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_11 * 128 + ((ld_ch ^ row_o_11 & 7) << 4))));
    int orow_11 = row0 + row_o_11;
    if (orow_11 < n_valid) {
        long long out_i_11 = ((long long)(tok0 + orow_11) * (long long)num_v_heads + (long long)hv) * 128 + 0 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i_11 / 2) = _iv4;
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
    int row_o_12 = ld_row;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_12 * 128 + ((ld_ch ^ row_o_12 & 7) << 4))));
    int orow_12 = row0 + row_o_12;
    if (orow_12 < n_valid) {
        long long out_i_12 = ((long long)(tok0 + orow_12) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i_12 / 2) = _iv4;
        }
    }
    int row_o_13 = ld_row + 4;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_13 * 128 + ((ld_ch ^ row_o_13 & 7) << 4))));
    int orow_13 = row0 + row_o_13;
    if (orow_13 < n_valid) {
        long long out_i_13 = ((long long)(tok0 + orow_13) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i_13 / 2) = _iv4;
        }
    }
    int row_o_14 = ld_row + 8;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_14 * 128 + ((ld_ch ^ row_o_14 & 7) << 4))));
    int orow_14 = row0 + row_o_14;
    if (orow_14 < n_valid) {
        long long out_i_14 = ((long long)(tok0 + orow_14) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i_14 / 2) = _iv4;
        }
    }
    int row_o_15 = ld_row + 12;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_o[(0) + 3]))
        : "r"(slot + (unsigned int)(row_o_15 * 128 + ((ld_ch ^ row_o_15 & 7) << 4))));
    int orow_15 = row0 + row_o_15;
    if (orow_15 < n_valid) {
        long long out_i_15 = ((long long)(tok0 + orow_15) * (long long)num_v_heads + (long long)hv) * 128 + 64 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_o[0 + 0], words_o[0 + 1], words_o[0 + 2], words_o[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_i_15 / 2) = _iv4;
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
