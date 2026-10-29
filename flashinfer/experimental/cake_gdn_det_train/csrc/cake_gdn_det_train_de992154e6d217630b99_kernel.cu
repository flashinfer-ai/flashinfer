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
#define SMEM_S_DV_OFF 81920
#define SMEM_S_DV_STAGE_BYTES 16384
#define SMEM_S_DV_STRIDE 16384
#define SMEM_S_DS_OFF 98304
#define SMEM_S_DS_STAGE_BYTES 8192
#define SMEM_S_DS_STRIDE 8192
#define SMEM_S_DST_OFF 98304
#define SMEM_S_DST_STAGE_BYTES 8192
#define SMEM_S_DST_STRIDE 8192
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
#define SMEM_TOTAL 108032
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
kernel_cake_gdn_det_train_de992154e6d217630b99(__nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ v_new, __nv_bfloat16* __restrict__ do_, __nv_bfloat16* __restrict__ dv2, __nv_bfloat16* __restrict__ h, __nv_bfloat16* __restrict__ dh, float* __restrict__ g_cs, __nv_bfloat16* __restrict__ dq_out, __nv_bfloat16* __restrict__ dk_out, __nv_bfloat16* __restrict__ dw_out, float* __restrict__ dg_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, float scale)
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
    #define l1_addr (mbar_base + 48)
    #define l2_addr (mbar_base + 56)
    #define l3_addr (mbar_base + 64)
    #define l4_addr (mbar_base + 72)
    #define l5_addr (mbar_base + 80)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* s_do = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_do_addr = smem + 0;
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
    __nv_bfloat16* s_dh = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_dh_addr = smem + 65536;
    __nv_bfloat16* s_dhT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_dhT_addr = smem + 65536;
    __nv_bfloat16* s_qn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_qn_addr = smem + 65536;
    __nv_bfloat16* s_qnT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_qnT_addr = smem + 65536;
    __nv_bfloat16* s_dv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 81920);
    const int s_dv_addr = smem + 81920;
    __nv_bfloat16* s_ds = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int s_ds_addr = smem + 98304;
    __nv_bfloat16* s_dsT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int s_dsT_addr = smem + 98304;
    float* s_g = reinterpret_cast<float*>(smem_raw + 106496);
    const int s_g_addr = smem + 106496;
    float* s_dgq = reinterpret_cast<float*>(smem_raw + 106752);
    const int s_dgq_addr = smem + 106752;
    float* s_dgk = reinterpret_cast<float*>(smem_raw + 107264);
    const int s_dgk_addr = smem + 107264;
    float* s_red = reinterpret_cast<float*>(smem_raw + 107776);
    const int s_red_addr = smem + 107776;

    // Mbarrier init (11 pipeline groups, 0 ordered-sequence groups, 11 barriers)
    // Mbarriers at smem_raw[107904..107992)

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
            // l1: 1 barriers, init_count=256
            mbarrier_init(smem + 107952, 256);
            // l2: 1 barriers, init_count=256
            mbarrier_init(smem + 107960, 256);
            // l3: 1 barriers, init_count=256
            mbarrier_init(smem + 107968, 256);
            // l4: 1 barriers, init_count=256
            mbarrier_init(smem + 107976, 256);
            // l5: 1 barriers, init_count=256
            mbarrier_init(smem + 107984, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 107992);
    if (warp == 0) {
        int _tmem_hold = smem + 107992;
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
    if (tid_2 < 64) {
        float gate_val = 0.0f;
        if (tid_2 < n_valid) {
            gate_val = g_cs[(long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv];
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
    float acc_s[16];
    float acc[16];
    float vals[8];
    unsigned int pk[4];
    unsigned int pair_a[1];
    unsigned int pair_b[1];
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
    asm volatile("cp.async.wait_group 1;");
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
    unsigned int _phase_m5_0 = 0;
    mbarrier_wait(m5_addr, _phase_m5_0);
    _phase_m5_0 ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
    long long out5_a = ((long long)(tok0 + row_a) * (long long)num_v_heads + (long long)hv) * 128 + (long long)n0 + (long long)col_q;
    long long out5_b = ((long long)(tok0 + row_b) * (long long)num_v_heads + (long long)hv) * 128 + (long long)n0 + (long long)col_q;
    #pragma unroll
    for (int half2 = 0; half2 < 2; half2++) {
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
            : "r"(ld_w + 1048576 + half2 * 32));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int i2 = 0; i2 < 4; i2++) {
            if (row_a < n_valid) {
                unsigned int _packed_0[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + i2 * 4)[_lp*2 + 0], (acc + i2 * 4)[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dq_out + (out5_a + (long long)((half2 * 4 + i2) * 8)))[0] = reinterpret_cast<unsigned*>(_packed_0)[0];
            }
            if (row_b < n_valid) {
                unsigned int _packed_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + i2 * 4 + 2)[_lp*2 + 0], (acc + i2 * 4 + 2)[_lp*2+1 + 0]));
                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dq_out + (out5_b + (long long)((half2 * 4 + i2) * 8)))[0] = reinterpret_cast<unsigned*>(_packed_1)[0];
            }
        }
    }
    asm volatile("cp.async.wait_group 0;");
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
    long long out3_a = ((long long)(tok0 + row_a) * (long long)num_v_heads + (long long)hv) * 128 + (long long)n0 + (long long)col_q;
    long long out3_b = ((long long)(tok0 + row_b) * (long long)num_v_heads + (long long)hv) * 128 + (long long)n0 + (long long)col_q;
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
            if (row_a < n_valid) {
                unsigned int _packed_2[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + i4 * 4)[_lp*2 + 0], (acc + i4 * 4)[_lp*2+1 + 0]));
                    _packed_2[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dk_out + (out3_a + (long long)((half4 * 4 + i4) * 8)))[0] = reinterpret_cast<unsigned*>(_packed_2)[0];
            }
            if (row_b < n_valid) {
                unsigned int _packed_3[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + i4 * 4 + 2)[_lp*2 + 0], (acc + i4 * 4 + 2)[_lp*2+1 + 0]));
                    _packed_3[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dk_out + (out3_b + (long long)((half4 * 4 + i4) * 8)))[0] = reinterpret_cast<unsigned*>(_packed_3)[0];
            }
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
        for (int i5 = 0; i5 < 4; i5++) {
            if (row_a < n_valid) {
                unsigned int _packed_4[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + i5 * 4)[_lp*2 + 0], (acc + i5 * 4)[_lp*2+1 + 0]));
                    _packed_4[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dw_out + (out3_a + (long long)((half5 * 4 + i5) * 8)))[0] = reinterpret_cast<unsigned*>(_packed_4)[0];
            }
            if (row_b < n_valid) {
                unsigned int _packed_5[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc + i5 * 4 + 2)[_lp*2 + 0], (acc + i5 * 4 + 2)[_lp*2+1 + 0]));
                    _packed_5[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(dw_out + (out3_b + (long long)((half5 * 4 + i5) * 8)))[0] = reinterpret_cast<unsigned*>(_packed_5)[0];
            }
        }
    }
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
        int _max_42 = ((n_valid) > (1) ? (n_valid) : (1));
        float g6_last = s_g[_max_42 - 1];
        float _exp2_43 = approx_exp2(g6_last);
        float dg_last = __fmaf_rn(hdh_sum, _exp2_43, part_sum);
        float dg_val = s_dgq[tid_2] + s_dgq[64 + tid_2] - (s_dgk[tid_2] + s_dgk[64 + tid_2]);
        if (tid_2 == n_valid - 1) {
            dg_val = dg_val + dg_last;
        }
        dg_out[(long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv] = dg_val;
    }
    if (warp_1 == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
}

} // extern "C"
