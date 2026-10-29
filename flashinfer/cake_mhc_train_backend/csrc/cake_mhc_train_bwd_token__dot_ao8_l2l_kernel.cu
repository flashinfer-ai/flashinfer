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
#define NUM_MAIN_STAGES 1
#define SMEM_DPRE_PART_OFF 1024
#define SMEM_DPRE_PART_STAGE_BYTES 2048
#define SMEM_DPRE_PART_STRIDE 2048
#define SMEM_FOLD_OFF 3072
#define SMEM_FOLD_STAGE_BYTES 4096
#define SMEM_FOLD_STRIDE 4096
#define SMEM_GSUM_OFF 7168
#define SMEM_GSUM_STAGE_BYTES 128
#define SMEM_GSUM_STRIDE 128
#define SMEM_HIST_POOL_OFF 8192
#define SMEM_HIST_POOL_STAGE_BYTES 128
#define SMEM_HIST_POOL_STRIDE 128
#define SMEM_RING_OFF 9216
#define SMEM_RING_STAGE_BYTES 81920
#define SMEM_RING_STRIDE 81920
#define SMEM_TOTAL 91136
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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


extern "C" __device__ float __nv_expf(float);




__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_mhc_train_bwd_token__dot_ao8_l2l(__nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ dU, float* __restrict__ stats, float* __restrict__ gstats, float* __restrict__ scale, float* __restrict__ base, __nv_bfloat16* __restrict__ dzp, float* __restrict__ tokp, float* __restrict__ param_partial, int T, int t_pad, int rows_per_cta, float hc_eps)
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
    #define ring_bar_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* dpre_part = reinterpret_cast<float*>(smem_raw + 1024);
    const int dpre_part_addr = smem + 1024;
    float* fold = reinterpret_cast<float*>(smem_raw + 3072);
    const int fold_addr = smem + 3072;
    float* gsum = reinterpret_cast<float*>(smem_raw + 7168);
    const int gsum_addr = smem + 7168;
    float* hist_pool = reinterpret_cast<float*>(smem_raw + 8192);
    const int hist_pool_addr = smem + 8192;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int ring_addr = smem + 9216;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // ring_bar: 16 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int warp_0 = (int)warp;
    int lane_1 = (int)lane;
    int tid_2 = tid;
    int t0 = bid * rows_per_cta;
    int t_end = t0 + rows_per_cta;
    int t1 = ((t_end < T) ? t_end : T);
    int t_last = t1 - 1;
    int n = ((lane_1 < 24) ? lane_1 : 23);
    long long n64 = (long long)n;
    long long lane64 = (long long)lane_1;
    float sc0 = scale[0];
    float sc1 = scale[1];
    float sc2 = scale[2];
    float sc = ((lane_1 < 4) ? sc0 : ((lane_1 < 8) ? sc1 : sc2));
    float bs = base[n64];
    int lane8 = lane_1 & 7;
    int lane_hi = lane_1 >> 3;
    int lc4 = ((lane_1 < 4) ? lane_1 : 3);
    int gcol = ((lane_1 >= 4) ? lane_1 - 4 : 0);
    long long col_off = (long long)tid_2 * 16;
    float acc[2];
    float comp[2];
    #pragma unroll
    for (int e = 0; e < 2; e++) {
        acc[e] = 0.0f;
        comp[e] = 0.0f;
    }
    long long wcol = (long long)warp_0 * 512;
    unsigned int lane32 = (unsigned int)(lane_1 * 32);
    int st__1_0 = warp_0 * 2;
    unsigned int sa__1_0 = (unsigned int)(ring_addr + (unsigned int)(st__1_0 * 5120));
    int st__1_1 = warp_0 * 2 + 1;
    unsigned int sa__1_1 = (unsigned int)(ring_addr + (unsigned int)(st__1_1 * 5120));
    if (t0 < t1) {
        int tp_1_0 = t0;
        int tpc_2_0 = ((tp_1_0 < t1) ? tp_1_0 : t_last);
        long long tp_t_0 = (long long)tpc_2_0;
        int tp_1_1 = t0 + 1;
        int tpc_2_1 = ((tp_1_1 < t1) ? tp_1_1 : t_last);
        long long tp_t_1 = (long long)tpc_2_1;
        int tp_1_2 = t0 + 2;
        int tpc_2_2 = ((tp_1_2 < t1) ? tp_1_2 : t_last);
        long long tp_t_2 = (long long)tpc_2_2;
        int tp_1_3 = t0 + 3;
        int tpc_2_3 = ((tp_1_3 < t1) ? tp_1_3 : t_last);
        long long tp_t_3 = (long long)tpc_2_3;
        int tp_1_4 = t0 + 4;
        int tpc_2_4 = ((tp_1_4 < t1) ? tp_1_4 : t_last);
        long long tp_t_4 = (long long)tpc_2_4;
        int tp_1_5 = t0 + 5;
        int tpc_2_5 = ((tp_1_5 < t1) ? tp_1_5 : t_last);
        long long tp_t_5 = (long long)tpc_2_5;
        int tp_1_6 = t0 + 6;
        int tpc_2_6 = ((tp_1_6 < t1) ? tp_1_6 : t_last);
        long long tp_t_6 = (long long)tpc_2_6;
        int tp_1_7 = t0 + 7;
        int tpc_2_7 = ((tp_1_7 < t1) ? tp_1_7 : t_last);
        long long tp_t_7 = (long long)tpc_2_7;
        int tp_1_8 = t0 + 8;
        int tpc_2_8 = ((tp_1_8 < t1) ? tp_1_8 : t_last);
        long long tp_t_8 = (long long)tpc_2_8;
        int tp_1_9 = t0 + 9;
        int tpc_2_9 = ((tp_1_9 < t1) ? tp_1_9 : t_last);
        long long tp_t_9 = (long long)tpc_2_9;
        int pro_base = warp_0 * 4 + lane_hi;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_0) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0), "l"(dU + (tp_t_0 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 1024), "l"(X + (tp_t_0 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 2048), "l"(X + ((tp_t_0 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 3072), "l"(X + ((tp_t_0 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 4096), "l"(X + ((tp_t_0 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_1) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1), "l"(dU + (tp_t_1 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 1024), "l"(X + (tp_t_1 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 2048), "l"(X + ((tp_t_1 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 3072), "l"(X + ((tp_t_1 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 4096), "l"(X + ((tp_t_1 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
    }
    int gi[1];
    gi[0] = 0;
    int ring_grp = 0;
    #pragma unroll 1
    for (int g0 = t0; g0 < t1; g0 += 8) {
        int cur_base = (gi[0] & 1) * 256 + warp_0 * 32;
        int nxt_base = (gi[0] + 1 & 1) * 256 + warp_0 * 4 + lane_hi;
        int tok = g0 + warp_0;
        int tn_1_0 = g0 + 8;
        int tnc_2_0 = ((tn_1_0 < t1) ? tn_1_0 : t_last);
        long long tn64_1_0 = (long long)tnc_2_0;
        int th_1_0 = g0;
        int thc_1_0 = ((th_1_0 < t1) ? th_1_0 : t_last);
        long long th2_0 = (long long)thc_1_0;
        int tn_1_1 = g0 + 9;
        int tnc_2_1 = ((tn_1_1 < t1) ? tn_1_1 : t_last);
        long long tn64_1_1 = (long long)tnc_2_1;
        int th_1_1 = g0 + 1;
        int thc_1_1 = ((th_1_1 < t1) ? th_1_1 : t_last);
        long long th2_1 = (long long)thc_1_1;
        int tn_1_2 = g0 + 10;
        int tnc_2_2 = ((tn_1_2 < t1) ? tn_1_2 : t_last);
        long long tn64_1_2 = (long long)tnc_2_2;
        int th_1_2 = g0 + 2;
        int thc_1_2 = ((th_1_2 < t1) ? th_1_2 : t_last);
        long long th2_2 = (long long)thc_1_2;
        int tn_1_3 = g0 + 11;
        int tnc_2_3 = ((tn_1_3 < t1) ? tn_1_3 : t_last);
        long long tn64_1_3 = (long long)tnc_2_3;
        int th_1_3 = g0 + 3;
        int thc_1_3 = ((th_1_3 < t1) ? th_1_3 : t_last);
        long long th2_3 = (long long)thc_1_3;
        int tn_1_4 = g0 + 12;
        int tnc_2_4 = ((tn_1_4 < t1) ? tn_1_4 : t_last);
        long long tn64_1_4 = (long long)tnc_2_4;
        int th_1_4 = g0 + 4;
        int thc_1_4 = ((th_1_4 < t1) ? th_1_4 : t_last);
        long long th2_4 = (long long)thc_1_4;
        int tn_1_5 = g0 + 13;
        int tnc_2_5 = ((tn_1_5 < t1) ? tn_1_5 : t_last);
        long long tn64_1_5 = (long long)tnc_2_5;
        int th_1_5 = g0 + 5;
        int thc_1_5 = ((th_1_5 < t1) ? th_1_5 : t_last);
        long long th2_5 = (long long)thc_1_5;
        int tn_1_6 = g0 + 14;
        int tnc_2_6 = ((tn_1_6 < t1) ? tn_1_6 : t_last);
        long long tn64_1_6 = (long long)tnc_2_6;
        int th_1_6 = g0 + 6;
        int thc_1_6 = ((th_1_6 < t1) ? th_1_6 : t_last);
        long long th2_6 = (long long)thc_1_6;
        int tn_1_7 = g0 + 15;
        int tnc_2_7 = ((tn_1_7 < t1) ? tn_1_7 : t_last);
        long long tn64_1_7 = (long long)tnc_2_7;
        int th_1_7 = g0 + 7;
        int thc_1_7 = ((th_1_7 < t1) ? th_1_7 : t_last);
        long long th2_7 = (long long)thc_1_7;
        int tn_1_8 = g0 + 16;
        int tnc_2_8 = ((tn_1_8 < t1) ? tn_1_8 : t_last);
        long long tn64_1_8 = (long long)tnc_2_8;
        int th_1_8 = g0 + 8;
        int thc_1_8 = ((th_1_8 < t1) ? th_1_8 : t_last);
        long long th2_8 = (long long)thc_1_8;
        int tn_1_9 = g0 + 17;
        int tnc_2_9 = ((tn_1_9 < t1) ? tn_1_9 : t_last);
        long long tn64_1_9 = (long long)tnc_2_9;
        int th_1_9 = g0 + 9;
        int thc_1_9 = ((th_1_9 < t1) ? th_1_9 : t_last);
        long long th2_9 = (long long)thc_1_9;
        int own_base = (gi[0] & 1) * 256 + warp_0 * 4 + lane_hi;
        int tokc = ((tok < t1) ? tok : t_last);
        long long tok64 = (long long)tokc;
        float dp[1];
        long long srow = tok64 * 64;
        float r = stats[srow];
        float z_load = stats[srow + 1 + n64];
        float gval = gstats[tok64 * 32 + (long long)gcol];
        float z = ((lane_1 < 24) ? z_load : 0.0f);
        float a_map;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a_map) : "f"(z), "f"(sc), "f"(bs));
        float a = ((lane_1 < 24) ? a_map : 0.0f);
        float gtwo;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(gtwo) : "f"(gval), "f"(2.0f));
        float hist[40];
        float _sub_0;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(0.0f), "f"(fabsf(a)));
        float _exp_0 = __nv_expf(_sub_0);
        float _add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(1.0f), "f"(_exp_0));
        float _fdiv_rn_0 = __fdiv_rn(1.0f, _add_0);
        float _fdiv_rn_1 = __fdiv_rn(_exp_0, _add_0);
        float sig = ((a >= 0.0f) ? _fdiv_rn_0 : _fdiv_rn_1);
        float sneg = ((a >= 0.0f) ? _fdiv_rn_1 : _fdiv_rn_0);
        float _add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(sig), "f"(hc_eps));
        float _mul_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(2.0f), "f"(sig));
        float logit = ((lane_1 >= 8 && lane_1 < 24) ? a : 0.0f);
        float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
        float _max_0 = fmaxf(logit, _shfl_xor_sync_1);
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, _max_0, 2);
        float _sub_2;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2) : "f"(logit), "f"(fmaxf(_max_0, _shfl_xor_sync_3)));
        float _exp_1 = __nv_expf(_sub_2);
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
        float add;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(_exp_1), "f"(_shfl_xor_sync_4));
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, add, 2);
        float _add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(add), "f"(_shfl_xor_sync_6));
        float _fdiv_rn_2 = __fdiv_rn(_exp_1, _add_5);
        hist[0] = _fdiv_rn_2;
        float _add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(_fdiv_rn_2), "f"(hc_eps));
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, _add_3, 4);
        float s;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s) : "f"(_add_3), "f"(_shfl_xor_sync_7));
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, s, 24);
        float _add_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(s), "f"(_shfl_xor_sync_10));
        float _add_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(_add_9), "f"(hc_eps));
        float c = __fdiv_rn(_add_3, _add_8);
        hist[1] = c;
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, c, 1);
        float s_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_0) : "f"(c), "f"(_shfl_xor_sync_11));
        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, s_0, 2);
        float _add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(s_0), "f"(_shfl_xor_sync_14));
        float _add_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(_add_13), "f"(hc_eps));
        float c_0 = __fdiv_rn(c, _add_12);
        hist[2] = c_0;
        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, c_0, 4);
        float s_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_1) : "f"(c_0), "f"(_shfl_xor_sync_15));
        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, s_1, 24);
        float _add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(s_1), "f"(_shfl_xor_sync_18));
        float _add_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(_add_17), "f"(hc_eps));
        float c_1 = __fdiv_rn(c_0, _add_16);
        hist[3] = c_1;
        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, c_1, 1);
        float s_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_2) : "f"(c_1), "f"(_shfl_xor_sync_19));
        float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, s_2, 2);
        float _add_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(s_2), "f"(_shfl_xor_sync_22));
        float _add_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(_add_21), "f"(hc_eps));
        float c_2 = __fdiv_rn(c_1, _add_20);
        hist[4] = c_2;
        float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, c_2, 4);
        float s_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_3) : "f"(c_2), "f"(_shfl_xor_sync_23));
        float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, s_3, 24);
        float _add_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(s_3), "f"(_shfl_xor_sync_26));
        float _add_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(_add_25), "f"(hc_eps));
        float c_3 = __fdiv_rn(c_2, _add_24);
        hist[5] = c_3;
        float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, c_3, 1);
        float s_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_4) : "f"(c_3), "f"(_shfl_xor_sync_27));
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, s_4, 2);
        float _add_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(s_4), "f"(_shfl_xor_sync_30));
        float _add_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(_add_29), "f"(hc_eps));
        float c_4 = __fdiv_rn(c_3, _add_28);
        hist[6] = c_4;
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, c_4, 4);
        float s_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_5) : "f"(c_4), "f"(_shfl_xor_sync_31));
        float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, s_5, 24);
        float _add_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(s_5), "f"(_shfl_xor_sync_34));
        float _add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(_add_33), "f"(hc_eps));
        float c_5 = __fdiv_rn(c_4, _add_32);
        hist[7] = c_5;
        float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, c_5, 1);
        float s_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_6) : "f"(c_5), "f"(_shfl_xor_sync_35));
        float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, s_6, 2);
        float _add_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(s_6), "f"(_shfl_xor_sync_38));
        float _add_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(_add_37), "f"(hc_eps));
        float c_6 = __fdiv_rn(c_5, _add_36);
        hist[8] = c_6;
        float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, c_6, 4);
        float s_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_7) : "f"(c_6), "f"(_shfl_xor_sync_39));
        float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, s_7, 24);
        float _add_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(s_7), "f"(_shfl_xor_sync_42));
        float _add_40;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(_add_41), "f"(hc_eps));
        float c_7 = __fdiv_rn(c_6, _add_40);
        hist[9] = c_7;
        mbarrier_wait(ring_bar_addr + (st__1_0) * 8, 0);
        unsigned int ring_pk[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(0) + 3]))
            : "r"(sa__1_0 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(4) + 3]))
            : "r"(sa__1_0 + 16 + lane32));
        unsigned int ring_pk_0[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[(0) + 3]))
            : "r"(sa__1_0 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[(4) + 3]))
            : "r"(sa__1_0 + 1040 + lane32));
        float dots_p[4];
        float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, c_7, 1);
        float s_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_8) : "f"(c_7), "f"(_shfl_xor_sync_43));
        float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, s_8, 2);
        float _add_45;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(s_8), "f"(_shfl_xor_sync_46));
        float _add_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(_add_45), "f"(hc_eps));
        float c_8 = __fdiv_rn(c_7, _add_44);
        hist[10] = c_8;
        float _dot_47;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_47) : "r"(ring_pk_0[0]), "r"(ring_pk[0]), "f"(-0.0f));
        dots_p[0] = _dot_47;
        float _dot_48;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_48) : "r"(ring_pk_0[1]), "r"(ring_pk[1]), "f"(dots_p[0]));
        dots_p[0] = _dot_48;
        float _dot_49;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_49) : "r"(ring_pk_0[2]), "r"(ring_pk[2]), "f"(dots_p[0]));
        dots_p[0] = _dot_49;
        float _dot_50;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_50) : "r"(ring_pk_0[3]), "r"(ring_pk[3]), "f"(dots_p[0]));
        dots_p[0] = _dot_50;
        float _dot_51;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_51) : "r"(ring_pk_0[4]), "r"(ring_pk[4]), "f"(dots_p[0]));
        dots_p[0] = _dot_51;
        float _dot_52;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_52) : "r"(ring_pk_0[5]), "r"(ring_pk[5]), "f"(dots_p[0]));
        dots_p[0] = _dot_52;
        float _dot_53;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_53) : "r"(ring_pk_0[6]), "r"(ring_pk[6]), "f"(dots_p[0]));
        dots_p[0] = _dot_53;
        float _dot_54;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_54) : "r"(ring_pk_0[7]), "r"(ring_pk[7]), "f"(dots_p[0]));
        dots_p[0] = _dot_54;
        unsigned int ring_pk_1[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(0) + 3]))
            : "r"(sa__1_0 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(4) + 3]))
            : "r"(sa__1_0 + 2064 + lane32));
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, c_8, 4);
        float s_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_9) : "f"(c_8), "f"(_shfl_xor_sync_55));
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, s_9, 24);
        float _add_57;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(s_9), "f"(_shfl_xor_sync_58));
        float _add_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(_add_57), "f"(hc_eps));
        float c_9 = __fdiv_rn(c_8, _add_56);
        hist[11] = c_9;
        float _dot_59;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_59) : "r"(ring_pk_1[0]), "r"(ring_pk[0]), "f"(-0.0f));
        dots_p[1] = _dot_59;
        float _dot_60;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_60) : "r"(ring_pk_1[1]), "r"(ring_pk[1]), "f"(dots_p[1]));
        dots_p[1] = _dot_60;
        float _dot_61;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_61) : "r"(ring_pk_1[2]), "r"(ring_pk[2]), "f"(dots_p[1]));
        dots_p[1] = _dot_61;
        float _dot_62;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_62) : "r"(ring_pk_1[3]), "r"(ring_pk[3]), "f"(dots_p[1]));
        dots_p[1] = _dot_62;
        float _dot_63;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_63) : "r"(ring_pk_1[4]), "r"(ring_pk[4]), "f"(dots_p[1]));
        dots_p[1] = _dot_63;
        float _dot_64;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_64) : "r"(ring_pk_1[5]), "r"(ring_pk[5]), "f"(dots_p[1]));
        dots_p[1] = _dot_64;
        float _dot_65;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_65) : "r"(ring_pk_1[6]), "r"(ring_pk[6]), "f"(dots_p[1]));
        dots_p[1] = _dot_65;
        float _dot_66;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_66) : "r"(ring_pk_1[7]), "r"(ring_pk[7]), "f"(dots_p[1]));
        dots_p[1] = _dot_66;
        unsigned int ring_pk_2[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(0) + 3]))
            : "r"(sa__1_0 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(4) + 3]))
            : "r"(sa__1_0 + 3088 + lane32));
        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, c_9, 1);
        float s_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_10) : "f"(c_9), "f"(_shfl_xor_sync_67));
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, s_10, 2);
        float _add_69;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(s_10), "f"(_shfl_xor_sync_70));
        float _add_68;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(_add_69), "f"(hc_eps));
        float c_10 = __fdiv_rn(c_9, _add_68);
        hist[12] = c_10;
        float _dot_71;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_71) : "r"(ring_pk_2[0]), "r"(ring_pk[0]), "f"(-0.0f));
        dots_p[2] = _dot_71;
        float _dot_72;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_72) : "r"(ring_pk_2[1]), "r"(ring_pk[1]), "f"(dots_p[2]));
        dots_p[2] = _dot_72;
        float _dot_73;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_73) : "r"(ring_pk_2[2]), "r"(ring_pk[2]), "f"(dots_p[2]));
        dots_p[2] = _dot_73;
        float _dot_74;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_74) : "r"(ring_pk_2[3]), "r"(ring_pk[3]), "f"(dots_p[2]));
        dots_p[2] = _dot_74;
        float _dot_75;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_75) : "r"(ring_pk_2[4]), "r"(ring_pk[4]), "f"(dots_p[2]));
        dots_p[2] = _dot_75;
        float _dot_76;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_76) : "r"(ring_pk_2[5]), "r"(ring_pk[5]), "f"(dots_p[2]));
        dots_p[2] = _dot_76;
        float _dot_77;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_77) : "r"(ring_pk_2[6]), "r"(ring_pk[6]), "f"(dots_p[2]));
        dots_p[2] = _dot_77;
        float _dot_78;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_78) : "r"(ring_pk_2[7]), "r"(ring_pk[7]), "f"(dots_p[2]));
        dots_p[2] = _dot_78;
        unsigned int ring_pk_3[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(0) + 3]))
            : "r"(sa__1_0 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(4) + 3]))
            : "r"(sa__1_0 + 4112 + lane32));
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, c_10, 4);
        float s_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_11) : "f"(c_10), "f"(_shfl_xor_sync_79));
        float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, s_11, 24);
        float _add_81;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(s_11), "f"(_shfl_xor_sync_82));
        float _add_80;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(_add_81), "f"(hc_eps));
        float c_11 = __fdiv_rn(c_10, _add_80);
        hist[13] = c_11;
        float _dot_83;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_83) : "r"(ring_pk_3[0]), "r"(ring_pk[0]), "f"(-0.0f));
        dots_p[3] = _dot_83;
        float _dot_84;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_84) : "r"(ring_pk_3[1]), "r"(ring_pk[1]), "f"(dots_p[3]));
        dots_p[3] = _dot_84;
        float _dot_85;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_85) : "r"(ring_pk_3[2]), "r"(ring_pk[2]), "f"(dots_p[3]));
        dots_p[3] = _dot_85;
        float _dot_86;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_86) : "r"(ring_pk_3[3]), "r"(ring_pk[3]), "f"(dots_p[3]));
        dots_p[3] = _dot_86;
        float _dot_87;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_87) : "r"(ring_pk_3[4]), "r"(ring_pk[4]), "f"(dots_p[3]));
        dots_p[3] = _dot_87;
        float _dot_88;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_88) : "r"(ring_pk_3[5]), "r"(ring_pk[5]), "f"(dots_p[3]));
        dots_p[3] = _dot_88;
        float _dot_89;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_89) : "r"(ring_pk_3[6]), "r"(ring_pk[6]), "f"(dots_p[3]));
        dots_p[3] = _dot_89;
        float _dot_90;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_90) : "r"(ring_pk_3[7]), "r"(ring_pk[7]), "f"(dots_p[3]));
        dots_p[3] = _dot_90;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_0) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0), "l"(dU + (th2_2 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 1024), "l"(X + (th2_2 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 2048), "l"(X + ((th2_2 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 3072), "l"(X + ((th2_2 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 4096), "l"(X + ((th2_2 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, c_11, 1);
        float s_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_12) : "f"(c_11), "f"(_shfl_xor_sync_91));
        float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, s_12, 2);
        float _add_93;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(s_12), "f"(_shfl_xor_sync_94));
        float _add_92;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(_add_93), "f"(hc_eps));
        float c_12 = __fdiv_rn(c_11, _add_92);
        hist[14] = c_12;
        int upper = lane_1 & 16;
        float send = ((upper != 0) ? dots_p[0] : dots_p[2]);
        float keep = ((upper != 0) ? dots_p[2] : dots_p[0]);
        float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
        float add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_0) : "f"(keep), "f"(_shfl_xor_sync_95));
        float send_1 = ((upper != 0) ? dots_p[1] : dots_p[3]);
        float keep_1 = ((upper != 0) ? dots_p[3] : dots_p[1]);
        float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, send_1, 16);
        float add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(keep_1), "f"(_shfl_xor_sync_96));
        int upper_1 = lane_1 & 8;
        float send_2 = ((upper_1 != 0) ? add_0 : add_1);
        float keep_2 = ((upper_1 != 0) ? add_1 : add_0);
        float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, send_2, 8);
        float add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(keep_2), "f"(_shfl_xor_sync_97));
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, c_12, 4);
        float s_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_13) : "f"(c_12), "f"(_shfl_xor_sync_98));
        float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, s_13, 24);
        float _add_100;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(s_13), "f"(_shfl_xor_sync_101));
        float _add_99;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(_add_100), "f"(hc_eps));
        float c_13 = __fdiv_rn(c_12, _add_99);
        hist[15] = c_13;
        float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, add_2, 4);
        float add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(add_2), "f"(_shfl_xor_sync_102));
        float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, add_3, 2);
        float add_1_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_0) : "f"(add_3), "f"(_shfl_xor_sync_103));
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, add_1_0, 1);
        float _add_104;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(add_1_0), "f"(_shfl_xor_sync_105));
        dpre_part[own_base] = _add_104;
        float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, c_13, 1);
        float s_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_14) : "f"(c_13), "f"(_shfl_xor_sync_106));
        float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, s_14, 2);
        float _add_108;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(s_14), "f"(_shfl_xor_sync_109));
        float _add_107;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(_add_108), "f"(hc_eps));
        float c_14 = __fdiv_rn(c_13, _add_107);
        hist[16] = c_14;
        float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, c_14, 4);
        float s_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_15) : "f"(c_14), "f"(_shfl_xor_sync_110));
        float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, s_15, 24);
        float _add_112;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(s_15), "f"(_shfl_xor_sync_113));
        float _add_111;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(_add_112), "f"(hc_eps));
        float c_15 = __fdiv_rn(c_14, _add_111);
        hist[17] = c_15;
        float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, c_15, 1);
        float s_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_16) : "f"(c_15), "f"(_shfl_xor_sync_114));
        float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, s_16, 2);
        float _add_116;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(s_16), "f"(_shfl_xor_sync_117));
        float _add_115;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(_add_116), "f"(hc_eps));
        float c_16 = __fdiv_rn(c_15, _add_115);
        hist[18] = c_16;
        mbarrier_wait(ring_bar_addr + (st__1_1) * 8, 0);
        unsigned int ring_pk_4[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[(0) + 3]))
            : "r"(sa__1_1 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[(4) + 3]))
            : "r"(sa__1_1 + 16 + lane32));
        unsigned int ring_pk_5[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[(0) + 3]))
            : "r"(sa__1_1 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[(4) + 3]))
            : "r"(sa__1_1 + 1040 + lane32));
        float dots_p_0[4];
        float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, c_16, 4);
        float s_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_17) : "f"(c_16), "f"(_shfl_xor_sync_118));
        float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, s_17, 24);
        float _add_120;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(s_17), "f"(_shfl_xor_sync_121));
        float _add_119;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(_add_120), "f"(hc_eps));
        float c_17 = __fdiv_rn(c_16, _add_119);
        hist[19] = c_17;
        float _dot_122;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_122) : "r"(ring_pk_5[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[0] = _dot_122;
        float _dot_123;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_123) : "r"(ring_pk_5[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_123;
        float _dot_124;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_124) : "r"(ring_pk_5[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_124;
        float _dot_125;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_125) : "r"(ring_pk_5[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_125;
        float _dot_126;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_126) : "r"(ring_pk_5[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_126;
        float _dot_127;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_127) : "r"(ring_pk_5[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_127;
        float _dot_128;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_128) : "r"(ring_pk_5[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_128;
        float _dot_129;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_129) : "r"(ring_pk_5[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_129;
        unsigned int ring_pk_6[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(0) + 3]))
            : "r"(sa__1_1 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(4) + 3]))
            : "r"(sa__1_1 + 2064 + lane32));
        float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, c_17, 1);
        float s_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_18) : "f"(c_17), "f"(_shfl_xor_sync_130));
        float _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, s_18, 2);
        float _add_132;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(s_18), "f"(_shfl_xor_sync_133));
        float _add_131;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(_add_132), "f"(hc_eps));
        float c_18 = __fdiv_rn(c_17, _add_131);
        hist[20] = c_18;
        float _dot_134;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_134) : "r"(ring_pk_6[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[1] = _dot_134;
        float _dot_135;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_135) : "r"(ring_pk_6[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_135;
        float _dot_136;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_136) : "r"(ring_pk_6[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_136;
        float _dot_137;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_137) : "r"(ring_pk_6[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_137;
        float _dot_138;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_138) : "r"(ring_pk_6[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_138;
        float _dot_139;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_139) : "r"(ring_pk_6[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_139;
        float _dot_140;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_140) : "r"(ring_pk_6[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_140;
        float _dot_141;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_141) : "r"(ring_pk_6[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_141;
        unsigned int ring_pk_7[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(0) + 3]))
            : "r"(sa__1_1 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(4) + 3]))
            : "r"(sa__1_1 + 3088 + lane32));
        float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, c_18, 4);
        float s_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_19) : "f"(c_18), "f"(_shfl_xor_sync_142));
        float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, s_19, 24);
        float _add_144;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(s_19), "f"(_shfl_xor_sync_145));
        float _add_143;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(_add_144), "f"(hc_eps));
        float c_19 = __fdiv_rn(c_18, _add_143);
        hist[21] = c_19;
        float _dot_146;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_146) : "r"(ring_pk_7[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[2] = _dot_146;
        float _dot_147;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_147) : "r"(ring_pk_7[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_147;
        float _dot_148;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_148) : "r"(ring_pk_7[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_148;
        float _dot_149;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_149) : "r"(ring_pk_7[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_149;
        float _dot_150;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_150) : "r"(ring_pk_7[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_150;
        float _dot_151;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_151) : "r"(ring_pk_7[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_151;
        float _dot_152;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_152) : "r"(ring_pk_7[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_152;
        float _dot_153;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_153) : "r"(ring_pk_7[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_153;
        unsigned int ring_pk_8[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(0) + 3]))
            : "r"(sa__1_1 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(4) + 3]))
            : "r"(sa__1_1 + 4112 + lane32));
        float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, c_19, 1);
        float s_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_20) : "f"(c_19), "f"(_shfl_xor_sync_154));
        float _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, s_20, 2);
        float _add_156;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(s_20), "f"(_shfl_xor_sync_157));
        float _add_155;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155) : "f"(_add_156), "f"(hc_eps));
        float c_20 = __fdiv_rn(c_19, _add_155);
        hist[22] = c_20;
        float _dot_158;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_158) : "r"(ring_pk_8[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[3] = _dot_158;
        float _dot_159;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_159) : "r"(ring_pk_8[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_159;
        float _dot_160;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_160) : "r"(ring_pk_8[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_160;
        float _dot_161;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_161) : "r"(ring_pk_8[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_161;
        float _dot_162;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_162) : "r"(ring_pk_8[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_162;
        float _dot_163;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_163) : "r"(ring_pk_8[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_163;
        float _dot_164;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_164) : "r"(ring_pk_8[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_164;
        float _dot_165;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_165) : "r"(ring_pk_8[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_165;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_1) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1), "l"(dU + (th2_3 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 1024), "l"(X + (th2_3 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 2048), "l"(X + ((th2_3 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 3072), "l"(X + ((th2_3 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 4096), "l"(X + ((th2_3 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, c_20, 4);
        float s_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_21) : "f"(c_20), "f"(_shfl_xor_sync_166));
        float _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, s_21, 24);
        float _add_168;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(s_21), "f"(_shfl_xor_sync_169));
        float _add_167;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(_add_168), "f"(hc_eps));
        float c_21 = __fdiv_rn(c_20, _add_167);
        hist[23] = c_21;
        int upper_0 = lane_1 & 16;
        float send_0 = ((upper_0 != 0) ? dots_p_0[0] : dots_p_0[2]);
        float keep_0 = ((upper_0 != 0) ? dots_p_0[2] : dots_p_0[0]);
        float _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, send_0, 16);
        float add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(keep_0), "f"(_shfl_xor_sync_170));
        float send_1_0 = ((upper_0 != 0) ? dots_p_0[1] : dots_p_0[3]);
        float keep_1_0 = ((upper_0 != 0) ? dots_p_0[3] : dots_p_0[1]);
        float _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, send_1_0, 16);
        float add_1_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_1) : "f"(keep_1_0), "f"(_shfl_xor_sync_171));
        int upper_1_0 = lane_1 & 8;
        float send_2_0 = ((upper_1_0 != 0) ? add_4 : add_1_1);
        float keep_2_0 = ((upper_1_0 != 0) ? add_1_1 : add_4);
        float _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, send_2_0, 8);
        float add_2_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_0) : "f"(keep_2_0), "f"(_shfl_xor_sync_172));
        float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, c_21, 1);
        float s_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_22) : "f"(c_21), "f"(_shfl_xor_sync_173));
        float _shfl_xor_sync_176 = __shfl_xor_sync(0xFFFFFFFF, s_22, 2);
        float _add_175;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_175) : "f"(s_22), "f"(_shfl_xor_sync_176));
        float _add_174;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_174) : "f"(_add_175), "f"(hc_eps));
        float c_22 = __fdiv_rn(c_21, _add_174);
        hist[24] = c_22;
        float _shfl_xor_sync_177 = __shfl_xor_sync(0xFFFFFFFF, add_2_0, 4);
        float add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(add_2_0), "f"(_shfl_xor_sync_177));
        float _shfl_xor_sync_178 = __shfl_xor_sync(0xFFFFFFFF, add_5, 2);
        float add_1_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_2) : "f"(add_5), "f"(_shfl_xor_sync_178));
        float _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, add_1_2, 1);
        float _add_179;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_179) : "f"(add_1_2), "f"(_shfl_xor_sync_180));
        dpre_part[own_base + 32] = _add_179;
        float _shfl_xor_sync_181 = __shfl_xor_sync(0xFFFFFFFF, c_22, 4);
        float s_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_23) : "f"(c_22), "f"(_shfl_xor_sync_181));
        float _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, s_23, 24);
        float _add_183;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183) : "f"(s_23), "f"(_shfl_xor_sync_184));
        float _add_182;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_182) : "f"(_add_183), "f"(hc_eps));
        float c_23 = __fdiv_rn(c_22, _add_182);
        hist[25] = c_23;
        float _shfl_xor_sync_185 = __shfl_xor_sync(0xFFFFFFFF, c_23, 1);
        float s_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_24) : "f"(c_23), "f"(_shfl_xor_sync_185));
        float _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, s_24, 2);
        float _add_187;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_187) : "f"(s_24), "f"(_shfl_xor_sync_188));
        float _add_186;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_186) : "f"(_add_187), "f"(hc_eps));
        float c_24 = __fdiv_rn(c_23, _add_186);
        hist[26] = c_24;
        float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, c_24, 4);
        float s_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_25) : "f"(c_24), "f"(_shfl_xor_sync_189));
        float _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, s_25, 24);
        float _add_191;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191) : "f"(s_25), "f"(_shfl_xor_sync_192));
        float _add_190;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_190) : "f"(_add_191), "f"(hc_eps));
        float c_25 = __fdiv_rn(c_24, _add_190);
        hist[27] = c_25;
        mbarrier_wait(ring_bar_addr + (st__1_0) * 8, 1);
        unsigned int ring_pk_9[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[(0) + 3]))
            : "r"(sa__1_0 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[(4) + 3]))
            : "r"(sa__1_0 + 16 + lane32));
        unsigned int ring_pk_10[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[(0) + 3]))
            : "r"(sa__1_0 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[(4) + 3]))
            : "r"(sa__1_0 + 1040 + lane32));
        float dots_p_1[4];
        float _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, c_25, 1);
        float s_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_26) : "f"(c_25), "f"(_shfl_xor_sync_193));
        float _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, s_26, 2);
        float _add_195;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(s_26), "f"(_shfl_xor_sync_196));
        float _add_194;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_194) : "f"(_add_195), "f"(hc_eps));
        float c_26 = __fdiv_rn(c_25, _add_194);
        hist[28] = c_26;
        float _dot_197;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_197) : "r"(ring_pk_10[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[0] = _dot_197;
        float _dot_198;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_198) : "r"(ring_pk_10[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_198;
        float _dot_199;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_199) : "r"(ring_pk_10[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_199;
        float _dot_200;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_200) : "r"(ring_pk_10[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_200;
        float _dot_201;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_201) : "r"(ring_pk_10[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_201;
        float _dot_202;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_202) : "r"(ring_pk_10[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_202;
        float _dot_203;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_203) : "r"(ring_pk_10[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_203;
        float _dot_204;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_204) : "r"(ring_pk_10[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_204;
        unsigned int ring_pk_11[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(0) + 3]))
            : "r"(sa__1_0 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(4) + 3]))
            : "r"(sa__1_0 + 2064 + lane32));
        float _shfl_xor_sync_205 = __shfl_xor_sync(0xFFFFFFFF, c_26, 4);
        float s_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_27) : "f"(c_26), "f"(_shfl_xor_sync_205));
        float _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, s_27, 24);
        float _add_207;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_207) : "f"(s_27), "f"(_shfl_xor_sync_208));
        float _add_206;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_206) : "f"(_add_207), "f"(hc_eps));
        float c_27 = __fdiv_rn(c_26, _add_206);
        hist[29] = c_27;
        float _dot_209;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_209) : "r"(ring_pk_11[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[1] = _dot_209;
        float _dot_210;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_210) : "r"(ring_pk_11[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_210;
        float _dot_211;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_211) : "r"(ring_pk_11[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_211;
        float _dot_212;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_212) : "r"(ring_pk_11[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_212;
        float _dot_213;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_213) : "r"(ring_pk_11[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_213;
        float _dot_214;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_214) : "r"(ring_pk_11[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_214;
        float _dot_215;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_215) : "r"(ring_pk_11[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_215;
        float _dot_216;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_216) : "r"(ring_pk_11[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_216;
        unsigned int ring_pk_12[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(0) + 3]))
            : "r"(sa__1_0 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(4) + 3]))
            : "r"(sa__1_0 + 3088 + lane32));
        float _shfl_xor_sync_217 = __shfl_xor_sync(0xFFFFFFFF, c_27, 1);
        float s_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_28) : "f"(c_27), "f"(_shfl_xor_sync_217));
        float _shfl_xor_sync_220 = __shfl_xor_sync(0xFFFFFFFF, s_28, 2);
        float _add_219;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_219) : "f"(s_28), "f"(_shfl_xor_sync_220));
        float _add_218;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_218) : "f"(_add_219), "f"(hc_eps));
        float c_28 = __fdiv_rn(c_27, _add_218);
        hist[30] = c_28;
        float _dot_221;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_221) : "r"(ring_pk_12[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[2] = _dot_221;
        float _dot_222;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_222) : "r"(ring_pk_12[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_222;
        float _dot_223;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_223) : "r"(ring_pk_12[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_223;
        float _dot_224;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_224) : "r"(ring_pk_12[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_224;
        float _dot_225;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_225) : "r"(ring_pk_12[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_225;
        float _dot_226;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_226) : "r"(ring_pk_12[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_226;
        float _dot_227;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_227) : "r"(ring_pk_12[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_227;
        float _dot_228;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_228) : "r"(ring_pk_12[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_228;
        unsigned int ring_pk_13[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(0) + 3]))
            : "r"(sa__1_0 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(4) + 3]))
            : "r"(sa__1_0 + 4112 + lane32));
        float _shfl_xor_sync_229 = __shfl_xor_sync(0xFFFFFFFF, c_28, 4);
        float s_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_29) : "f"(c_28), "f"(_shfl_xor_sync_229));
        float _shfl_xor_sync_232 = __shfl_xor_sync(0xFFFFFFFF, s_29, 24);
        float _add_231;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_231) : "f"(s_29), "f"(_shfl_xor_sync_232));
        float _add_230;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_230) : "f"(_add_231), "f"(hc_eps));
        float c_29 = __fdiv_rn(c_28, _add_230);
        hist[31] = c_29;
        float _dot_233;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_233) : "r"(ring_pk_13[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[3] = _dot_233;
        float _dot_234;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_234) : "r"(ring_pk_13[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_234;
        float _dot_235;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_235) : "r"(ring_pk_13[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_235;
        float _dot_236;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_236) : "r"(ring_pk_13[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_236;
        float _dot_237;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_237) : "r"(ring_pk_13[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_237;
        float _dot_238;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_238) : "r"(ring_pk_13[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_238;
        float _dot_239;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_239) : "r"(ring_pk_13[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_239;
        float _dot_240;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_240) : "r"(ring_pk_13[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_240;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_0) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0), "l"(dU + (th2_4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 1024), "l"(X + (th2_4 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 2048), "l"(X + ((th2_4 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 3072), "l"(X + ((th2_4 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 4096), "l"(X + ((th2_4 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_241 = __shfl_xor_sync(0xFFFFFFFF, c_29, 1);
        float s_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_30) : "f"(c_29), "f"(_shfl_xor_sync_241));
        float _shfl_xor_sync_244 = __shfl_xor_sync(0xFFFFFFFF, s_30, 2);
        float _add_243;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_243) : "f"(s_30), "f"(_shfl_xor_sync_244));
        float _add_242;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_242) : "f"(_add_243), "f"(hc_eps));
        float c_30 = __fdiv_rn(c_29, _add_242);
        hist[32] = c_30;
        int upper_2 = lane_1 & 16;
        float send_3 = ((upper_2 != 0) ? dots_p_1[0] : dots_p_1[2]);
        float keep_3 = ((upper_2 != 0) ? dots_p_1[2] : dots_p_1[0]);
        float _shfl_xor_sync_245 = __shfl_xor_sync(0xFFFFFFFF, send_3, 16);
        float add_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_6) : "f"(keep_3), "f"(_shfl_xor_sync_245));
        float send_1_1 = ((upper_2 != 0) ? dots_p_1[1] : dots_p_1[3]);
        float keep_1_1 = ((upper_2 != 0) ? dots_p_1[3] : dots_p_1[1]);
        float _shfl_xor_sync_246 = __shfl_xor_sync(0xFFFFFFFF, send_1_1, 16);
        float add_1_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_3) : "f"(keep_1_1), "f"(_shfl_xor_sync_246));
        int upper_1_1 = lane_1 & 8;
        float send_2_1 = ((upper_1_1 != 0) ? add_6 : add_1_3);
        float keep_2_1 = ((upper_1_1 != 0) ? add_1_3 : add_6);
        float _shfl_xor_sync_247 = __shfl_xor_sync(0xFFFFFFFF, send_2_1, 8);
        float add_2_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_1) : "f"(keep_2_1), "f"(_shfl_xor_sync_247));
        float _shfl_xor_sync_248 = __shfl_xor_sync(0xFFFFFFFF, c_30, 4);
        float s_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_31) : "f"(c_30), "f"(_shfl_xor_sync_248));
        float _shfl_xor_sync_251 = __shfl_xor_sync(0xFFFFFFFF, s_31, 24);
        float _add_250;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_250) : "f"(s_31), "f"(_shfl_xor_sync_251));
        float _add_249;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_249) : "f"(_add_250), "f"(hc_eps));
        float c_31 = __fdiv_rn(c_30, _add_249);
        hist[33] = c_31;
        float _shfl_xor_sync_252 = __shfl_xor_sync(0xFFFFFFFF, add_2_1, 4);
        float add_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7) : "f"(add_2_1), "f"(_shfl_xor_sync_252));
        float _shfl_xor_sync_253 = __shfl_xor_sync(0xFFFFFFFF, add_7, 2);
        float add_1_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_4) : "f"(add_7), "f"(_shfl_xor_sync_253));
        float _shfl_xor_sync_255 = __shfl_xor_sync(0xFFFFFFFF, add_1_4, 1);
        float _add_254;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_254) : "f"(add_1_4), "f"(_shfl_xor_sync_255));
        dpre_part[own_base + 64] = _add_254;
        float _shfl_xor_sync_256 = __shfl_xor_sync(0xFFFFFFFF, c_31, 1);
        float s_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_32) : "f"(c_31), "f"(_shfl_xor_sync_256));
        float _shfl_xor_sync_259 = __shfl_xor_sync(0xFFFFFFFF, s_32, 2);
        float _add_258;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_258) : "f"(s_32), "f"(_shfl_xor_sync_259));
        float _add_257;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_257) : "f"(_add_258), "f"(hc_eps));
        float c_32 = __fdiv_rn(c_31, _add_257);
        hist[34] = c_32;
        float _shfl_xor_sync_260 = __shfl_xor_sync(0xFFFFFFFF, c_32, 4);
        float s_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_33) : "f"(c_32), "f"(_shfl_xor_sync_260));
        float _shfl_xor_sync_263 = __shfl_xor_sync(0xFFFFFFFF, s_33, 24);
        float _add_262;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_262) : "f"(s_33), "f"(_shfl_xor_sync_263));
        float _add_261;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_261) : "f"(_add_262), "f"(hc_eps));
        float c_33 = __fdiv_rn(c_32, _add_261);
        hist[35] = c_33;
        float _shfl_xor_sync_264 = __shfl_xor_sync(0xFFFFFFFF, c_33, 1);
        float s_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_34) : "f"(c_33), "f"(_shfl_xor_sync_264));
        float _shfl_xor_sync_267 = __shfl_xor_sync(0xFFFFFFFF, s_34, 2);
        float _add_266;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_266) : "f"(s_34), "f"(_shfl_xor_sync_267));
        float _add_265;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_265) : "f"(_add_266), "f"(hc_eps));
        float c_34 = __fdiv_rn(c_33, _add_265);
        hist[36] = c_34;
        mbarrier_wait(ring_bar_addr + (st__1_1) * 8, 1);
        unsigned int ring_pk_14[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[(0) + 3]))
            : "r"(sa__1_1 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[(4) + 3]))
            : "r"(sa__1_1 + 16 + lane32));
        unsigned int ring_pk_15[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[(0) + 3]))
            : "r"(sa__1_1 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[(4) + 3]))
            : "r"(sa__1_1 + 1040 + lane32));
        float dots_p_2[4];
        float _shfl_xor_sync_268 = __shfl_xor_sync(0xFFFFFFFF, c_34, 4);
        float s_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_35) : "f"(c_34), "f"(_shfl_xor_sync_268));
        float _shfl_xor_sync_271 = __shfl_xor_sync(0xFFFFFFFF, s_35, 24);
        float _add_270;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_270) : "f"(s_35), "f"(_shfl_xor_sync_271));
        float _add_269;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_269) : "f"(_add_270), "f"(hc_eps));
        float c_35 = __fdiv_rn(c_34, _add_269);
        hist[37] = c_35;
        float _dot_272;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_272) : "r"(ring_pk_15[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[0] = _dot_272;
        float _dot_273;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_273) : "r"(ring_pk_15[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_273;
        float _dot_274;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_274) : "r"(ring_pk_15[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_274;
        float _dot_275;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_275) : "r"(ring_pk_15[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_275;
        float _dot_276;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_276) : "r"(ring_pk_15[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_276;
        float _dot_277;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_277) : "r"(ring_pk_15[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_277;
        float _dot_278;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_278) : "r"(ring_pk_15[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_278;
        float _dot_279;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_279) : "r"(ring_pk_15[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_279;
        unsigned int ring_pk_16[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(0) + 3]))
            : "r"(sa__1_1 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(4) + 3]))
            : "r"(sa__1_1 + 2064 + lane32));
        float _shfl_xor_sync_280 = __shfl_xor_sync(0xFFFFFFFF, c_35, 1);
        float s_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_36) : "f"(c_35), "f"(_shfl_xor_sync_280));
        float _shfl_xor_sync_283 = __shfl_xor_sync(0xFFFFFFFF, s_36, 2);
        float _add_282;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_282) : "f"(s_36), "f"(_shfl_xor_sync_283));
        float _add_281;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_281) : "f"(_add_282), "f"(hc_eps));
        float c_36 = __fdiv_rn(c_35, _add_281);
        hist[38] = c_36;
        float _dot_284;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_284) : "r"(ring_pk_16[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[1] = _dot_284;
        float _dot_285;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_285) : "r"(ring_pk_16[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_285;
        float _dot_286;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_286) : "r"(ring_pk_16[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_286;
        float _dot_287;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_287) : "r"(ring_pk_16[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_287;
        float _dot_288;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_288) : "r"(ring_pk_16[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_288;
        float _dot_289;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_289) : "r"(ring_pk_16[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_289;
        float _dot_290;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_290) : "r"(ring_pk_16[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_290;
        float _dot_291;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_291) : "r"(ring_pk_16[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_291;
        unsigned int ring_pk_17[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(0) + 3]))
            : "r"(sa__1_1 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(4) + 3]))
            : "r"(sa__1_1 + 3088 + lane32));
        float _shfl_xor_sync_292 = __shfl_xor_sync(0xFFFFFFFF, c_36, 4);
        float s_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_37) : "f"(c_36), "f"(_shfl_xor_sync_292));
        float _shfl_xor_sync_295 = __shfl_xor_sync(0xFFFFFFFF, s_37, 24);
        float _add_294;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_294) : "f"(s_37), "f"(_shfl_xor_sync_295));
        float _add_293;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_293) : "f"(_add_294), "f"(hc_eps));
        float c_37 = __fdiv_rn(c_36, _add_293);
        hist[39] = c_37;
        float _dot_296;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_296) : "r"(ring_pk_17[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[2] = _dot_296;
        float _dot_297;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_297) : "r"(ring_pk_17[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_297;
        float _dot_298;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_298) : "r"(ring_pk_17[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_298;
        float _dot_299;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_299) : "r"(ring_pk_17[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_299;
        float _dot_300;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_300) : "r"(ring_pk_17[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_300;
        float _dot_301;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_301) : "r"(ring_pk_17[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_301;
        float _dot_302;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_302) : "r"(ring_pk_17[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_302;
        float _dot_303;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_303) : "r"(ring_pk_17[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_303;
        unsigned int ring_pk_18[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(0) + 3]))
            : "r"(sa__1_1 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(4) + 3]))
            : "r"(sa__1_1 + 4112 + lane32));
        float dsig;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dsig) : "f"(sig), "f"(sneg));
        float G = ((lane_1 >= 8 && lane_1 < 24) ? gval : 0.0f);
        float _shfl_xor_sync_304 = __shfl_xor_sync(0xFFFFFFFF, hist[38], 4);
        float s_38;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_38) : "f"(hist[38]), "f"(_shfl_xor_sync_304));
        float _shfl_xor_sync_306 = __shfl_xor_sync(0xFFFFFFFF, s_38, 24);
        float _add_305;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_305) : "f"(s_38), "f"(_shfl_xor_sync_306));
        float v;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(_add_305), "f"(hc_eps));
        float m;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m) : "f"(G), "f"(hist[39]));
        float _shfl_xor_sync_307 = __shfl_xor_sync(0xFFFFFFFF, m, 4);
        float ip;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip) : "f"(m), "f"(_shfl_xor_sync_307));
        float _shfl_xor_sync_310 = __shfl_xor_sync(0xFFFFFFFF, ip, 24);
        float _add_309;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_309) : "f"(ip), "f"(_shfl_xor_sync_310));
        float _sub_308;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_308) : "f"(G), "f"(_add_309));
        float g = __fdiv_rn(_sub_308, v);
        float _dot_311;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_311) : "r"(ring_pk_18[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[3] = _dot_311;
        float _dot_312;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_312) : "r"(ring_pk_18[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_312;
        float _dot_313;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_313) : "r"(ring_pk_18[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_313;
        float _dot_314;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_314) : "r"(ring_pk_18[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_314;
        float _dot_315;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_315) : "r"(ring_pk_18[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_315;
        float _dot_316;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_316) : "r"(ring_pk_18[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_316;
        float _dot_317;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_317) : "r"(ring_pk_18[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_317;
        float _dot_318;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_318) : "r"(ring_pk_18[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_318;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_1) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1), "l"(dU + (th2_5 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 1024), "l"(X + (th2_5 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 2048), "l"(X + ((th2_5 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 3072), "l"(X + ((th2_5 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 4096), "l"(X + ((th2_5 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_319 = __shfl_xor_sync(0xFFFFFFFF, hist[37], 1);
        float s_39;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_39) : "f"(hist[37]), "f"(_shfl_xor_sync_319));
        float _shfl_xor_sync_321 = __shfl_xor_sync(0xFFFFFFFF, s_39, 2);
        float _add_320;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_320) : "f"(s_39), "f"(_shfl_xor_sync_321));
        float v_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_0) : "f"(_add_320), "f"(hc_eps));
        float m_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_0) : "f"(g), "f"(hist[38]));
        float _shfl_xor_sync_322 = __shfl_xor_sync(0xFFFFFFFF, m_0, 1);
        float ip_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_0) : "f"(m_0), "f"(_shfl_xor_sync_322));
        float _shfl_xor_sync_325 = __shfl_xor_sync(0xFFFFFFFF, ip_0, 2);
        float _add_324;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_324) : "f"(ip_0), "f"(_shfl_xor_sync_325));
        float _sub_323;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_323) : "f"(g), "f"(_add_324));
        float g_0 = __fdiv_rn(_sub_323, v_0);
        int upper_3 = lane_1 & 16;
        float send_4 = ((upper_3 != 0) ? dots_p_2[0] : dots_p_2[2]);
        float keep_4 = ((upper_3 != 0) ? dots_p_2[2] : dots_p_2[0]);
        float _shfl_xor_sync_326 = __shfl_xor_sync(0xFFFFFFFF, send_4, 16);
        float add_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8) : "f"(keep_4), "f"(_shfl_xor_sync_326));
        float send_1_2 = ((upper_3 != 0) ? dots_p_2[1] : dots_p_2[3]);
        float keep_1_2 = ((upper_3 != 0) ? dots_p_2[3] : dots_p_2[1]);
        float _shfl_xor_sync_327 = __shfl_xor_sync(0xFFFFFFFF, send_1_2, 16);
        float add_1_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_5) : "f"(keep_1_2), "f"(_shfl_xor_sync_327));
        int upper_1_2 = lane_1 & 8;
        float send_2_2 = ((upper_1_2 != 0) ? add_8 : add_1_5);
        float keep_2_2 = ((upper_1_2 != 0) ? add_1_5 : add_8);
        float _shfl_xor_sync_328 = __shfl_xor_sync(0xFFFFFFFF, send_2_2, 8);
        float add_2_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_2) : "f"(keep_2_2), "f"(_shfl_xor_sync_328));
        float _shfl_xor_sync_329 = __shfl_xor_sync(0xFFFFFFFF, hist[36], 4);
        float s_40;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_40) : "f"(hist[36]), "f"(_shfl_xor_sync_329));
        float _shfl_xor_sync_331 = __shfl_xor_sync(0xFFFFFFFF, s_40, 24);
        float _add_330;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_330) : "f"(s_40), "f"(_shfl_xor_sync_331));
        float v_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_1) : "f"(_add_330), "f"(hc_eps));
        float m_1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_1) : "f"(g_0), "f"(hist[37]));
        float _shfl_xor_sync_332 = __shfl_xor_sync(0xFFFFFFFF, m_1, 4);
        float ip_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_1) : "f"(m_1), "f"(_shfl_xor_sync_332));
        float _shfl_xor_sync_335 = __shfl_xor_sync(0xFFFFFFFF, ip_1, 24);
        float _add_334;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_334) : "f"(ip_1), "f"(_shfl_xor_sync_335));
        float _sub_333;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_333) : "f"(g_0), "f"(_add_334));
        float g_1 = __fdiv_rn(_sub_333, v_1);
        float _shfl_xor_sync_336 = __shfl_xor_sync(0xFFFFFFFF, add_2_2, 4);
        float add_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_9) : "f"(add_2_2), "f"(_shfl_xor_sync_336));
        float _shfl_xor_sync_337 = __shfl_xor_sync(0xFFFFFFFF, add_9, 2);
        float add_1_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_6) : "f"(add_9), "f"(_shfl_xor_sync_337));
        float _shfl_xor_sync_339 = __shfl_xor_sync(0xFFFFFFFF, add_1_6, 1);
        float _add_338;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_338) : "f"(add_1_6), "f"(_shfl_xor_sync_339));
        dpre_part[own_base + 96] = _add_338;
        float _shfl_xor_sync_340 = __shfl_xor_sync(0xFFFFFFFF, hist[35], 1);
        float s_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_41) : "f"(hist[35]), "f"(_shfl_xor_sync_340));
        float _shfl_xor_sync_342 = __shfl_xor_sync(0xFFFFFFFF, s_41, 2);
        float _add_341;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_341) : "f"(s_41), "f"(_shfl_xor_sync_342));
        float v_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_2) : "f"(_add_341), "f"(hc_eps));
        float m_2;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_2) : "f"(g_1), "f"(hist[36]));
        float _shfl_xor_sync_343 = __shfl_xor_sync(0xFFFFFFFF, m_2, 1);
        float ip_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_2) : "f"(m_2), "f"(_shfl_xor_sync_343));
        float _shfl_xor_sync_346 = __shfl_xor_sync(0xFFFFFFFF, ip_2, 2);
        float _add_345;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_345) : "f"(ip_2), "f"(_shfl_xor_sync_346));
        float _sub_344;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_344) : "f"(g_1), "f"(_add_345));
        float g_2 = __fdiv_rn(_sub_344, v_2);
        float _shfl_xor_sync_347 = __shfl_xor_sync(0xFFFFFFFF, hist[34], 4);
        float s_42;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_42) : "f"(hist[34]), "f"(_shfl_xor_sync_347));
        float _shfl_xor_sync_349 = __shfl_xor_sync(0xFFFFFFFF, s_42, 24);
        float _add_348;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_348) : "f"(s_42), "f"(_shfl_xor_sync_349));
        float v_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_3) : "f"(_add_348), "f"(hc_eps));
        float m_3;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_3) : "f"(g_2), "f"(hist[35]));
        float _shfl_xor_sync_350 = __shfl_xor_sync(0xFFFFFFFF, m_3, 4);
        float ip_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_3) : "f"(m_3), "f"(_shfl_xor_sync_350));
        float _shfl_xor_sync_353 = __shfl_xor_sync(0xFFFFFFFF, ip_3, 24);
        float _add_352;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_352) : "f"(ip_3), "f"(_shfl_xor_sync_353));
        float _sub_351;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_351) : "f"(g_2), "f"(_add_352));
        float g_3 = __fdiv_rn(_sub_351, v_3);
        float _shfl_xor_sync_354 = __shfl_xor_sync(0xFFFFFFFF, hist[33], 1);
        float s_43;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_43) : "f"(hist[33]), "f"(_shfl_xor_sync_354));
        float _shfl_xor_sync_356 = __shfl_xor_sync(0xFFFFFFFF, s_43, 2);
        float _add_355;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_355) : "f"(s_43), "f"(_shfl_xor_sync_356));
        float v_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_4) : "f"(_add_355), "f"(hc_eps));
        float m_4;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_4) : "f"(g_3), "f"(hist[34]));
        float _shfl_xor_sync_357 = __shfl_xor_sync(0xFFFFFFFF, m_4, 1);
        float ip_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_4) : "f"(m_4), "f"(_shfl_xor_sync_357));
        float _shfl_xor_sync_360 = __shfl_xor_sync(0xFFFFFFFF, ip_4, 2);
        float _add_359;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_359) : "f"(ip_4), "f"(_shfl_xor_sync_360));
        float _sub_358;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_358) : "f"(g_3), "f"(_add_359));
        float g_4 = __fdiv_rn(_sub_358, v_4);
        mbarrier_wait(ring_bar_addr + (st__1_0) * 8, 0);
        unsigned int ring_pk_19[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[(0) + 3]))
            : "r"(sa__1_0 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[(4) + 3]))
            : "r"(sa__1_0 + 16 + lane32));
        unsigned int ring_pk_20[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[(0) + 3]))
            : "r"(sa__1_0 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[(4) + 3]))
            : "r"(sa__1_0 + 1040 + lane32));
        float dots_p_3[4];
        float _shfl_xor_sync_361 = __shfl_xor_sync(0xFFFFFFFF, hist[32], 4);
        float s_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_44) : "f"(hist[32]), "f"(_shfl_xor_sync_361));
        float _shfl_xor_sync_363 = __shfl_xor_sync(0xFFFFFFFF, s_44, 24);
        float _add_362;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_362) : "f"(s_44), "f"(_shfl_xor_sync_363));
        float v_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_5) : "f"(_add_362), "f"(hc_eps));
        float m_5;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_5) : "f"(g_4), "f"(hist[33]));
        float _shfl_xor_sync_364 = __shfl_xor_sync(0xFFFFFFFF, m_5, 4);
        float ip_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_5) : "f"(m_5), "f"(_shfl_xor_sync_364));
        float _shfl_xor_sync_367 = __shfl_xor_sync(0xFFFFFFFF, ip_5, 24);
        float _add_366;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_366) : "f"(ip_5), "f"(_shfl_xor_sync_367));
        float _sub_365;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_365) : "f"(g_4), "f"(_add_366));
        float g_5 = __fdiv_rn(_sub_365, v_5);
        float _dot_368;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_368) : "r"(ring_pk_20[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[0] = _dot_368;
        float _dot_369;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_369) : "r"(ring_pk_20[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_369;
        float _dot_370;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_370) : "r"(ring_pk_20[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_370;
        float _dot_371;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_371) : "r"(ring_pk_20[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_371;
        float _dot_372;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_372) : "r"(ring_pk_20[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_372;
        float _dot_373;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_373) : "r"(ring_pk_20[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_373;
        float _dot_374;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_374) : "r"(ring_pk_20[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_374;
        float _dot_375;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_375) : "r"(ring_pk_20[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_375;
        unsigned int ring_pk_21[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(0) + 3]))
            : "r"(sa__1_0 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(4) + 3]))
            : "r"(sa__1_0 + 2064 + lane32));
        float _shfl_xor_sync_376 = __shfl_xor_sync(0xFFFFFFFF, hist[31], 1);
        float s_45;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_45) : "f"(hist[31]), "f"(_shfl_xor_sync_376));
        float _shfl_xor_sync_378 = __shfl_xor_sync(0xFFFFFFFF, s_45, 2);
        float _add_377;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_377) : "f"(s_45), "f"(_shfl_xor_sync_378));
        float v_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_6) : "f"(_add_377), "f"(hc_eps));
        float m_6;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_6) : "f"(g_5), "f"(hist[32]));
        float _shfl_xor_sync_379 = __shfl_xor_sync(0xFFFFFFFF, m_6, 1);
        float ip_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_6) : "f"(m_6), "f"(_shfl_xor_sync_379));
        float _shfl_xor_sync_382 = __shfl_xor_sync(0xFFFFFFFF, ip_6, 2);
        float _add_381;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_381) : "f"(ip_6), "f"(_shfl_xor_sync_382));
        float _sub_380;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_380) : "f"(g_5), "f"(_add_381));
        float g_6 = __fdiv_rn(_sub_380, v_6);
        float _dot_383;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_383) : "r"(ring_pk_21[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[1] = _dot_383;
        float _dot_384;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_384) : "r"(ring_pk_21[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_384;
        float _dot_385;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_385) : "r"(ring_pk_21[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_385;
        float _dot_386;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_386) : "r"(ring_pk_21[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_386;
        float _dot_387;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_387) : "r"(ring_pk_21[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_387;
        float _dot_388;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_388) : "r"(ring_pk_21[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_388;
        float _dot_389;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_389) : "r"(ring_pk_21[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_389;
        float _dot_390;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_390) : "r"(ring_pk_21[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_390;
        unsigned int ring_pk_22[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(0) + 3]))
            : "r"(sa__1_0 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(4) + 3]))
            : "r"(sa__1_0 + 3088 + lane32));
        float _shfl_xor_sync_391 = __shfl_xor_sync(0xFFFFFFFF, hist[30], 4);
        float s_46;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_46) : "f"(hist[30]), "f"(_shfl_xor_sync_391));
        float _shfl_xor_sync_393 = __shfl_xor_sync(0xFFFFFFFF, s_46, 24);
        float _add_392;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_392) : "f"(s_46), "f"(_shfl_xor_sync_393));
        float v_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_7) : "f"(_add_392), "f"(hc_eps));
        float m_7;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_7) : "f"(g_6), "f"(hist[31]));
        float _shfl_xor_sync_394 = __shfl_xor_sync(0xFFFFFFFF, m_7, 4);
        float ip_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_7) : "f"(m_7), "f"(_shfl_xor_sync_394));
        float _shfl_xor_sync_397 = __shfl_xor_sync(0xFFFFFFFF, ip_7, 24);
        float _add_396;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_396) : "f"(ip_7), "f"(_shfl_xor_sync_397));
        float _sub_395;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_395) : "f"(g_6), "f"(_add_396));
        float g_7 = __fdiv_rn(_sub_395, v_7);
        float _dot_398;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_398) : "r"(ring_pk_22[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[2] = _dot_398;
        float _dot_399;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_399) : "r"(ring_pk_22[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_399;
        float _dot_400;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_400) : "r"(ring_pk_22[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_400;
        float _dot_401;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_401) : "r"(ring_pk_22[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_401;
        float _dot_402;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_402) : "r"(ring_pk_22[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_402;
        float _dot_403;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_403) : "r"(ring_pk_22[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_403;
        float _dot_404;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_404) : "r"(ring_pk_22[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_404;
        float _dot_405;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_405) : "r"(ring_pk_22[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_405;
        unsigned int ring_pk_23[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(0) + 3]))
            : "r"(sa__1_0 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(4) + 3]))
            : "r"(sa__1_0 + 4112 + lane32));
        float _shfl_xor_sync_406 = __shfl_xor_sync(0xFFFFFFFF, hist[29], 1);
        float s_47;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_47) : "f"(hist[29]), "f"(_shfl_xor_sync_406));
        float _shfl_xor_sync_408 = __shfl_xor_sync(0xFFFFFFFF, s_47, 2);
        float _add_407;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_407) : "f"(s_47), "f"(_shfl_xor_sync_408));
        float v_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_8) : "f"(_add_407), "f"(hc_eps));
        float m_8;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_8) : "f"(g_7), "f"(hist[30]));
        float _shfl_xor_sync_409 = __shfl_xor_sync(0xFFFFFFFF, m_8, 1);
        float ip_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_8) : "f"(m_8), "f"(_shfl_xor_sync_409));
        float _shfl_xor_sync_412 = __shfl_xor_sync(0xFFFFFFFF, ip_8, 2);
        float _add_411;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_411) : "f"(ip_8), "f"(_shfl_xor_sync_412));
        float _sub_410;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_410) : "f"(g_7), "f"(_add_411));
        float g_8 = __fdiv_rn(_sub_410, v_8);
        float _dot_413;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_413) : "r"(ring_pk_23[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[3] = _dot_413;
        float _dot_414;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_414) : "r"(ring_pk_23[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_414;
        float _dot_415;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_415) : "r"(ring_pk_23[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_415;
        float _dot_416;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_416) : "r"(ring_pk_23[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_416;
        float _dot_417;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_417) : "r"(ring_pk_23[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_417;
        float _dot_418;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_418) : "r"(ring_pk_23[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_418;
        float _dot_419;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_419) : "r"(ring_pk_23[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_419;
        float _dot_420;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_420) : "r"(ring_pk_23[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_420;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_0) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0), "l"(dU + (th2_6 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 1024), "l"(X + (th2_6 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 2048), "l"(X + ((th2_6 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 3072), "l"(X + ((th2_6 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 4096), "l"(X + ((th2_6 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_421 = __shfl_xor_sync(0xFFFFFFFF, hist[28], 4);
        float s_48;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_48) : "f"(hist[28]), "f"(_shfl_xor_sync_421));
        float _shfl_xor_sync_423 = __shfl_xor_sync(0xFFFFFFFF, s_48, 24);
        float _add_422;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_422) : "f"(s_48), "f"(_shfl_xor_sync_423));
        float v_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_9) : "f"(_add_422), "f"(hc_eps));
        float m_9;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_9) : "f"(g_8), "f"(hist[29]));
        float _shfl_xor_sync_424 = __shfl_xor_sync(0xFFFFFFFF, m_9, 4);
        float ip_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_9) : "f"(m_9), "f"(_shfl_xor_sync_424));
        float _shfl_xor_sync_427 = __shfl_xor_sync(0xFFFFFFFF, ip_9, 24);
        float _add_426;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_426) : "f"(ip_9), "f"(_shfl_xor_sync_427));
        float _sub_425;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_425) : "f"(g_8), "f"(_add_426));
        float g_9 = __fdiv_rn(_sub_425, v_9);
        int upper_4 = lane_1 & 16;
        float send_5 = ((upper_4 != 0) ? dots_p_3[0] : dots_p_3[2]);
        float keep_5 = ((upper_4 != 0) ? dots_p_3[2] : dots_p_3[0]);
        float _shfl_xor_sync_428 = __shfl_xor_sync(0xFFFFFFFF, send_5, 16);
        float add_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_10) : "f"(keep_5), "f"(_shfl_xor_sync_428));
        float send_1_3 = ((upper_4 != 0) ? dots_p_3[1] : dots_p_3[3]);
        float keep_1_3 = ((upper_4 != 0) ? dots_p_3[3] : dots_p_3[1]);
        float _shfl_xor_sync_429 = __shfl_xor_sync(0xFFFFFFFF, send_1_3, 16);
        float add_1_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_7) : "f"(keep_1_3), "f"(_shfl_xor_sync_429));
        int upper_1_3 = lane_1 & 8;
        float send_2_3 = ((upper_1_3 != 0) ? add_10 : add_1_7);
        float keep_2_3 = ((upper_1_3 != 0) ? add_1_7 : add_10);
        float _shfl_xor_sync_430 = __shfl_xor_sync(0xFFFFFFFF, send_2_3, 8);
        float add_2_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_3) : "f"(keep_2_3), "f"(_shfl_xor_sync_430));
        float _shfl_xor_sync_431 = __shfl_xor_sync(0xFFFFFFFF, hist[27], 1);
        float s_49;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_49) : "f"(hist[27]), "f"(_shfl_xor_sync_431));
        float _shfl_xor_sync_433 = __shfl_xor_sync(0xFFFFFFFF, s_49, 2);
        float _add_432;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_432) : "f"(s_49), "f"(_shfl_xor_sync_433));
        float v_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_10) : "f"(_add_432), "f"(hc_eps));
        float m_10;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_10) : "f"(g_9), "f"(hist[28]));
        float _shfl_xor_sync_434 = __shfl_xor_sync(0xFFFFFFFF, m_10, 1);
        float ip_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_10) : "f"(m_10), "f"(_shfl_xor_sync_434));
        float _shfl_xor_sync_437 = __shfl_xor_sync(0xFFFFFFFF, ip_10, 2);
        float _add_436;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_436) : "f"(ip_10), "f"(_shfl_xor_sync_437));
        float _sub_435;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_435) : "f"(g_9), "f"(_add_436));
        float g_10 = __fdiv_rn(_sub_435, v_10);
        float _shfl_xor_sync_438 = __shfl_xor_sync(0xFFFFFFFF, add_2_3, 4);
        float add_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_11) : "f"(add_2_3), "f"(_shfl_xor_sync_438));
        float _shfl_xor_sync_439 = __shfl_xor_sync(0xFFFFFFFF, add_11, 2);
        float add_1_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_8) : "f"(add_11), "f"(_shfl_xor_sync_439));
        float _shfl_xor_sync_441 = __shfl_xor_sync(0xFFFFFFFF, add_1_8, 1);
        float _add_440;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_440) : "f"(add_1_8), "f"(_shfl_xor_sync_441));
        dpre_part[own_base + 128] = _add_440;
        float _shfl_xor_sync_442 = __shfl_xor_sync(0xFFFFFFFF, hist[26], 4);
        float s_50;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_50) : "f"(hist[26]), "f"(_shfl_xor_sync_442));
        float _shfl_xor_sync_444 = __shfl_xor_sync(0xFFFFFFFF, s_50, 24);
        float _add_443;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_443) : "f"(s_50), "f"(_shfl_xor_sync_444));
        float v_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_11) : "f"(_add_443), "f"(hc_eps));
        float m_11;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_11) : "f"(g_10), "f"(hist[27]));
        float _shfl_xor_sync_445 = __shfl_xor_sync(0xFFFFFFFF, m_11, 4);
        float ip_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_11) : "f"(m_11), "f"(_shfl_xor_sync_445));
        float _shfl_xor_sync_448 = __shfl_xor_sync(0xFFFFFFFF, ip_11, 24);
        float _add_447;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_447) : "f"(ip_11), "f"(_shfl_xor_sync_448));
        float _sub_446;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_446) : "f"(g_10), "f"(_add_447));
        float g_11 = __fdiv_rn(_sub_446, v_11);
        float _shfl_xor_sync_449 = __shfl_xor_sync(0xFFFFFFFF, hist[25], 1);
        float s_51;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_51) : "f"(hist[25]), "f"(_shfl_xor_sync_449));
        float _shfl_xor_sync_451 = __shfl_xor_sync(0xFFFFFFFF, s_51, 2);
        float _add_450;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_450) : "f"(s_51), "f"(_shfl_xor_sync_451));
        float v_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_12) : "f"(_add_450), "f"(hc_eps));
        float m_12;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_12) : "f"(g_11), "f"(hist[26]));
        float _shfl_xor_sync_452 = __shfl_xor_sync(0xFFFFFFFF, m_12, 1);
        float ip_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_12) : "f"(m_12), "f"(_shfl_xor_sync_452));
        float _shfl_xor_sync_455 = __shfl_xor_sync(0xFFFFFFFF, ip_12, 2);
        float _add_454;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_454) : "f"(ip_12), "f"(_shfl_xor_sync_455));
        float _sub_453;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_453) : "f"(g_11), "f"(_add_454));
        float g_12 = __fdiv_rn(_sub_453, v_12);
        float _shfl_xor_sync_456 = __shfl_xor_sync(0xFFFFFFFF, hist[24], 4);
        float s_52;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_52) : "f"(hist[24]), "f"(_shfl_xor_sync_456));
        float _shfl_xor_sync_458 = __shfl_xor_sync(0xFFFFFFFF, s_52, 24);
        float _add_457;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_457) : "f"(s_52), "f"(_shfl_xor_sync_458));
        float v_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_13) : "f"(_add_457), "f"(hc_eps));
        float m_13;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_13) : "f"(g_12), "f"(hist[25]));
        float _shfl_xor_sync_459 = __shfl_xor_sync(0xFFFFFFFF, m_13, 4);
        float ip_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_13) : "f"(m_13), "f"(_shfl_xor_sync_459));
        float _shfl_xor_sync_462 = __shfl_xor_sync(0xFFFFFFFF, ip_13, 24);
        float _add_461;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_461) : "f"(ip_13), "f"(_shfl_xor_sync_462));
        float _sub_460;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_460) : "f"(g_12), "f"(_add_461));
        float g_13 = __fdiv_rn(_sub_460, v_13);
        mbarrier_wait(ring_bar_addr + (st__1_1) * 8, 0);
        unsigned int ring_pk_24[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[(0) + 3]))
            : "r"(sa__1_1 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[(4) + 3]))
            : "r"(sa__1_1 + 16 + lane32));
        unsigned int ring_pk_25[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[(0) + 3]))
            : "r"(sa__1_1 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[(4) + 3]))
            : "r"(sa__1_1 + 1040 + lane32));
        float dots_p_4[4];
        float _shfl_xor_sync_463 = __shfl_xor_sync(0xFFFFFFFF, hist[23], 1);
        float s_53;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_53) : "f"(hist[23]), "f"(_shfl_xor_sync_463));
        float _shfl_xor_sync_465 = __shfl_xor_sync(0xFFFFFFFF, s_53, 2);
        float _add_464;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_464) : "f"(s_53), "f"(_shfl_xor_sync_465));
        float v_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_14) : "f"(_add_464), "f"(hc_eps));
        float m_14;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_14) : "f"(g_13), "f"(hist[24]));
        float _shfl_xor_sync_466 = __shfl_xor_sync(0xFFFFFFFF, m_14, 1);
        float ip_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_14) : "f"(m_14), "f"(_shfl_xor_sync_466));
        float _shfl_xor_sync_469 = __shfl_xor_sync(0xFFFFFFFF, ip_14, 2);
        float _add_468;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_468) : "f"(ip_14), "f"(_shfl_xor_sync_469));
        float _sub_467;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_467) : "f"(g_13), "f"(_add_468));
        float g_14 = __fdiv_rn(_sub_467, v_14);
        float _dot_470;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_470) : "r"(ring_pk_25[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[0] = _dot_470;
        float _dot_471;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_471) : "r"(ring_pk_25[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_471;
        float _dot_472;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_472) : "r"(ring_pk_25[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_472;
        float _dot_473;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_473) : "r"(ring_pk_25[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_473;
        float _dot_474;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_474) : "r"(ring_pk_25[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_474;
        float _dot_475;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_475) : "r"(ring_pk_25[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_475;
        float _dot_476;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_476) : "r"(ring_pk_25[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_476;
        float _dot_477;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_477) : "r"(ring_pk_25[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_477;
        unsigned int ring_pk_26[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(0) + 3]))
            : "r"(sa__1_1 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(4) + 3]))
            : "r"(sa__1_1 + 2064 + lane32));
        float _shfl_xor_sync_478 = __shfl_xor_sync(0xFFFFFFFF, hist[22], 4);
        float s_54;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_54) : "f"(hist[22]), "f"(_shfl_xor_sync_478));
        float _shfl_xor_sync_480 = __shfl_xor_sync(0xFFFFFFFF, s_54, 24);
        float _add_479;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_479) : "f"(s_54), "f"(_shfl_xor_sync_480));
        float v_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_15) : "f"(_add_479), "f"(hc_eps));
        float m_15;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_15) : "f"(g_14), "f"(hist[23]));
        float _shfl_xor_sync_481 = __shfl_xor_sync(0xFFFFFFFF, m_15, 4);
        float ip_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_15) : "f"(m_15), "f"(_shfl_xor_sync_481));
        float _shfl_xor_sync_484 = __shfl_xor_sync(0xFFFFFFFF, ip_15, 24);
        float _add_483;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_483) : "f"(ip_15), "f"(_shfl_xor_sync_484));
        float _sub_482;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_482) : "f"(g_14), "f"(_add_483));
        float g_15 = __fdiv_rn(_sub_482, v_15);
        float _dot_485;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_485) : "r"(ring_pk_26[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[1] = _dot_485;
        float _dot_486;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_486) : "r"(ring_pk_26[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_486;
        float _dot_487;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_487) : "r"(ring_pk_26[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_487;
        float _dot_488;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_488) : "r"(ring_pk_26[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_488;
        float _dot_489;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_489) : "r"(ring_pk_26[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_489;
        float _dot_490;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_490) : "r"(ring_pk_26[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_490;
        float _dot_491;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_491) : "r"(ring_pk_26[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_491;
        float _dot_492;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_492) : "r"(ring_pk_26[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_492;
        unsigned int ring_pk_27[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(0) + 3]))
            : "r"(sa__1_1 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(4) + 3]))
            : "r"(sa__1_1 + 3088 + lane32));
        float _shfl_xor_sync_493 = __shfl_xor_sync(0xFFFFFFFF, hist[21], 1);
        float s_55;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_55) : "f"(hist[21]), "f"(_shfl_xor_sync_493));
        float _shfl_xor_sync_495 = __shfl_xor_sync(0xFFFFFFFF, s_55, 2);
        float _add_494;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_494) : "f"(s_55), "f"(_shfl_xor_sync_495));
        float v_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_16) : "f"(_add_494), "f"(hc_eps));
        float m_16;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_16) : "f"(g_15), "f"(hist[22]));
        float _shfl_xor_sync_496 = __shfl_xor_sync(0xFFFFFFFF, m_16, 1);
        float ip_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_16) : "f"(m_16), "f"(_shfl_xor_sync_496));
        float _shfl_xor_sync_499 = __shfl_xor_sync(0xFFFFFFFF, ip_16, 2);
        float _add_498;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_498) : "f"(ip_16), "f"(_shfl_xor_sync_499));
        float _sub_497;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_497) : "f"(g_15), "f"(_add_498));
        float g_16 = __fdiv_rn(_sub_497, v_16);
        float _dot_500;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_500) : "r"(ring_pk_27[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[2] = _dot_500;
        float _dot_501;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_501) : "r"(ring_pk_27[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_501;
        float _dot_502;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_502) : "r"(ring_pk_27[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_502;
        float _dot_503;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_503) : "r"(ring_pk_27[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_503;
        float _dot_504;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_504) : "r"(ring_pk_27[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_504;
        float _dot_505;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_505) : "r"(ring_pk_27[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_505;
        float _dot_506;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_506) : "r"(ring_pk_27[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_506;
        float _dot_507;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_507) : "r"(ring_pk_27[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_507;
        unsigned int ring_pk_28[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(0) + 3]))
            : "r"(sa__1_1 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(4) + 3]))
            : "r"(sa__1_1 + 4112 + lane32));
        float _shfl_xor_sync_508 = __shfl_xor_sync(0xFFFFFFFF, hist[20], 4);
        float s_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_56) : "f"(hist[20]), "f"(_shfl_xor_sync_508));
        float _shfl_xor_sync_510 = __shfl_xor_sync(0xFFFFFFFF, s_56, 24);
        float _add_509;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_509) : "f"(s_56), "f"(_shfl_xor_sync_510));
        float v_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_17) : "f"(_add_509), "f"(hc_eps));
        float m_17;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_17) : "f"(g_16), "f"(hist[21]));
        float _shfl_xor_sync_511 = __shfl_xor_sync(0xFFFFFFFF, m_17, 4);
        float ip_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_17) : "f"(m_17), "f"(_shfl_xor_sync_511));
        float _shfl_xor_sync_514 = __shfl_xor_sync(0xFFFFFFFF, ip_17, 24);
        float _add_513;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_513) : "f"(ip_17), "f"(_shfl_xor_sync_514));
        float _sub_512;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_512) : "f"(g_16), "f"(_add_513));
        float g_17 = __fdiv_rn(_sub_512, v_17);
        float _dot_515;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_515) : "r"(ring_pk_28[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[3] = _dot_515;
        float _dot_516;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_516) : "r"(ring_pk_28[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_516;
        float _dot_517;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_517) : "r"(ring_pk_28[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_517;
        float _dot_518;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_518) : "r"(ring_pk_28[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_518;
        float _dot_519;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_519) : "r"(ring_pk_28[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_519;
        float _dot_520;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_520) : "r"(ring_pk_28[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_520;
        float _dot_521;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_521) : "r"(ring_pk_28[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_521;
        float _dot_522;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_522) : "r"(ring_pk_28[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_522;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_1) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1), "l"(dU + (th2_7 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 1024), "l"(X + (th2_7 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 2048), "l"(X + ((th2_7 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 3072), "l"(X + ((th2_7 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 4096), "l"(X + ((th2_7 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_523 = __shfl_xor_sync(0xFFFFFFFF, hist[19], 1);
        float s_57;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_57) : "f"(hist[19]), "f"(_shfl_xor_sync_523));
        float _shfl_xor_sync_525 = __shfl_xor_sync(0xFFFFFFFF, s_57, 2);
        float _add_524;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_524) : "f"(s_57), "f"(_shfl_xor_sync_525));
        float v_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_18) : "f"(_add_524), "f"(hc_eps));
        float m_18;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_18) : "f"(g_17), "f"(hist[20]));
        float _shfl_xor_sync_526 = __shfl_xor_sync(0xFFFFFFFF, m_18, 1);
        float ip_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_18) : "f"(m_18), "f"(_shfl_xor_sync_526));
        float _shfl_xor_sync_529 = __shfl_xor_sync(0xFFFFFFFF, ip_18, 2);
        float _add_528;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_528) : "f"(ip_18), "f"(_shfl_xor_sync_529));
        float _sub_527;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_527) : "f"(g_17), "f"(_add_528));
        float g_18 = __fdiv_rn(_sub_527, v_18);
        int upper_5 = lane_1 & 16;
        float send_6 = ((upper_5 != 0) ? dots_p_4[0] : dots_p_4[2]);
        float keep_6 = ((upper_5 != 0) ? dots_p_4[2] : dots_p_4[0]);
        float _shfl_xor_sync_530 = __shfl_xor_sync(0xFFFFFFFF, send_6, 16);
        float add_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_12) : "f"(keep_6), "f"(_shfl_xor_sync_530));
        float send_1_4 = ((upper_5 != 0) ? dots_p_4[1] : dots_p_4[3]);
        float keep_1_4 = ((upper_5 != 0) ? dots_p_4[3] : dots_p_4[1]);
        float _shfl_xor_sync_531 = __shfl_xor_sync(0xFFFFFFFF, send_1_4, 16);
        float add_1_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_9) : "f"(keep_1_4), "f"(_shfl_xor_sync_531));
        int upper_1_4 = lane_1 & 8;
        float send_2_4 = ((upper_1_4 != 0) ? add_12 : add_1_9);
        float keep_2_4 = ((upper_1_4 != 0) ? add_1_9 : add_12);
        float _shfl_xor_sync_532 = __shfl_xor_sync(0xFFFFFFFF, send_2_4, 8);
        float add_2_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_4) : "f"(keep_2_4), "f"(_shfl_xor_sync_532));
        float _shfl_xor_sync_533 = __shfl_xor_sync(0xFFFFFFFF, hist[18], 4);
        float s_58;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_58) : "f"(hist[18]), "f"(_shfl_xor_sync_533));
        float _shfl_xor_sync_535 = __shfl_xor_sync(0xFFFFFFFF, s_58, 24);
        float _add_534;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_534) : "f"(s_58), "f"(_shfl_xor_sync_535));
        float v_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_19) : "f"(_add_534), "f"(hc_eps));
        float m_19;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_19) : "f"(g_18), "f"(hist[19]));
        float _shfl_xor_sync_536 = __shfl_xor_sync(0xFFFFFFFF, m_19, 4);
        float ip_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_19) : "f"(m_19), "f"(_shfl_xor_sync_536));
        float _shfl_xor_sync_539 = __shfl_xor_sync(0xFFFFFFFF, ip_19, 24);
        float _add_538;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_538) : "f"(ip_19), "f"(_shfl_xor_sync_539));
        float _sub_537;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_537) : "f"(g_18), "f"(_add_538));
        float g_19 = __fdiv_rn(_sub_537, v_19);
        float _shfl_xor_sync_540 = __shfl_xor_sync(0xFFFFFFFF, add_2_4, 4);
        float add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_13) : "f"(add_2_4), "f"(_shfl_xor_sync_540));
        float _shfl_xor_sync_541 = __shfl_xor_sync(0xFFFFFFFF, add_13, 2);
        float add_1_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_10) : "f"(add_13), "f"(_shfl_xor_sync_541));
        float _shfl_xor_sync_543 = __shfl_xor_sync(0xFFFFFFFF, add_1_10, 1);
        float _add_542;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_542) : "f"(add_1_10), "f"(_shfl_xor_sync_543));
        dpre_part[own_base + 160] = _add_542;
        float _shfl_xor_sync_544 = __shfl_xor_sync(0xFFFFFFFF, hist[17], 1);
        float s_59;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_59) : "f"(hist[17]), "f"(_shfl_xor_sync_544));
        float _shfl_xor_sync_546 = __shfl_xor_sync(0xFFFFFFFF, s_59, 2);
        float _add_545;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_545) : "f"(s_59), "f"(_shfl_xor_sync_546));
        float v_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_20) : "f"(_add_545), "f"(hc_eps));
        float m_20;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_20) : "f"(g_19), "f"(hist[18]));
        float _shfl_xor_sync_547 = __shfl_xor_sync(0xFFFFFFFF, m_20, 1);
        float ip_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_20) : "f"(m_20), "f"(_shfl_xor_sync_547));
        float _shfl_xor_sync_550 = __shfl_xor_sync(0xFFFFFFFF, ip_20, 2);
        float _add_549;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_549) : "f"(ip_20), "f"(_shfl_xor_sync_550));
        float _sub_548;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_548) : "f"(g_19), "f"(_add_549));
        float g_20 = __fdiv_rn(_sub_548, v_20);
        float _shfl_xor_sync_551 = __shfl_xor_sync(0xFFFFFFFF, hist[16], 4);
        float s_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_60) : "f"(hist[16]), "f"(_shfl_xor_sync_551));
        float _shfl_xor_sync_553 = __shfl_xor_sync(0xFFFFFFFF, s_60, 24);
        float _add_552;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_552) : "f"(s_60), "f"(_shfl_xor_sync_553));
        float v_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_21) : "f"(_add_552), "f"(hc_eps));
        float m_21;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_21) : "f"(g_20), "f"(hist[17]));
        float _shfl_xor_sync_554 = __shfl_xor_sync(0xFFFFFFFF, m_21, 4);
        float ip_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_21) : "f"(m_21), "f"(_shfl_xor_sync_554));
        float _shfl_xor_sync_557 = __shfl_xor_sync(0xFFFFFFFF, ip_21, 24);
        float _add_556;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_556) : "f"(ip_21), "f"(_shfl_xor_sync_557));
        float _sub_555;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_555) : "f"(g_20), "f"(_add_556));
        float g_21 = __fdiv_rn(_sub_555, v_21);
        float _shfl_xor_sync_558 = __shfl_xor_sync(0xFFFFFFFF, hist[15], 1);
        float s_61;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_61) : "f"(hist[15]), "f"(_shfl_xor_sync_558));
        float _shfl_xor_sync_560 = __shfl_xor_sync(0xFFFFFFFF, s_61, 2);
        float _add_559;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_559) : "f"(s_61), "f"(_shfl_xor_sync_560));
        float v_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_22) : "f"(_add_559), "f"(hc_eps));
        float m_22;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_22) : "f"(g_21), "f"(hist[16]));
        float _shfl_xor_sync_561 = __shfl_xor_sync(0xFFFFFFFF, m_22, 1);
        float ip_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_22) : "f"(m_22), "f"(_shfl_xor_sync_561));
        float _shfl_xor_sync_564 = __shfl_xor_sync(0xFFFFFFFF, ip_22, 2);
        float _add_563;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_563) : "f"(ip_22), "f"(_shfl_xor_sync_564));
        float _sub_562;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_562) : "f"(g_21), "f"(_add_563));
        float g_22 = __fdiv_rn(_sub_562, v_22);
        mbarrier_wait(ring_bar_addr + (st__1_0) * 8, 1);
        unsigned int ring_pk_29[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[(0) + 3]))
            : "r"(sa__1_0 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[(4) + 3]))
            : "r"(sa__1_0 + 16 + lane32));
        unsigned int ring_pk_30[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[(0) + 3]))
            : "r"(sa__1_0 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[(4) + 3]))
            : "r"(sa__1_0 + 1040 + lane32));
        float dots_p_5[4];
        float _shfl_xor_sync_565 = __shfl_xor_sync(0xFFFFFFFF, hist[14], 4);
        float s_62;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_62) : "f"(hist[14]), "f"(_shfl_xor_sync_565));
        float _shfl_xor_sync_567 = __shfl_xor_sync(0xFFFFFFFF, s_62, 24);
        float _add_566;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_566) : "f"(s_62), "f"(_shfl_xor_sync_567));
        float v_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_23) : "f"(_add_566), "f"(hc_eps));
        float m_23;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_23) : "f"(g_22), "f"(hist[15]));
        float _shfl_xor_sync_568 = __shfl_xor_sync(0xFFFFFFFF, m_23, 4);
        float ip_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_23) : "f"(m_23), "f"(_shfl_xor_sync_568));
        float _shfl_xor_sync_571 = __shfl_xor_sync(0xFFFFFFFF, ip_23, 24);
        float _add_570;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_570) : "f"(ip_23), "f"(_shfl_xor_sync_571));
        float _sub_569;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_569) : "f"(g_22), "f"(_add_570));
        float g_23 = __fdiv_rn(_sub_569, v_23);
        float _dot_572;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_572) : "r"(ring_pk_30[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[0] = _dot_572;
        float _dot_573;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_573) : "r"(ring_pk_30[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_573;
        float _dot_574;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_574) : "r"(ring_pk_30[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_574;
        float _dot_575;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_575) : "r"(ring_pk_30[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_575;
        float _dot_576;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_576) : "r"(ring_pk_30[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_576;
        float _dot_577;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_577) : "r"(ring_pk_30[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_577;
        float _dot_578;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_578) : "r"(ring_pk_30[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_578;
        float _dot_579;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_579) : "r"(ring_pk_30[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_579;
        unsigned int ring_pk_31[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(0) + 3]))
            : "r"(sa__1_0 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(4) + 3]))
            : "r"(sa__1_0 + 2064 + lane32));
        float _shfl_xor_sync_580 = __shfl_xor_sync(0xFFFFFFFF, hist[13], 1);
        float s_63;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_63) : "f"(hist[13]), "f"(_shfl_xor_sync_580));
        float _shfl_xor_sync_582 = __shfl_xor_sync(0xFFFFFFFF, s_63, 2);
        float _add_581;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_581) : "f"(s_63), "f"(_shfl_xor_sync_582));
        float v_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_24) : "f"(_add_581), "f"(hc_eps));
        float m_24;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_24) : "f"(g_23), "f"(hist[14]));
        float _shfl_xor_sync_583 = __shfl_xor_sync(0xFFFFFFFF, m_24, 1);
        float ip_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_24) : "f"(m_24), "f"(_shfl_xor_sync_583));
        float _shfl_xor_sync_586 = __shfl_xor_sync(0xFFFFFFFF, ip_24, 2);
        float _add_585;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_585) : "f"(ip_24), "f"(_shfl_xor_sync_586));
        float _sub_584;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_584) : "f"(g_23), "f"(_add_585));
        float g_24 = __fdiv_rn(_sub_584, v_24);
        float _dot_587;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_587) : "r"(ring_pk_31[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[1] = _dot_587;
        float _dot_588;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_588) : "r"(ring_pk_31[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_588;
        float _dot_589;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_589) : "r"(ring_pk_31[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_589;
        float _dot_590;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_590) : "r"(ring_pk_31[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_590;
        float _dot_591;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_591) : "r"(ring_pk_31[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_591;
        float _dot_592;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_592) : "r"(ring_pk_31[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_592;
        float _dot_593;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_593) : "r"(ring_pk_31[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_593;
        float _dot_594;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_594) : "r"(ring_pk_31[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_594;
        unsigned int ring_pk_32[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(0) + 3]))
            : "r"(sa__1_0 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(4) + 3]))
            : "r"(sa__1_0 + 3088 + lane32));
        float _shfl_xor_sync_595 = __shfl_xor_sync(0xFFFFFFFF, hist[12], 4);
        float s_64;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_64) : "f"(hist[12]), "f"(_shfl_xor_sync_595));
        float _shfl_xor_sync_597 = __shfl_xor_sync(0xFFFFFFFF, s_64, 24);
        float _add_596;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_596) : "f"(s_64), "f"(_shfl_xor_sync_597));
        float v_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_25) : "f"(_add_596), "f"(hc_eps));
        float m_25;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_25) : "f"(g_24), "f"(hist[13]));
        float _shfl_xor_sync_598 = __shfl_xor_sync(0xFFFFFFFF, m_25, 4);
        float ip_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_25) : "f"(m_25), "f"(_shfl_xor_sync_598));
        float _shfl_xor_sync_601 = __shfl_xor_sync(0xFFFFFFFF, ip_25, 24);
        float _add_600;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_600) : "f"(ip_25), "f"(_shfl_xor_sync_601));
        float _sub_599;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_599) : "f"(g_24), "f"(_add_600));
        float g_25 = __fdiv_rn(_sub_599, v_25);
        float _dot_602;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_602) : "r"(ring_pk_32[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[2] = _dot_602;
        float _dot_603;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_603) : "r"(ring_pk_32[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_603;
        float _dot_604;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_604) : "r"(ring_pk_32[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_604;
        float _dot_605;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_605) : "r"(ring_pk_32[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_605;
        float _dot_606;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_606) : "r"(ring_pk_32[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_606;
        float _dot_607;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_607) : "r"(ring_pk_32[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_607;
        float _dot_608;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_608) : "r"(ring_pk_32[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_608;
        float _dot_609;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_609) : "r"(ring_pk_32[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_609;
        unsigned int ring_pk_33[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(0) + 3]))
            : "r"(sa__1_0 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(4) + 3]))
            : "r"(sa__1_0 + 4112 + lane32));
        float _shfl_xor_sync_610 = __shfl_xor_sync(0xFFFFFFFF, hist[11], 1);
        float s_65;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_65) : "f"(hist[11]), "f"(_shfl_xor_sync_610));
        float _shfl_xor_sync_612 = __shfl_xor_sync(0xFFFFFFFF, s_65, 2);
        float _add_611;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_611) : "f"(s_65), "f"(_shfl_xor_sync_612));
        float v_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_26) : "f"(_add_611), "f"(hc_eps));
        float m_26;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_26) : "f"(g_25), "f"(hist[12]));
        float _shfl_xor_sync_613 = __shfl_xor_sync(0xFFFFFFFF, m_26, 1);
        float ip_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_26) : "f"(m_26), "f"(_shfl_xor_sync_613));
        float _shfl_xor_sync_616 = __shfl_xor_sync(0xFFFFFFFF, ip_26, 2);
        float _add_615;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_615) : "f"(ip_26), "f"(_shfl_xor_sync_616));
        float _sub_614;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_614) : "f"(g_25), "f"(_add_615));
        float g_26 = __fdiv_rn(_sub_614, v_26);
        float _dot_617;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_617) : "r"(ring_pk_33[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[3] = _dot_617;
        float _dot_618;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_618) : "r"(ring_pk_33[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_618;
        float _dot_619;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_619) : "r"(ring_pk_33[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_619;
        float _dot_620;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_620) : "r"(ring_pk_33[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_620;
        float _dot_621;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_621) : "r"(ring_pk_33[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_621;
        float _dot_622;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_622) : "r"(ring_pk_33[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_622;
        float _dot_623;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_623) : "r"(ring_pk_33[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_623;
        float _dot_624;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_624) : "r"(ring_pk_33[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_624;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_0) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0), "l"(dU + (th2_8 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 1024), "l"(X + (th2_8 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 2048), "l"(X + ((th2_8 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 3072), "l"(X + ((th2_8 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 4096), "l"(X + ((th2_8 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_625 = __shfl_xor_sync(0xFFFFFFFF, hist[10], 4);
        float s_66;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_66) : "f"(hist[10]), "f"(_shfl_xor_sync_625));
        float _shfl_xor_sync_627 = __shfl_xor_sync(0xFFFFFFFF, s_66, 24);
        float _add_626;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_626) : "f"(s_66), "f"(_shfl_xor_sync_627));
        float v_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_27) : "f"(_add_626), "f"(hc_eps));
        float m_27;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_27) : "f"(g_26), "f"(hist[11]));
        float _shfl_xor_sync_628 = __shfl_xor_sync(0xFFFFFFFF, m_27, 4);
        float ip_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_27) : "f"(m_27), "f"(_shfl_xor_sync_628));
        float _shfl_xor_sync_631 = __shfl_xor_sync(0xFFFFFFFF, ip_27, 24);
        float _add_630;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_630) : "f"(ip_27), "f"(_shfl_xor_sync_631));
        float _sub_629;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_629) : "f"(g_26), "f"(_add_630));
        float g_27 = __fdiv_rn(_sub_629, v_27);
        int upper_6 = lane_1 & 16;
        float send_7 = ((upper_6 != 0) ? dots_p_5[0] : dots_p_5[2]);
        float keep_7 = ((upper_6 != 0) ? dots_p_5[2] : dots_p_5[0]);
        float _shfl_xor_sync_632 = __shfl_xor_sync(0xFFFFFFFF, send_7, 16);
        float add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_14) : "f"(keep_7), "f"(_shfl_xor_sync_632));
        float send_1_5 = ((upper_6 != 0) ? dots_p_5[1] : dots_p_5[3]);
        float keep_1_5 = ((upper_6 != 0) ? dots_p_5[3] : dots_p_5[1]);
        float _shfl_xor_sync_633 = __shfl_xor_sync(0xFFFFFFFF, send_1_5, 16);
        float add_1_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_11) : "f"(keep_1_5), "f"(_shfl_xor_sync_633));
        int upper_1_5 = lane_1 & 8;
        float send_2_5 = ((upper_1_5 != 0) ? add_14 : add_1_11);
        float keep_2_5 = ((upper_1_5 != 0) ? add_1_11 : add_14);
        float _shfl_xor_sync_634 = __shfl_xor_sync(0xFFFFFFFF, send_2_5, 8);
        float add_2_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_5) : "f"(keep_2_5), "f"(_shfl_xor_sync_634));
        float _shfl_xor_sync_635 = __shfl_xor_sync(0xFFFFFFFF, hist[9], 1);
        float s_67;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_67) : "f"(hist[9]), "f"(_shfl_xor_sync_635));
        float _shfl_xor_sync_637 = __shfl_xor_sync(0xFFFFFFFF, s_67, 2);
        float _add_636;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_636) : "f"(s_67), "f"(_shfl_xor_sync_637));
        float v_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_28) : "f"(_add_636), "f"(hc_eps));
        float m_28;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_28) : "f"(g_27), "f"(hist[10]));
        float _shfl_xor_sync_638 = __shfl_xor_sync(0xFFFFFFFF, m_28, 1);
        float ip_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_28) : "f"(m_28), "f"(_shfl_xor_sync_638));
        float _shfl_xor_sync_641 = __shfl_xor_sync(0xFFFFFFFF, ip_28, 2);
        float _add_640;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_640) : "f"(ip_28), "f"(_shfl_xor_sync_641));
        float _sub_639;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_639) : "f"(g_27), "f"(_add_640));
        float g_28 = __fdiv_rn(_sub_639, v_28);
        float _shfl_xor_sync_642 = __shfl_xor_sync(0xFFFFFFFF, add_2_5, 4);
        float add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_15) : "f"(add_2_5), "f"(_shfl_xor_sync_642));
        float _shfl_xor_sync_643 = __shfl_xor_sync(0xFFFFFFFF, add_15, 2);
        float add_1_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_12) : "f"(add_15), "f"(_shfl_xor_sync_643));
        float _shfl_xor_sync_645 = __shfl_xor_sync(0xFFFFFFFF, add_1_12, 1);
        float _add_644;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_644) : "f"(add_1_12), "f"(_shfl_xor_sync_645));
        dpre_part[own_base + 192] = _add_644;
        float _shfl_xor_sync_646 = __shfl_xor_sync(0xFFFFFFFF, hist[8], 4);
        float s_68;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_68) : "f"(hist[8]), "f"(_shfl_xor_sync_646));
        float _shfl_xor_sync_648 = __shfl_xor_sync(0xFFFFFFFF, s_68, 24);
        float _add_647;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_647) : "f"(s_68), "f"(_shfl_xor_sync_648));
        float v_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_29) : "f"(_add_647), "f"(hc_eps));
        float m_29;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_29) : "f"(g_28), "f"(hist[9]));
        float _shfl_xor_sync_649 = __shfl_xor_sync(0xFFFFFFFF, m_29, 4);
        float ip_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_29) : "f"(m_29), "f"(_shfl_xor_sync_649));
        float _shfl_xor_sync_652 = __shfl_xor_sync(0xFFFFFFFF, ip_29, 24);
        float _add_651;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_651) : "f"(ip_29), "f"(_shfl_xor_sync_652));
        float _sub_650;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_650) : "f"(g_28), "f"(_add_651));
        float g_29 = __fdiv_rn(_sub_650, v_29);
        float _shfl_xor_sync_653 = __shfl_xor_sync(0xFFFFFFFF, hist[7], 1);
        float s_69;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_69) : "f"(hist[7]), "f"(_shfl_xor_sync_653));
        float _shfl_xor_sync_655 = __shfl_xor_sync(0xFFFFFFFF, s_69, 2);
        float _add_654;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_654) : "f"(s_69), "f"(_shfl_xor_sync_655));
        float v_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_30) : "f"(_add_654), "f"(hc_eps));
        float m_30;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_30) : "f"(g_29), "f"(hist[8]));
        float _shfl_xor_sync_656 = __shfl_xor_sync(0xFFFFFFFF, m_30, 1);
        float ip_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_30) : "f"(m_30), "f"(_shfl_xor_sync_656));
        float _shfl_xor_sync_659 = __shfl_xor_sync(0xFFFFFFFF, ip_30, 2);
        float _add_658;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_658) : "f"(ip_30), "f"(_shfl_xor_sync_659));
        float _sub_657;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_657) : "f"(g_29), "f"(_add_658));
        float g_30 = __fdiv_rn(_sub_657, v_30);
        float _shfl_xor_sync_660 = __shfl_xor_sync(0xFFFFFFFF, hist[6], 4);
        float s_70;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_70) : "f"(hist[6]), "f"(_shfl_xor_sync_660));
        float _shfl_xor_sync_662 = __shfl_xor_sync(0xFFFFFFFF, s_70, 24);
        float _add_661;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_661) : "f"(s_70), "f"(_shfl_xor_sync_662));
        float v_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_31) : "f"(_add_661), "f"(hc_eps));
        float m_31;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_31) : "f"(g_30), "f"(hist[7]));
        float _shfl_xor_sync_663 = __shfl_xor_sync(0xFFFFFFFF, m_31, 4);
        float ip_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_31) : "f"(m_31), "f"(_shfl_xor_sync_663));
        float _shfl_xor_sync_666 = __shfl_xor_sync(0xFFFFFFFF, ip_31, 24);
        float _add_665;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_665) : "f"(ip_31), "f"(_shfl_xor_sync_666));
        float _sub_664;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_664) : "f"(g_30), "f"(_add_665));
        float g_31 = __fdiv_rn(_sub_664, v_31);
        mbarrier_wait(ring_bar_addr + (st__1_1) * 8, 1);
        unsigned int ring_pk_34[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[(0) + 3]))
            : "r"(sa__1_1 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[(4) + 3]))
            : "r"(sa__1_1 + 16 + lane32));
        unsigned int ring_pk_35[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[(0) + 3]))
            : "r"(sa__1_1 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[(4) + 3]))
            : "r"(sa__1_1 + 1040 + lane32));
        float dots_p_6[4];
        float _shfl_xor_sync_667 = __shfl_xor_sync(0xFFFFFFFF, hist[5], 1);
        float s_71;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_71) : "f"(hist[5]), "f"(_shfl_xor_sync_667));
        float _shfl_xor_sync_669 = __shfl_xor_sync(0xFFFFFFFF, s_71, 2);
        float _add_668;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_668) : "f"(s_71), "f"(_shfl_xor_sync_669));
        float v_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_32) : "f"(_add_668), "f"(hc_eps));
        float m_32;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_32) : "f"(g_31), "f"(hist[6]));
        float _shfl_xor_sync_670 = __shfl_xor_sync(0xFFFFFFFF, m_32, 1);
        float ip_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_32) : "f"(m_32), "f"(_shfl_xor_sync_670));
        float _shfl_xor_sync_673 = __shfl_xor_sync(0xFFFFFFFF, ip_32, 2);
        float _add_672;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_672) : "f"(ip_32), "f"(_shfl_xor_sync_673));
        float _sub_671;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_671) : "f"(g_31), "f"(_add_672));
        float g_32 = __fdiv_rn(_sub_671, v_32);
        float _dot_674;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_674) : "r"(ring_pk_35[0]), "r"(ring_pk_34[0]), "f"(-0.0f));
        dots_p_6[0] = _dot_674;
        float _dot_675;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_675) : "r"(ring_pk_35[1]), "r"(ring_pk_34[1]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_675;
        float _dot_676;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_676) : "r"(ring_pk_35[2]), "r"(ring_pk_34[2]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_676;
        float _dot_677;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_677) : "r"(ring_pk_35[3]), "r"(ring_pk_34[3]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_677;
        float _dot_678;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_678) : "r"(ring_pk_35[4]), "r"(ring_pk_34[4]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_678;
        float _dot_679;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_679) : "r"(ring_pk_35[5]), "r"(ring_pk_34[5]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_679;
        float _dot_680;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_680) : "r"(ring_pk_35[6]), "r"(ring_pk_34[6]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_680;
        float _dot_681;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_681) : "r"(ring_pk_35[7]), "r"(ring_pk_34[7]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_681;
        unsigned int ring_pk_36[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[(0) + 3]))
            : "r"(sa__1_1 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[(4) + 3]))
            : "r"(sa__1_1 + 2064 + lane32));
        float _shfl_xor_sync_682 = __shfl_xor_sync(0xFFFFFFFF, hist[4], 4);
        float s_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_72) : "f"(hist[4]), "f"(_shfl_xor_sync_682));
        float _shfl_xor_sync_684 = __shfl_xor_sync(0xFFFFFFFF, s_72, 24);
        float _add_683;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_683) : "f"(s_72), "f"(_shfl_xor_sync_684));
        float v_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_33) : "f"(_add_683), "f"(hc_eps));
        float m_33;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_33) : "f"(g_32), "f"(hist[5]));
        float _shfl_xor_sync_685 = __shfl_xor_sync(0xFFFFFFFF, m_33, 4);
        float ip_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_33) : "f"(m_33), "f"(_shfl_xor_sync_685));
        float _shfl_xor_sync_688 = __shfl_xor_sync(0xFFFFFFFF, ip_33, 24);
        float _add_687;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_687) : "f"(ip_33), "f"(_shfl_xor_sync_688));
        float _sub_686;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_686) : "f"(g_32), "f"(_add_687));
        float g_33 = __fdiv_rn(_sub_686, v_33);
        float _dot_689;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_689) : "r"(ring_pk_36[0]), "r"(ring_pk_34[0]), "f"(-0.0f));
        dots_p_6[1] = _dot_689;
        float _dot_690;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_690) : "r"(ring_pk_36[1]), "r"(ring_pk_34[1]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_690;
        float _dot_691;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_691) : "r"(ring_pk_36[2]), "r"(ring_pk_34[2]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_691;
        float _dot_692;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_692) : "r"(ring_pk_36[3]), "r"(ring_pk_34[3]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_692;
        float _dot_693;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_693) : "r"(ring_pk_36[4]), "r"(ring_pk_34[4]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_693;
        float _dot_694;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_694) : "r"(ring_pk_36[5]), "r"(ring_pk_34[5]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_694;
        float _dot_695;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_695) : "r"(ring_pk_36[6]), "r"(ring_pk_34[6]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_695;
        float _dot_696;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_696) : "r"(ring_pk_36[7]), "r"(ring_pk_34[7]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_696;
        unsigned int ring_pk_37[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[(0) + 3]))
            : "r"(sa__1_1 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[(4) + 3]))
            : "r"(sa__1_1 + 3088 + lane32));
        float _shfl_xor_sync_697 = __shfl_xor_sync(0xFFFFFFFF, hist[3], 1);
        float s_73;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_73) : "f"(hist[3]), "f"(_shfl_xor_sync_697));
        float _shfl_xor_sync_699 = __shfl_xor_sync(0xFFFFFFFF, s_73, 2);
        float _add_698;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_698) : "f"(s_73), "f"(_shfl_xor_sync_699));
        float v_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_34) : "f"(_add_698), "f"(hc_eps));
        float m_34;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_34) : "f"(g_33), "f"(hist[4]));
        float _shfl_xor_sync_700 = __shfl_xor_sync(0xFFFFFFFF, m_34, 1);
        float ip_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_34) : "f"(m_34), "f"(_shfl_xor_sync_700));
        float _shfl_xor_sync_703 = __shfl_xor_sync(0xFFFFFFFF, ip_34, 2);
        float _add_702;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_702) : "f"(ip_34), "f"(_shfl_xor_sync_703));
        float _sub_701;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_701) : "f"(g_33), "f"(_add_702));
        float g_34 = __fdiv_rn(_sub_701, v_34);
        float _dot_704;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_704) : "r"(ring_pk_37[0]), "r"(ring_pk_34[0]), "f"(-0.0f));
        dots_p_6[2] = _dot_704;
        float _dot_705;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_705) : "r"(ring_pk_37[1]), "r"(ring_pk_34[1]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_705;
        float _dot_706;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_706) : "r"(ring_pk_37[2]), "r"(ring_pk_34[2]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_706;
        float _dot_707;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_707) : "r"(ring_pk_37[3]), "r"(ring_pk_34[3]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_707;
        float _dot_708;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_708) : "r"(ring_pk_37[4]), "r"(ring_pk_34[4]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_708;
        float _dot_709;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_709) : "r"(ring_pk_37[5]), "r"(ring_pk_34[5]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_709;
        float _dot_710;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_710) : "r"(ring_pk_37[6]), "r"(ring_pk_34[6]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_710;
        float _dot_711;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_711) : "r"(ring_pk_37[7]), "r"(ring_pk_34[7]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_711;
        unsigned int ring_pk_38[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[(0) + 3]))
            : "r"(sa__1_1 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[(4) + 3]))
            : "r"(sa__1_1 + 4112 + lane32));
        float _shfl_xor_sync_712 = __shfl_xor_sync(0xFFFFFFFF, hist[2], 4);
        float s_74;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_74) : "f"(hist[2]), "f"(_shfl_xor_sync_712));
        float _shfl_xor_sync_714 = __shfl_xor_sync(0xFFFFFFFF, s_74, 24);
        float _add_713;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_713) : "f"(s_74), "f"(_shfl_xor_sync_714));
        float v_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_35) : "f"(_add_713), "f"(hc_eps));
        float m_35;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_35) : "f"(g_34), "f"(hist[3]));
        float _shfl_xor_sync_715 = __shfl_xor_sync(0xFFFFFFFF, m_35, 4);
        float ip_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_35) : "f"(m_35), "f"(_shfl_xor_sync_715));
        float _shfl_xor_sync_718 = __shfl_xor_sync(0xFFFFFFFF, ip_35, 24);
        float _add_717;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_717) : "f"(ip_35), "f"(_shfl_xor_sync_718));
        float _sub_716;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_716) : "f"(g_34), "f"(_add_717));
        float g_35 = __fdiv_rn(_sub_716, v_35);
        float _dot_719;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_719) : "r"(ring_pk_38[0]), "r"(ring_pk_34[0]), "f"(-0.0f));
        dots_p_6[3] = _dot_719;
        float _dot_720;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_720) : "r"(ring_pk_38[1]), "r"(ring_pk_34[1]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_720;
        float _dot_721;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_721) : "r"(ring_pk_38[2]), "r"(ring_pk_34[2]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_721;
        float _dot_722;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_722) : "r"(ring_pk_38[3]), "r"(ring_pk_34[3]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_722;
        float _dot_723;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_723) : "r"(ring_pk_38[4]), "r"(ring_pk_34[4]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_723;
        float _dot_724;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_724) : "r"(ring_pk_38[5]), "r"(ring_pk_34[5]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_724;
        float _dot_725;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_725) : "r"(ring_pk_38[6]), "r"(ring_pk_34[6]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_725;
        float _dot_726;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_726) : "r"(ring_pk_38[7]), "r"(ring_pk_34[7]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_726;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_1) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1), "l"(dU + (th2_9 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 1024), "l"(X + (th2_9 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 2048), "l"(X + ((th2_9 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 3072), "l"(X + ((th2_9 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 4096), "l"(X + ((th2_9 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_727 = __shfl_xor_sync(0xFFFFFFFF, hist[1], 1);
        float s_75;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_75) : "f"(hist[1]), "f"(_shfl_xor_sync_727));
        float _shfl_xor_sync_729 = __shfl_xor_sync(0xFFFFFFFF, s_75, 2);
        float _add_728;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_728) : "f"(s_75), "f"(_shfl_xor_sync_729));
        float v_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_36) : "f"(_add_728), "f"(hc_eps));
        float m_36;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_36) : "f"(g_35), "f"(hist[2]));
        float _shfl_xor_sync_730 = __shfl_xor_sync(0xFFFFFFFF, m_36, 1);
        float ip_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_36) : "f"(m_36), "f"(_shfl_xor_sync_730));
        float _shfl_xor_sync_733 = __shfl_xor_sync(0xFFFFFFFF, ip_36, 2);
        float _add_732;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_732) : "f"(ip_36), "f"(_shfl_xor_sync_733));
        float _sub_731;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_731) : "f"(g_35), "f"(_add_732));
        float g_36 = __fdiv_rn(_sub_731, v_36);
        int upper_7 = lane_1 & 16;
        float send_8 = ((upper_7 != 0) ? dots_p_6[0] : dots_p_6[2]);
        float keep_8 = ((upper_7 != 0) ? dots_p_6[2] : dots_p_6[0]);
        float _shfl_xor_sync_734 = __shfl_xor_sync(0xFFFFFFFF, send_8, 16);
        float add_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_16) : "f"(keep_8), "f"(_shfl_xor_sync_734));
        float send_1_6 = ((upper_7 != 0) ? dots_p_6[1] : dots_p_6[3]);
        float keep_1_6 = ((upper_7 != 0) ? dots_p_6[3] : dots_p_6[1]);
        float _shfl_xor_sync_735 = __shfl_xor_sync(0xFFFFFFFF, send_1_6, 16);
        float add_1_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_13) : "f"(keep_1_6), "f"(_shfl_xor_sync_735));
        int upper_1_6 = lane_1 & 8;
        float send_2_6 = ((upper_1_6 != 0) ? add_16 : add_1_13);
        float keep_2_6 = ((upper_1_6 != 0) ? add_1_13 : add_16);
        float _shfl_xor_sync_736 = __shfl_xor_sync(0xFFFFFFFF, send_2_6, 8);
        float add_2_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_6) : "f"(keep_2_6), "f"(_shfl_xor_sync_736));
        float _add_195_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195_1) : "f"(hist[0]), "f"(hc_eps));
        float _shfl_xor_sync_737 = __shfl_xor_sync(0xFFFFFFFF, _add_195_1, 4);
        float add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_17) : "f"(_add_195_1), "f"(_shfl_xor_sync_737));
        float _shfl_xor_sync_739 = __shfl_xor_sync(0xFFFFFFFF, add_17, 24);
        float _add_738;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_738) : "f"(add_17), "f"(_shfl_xor_sync_739));
        float _add_197;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(_add_738), "f"(hc_eps));
        float _mul_39;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(g_36), "f"(hist[1]));
        float _shfl_xor_sync_740 = __shfl_xor_sync(0xFFFFFFFF, _mul_39, 4);
        float add_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_18) : "f"(_mul_39), "f"(_shfl_xor_sync_740));
        float _shfl_xor_sync_743 = __shfl_xor_sync(0xFFFFFFFF, add_18, 24);
        float _add_742;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_742) : "f"(add_18), "f"(_shfl_xor_sync_743));
        float _sub_741;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_741) : "f"(g_36), "f"(_add_742));
        float _fdiv_rn_80 = __fdiv_rn(_sub_741, _add_197);
        float _shfl_xor_sync_744 = __shfl_xor_sync(0xFFFFFFFF, add_2_6, 4);
        float add_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_19) : "f"(add_2_6), "f"(_shfl_xor_sync_744));
        float _shfl_xor_sync_745 = __shfl_xor_sync(0xFFFFFFFF, add_19, 2);
        float add_1_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_14) : "f"(add_19), "f"(_shfl_xor_sync_745));
        float _shfl_xor_sync_747 = __shfl_xor_sync(0xFFFFFFFF, add_1_14, 1);
        float _add_746;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_746) : "f"(add_1_14), "f"(_shfl_xor_sync_747));
        dpre_part[own_base + 224] = _add_746;
        float acc_162 = 0.0f;
        float _shfl_xor_sync_748 = __shfl_xor_sync(0xFFFFFFFF, hist[0], 2);
        float _shfl_xor_sync_750 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_80, 2);
        float _sub_749;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_749) : "f"(_fdiv_rn_80), "f"(_shfl_xor_sync_750));
        float _shfl_xor_sync_752 = __shfl_xor_sync(0xFFFFFFFF, hist[0], 1);
        float _shfl_xor_sync_754 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_80, 1);
        float _sub_753;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_753) : "f"(_fdiv_rn_80), "f"(_shfl_xor_sync_754));
        float _fma_751;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_751) : "f"(_shfl_xor_sync_752), "f"(_sub_753), "f"(acc_162));
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(_shfl_xor_sync_748), "f"(_sub_749), "f"(_fma_751));
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, hist[0], 3);
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_80, 3);
        float _sub_756;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_756) : "f"(_fdiv_rn_80), "f"(_shfl_xor_1));
        float _fma_755;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_755) : "f"(_shfl_xor_0), "f"(_sub_756), "f"(_fma_0));
        float da = ((lane_1 >= 8 && lane_1 < 24) ? mul_f32_rn_noftz(hist[0], _fma_755) : mul_f32_rn_noftz(1.0f, dsig));
        __syncthreads();
        dp[0] = dpre_part[cur_base + lc4];
        float _add_757;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_757) : "f"(dp[0]), "f"(dpre_part[cur_base + 4 + lc4]));
        dp[0] = _add_757;
        float _add_758;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_758) : "f"(dp[0]), "f"(dpre_part[cur_base + 8 + lc4]));
        dp[0] = _add_758;
        float _add_759;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_759) : "f"(dp[0]), "f"(dpre_part[cur_base + 12 + lc4]));
        dp[0] = _add_759;
        float _add_760;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_760) : "f"(dp[0]), "f"(dpre_part[cur_base + 16 + lc4]));
        dp[0] = _add_760;
        float _add_761;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_761) : "f"(dp[0]), "f"(dpre_part[cur_base + 20 + lc4]));
        dp[0] = _add_761;
        float _add_762;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_762) : "f"(dp[0]), "f"(dpre_part[cur_base + 24 + lc4]));
        dp[0] = _add_762;
        float _add_763;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_763) : "f"(dp[0]), "f"(dpre_part[cur_base + 28 + lc4]));
        dp[0] = _add_763;
        float gsig = ((lane_1 < 4) ? dp[0] : gtwo);
        float da_sig;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(da_sig) : "f"(gsig), "f"(da));
        float da_late = ((lane_1 >= 8 && lane_1 < 24) ? da : da_sig);
        float da_0 = ((lane_1 < 24) ? da_late : 0.0f);
        float dz;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz) : "f"(da_0), "f"(sc));
        float daz;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(daz) : "f"(da_0), "f"(z));
        float dzz;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dzz) : "f"(dz), "f"(z));
        float _warp_reduce_0 = dzz;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        float rr;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rr) : "f"(r), "f"(r));
        float c_t;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(c_t) : "f"(rr), "f"(mul_f32_rn_noftz(_warp_reduce_0, 6.1035156e-05f)));
        float dzr;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dzr) : "f"(dz), "f"(r));
        float cvt_f32 = (float)(__nv_bfloat16)dzr;
        float sub;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub) : "f"(dzr), "f"(cvt_f32));
        float cvt_f32_1 = (float)(__nv_bfloat16)sub;
        float _sub_764;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_764) : "f"(sub), "f"(cvt_f32_1));
        float cvt_f32_2 = (float)(__nv_bfloat16)_sub_764;
        float p1 = ((lane_1 < 24) ? cvt_f32 : 0.0f);
        float p2 = ((lane_1 < 24) ? cvt_f32_1 : 0.0f);
        float p3 = ((lane_1 < 24) ? cvt_f32_2 : 0.0f);
        long long drow = tok64 * 96 + lane64;
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow])), "h"(__bfloat16_as_ushort((__nv_bfloat16)p1)), "r"((unsigned int)(tok < t1)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow + 32])), "h"(__bfloat16_as_ushort((__nv_bfloat16)p2)), "r"((unsigned int)(tok < t1)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow + 64])), "h"(__bfloat16_as_ushort((__nv_bfloat16)p3)), "r"((unsigned int)(tok < t1)) : "memory");
        long long trow = tok64 * 24;
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow])), "f"(r), "r"((unsigned int)(tok < t1 && lane_1 == 0)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + 1])), "f"(c_t), "r"((unsigned int)(tok < t1 && lane_1 == 0)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + 2 + lane64])), "f"(_add_1), "r"((unsigned int)(tok < t1 && lane_1 < 4)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + lane64 - 2])), "f"(c_37), "r"((unsigned int)(tok < t1 && (lane_1 >= 8 && lane_1 < 24))) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + lane64 - 2])), "f"(0.0f), "r"((unsigned int)(tok < t1 && (lane_1 >= 24 && lane_1 < 26))) : "memory");
        float _sub_0_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0_1) : "f"(da_0), "f"(comp[0]));
        float _add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(acc[0]), "f"(_sub_0_1));
        float _sub_766;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_766) : "f"(_add_2), "f"(acc[0]));
        float _sub_765;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_765) : "f"(_sub_766), "f"(_sub_0_1));
        comp[0] = ((tok < t1) ? _sub_765 : comp[0]);
        acc[0] = ((tok < t1) ? _add_2 : acc[0]);
        float _sub_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(daz), "f"(comp[1]));
        float _add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(acc[1]), "f"(_sub_1));
        float _sub_768;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_768) : "f"(_add_4), "f"(acc[1]));
        float _sub_767;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_767) : "f"(_sub_768), "f"(_sub_1));
        comp[1] = ((tok < t1) ? _sub_767 : comp[1]);
        acc[1] = ((tok < t1) ? _add_4 : acc[1]);
        gi[0] = gi[0] + 1;
        ring_grp += 1;
    }
    if (t0 < t1) {
        mbarrier_wait(ring_bar_addr + (st__1_0) * 8, 0);
        mbarrier_wait(ring_bar_addr + (st__1_1) * 8, 0);
    }
    int fbase = (warp_0 * 32 + lane_1) * 4;
    fold[fbase] = acc[0];
    fold[fbase + 1] = comp[0];
    fold[fbase + 2] = acc[1];
    fold[fbase + 3] = comp[1];
    __syncthreads();
    if (warp_0 == 0) {
        long long prow = (long long)bid * 32;
        float o[2];
        float oc[2];
        #pragma unroll
        for (int e_1 = 0; e_1 < 2; e_1++) {
            o[e_1] = 0.0f;
            oc[e_1] = 0.0f;
        }
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            int src = (w * 32 + lane_1) * 4;
            float sub_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1) : "f"(fold[src]), "f"(oc[0]));
            float add_20;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_20) : "f"(o[0]), "f"(sub_1));
            float _sub_770;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_770) : "f"(add_20), "f"(o[0]));
            float _sub_769;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_769) : "f"(_sub_770), "f"(sub_1));
            oc[0] = _sub_769;
            o[0] = add_20;
            float _sub_771;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_771) : "f"(0.0f), "f"(fold[src + 1]));
            float sub_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_0) : "f"(_sub_771), "f"(oc[0]));
            float add_20_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_20_1) : "f"(o[0]), "f"(sub_0));
            float _sub_773;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_773) : "f"(add_20_1), "f"(o[0]));
            float _sub_772;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_772) : "f"(_sub_773), "f"(sub_0));
            oc[0] = _sub_772;
            o[0] = add_20_1;
            float sub_1_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1_1) : "f"(fold[src + 2]), "f"(oc[1]));
            float add_21;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_21) : "f"(o[1]), "f"(sub_1_1));
            float _sub_775;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_775) : "f"(add_21), "f"(o[1]));
            float _sub_774;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_774) : "f"(_sub_775), "f"(sub_1_1));
            oc[1] = _sub_774;
            o[1] = add_21;
            float _sub_776;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_776) : "f"(0.0f), "f"(fold[src + 3]));
            float sub_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_2) : "f"(_sub_776), "f"(oc[1]));
            float add_22;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_22) : "f"(o[1]), "f"(sub_2));
            float _sub_778;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_778) : "f"(add_22), "f"(o[1]));
            float _sub_777;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_777) : "f"(_sub_778), "f"(sub_2));
            oc[1] = _sub_777;
            o[1] = add_22;
        }
        float dbase_n;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(dbase_n) : "f"(o[0]), "f"(oc[0]));
        float dsz_n;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(dsz_n) : "f"(o[1]), "f"(oc[1]));
        gsum[lane_1] = dsz_n;
        __syncwarp();
        if (lane_1 < 24) {
            *(reinterpret_cast<float*>(param_partial + (prow + lane64)) + (0)) = dbase_n;
        }
        if (lane_1 < 3) {
            int glo = ((lane_1 < 2) ? lane_1 * 4 : 8);
            int ghi = ((lane_1 < 2) ? lane_1 * 4 + 4 : 24);
            float ga[1];
            float gc[1];
            ga[0] = 0.0f;
            gc[0] = 0.0f;
            #pragma unroll 1
            for (int mrow = glo; mrow < ghi; mrow++) {
                float sub_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_3) : "f"(gsum[mrow]), "f"(gc[0]));
                float add_23;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_23) : "f"(ga[0]), "f"(sub_3));
                float _sub_780;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_780) : "f"(add_23), "f"(ga[0]));
                float _sub_779;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_779) : "f"(_sub_780), "f"(sub_3));
                gc[0] = _sub_779;
                ga[0] = add_23;
            }
            float _sub_781;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_781) : "f"(ga[0]), "f"(gc[0]));
            *(reinterpret_cast<float*>(param_partial + (prow + 24 + lane64)) + (0)) = _sub_781;
        }
        if (lane_1 >= 27) {
            *(reinterpret_cast<float*>(param_partial + (prow + lane64)) + (0)) = 0.0f;
        }
    }
    if (t0 < T && t_end >= T) {
        float zero = 0.0f;
        int npad = t_pad - T;
        long long t64 = (long long)T;
        int nz = npad * 96;
        int trips_z = (nz + 255) / 256;
        #pragma unroll 1
        for (int k = 0; k < trips_z; k++) {
            int ez = k * 256 + tid_2;
            if (ez < nz) {
                *(reinterpret_cast<__nv_bfloat16*>(dzp + (t64 * 96 + (long long)ez)) + (0)) = (__nv_bfloat16)zero;
            }
        }
        int nt = npad * 24;
        int trips_t = (nt + 255) / 256;
        #pragma unroll 1
        for (int k_1 = 0; k_1 < trips_t; k_1++) {
            int et = k_1 * 256 + tid_2;
            if (et < nt) {
                *(reinterpret_cast<float*>(tokp + (t64 * 24 + (long long)et)) + (0)) = zero;
            }
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
