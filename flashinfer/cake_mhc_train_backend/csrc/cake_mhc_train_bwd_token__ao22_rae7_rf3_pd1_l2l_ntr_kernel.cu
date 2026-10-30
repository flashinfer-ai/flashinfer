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
kernel_cake_mhc_train_bwd_token__ao22_rae7_rf3_pd1_l2l_ntr(__nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ dU, float* __restrict__ stats, float* __restrict__ gstats, float* __restrict__ scale, float* __restrict__ base, __nv_bfloat16* __restrict__ dzp, float* __restrict__ tokp, float* __restrict__ param_partial, int T, int t_pad, int rows_per_cta, float hc_eps)
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
        bool tail_pred = t1 > g0 + 8;
        bool first_grp = gi[0] == 0;
        int td = t0 + 1;
        int tdc = ((td < t1) ? td : t_last);
        long long td64 = (long long)tdc;
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
        __syncwarp();
        bool elected = elect_sync();
        bool defer_el = elected && first_grp;
        if (defer_el) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_1) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1), "l"(dU + (td64 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 1024), "l"(X + (td64 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 2048), "l"(X + ((td64 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 3072), "l"(X + ((td64 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 4096), "l"(X + ((td64 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
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
        float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, c_8, 4);
        float s_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_9) : "f"(c_8), "f"(_shfl_xor_sync_47));
        float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, s_9, 24);
        float _add_49;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(s_9), "f"(_shfl_xor_sync_50));
        float _add_48;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(_add_49), "f"(hc_eps));
        float c_9 = __fdiv_rn(c_8, _add_48);
        hist[11] = c_9;
        float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, c_9, 1);
        float s_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_10) : "f"(c_9), "f"(_shfl_xor_sync_51));
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, s_10, 2);
        float _add_53;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(s_10), "f"(_shfl_xor_sync_54));
        float _add_52;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(_add_53), "f"(hc_eps));
        float c_10 = __fdiv_rn(c_9, _add_52);
        hist[12] = c_10;
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, c_10, 4);
        float s_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_11) : "f"(c_10), "f"(_shfl_xor_sync_55));
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, s_11, 24);
        float _add_57;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(s_11), "f"(_shfl_xor_sync_58));
        float _add_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(_add_57), "f"(hc_eps));
        float c_11 = __fdiv_rn(c_10, _add_56);
        hist[13] = c_11;
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, c_11, 1);
        float s_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_12) : "f"(c_11), "f"(_shfl_xor_sync_59));
        float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, s_12, 2);
        float _add_61;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(s_12), "f"(_shfl_xor_sync_62));
        float _add_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(_add_61), "f"(hc_eps));
        float c_12 = __fdiv_rn(c_11, _add_60);
        hist[14] = c_12;
        float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, c_12, 4);
        float s_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_13) : "f"(c_12), "f"(_shfl_xor_sync_63));
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, s_13, 24);
        float _add_65;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(s_13), "f"(_shfl_xor_sync_66));
        float _add_64;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(_add_65), "f"(hc_eps));
        float c_13 = __fdiv_rn(c_12, _add_64);
        hist[15] = c_13;
        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, c_13, 1);
        float s_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_14) : "f"(c_13), "f"(_shfl_xor_sync_67));
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, s_14, 2);
        float _add_69;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(s_14), "f"(_shfl_xor_sync_70));
        float _add_68;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(_add_69), "f"(hc_eps));
        float c_14 = __fdiv_rn(c_13, _add_68);
        hist[16] = c_14;
        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, c_14, 4);
        float s_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_15) : "f"(c_14), "f"(_shfl_xor_sync_71));
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, s_15, 24);
        float _add_73;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(s_15), "f"(_shfl_xor_sync_74));
        float _add_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(_add_73), "f"(hc_eps));
        float c_15 = __fdiv_rn(c_14, _add_72);
        hist[17] = c_15;
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, c_15, 1);
        float s_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_16) : "f"(c_15), "f"(_shfl_xor_sync_75));
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, s_16, 2);
        float _add_77;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(s_16), "f"(_shfl_xor_sync_78));
        float _add_76;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(_add_77), "f"(hc_eps));
        float c_16 = __fdiv_rn(c_15, _add_76);
        hist[18] = c_16;
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, c_16, 4);
        float s_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_17) : "f"(c_16), "f"(_shfl_xor_sync_79));
        float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, s_17, 24);
        float _add_81;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(s_17), "f"(_shfl_xor_sync_82));
        float _add_80;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(_add_81), "f"(hc_eps));
        float c_17 = __fdiv_rn(c_16, _add_80);
        hist[19] = c_17;
        float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, c_17, 1);
        float s_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_18) : "f"(c_17), "f"(_shfl_xor_sync_83));
        float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, s_18, 2);
        float _add_85;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(s_18), "f"(_shfl_xor_sync_86));
        float _add_84;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(_add_85), "f"(hc_eps));
        float c_18 = __fdiv_rn(c_17, _add_84);
        hist[20] = c_18;
        float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, c_18, 4);
        float s_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_19) : "f"(c_18), "f"(_shfl_xor_sync_87));
        float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, s_19, 24);
        float _add_89;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(s_19), "f"(_shfl_xor_sync_90));
        float _add_88;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(_add_89), "f"(hc_eps));
        float c_19 = __fdiv_rn(c_18, _add_88);
        hist[21] = c_19;
        float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, c_19, 1);
        float s_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_20) : "f"(c_19), "f"(_shfl_xor_sync_91));
        float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, s_20, 2);
        float _add_93;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(s_20), "f"(_shfl_xor_sync_94));
        float _add_92;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(_add_93), "f"(hc_eps));
        float c_20 = __fdiv_rn(c_19, _add_92);
        hist[22] = c_20;
        float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, c_20, 4);
        float s_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_21) : "f"(c_20), "f"(_shfl_xor_sync_95));
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, s_21, 24);
        float _add_97;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(s_21), "f"(_shfl_xor_sync_98));
        float _add_96;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(_add_97), "f"(hc_eps));
        float c_21 = __fdiv_rn(c_20, _add_96);
        hist[23] = c_21;
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
        float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, c_21, 1);
        float s_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_22) : "f"(c_21), "f"(_shfl_xor_sync_99));
        float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, s_22, 2);
        float _add_101;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(s_22), "f"(_shfl_xor_sync_102));
        float _add_100;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(_add_101), "f"(hc_eps));
        float c_22 = __fdiv_rn(c_21, _add_100);
        hist[24] = c_22;
        float _dot_103;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_103) : "r"(ring_pk_0[0]), "r"(ring_pk[0]), "f"(-0.0f));
        dots_p[0] = _dot_103;
        float _dot_104;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_104) : "r"(ring_pk_0[1]), "r"(ring_pk[1]), "f"(dots_p[0]));
        dots_p[0] = _dot_104;
        float _dot_105;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_105) : "r"(ring_pk_0[2]), "r"(ring_pk[2]), "f"(dots_p[0]));
        dots_p[0] = _dot_105;
        float _dot_106;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_106) : "r"(ring_pk_0[3]), "r"(ring_pk[3]), "f"(dots_p[0]));
        dots_p[0] = _dot_106;
        float _dot_107;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_107) : "r"(ring_pk_0[4]), "r"(ring_pk[4]), "f"(dots_p[0]));
        dots_p[0] = _dot_107;
        float _dot_108;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_108) : "r"(ring_pk_0[5]), "r"(ring_pk[5]), "f"(dots_p[0]));
        dots_p[0] = _dot_108;
        float _dot_109;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_109) : "r"(ring_pk_0[6]), "r"(ring_pk[6]), "f"(dots_p[0]));
        dots_p[0] = _dot_109;
        float _dot_110;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_110) : "r"(ring_pk_0[7]), "r"(ring_pk[7]), "f"(dots_p[0]));
        dots_p[0] = _dot_110;
        unsigned int ring_pk_1[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(0) + 3]))
            : "r"(sa__1_0 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(4) + 3]))
            : "r"(sa__1_0 + 2064 + lane32));
        float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, c_22, 4);
        float s_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_23) : "f"(c_22), "f"(_shfl_xor_sync_111));
        float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, s_23, 24);
        float _add_113;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(s_23), "f"(_shfl_xor_sync_114));
        float _add_112;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(_add_113), "f"(hc_eps));
        float c_23 = __fdiv_rn(c_22, _add_112);
        hist[25] = c_23;
        float _dot_115;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_115) : "r"(ring_pk_1[0]), "r"(ring_pk[0]), "f"(-0.0f));
        dots_p[1] = _dot_115;
        float _dot_116;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_116) : "r"(ring_pk_1[1]), "r"(ring_pk[1]), "f"(dots_p[1]));
        dots_p[1] = _dot_116;
        float _dot_117;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_117) : "r"(ring_pk_1[2]), "r"(ring_pk[2]), "f"(dots_p[1]));
        dots_p[1] = _dot_117;
        float _dot_118;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_118) : "r"(ring_pk_1[3]), "r"(ring_pk[3]), "f"(dots_p[1]));
        dots_p[1] = _dot_118;
        float _dot_119;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_119) : "r"(ring_pk_1[4]), "r"(ring_pk[4]), "f"(dots_p[1]));
        dots_p[1] = _dot_119;
        float _dot_120;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_120) : "r"(ring_pk_1[5]), "r"(ring_pk[5]), "f"(dots_p[1]));
        dots_p[1] = _dot_120;
        float _dot_121;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_121) : "r"(ring_pk_1[6]), "r"(ring_pk[6]), "f"(dots_p[1]));
        dots_p[1] = _dot_121;
        float _dot_122;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_122) : "r"(ring_pk_1[7]), "r"(ring_pk[7]), "f"(dots_p[1]));
        dots_p[1] = _dot_122;
        unsigned int ring_pk_2[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(0) + 3]))
            : "r"(sa__1_0 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(4) + 3]))
            : "r"(sa__1_0 + 3088 + lane32));
        float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, c_23, 1);
        float s_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_24) : "f"(c_23), "f"(_shfl_xor_sync_123));
        float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, s_24, 2);
        float _add_125;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(s_24), "f"(_shfl_xor_sync_126));
        float _add_124;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(_add_125), "f"(hc_eps));
        float c_24 = __fdiv_rn(c_23, _add_124);
        hist[26] = c_24;
        float _dot_127;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_127) : "r"(ring_pk_2[0]), "r"(ring_pk[0]), "f"(-0.0f));
        dots_p[2] = _dot_127;
        float _dot_128;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_128) : "r"(ring_pk_2[1]), "r"(ring_pk[1]), "f"(dots_p[2]));
        dots_p[2] = _dot_128;
        float _dot_129;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_129) : "r"(ring_pk_2[2]), "r"(ring_pk[2]), "f"(dots_p[2]));
        dots_p[2] = _dot_129;
        float _dot_130;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_130) : "r"(ring_pk_2[3]), "r"(ring_pk[3]), "f"(dots_p[2]));
        dots_p[2] = _dot_130;
        float _dot_131;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_131) : "r"(ring_pk_2[4]), "r"(ring_pk[4]), "f"(dots_p[2]));
        dots_p[2] = _dot_131;
        float _dot_132;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_132) : "r"(ring_pk_2[5]), "r"(ring_pk[5]), "f"(dots_p[2]));
        dots_p[2] = _dot_132;
        float _dot_133;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_133) : "r"(ring_pk_2[6]), "r"(ring_pk[6]), "f"(dots_p[2]));
        dots_p[2] = _dot_133;
        float _dot_134;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_134) : "r"(ring_pk_2[7]), "r"(ring_pk[7]), "f"(dots_p[2]));
        dots_p[2] = _dot_134;
        unsigned int ring_pk_3[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(0) + 3]))
            : "r"(sa__1_0 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(4) + 3]))
            : "r"(sa__1_0 + 4112 + lane32));
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
        float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, c_24, 4);
        float s_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_25) : "f"(c_24), "f"(_shfl_xor_sync_135));
        float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, s_25, 24);
        float _add_137;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(s_25), "f"(_shfl_xor_sync_138));
        float _add_136;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(_add_137), "f"(hc_eps));
        float c_25 = __fdiv_rn(c_24, _add_136);
        hist[27] = c_25;
        float _dot_139;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_139) : "r"(ring_pk_3[0]), "r"(ring_pk[0]), "f"(-0.0f));
        dots_p[3] = _dot_139;
        float _dot_140;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_140) : "r"(ring_pk_3[1]), "r"(ring_pk[1]), "f"(dots_p[3]));
        dots_p[3] = _dot_140;
        float _dot_141;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_141) : "r"(ring_pk_3[2]), "r"(ring_pk[2]), "f"(dots_p[3]));
        dots_p[3] = _dot_141;
        float _dot_142;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_142) : "r"(ring_pk_3[3]), "r"(ring_pk[3]), "f"(dots_p[3]));
        dots_p[3] = _dot_142;
        float _dot_143;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_143) : "r"(ring_pk_3[4]), "r"(ring_pk[4]), "f"(dots_p[3]));
        dots_p[3] = _dot_143;
        float _dot_144;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_144) : "r"(ring_pk_3[5]), "r"(ring_pk[5]), "f"(dots_p[3]));
        dots_p[3] = _dot_144;
        float _dot_145;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_145) : "r"(ring_pk_3[6]), "r"(ring_pk[6]), "f"(dots_p[3]));
        dots_p[3] = _dot_145;
        float _dot_146;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_146) : "r"(ring_pk_3[7]), "r"(ring_pk[7]), "f"(dots_p[3]));
        dots_p[3] = _dot_146;
        float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, c_25, 1);
        float s_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_26) : "f"(c_25), "f"(_shfl_xor_sync_147));
        float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, s_26, 2);
        float _add_149;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_149) : "f"(s_26), "f"(_shfl_xor_sync_150));
        float _add_148;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(_add_149), "f"(hc_eps));
        float c_26 = __fdiv_rn(c_25, _add_148);
        hist[28] = c_26;
        int upper = lane_1 & 16;
        float send = ((upper != 0) ? dots_p[0] : dots_p[2]);
        float keep = ((upper != 0) ? dots_p[2] : dots_p[0]);
        float _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
        float add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_0) : "f"(keep), "f"(_shfl_xor_sync_151));
        float send_1 = ((upper != 0) ? dots_p[1] : dots_p[3]);
        float keep_1 = ((upper != 0) ? dots_p[3] : dots_p[1]);
        float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, send_1, 16);
        float add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(keep_1), "f"(_shfl_xor_sync_152));
        int upper_1 = lane_1 & 8;
        float send_2 = ((upper_1 != 0) ? add_0 : add_1);
        float keep_2 = ((upper_1 != 0) ? add_1 : add_0);
        float _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, send_2, 8);
        float add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(keep_2), "f"(_shfl_xor_sync_153));
        float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, c_26, 4);
        float s_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_27) : "f"(c_26), "f"(_shfl_xor_sync_154));
        float _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, s_27, 24);
        float _add_156;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(s_27), "f"(_shfl_xor_sync_157));
        float _add_155;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155) : "f"(_add_156), "f"(hc_eps));
        float c_27 = __fdiv_rn(c_26, _add_155);
        hist[29] = c_27;
        float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, add_2, 4);
        float add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(add_2), "f"(_shfl_xor_sync_158));
        float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, add_3, 2);
        float add_1_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_0) : "f"(add_3), "f"(_shfl_xor_sync_159));
        float _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, add_1_0, 1);
        float _add_160;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(add_1_0), "f"(_shfl_xor_sync_161));
        dpre_part[own_base] = _add_160;
        float _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, c_27, 1);
        float s_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_28) : "f"(c_27), "f"(_shfl_xor_sync_162));
        float _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, s_28, 2);
        float _add_164;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_164) : "f"(s_28), "f"(_shfl_xor_sync_165));
        float _add_163;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(_add_164), "f"(hc_eps));
        float c_28 = __fdiv_rn(c_27, _add_163);
        hist[30] = c_28;
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
        float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, c_28, 4);
        float s_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_29) : "f"(c_28), "f"(_shfl_xor_sync_166));
        float _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, s_29, 24);
        float _add_168;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(s_29), "f"(_shfl_xor_sync_169));
        float _add_167;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(_add_168), "f"(hc_eps));
        float c_29 = __fdiv_rn(c_28, _add_167);
        hist[31] = c_29;
        float _dot_170;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_170) : "r"(ring_pk_5[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[0] = _dot_170;
        float _dot_171;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_171) : "r"(ring_pk_5[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_171;
        float _dot_172;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_172) : "r"(ring_pk_5[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_172;
        float _dot_173;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_173) : "r"(ring_pk_5[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_173;
        float _dot_174;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_174) : "r"(ring_pk_5[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_174;
        float _dot_175;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_175) : "r"(ring_pk_5[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_175;
        float _dot_176;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_176) : "r"(ring_pk_5[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_176;
        float _dot_177;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_177) : "r"(ring_pk_5[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_177;
        unsigned int ring_pk_6[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(0) + 3]))
            : "r"(sa__1_1 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(4) + 3]))
            : "r"(sa__1_1 + 2064 + lane32));
        float _shfl_xor_sync_178 = __shfl_xor_sync(0xFFFFFFFF, c_29, 1);
        float s_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_30) : "f"(c_29), "f"(_shfl_xor_sync_178));
        float _shfl_xor_sync_181 = __shfl_xor_sync(0xFFFFFFFF, s_30, 2);
        float _add_180;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_180) : "f"(s_30), "f"(_shfl_xor_sync_181));
        float _add_179;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_179) : "f"(_add_180), "f"(hc_eps));
        float c_30 = __fdiv_rn(c_29, _add_179);
        hist[32] = c_30;
        float _dot_182;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_182) : "r"(ring_pk_6[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[1] = _dot_182;
        float _dot_183;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_183) : "r"(ring_pk_6[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_183;
        float _dot_184;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_184) : "r"(ring_pk_6[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_184;
        float _dot_185;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_185) : "r"(ring_pk_6[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_185;
        float _dot_186;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_186) : "r"(ring_pk_6[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_186;
        float _dot_187;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_187) : "r"(ring_pk_6[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_187;
        float _dot_188;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_188) : "r"(ring_pk_6[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_188;
        float _dot_189;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_189) : "r"(ring_pk_6[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_189;
        unsigned int ring_pk_7[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(0) + 3]))
            : "r"(sa__1_1 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(4) + 3]))
            : "r"(sa__1_1 + 3088 + lane32));
        float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, c_30, 4);
        float s_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_31) : "f"(c_30), "f"(_shfl_xor_sync_190));
        float _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, s_31, 24);
        float _add_192;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_192) : "f"(s_31), "f"(_shfl_xor_sync_193));
        float _add_191;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191) : "f"(_add_192), "f"(hc_eps));
        float c_31 = __fdiv_rn(c_30, _add_191);
        hist[33] = c_31;
        float _dot_194;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_194) : "r"(ring_pk_7[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[2] = _dot_194;
        float _dot_195;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_195) : "r"(ring_pk_7[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_195;
        float _dot_196;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_196) : "r"(ring_pk_7[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_196;
        float _dot_197;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_197) : "r"(ring_pk_7[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_197;
        float _dot_198;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_198) : "r"(ring_pk_7[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_198;
        float _dot_199;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_199) : "r"(ring_pk_7[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_199;
        float _dot_200;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_200) : "r"(ring_pk_7[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_200;
        float _dot_201;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_201) : "r"(ring_pk_7[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_201;
        unsigned int ring_pk_8[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(0) + 3]))
            : "r"(sa__1_1 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(4) + 3]))
            : "r"(sa__1_1 + 4112 + lane32));
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
        float _shfl_xor_sync_202 = __shfl_xor_sync(0xFFFFFFFF, c_31, 1);
        float s_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_32) : "f"(c_31), "f"(_shfl_xor_sync_202));
        float _shfl_xor_sync_205 = __shfl_xor_sync(0xFFFFFFFF, s_32, 2);
        float _add_204;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_204) : "f"(s_32), "f"(_shfl_xor_sync_205));
        float _add_203;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_203) : "f"(_add_204), "f"(hc_eps));
        float c_32 = __fdiv_rn(c_31, _add_203);
        hist[34] = c_32;
        float _dot_206;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_206) : "r"(ring_pk_8[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[3] = _dot_206;
        float _dot_207;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_207) : "r"(ring_pk_8[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_207;
        float _dot_208;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_208) : "r"(ring_pk_8[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_208;
        float _dot_209;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_209) : "r"(ring_pk_8[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_209;
        float _dot_210;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_210) : "r"(ring_pk_8[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_210;
        float _dot_211;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_211) : "r"(ring_pk_8[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_211;
        float _dot_212;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_212) : "r"(ring_pk_8[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_212;
        float _dot_213;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_213) : "r"(ring_pk_8[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_213;
        float _shfl_xor_sync_214 = __shfl_xor_sync(0xFFFFFFFF, c_32, 4);
        float s_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_33) : "f"(c_32), "f"(_shfl_xor_sync_214));
        float _shfl_xor_sync_217 = __shfl_xor_sync(0xFFFFFFFF, s_33, 24);
        float _add_216;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_216) : "f"(s_33), "f"(_shfl_xor_sync_217));
        float _add_215;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_215) : "f"(_add_216), "f"(hc_eps));
        float c_33 = __fdiv_rn(c_32, _add_215);
        hist[35] = c_33;
        int upper_0 = lane_1 & 16;
        float send_0 = ((upper_0 != 0) ? dots_p_0[0] : dots_p_0[2]);
        float keep_0 = ((upper_0 != 0) ? dots_p_0[2] : dots_p_0[0]);
        float _shfl_xor_sync_218 = __shfl_xor_sync(0xFFFFFFFF, send_0, 16);
        float add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(keep_0), "f"(_shfl_xor_sync_218));
        float send_1_0 = ((upper_0 != 0) ? dots_p_0[1] : dots_p_0[3]);
        float keep_1_0 = ((upper_0 != 0) ? dots_p_0[3] : dots_p_0[1]);
        float _shfl_xor_sync_219 = __shfl_xor_sync(0xFFFFFFFF, send_1_0, 16);
        float add_1_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_1) : "f"(keep_1_0), "f"(_shfl_xor_sync_219));
        int upper_1_0 = lane_1 & 8;
        float send_2_0 = ((upper_1_0 != 0) ? add_4 : add_1_1);
        float keep_2_0 = ((upper_1_0 != 0) ? add_1_1 : add_4);
        float _shfl_xor_sync_220 = __shfl_xor_sync(0xFFFFFFFF, send_2_0, 8);
        float add_2_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_0) : "f"(keep_2_0), "f"(_shfl_xor_sync_220));
        float _shfl_xor_sync_221 = __shfl_xor_sync(0xFFFFFFFF, c_33, 1);
        float s_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_34) : "f"(c_33), "f"(_shfl_xor_sync_221));
        float _shfl_xor_sync_224 = __shfl_xor_sync(0xFFFFFFFF, s_34, 2);
        float _add_223;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_223) : "f"(s_34), "f"(_shfl_xor_sync_224));
        float _add_222;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_222) : "f"(_add_223), "f"(hc_eps));
        float c_34 = __fdiv_rn(c_33, _add_222);
        hist[36] = c_34;
        float _shfl_xor_sync_225 = __shfl_xor_sync(0xFFFFFFFF, add_2_0, 4);
        float add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(add_2_0), "f"(_shfl_xor_sync_225));
        float _shfl_xor_sync_226 = __shfl_xor_sync(0xFFFFFFFF, add_5, 2);
        float add_1_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_2) : "f"(add_5), "f"(_shfl_xor_sync_226));
        float _shfl_xor_sync_228 = __shfl_xor_sync(0xFFFFFFFF, add_1_2, 1);
        float _add_227;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_227) : "f"(add_1_2), "f"(_shfl_xor_sync_228));
        dpre_part[own_base + 32] = _add_227;
        float _shfl_xor_sync_229 = __shfl_xor_sync(0xFFFFFFFF, c_34, 4);
        float s_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_35) : "f"(c_34), "f"(_shfl_xor_sync_229));
        float _shfl_xor_sync_232 = __shfl_xor_sync(0xFFFFFFFF, s_35, 24);
        float _add_231;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_231) : "f"(s_35), "f"(_shfl_xor_sync_232));
        float _add_230;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_230) : "f"(_add_231), "f"(hc_eps));
        float c_35 = __fdiv_rn(c_34, _add_230);
        hist[37] = c_35;
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
        float _shfl_xor_sync_233 = __shfl_xor_sync(0xFFFFFFFF, c_35, 1);
        float s_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_36) : "f"(c_35), "f"(_shfl_xor_sync_233));
        float _shfl_xor_sync_236 = __shfl_xor_sync(0xFFFFFFFF, s_36, 2);
        float _add_235;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_235) : "f"(s_36), "f"(_shfl_xor_sync_236));
        float _add_234;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_234) : "f"(_add_235), "f"(hc_eps));
        float c_36 = __fdiv_rn(c_35, _add_234);
        hist[38] = c_36;
        float _dot_237;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_237) : "r"(ring_pk_10[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[0] = _dot_237;
        float _dot_238;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_238) : "r"(ring_pk_10[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_238;
        float _dot_239;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_239) : "r"(ring_pk_10[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_239;
        float _dot_240;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_240) : "r"(ring_pk_10[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_240;
        float _dot_241;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_241) : "r"(ring_pk_10[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_241;
        float _dot_242;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_242) : "r"(ring_pk_10[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_242;
        float _dot_243;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_243) : "r"(ring_pk_10[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_243;
        float _dot_244;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_244) : "r"(ring_pk_10[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_244;
        unsigned int ring_pk_11[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(0) + 3]))
            : "r"(sa__1_0 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(4) + 3]))
            : "r"(sa__1_0 + 2064 + lane32));
        float _shfl_xor_sync_245 = __shfl_xor_sync(0xFFFFFFFF, c_36, 4);
        float s_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_37) : "f"(c_36), "f"(_shfl_xor_sync_245));
        float _shfl_xor_sync_248 = __shfl_xor_sync(0xFFFFFFFF, s_37, 24);
        float _add_247;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_247) : "f"(s_37), "f"(_shfl_xor_sync_248));
        float _add_246;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(_add_247), "f"(hc_eps));
        float c_37 = __fdiv_rn(c_36, _add_246);
        hist[39] = c_37;
        float _dot_249;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_249) : "r"(ring_pk_11[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[1] = _dot_249;
        float _dot_250;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_250) : "r"(ring_pk_11[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_250;
        float _dot_251;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_251) : "r"(ring_pk_11[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_251;
        float _dot_252;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_252) : "r"(ring_pk_11[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_252;
        float _dot_253;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_253) : "r"(ring_pk_11[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_253;
        float _dot_254;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_254) : "r"(ring_pk_11[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_254;
        float _dot_255;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_255) : "r"(ring_pk_11[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_255;
        float _dot_256;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_256) : "r"(ring_pk_11[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_256;
        unsigned int ring_pk_12[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(0) + 3]))
            : "r"(sa__1_0 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(4) + 3]))
            : "r"(sa__1_0 + 3088 + lane32));
        float dsig;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dsig) : "f"(sig), "f"(sneg));
        float G = ((lane_1 >= 8 && lane_1 < 24) ? gval : 0.0f);
        float _shfl_xor_sync_257 = __shfl_xor_sync(0xFFFFFFFF, hist[38], 4);
        float s_38;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_38) : "f"(hist[38]), "f"(_shfl_xor_sync_257));
        float _shfl_xor_sync_259 = __shfl_xor_sync(0xFFFFFFFF, s_38, 24);
        float _add_258;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_258) : "f"(s_38), "f"(_shfl_xor_sync_259));
        float v;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(_add_258), "f"(hc_eps));
        float m;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m) : "f"(G), "f"(hist[39]));
        float _shfl_xor_sync_260 = __shfl_xor_sync(0xFFFFFFFF, m, 4);
        float ip;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip) : "f"(m), "f"(_shfl_xor_sync_260));
        float _shfl_xor_sync_263 = __shfl_xor_sync(0xFFFFFFFF, ip, 24);
        float _add_262;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_262) : "f"(ip), "f"(_shfl_xor_sync_263));
        float _sub_261;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_261) : "f"(G), "f"(_add_262));
        float g = __fdiv_rn(_sub_261, v);
        float _dot_264;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_264) : "r"(ring_pk_12[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[2] = _dot_264;
        float _dot_265;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_265) : "r"(ring_pk_12[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_265;
        float _dot_266;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_266) : "r"(ring_pk_12[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_266;
        float _dot_267;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_267) : "r"(ring_pk_12[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_267;
        float _dot_268;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_268) : "r"(ring_pk_12[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_268;
        float _dot_269;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_269) : "r"(ring_pk_12[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_269;
        float _dot_270;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_270) : "r"(ring_pk_12[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_270;
        float _dot_271;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_271) : "r"(ring_pk_12[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_271;
        unsigned int ring_pk_13[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(0) + 3]))
            : "r"(sa__1_0 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(4) + 3]))
            : "r"(sa__1_0 + 4112 + lane32));
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
        float _shfl_xor_sync_272 = __shfl_xor_sync(0xFFFFFFFF, hist[37], 1);
        float s_39;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_39) : "f"(hist[37]), "f"(_shfl_xor_sync_272));
        float _shfl_xor_sync_274 = __shfl_xor_sync(0xFFFFFFFF, s_39, 2);
        float _add_273;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_273) : "f"(s_39), "f"(_shfl_xor_sync_274));
        float v_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_0) : "f"(_add_273), "f"(hc_eps));
        float m_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_0) : "f"(g), "f"(hist[38]));
        float _shfl_xor_sync_275 = __shfl_xor_sync(0xFFFFFFFF, m_0, 1);
        float ip_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_0) : "f"(m_0), "f"(_shfl_xor_sync_275));
        float _shfl_xor_sync_278 = __shfl_xor_sync(0xFFFFFFFF, ip_0, 2);
        float _add_277;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_277) : "f"(ip_0), "f"(_shfl_xor_sync_278));
        float _sub_276;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_276) : "f"(g), "f"(_add_277));
        float g_0 = __fdiv_rn(_sub_276, v_0);
        float _dot_279;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_279) : "r"(ring_pk_13[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[3] = _dot_279;
        float _dot_280;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_280) : "r"(ring_pk_13[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_280;
        float _dot_281;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_281) : "r"(ring_pk_13[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_281;
        float _dot_282;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_282) : "r"(ring_pk_13[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_282;
        float _dot_283;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_283) : "r"(ring_pk_13[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_283;
        float _dot_284;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_284) : "r"(ring_pk_13[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_284;
        float _dot_285;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_285) : "r"(ring_pk_13[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_285;
        float _dot_286;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_286) : "r"(ring_pk_13[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_286;
        float _shfl_xor_sync_287 = __shfl_xor_sync(0xFFFFFFFF, hist[36], 4);
        float s_40;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_40) : "f"(hist[36]), "f"(_shfl_xor_sync_287));
        float _shfl_xor_sync_289 = __shfl_xor_sync(0xFFFFFFFF, s_40, 24);
        float _add_288;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_288) : "f"(s_40), "f"(_shfl_xor_sync_289));
        float v_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_1) : "f"(_add_288), "f"(hc_eps));
        float m_1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_1) : "f"(g_0), "f"(hist[37]));
        float _shfl_xor_sync_290 = __shfl_xor_sync(0xFFFFFFFF, m_1, 4);
        float ip_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_1) : "f"(m_1), "f"(_shfl_xor_sync_290));
        float _shfl_xor_sync_293 = __shfl_xor_sync(0xFFFFFFFF, ip_1, 24);
        float _add_292;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_292) : "f"(ip_1), "f"(_shfl_xor_sync_293));
        float _sub_291;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_291) : "f"(g_0), "f"(_add_292));
        float g_1 = __fdiv_rn(_sub_291, v_1);
        int upper_2 = lane_1 & 16;
        float send_3 = ((upper_2 != 0) ? dots_p_1[0] : dots_p_1[2]);
        float keep_3 = ((upper_2 != 0) ? dots_p_1[2] : dots_p_1[0]);
        float _shfl_xor_sync_294 = __shfl_xor_sync(0xFFFFFFFF, send_3, 16);
        float add_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_6) : "f"(keep_3), "f"(_shfl_xor_sync_294));
        float send_1_1 = ((upper_2 != 0) ? dots_p_1[1] : dots_p_1[3]);
        float keep_1_1 = ((upper_2 != 0) ? dots_p_1[3] : dots_p_1[1]);
        float _shfl_xor_sync_295 = __shfl_xor_sync(0xFFFFFFFF, send_1_1, 16);
        float add_1_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_3) : "f"(keep_1_1), "f"(_shfl_xor_sync_295));
        int upper_1_1 = lane_1 & 8;
        float send_2_1 = ((upper_1_1 != 0) ? add_6 : add_1_3);
        float keep_2_1 = ((upper_1_1 != 0) ? add_1_3 : add_6);
        float _shfl_xor_sync_296 = __shfl_xor_sync(0xFFFFFFFF, send_2_1, 8);
        float add_2_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_1) : "f"(keep_2_1), "f"(_shfl_xor_sync_296));
        float _shfl_xor_sync_297 = __shfl_xor_sync(0xFFFFFFFF, hist[35], 1);
        float s_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_41) : "f"(hist[35]), "f"(_shfl_xor_sync_297));
        float _shfl_xor_sync_299 = __shfl_xor_sync(0xFFFFFFFF, s_41, 2);
        float _add_298;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_298) : "f"(s_41), "f"(_shfl_xor_sync_299));
        float v_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_2) : "f"(_add_298), "f"(hc_eps));
        float m_2;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_2) : "f"(g_1), "f"(hist[36]));
        float _shfl_xor_sync_300 = __shfl_xor_sync(0xFFFFFFFF, m_2, 1);
        float ip_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_2) : "f"(m_2), "f"(_shfl_xor_sync_300));
        float _shfl_xor_sync_303 = __shfl_xor_sync(0xFFFFFFFF, ip_2, 2);
        float _add_302;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_302) : "f"(ip_2), "f"(_shfl_xor_sync_303));
        float _sub_301;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_301) : "f"(g_1), "f"(_add_302));
        float g_2 = __fdiv_rn(_sub_301, v_2);
        float _shfl_xor_sync_304 = __shfl_xor_sync(0xFFFFFFFF, add_2_1, 4);
        float add_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7) : "f"(add_2_1), "f"(_shfl_xor_sync_304));
        float _shfl_xor_sync_305 = __shfl_xor_sync(0xFFFFFFFF, add_7, 2);
        float add_1_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_4) : "f"(add_7), "f"(_shfl_xor_sync_305));
        float _shfl_xor_sync_307 = __shfl_xor_sync(0xFFFFFFFF, add_1_4, 1);
        float _add_306;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_306) : "f"(add_1_4), "f"(_shfl_xor_sync_307));
        dpre_part[own_base + 64] = _add_306;
        float _shfl_xor_sync_308 = __shfl_xor_sync(0xFFFFFFFF, hist[34], 4);
        float s_42;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_42) : "f"(hist[34]), "f"(_shfl_xor_sync_308));
        float _shfl_xor_sync_310 = __shfl_xor_sync(0xFFFFFFFF, s_42, 24);
        float _add_309;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_309) : "f"(s_42), "f"(_shfl_xor_sync_310));
        float v_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_3) : "f"(_add_309), "f"(hc_eps));
        float m_3;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_3) : "f"(g_2), "f"(hist[35]));
        float _shfl_xor_sync_311 = __shfl_xor_sync(0xFFFFFFFF, m_3, 4);
        float ip_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_3) : "f"(m_3), "f"(_shfl_xor_sync_311));
        float _shfl_xor_sync_314 = __shfl_xor_sync(0xFFFFFFFF, ip_3, 24);
        float _add_313;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_313) : "f"(ip_3), "f"(_shfl_xor_sync_314));
        float _sub_312;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_312) : "f"(g_2), "f"(_add_313));
        float g_3 = __fdiv_rn(_sub_312, v_3);
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
        float _shfl_xor_sync_315 = __shfl_xor_sync(0xFFFFFFFF, hist[33], 1);
        float s_43;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_43) : "f"(hist[33]), "f"(_shfl_xor_sync_315));
        float _shfl_xor_sync_317 = __shfl_xor_sync(0xFFFFFFFF, s_43, 2);
        float _add_316;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_316) : "f"(s_43), "f"(_shfl_xor_sync_317));
        float v_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_4) : "f"(_add_316), "f"(hc_eps));
        float m_4;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_4) : "f"(g_3), "f"(hist[34]));
        float _shfl_xor_sync_318 = __shfl_xor_sync(0xFFFFFFFF, m_4, 1);
        float ip_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_4) : "f"(m_4), "f"(_shfl_xor_sync_318));
        float _shfl_xor_sync_321 = __shfl_xor_sync(0xFFFFFFFF, ip_4, 2);
        float _add_320;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_320) : "f"(ip_4), "f"(_shfl_xor_sync_321));
        float _sub_319;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_319) : "f"(g_3), "f"(_add_320));
        float g_4 = __fdiv_rn(_sub_319, v_4);
        float _dot_322;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_322) : "r"(ring_pk_15[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[0] = _dot_322;
        float _dot_323;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_323) : "r"(ring_pk_15[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_323;
        float _dot_324;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_324) : "r"(ring_pk_15[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_324;
        float _dot_325;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_325) : "r"(ring_pk_15[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_325;
        float _dot_326;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_326) : "r"(ring_pk_15[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_326;
        float _dot_327;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_327) : "r"(ring_pk_15[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_327;
        float _dot_328;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_328) : "r"(ring_pk_15[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_328;
        float _dot_329;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_329) : "r"(ring_pk_15[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_329;
        unsigned int ring_pk_16[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(0) + 3]))
            : "r"(sa__1_1 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(4) + 3]))
            : "r"(sa__1_1 + 2064 + lane32));
        float _shfl_xor_sync_330 = __shfl_xor_sync(0xFFFFFFFF, hist[32], 4);
        float s_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_44) : "f"(hist[32]), "f"(_shfl_xor_sync_330));
        float _shfl_xor_sync_332 = __shfl_xor_sync(0xFFFFFFFF, s_44, 24);
        float _add_331;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_331) : "f"(s_44), "f"(_shfl_xor_sync_332));
        float v_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_5) : "f"(_add_331), "f"(hc_eps));
        float m_5;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_5) : "f"(g_4), "f"(hist[33]));
        float _shfl_xor_sync_333 = __shfl_xor_sync(0xFFFFFFFF, m_5, 4);
        float ip_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_5) : "f"(m_5), "f"(_shfl_xor_sync_333));
        float _shfl_xor_sync_336 = __shfl_xor_sync(0xFFFFFFFF, ip_5, 24);
        float _add_335;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_335) : "f"(ip_5), "f"(_shfl_xor_sync_336));
        float _sub_334;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_334) : "f"(g_4), "f"(_add_335));
        float g_5 = __fdiv_rn(_sub_334, v_5);
        float _dot_337;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_337) : "r"(ring_pk_16[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[1] = _dot_337;
        float _dot_338;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_338) : "r"(ring_pk_16[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_338;
        float _dot_339;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_339) : "r"(ring_pk_16[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_339;
        float _dot_340;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_340) : "r"(ring_pk_16[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_340;
        float _dot_341;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_341) : "r"(ring_pk_16[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_341;
        float _dot_342;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_342) : "r"(ring_pk_16[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_342;
        float _dot_343;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_343) : "r"(ring_pk_16[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_343;
        float _dot_344;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_344) : "r"(ring_pk_16[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_344;
        unsigned int ring_pk_17[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(0) + 3]))
            : "r"(sa__1_1 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(4) + 3]))
            : "r"(sa__1_1 + 3088 + lane32));
        float _shfl_xor_sync_345 = __shfl_xor_sync(0xFFFFFFFF, hist[31], 1);
        float s_45;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_45) : "f"(hist[31]), "f"(_shfl_xor_sync_345));
        float _shfl_xor_sync_347 = __shfl_xor_sync(0xFFFFFFFF, s_45, 2);
        float _add_346;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_346) : "f"(s_45), "f"(_shfl_xor_sync_347));
        float v_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_6) : "f"(_add_346), "f"(hc_eps));
        float m_6;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_6) : "f"(g_5), "f"(hist[32]));
        float _shfl_xor_sync_348 = __shfl_xor_sync(0xFFFFFFFF, m_6, 1);
        float ip_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_6) : "f"(m_6), "f"(_shfl_xor_sync_348));
        float _shfl_xor_sync_351 = __shfl_xor_sync(0xFFFFFFFF, ip_6, 2);
        float _add_350;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_350) : "f"(ip_6), "f"(_shfl_xor_sync_351));
        float _sub_349;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_349) : "f"(g_5), "f"(_add_350));
        float g_6 = __fdiv_rn(_sub_349, v_6);
        float _dot_352;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_352) : "r"(ring_pk_17[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[2] = _dot_352;
        float _dot_353;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_353) : "r"(ring_pk_17[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_353;
        float _dot_354;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_354) : "r"(ring_pk_17[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_354;
        float _dot_355;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_355) : "r"(ring_pk_17[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_355;
        float _dot_356;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_356) : "r"(ring_pk_17[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_356;
        float _dot_357;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_357) : "r"(ring_pk_17[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_357;
        float _dot_358;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_358) : "r"(ring_pk_17[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_358;
        float _dot_359;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_359) : "r"(ring_pk_17[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_359;
        unsigned int ring_pk_18[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(0) + 3]))
            : "r"(sa__1_1 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(4) + 3]))
            : "r"(sa__1_1 + 4112 + lane32));
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
        float _shfl_xor_sync_360 = __shfl_xor_sync(0xFFFFFFFF, hist[30], 4);
        float s_46;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_46) : "f"(hist[30]), "f"(_shfl_xor_sync_360));
        float _shfl_xor_sync_362 = __shfl_xor_sync(0xFFFFFFFF, s_46, 24);
        float _add_361;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_361) : "f"(s_46), "f"(_shfl_xor_sync_362));
        float v_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_7) : "f"(_add_361), "f"(hc_eps));
        float m_7;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_7) : "f"(g_6), "f"(hist[31]));
        float _shfl_xor_sync_363 = __shfl_xor_sync(0xFFFFFFFF, m_7, 4);
        float ip_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_7) : "f"(m_7), "f"(_shfl_xor_sync_363));
        float _shfl_xor_sync_366 = __shfl_xor_sync(0xFFFFFFFF, ip_7, 24);
        float _add_365;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_365) : "f"(ip_7), "f"(_shfl_xor_sync_366));
        float _sub_364;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_364) : "f"(g_6), "f"(_add_365));
        float g_7 = __fdiv_rn(_sub_364, v_7);
        float _dot_367;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_367) : "r"(ring_pk_18[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[3] = _dot_367;
        float _dot_368;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_368) : "r"(ring_pk_18[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_368;
        float _dot_369;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_369) : "r"(ring_pk_18[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_369;
        float _dot_370;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_370) : "r"(ring_pk_18[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_370;
        float _dot_371;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_371) : "r"(ring_pk_18[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_371;
        float _dot_372;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_372) : "r"(ring_pk_18[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_372;
        float _dot_373;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_373) : "r"(ring_pk_18[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_373;
        float _dot_374;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_374) : "r"(ring_pk_18[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_374;
        float _shfl_xor_sync_375 = __shfl_xor_sync(0xFFFFFFFF, hist[29], 1);
        float s_47;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_47) : "f"(hist[29]), "f"(_shfl_xor_sync_375));
        float _shfl_xor_sync_377 = __shfl_xor_sync(0xFFFFFFFF, s_47, 2);
        float _add_376;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_376) : "f"(s_47), "f"(_shfl_xor_sync_377));
        float v_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_8) : "f"(_add_376), "f"(hc_eps));
        float m_8;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_8) : "f"(g_7), "f"(hist[30]));
        float _shfl_xor_sync_378 = __shfl_xor_sync(0xFFFFFFFF, m_8, 1);
        float ip_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_8) : "f"(m_8), "f"(_shfl_xor_sync_378));
        float _shfl_xor_sync_381 = __shfl_xor_sync(0xFFFFFFFF, ip_8, 2);
        float _add_380;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_380) : "f"(ip_8), "f"(_shfl_xor_sync_381));
        float _sub_379;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_379) : "f"(g_7), "f"(_add_380));
        float g_8 = __fdiv_rn(_sub_379, v_8);
        int upper_3 = lane_1 & 16;
        float send_4 = ((upper_3 != 0) ? dots_p_2[0] : dots_p_2[2]);
        float keep_4 = ((upper_3 != 0) ? dots_p_2[2] : dots_p_2[0]);
        float _shfl_xor_sync_382 = __shfl_xor_sync(0xFFFFFFFF, send_4, 16);
        float add_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8) : "f"(keep_4), "f"(_shfl_xor_sync_382));
        float send_1_2 = ((upper_3 != 0) ? dots_p_2[1] : dots_p_2[3]);
        float keep_1_2 = ((upper_3 != 0) ? dots_p_2[3] : dots_p_2[1]);
        float _shfl_xor_sync_383 = __shfl_xor_sync(0xFFFFFFFF, send_1_2, 16);
        float add_1_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_5) : "f"(keep_1_2), "f"(_shfl_xor_sync_383));
        int upper_1_2 = lane_1 & 8;
        float send_2_2 = ((upper_1_2 != 0) ? add_8 : add_1_5);
        float keep_2_2 = ((upper_1_2 != 0) ? add_1_5 : add_8);
        float _shfl_xor_sync_384 = __shfl_xor_sync(0xFFFFFFFF, send_2_2, 8);
        float add_2_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_2) : "f"(keep_2_2), "f"(_shfl_xor_sync_384));
        float _shfl_xor_sync_385 = __shfl_xor_sync(0xFFFFFFFF, hist[28], 4);
        float s_48;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_48) : "f"(hist[28]), "f"(_shfl_xor_sync_385));
        float _shfl_xor_sync_387 = __shfl_xor_sync(0xFFFFFFFF, s_48, 24);
        float _add_386;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_386) : "f"(s_48), "f"(_shfl_xor_sync_387));
        float v_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_9) : "f"(_add_386), "f"(hc_eps));
        float m_9;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_9) : "f"(g_8), "f"(hist[29]));
        float _shfl_xor_sync_388 = __shfl_xor_sync(0xFFFFFFFF, m_9, 4);
        float ip_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_9) : "f"(m_9), "f"(_shfl_xor_sync_388));
        float _shfl_xor_sync_391 = __shfl_xor_sync(0xFFFFFFFF, ip_9, 24);
        float _add_390;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_390) : "f"(ip_9), "f"(_shfl_xor_sync_391));
        float _sub_389;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_389) : "f"(g_8), "f"(_add_390));
        float g_9 = __fdiv_rn(_sub_389, v_9);
        float _shfl_xor_sync_392 = __shfl_xor_sync(0xFFFFFFFF, add_2_2, 4);
        float add_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_9) : "f"(add_2_2), "f"(_shfl_xor_sync_392));
        float _shfl_xor_sync_393 = __shfl_xor_sync(0xFFFFFFFF, add_9, 2);
        float add_1_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_6) : "f"(add_9), "f"(_shfl_xor_sync_393));
        float _shfl_xor_sync_395 = __shfl_xor_sync(0xFFFFFFFF, add_1_6, 1);
        float _add_394;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_394) : "f"(add_1_6), "f"(_shfl_xor_sync_395));
        dpre_part[own_base + 96] = _add_394;
        float _shfl_xor_sync_396 = __shfl_xor_sync(0xFFFFFFFF, hist[27], 1);
        float s_49;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_49) : "f"(hist[27]), "f"(_shfl_xor_sync_396));
        float _shfl_xor_sync_398 = __shfl_xor_sync(0xFFFFFFFF, s_49, 2);
        float _add_397;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_397) : "f"(s_49), "f"(_shfl_xor_sync_398));
        float v_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_10) : "f"(_add_397), "f"(hc_eps));
        float m_10;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_10) : "f"(g_9), "f"(hist[28]));
        float _shfl_xor_sync_399 = __shfl_xor_sync(0xFFFFFFFF, m_10, 1);
        float ip_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_10) : "f"(m_10), "f"(_shfl_xor_sync_399));
        float _shfl_xor_sync_402 = __shfl_xor_sync(0xFFFFFFFF, ip_10, 2);
        float _add_401;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_401) : "f"(ip_10), "f"(_shfl_xor_sync_402));
        float _sub_400;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_400) : "f"(g_9), "f"(_add_401));
        float g_10 = __fdiv_rn(_sub_400, v_10);
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
        float _shfl_xor_sync_403 = __shfl_xor_sync(0xFFFFFFFF, hist[26], 4);
        float s_50;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_50) : "f"(hist[26]), "f"(_shfl_xor_sync_403));
        float _shfl_xor_sync_405 = __shfl_xor_sync(0xFFFFFFFF, s_50, 24);
        float _add_404;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_404) : "f"(s_50), "f"(_shfl_xor_sync_405));
        float v_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_11) : "f"(_add_404), "f"(hc_eps));
        float m_11;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_11) : "f"(g_10), "f"(hist[27]));
        float _shfl_xor_sync_406 = __shfl_xor_sync(0xFFFFFFFF, m_11, 4);
        float ip_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_11) : "f"(m_11), "f"(_shfl_xor_sync_406));
        float _shfl_xor_sync_409 = __shfl_xor_sync(0xFFFFFFFF, ip_11, 24);
        float _add_408;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_408) : "f"(ip_11), "f"(_shfl_xor_sync_409));
        float _sub_407;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_407) : "f"(g_10), "f"(_add_408));
        float g_11 = __fdiv_rn(_sub_407, v_11);
        float _dot_410;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_410) : "r"(ring_pk_20[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[0] = _dot_410;
        float _dot_411;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_411) : "r"(ring_pk_20[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_411;
        float _dot_412;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_412) : "r"(ring_pk_20[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_412;
        float _dot_413;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_413) : "r"(ring_pk_20[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_413;
        float _dot_414;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_414) : "r"(ring_pk_20[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_414;
        float _dot_415;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_415) : "r"(ring_pk_20[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_415;
        float _dot_416;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_416) : "r"(ring_pk_20[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_416;
        float _dot_417;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_417) : "r"(ring_pk_20[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_417;
        unsigned int ring_pk_21[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(0) + 3]))
            : "r"(sa__1_0 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(4) + 3]))
            : "r"(sa__1_0 + 2064 + lane32));
        float _shfl_xor_sync_418 = __shfl_xor_sync(0xFFFFFFFF, hist[25], 1);
        float s_51;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_51) : "f"(hist[25]), "f"(_shfl_xor_sync_418));
        float _shfl_xor_sync_420 = __shfl_xor_sync(0xFFFFFFFF, s_51, 2);
        float _add_419;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_419) : "f"(s_51), "f"(_shfl_xor_sync_420));
        float v_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_12) : "f"(_add_419), "f"(hc_eps));
        float m_12;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_12) : "f"(g_11), "f"(hist[26]));
        float _shfl_xor_sync_421 = __shfl_xor_sync(0xFFFFFFFF, m_12, 1);
        float ip_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_12) : "f"(m_12), "f"(_shfl_xor_sync_421));
        float _shfl_xor_sync_424 = __shfl_xor_sync(0xFFFFFFFF, ip_12, 2);
        float _add_423;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_423) : "f"(ip_12), "f"(_shfl_xor_sync_424));
        float _sub_422;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_422) : "f"(g_11), "f"(_add_423));
        float g_12 = __fdiv_rn(_sub_422, v_12);
        float _dot_425;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_425) : "r"(ring_pk_21[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[1] = _dot_425;
        float _dot_426;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_426) : "r"(ring_pk_21[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_426;
        float _dot_427;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_427) : "r"(ring_pk_21[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_427;
        float _dot_428;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_428) : "r"(ring_pk_21[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_428;
        float _dot_429;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_429) : "r"(ring_pk_21[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_429;
        float _dot_430;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_430) : "r"(ring_pk_21[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_430;
        float _dot_431;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_431) : "r"(ring_pk_21[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_431;
        float _dot_432;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_432) : "r"(ring_pk_21[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_432;
        unsigned int ring_pk_22[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(0) + 3]))
            : "r"(sa__1_0 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(4) + 3]))
            : "r"(sa__1_0 + 3088 + lane32));
        float _shfl_xor_sync_433 = __shfl_xor_sync(0xFFFFFFFF, hist[24], 4);
        float s_52;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_52) : "f"(hist[24]), "f"(_shfl_xor_sync_433));
        float _shfl_xor_sync_435 = __shfl_xor_sync(0xFFFFFFFF, s_52, 24);
        float _add_434;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_434) : "f"(s_52), "f"(_shfl_xor_sync_435));
        float v_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_13) : "f"(_add_434), "f"(hc_eps));
        float m_13;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_13) : "f"(g_12), "f"(hist[25]));
        float _shfl_xor_sync_436 = __shfl_xor_sync(0xFFFFFFFF, m_13, 4);
        float ip_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_13) : "f"(m_13), "f"(_shfl_xor_sync_436));
        float _shfl_xor_sync_439 = __shfl_xor_sync(0xFFFFFFFF, ip_13, 24);
        float _add_438;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_438) : "f"(ip_13), "f"(_shfl_xor_sync_439));
        float _sub_437;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_437) : "f"(g_12), "f"(_add_438));
        float g_13 = __fdiv_rn(_sub_437, v_13);
        float _dot_440;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_440) : "r"(ring_pk_22[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[2] = _dot_440;
        float _dot_441;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_441) : "r"(ring_pk_22[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_441;
        float _dot_442;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_442) : "r"(ring_pk_22[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_442;
        float _dot_443;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_443) : "r"(ring_pk_22[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_443;
        float _dot_444;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_444) : "r"(ring_pk_22[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_444;
        float _dot_445;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_445) : "r"(ring_pk_22[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_445;
        float _dot_446;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_446) : "r"(ring_pk_22[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_446;
        float _dot_447;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_447) : "r"(ring_pk_22[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_447;
        unsigned int ring_pk_23[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(0) + 3]))
            : "r"(sa__1_0 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(4) + 3]))
            : "r"(sa__1_0 + 4112 + lane32));
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
        float _shfl_xor_sync_448 = __shfl_xor_sync(0xFFFFFFFF, hist[23], 1);
        float s_53;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_53) : "f"(hist[23]), "f"(_shfl_xor_sync_448));
        float _shfl_xor_sync_450 = __shfl_xor_sync(0xFFFFFFFF, s_53, 2);
        float _add_449;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_449) : "f"(s_53), "f"(_shfl_xor_sync_450));
        float v_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_14) : "f"(_add_449), "f"(hc_eps));
        float m_14;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_14) : "f"(g_13), "f"(hist[24]));
        float _shfl_xor_sync_451 = __shfl_xor_sync(0xFFFFFFFF, m_14, 1);
        float ip_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_14) : "f"(m_14), "f"(_shfl_xor_sync_451));
        float _shfl_xor_sync_454 = __shfl_xor_sync(0xFFFFFFFF, ip_14, 2);
        float _add_453;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_453) : "f"(ip_14), "f"(_shfl_xor_sync_454));
        float _sub_452;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_452) : "f"(g_13), "f"(_add_453));
        float g_14 = __fdiv_rn(_sub_452, v_14);
        float _dot_455;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_455) : "r"(ring_pk_23[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[3] = _dot_455;
        float _dot_456;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_456) : "r"(ring_pk_23[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_456;
        float _dot_457;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_457) : "r"(ring_pk_23[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_457;
        float _dot_458;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_458) : "r"(ring_pk_23[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_458;
        float _dot_459;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_459) : "r"(ring_pk_23[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_459;
        float _dot_460;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_460) : "r"(ring_pk_23[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_460;
        float _dot_461;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_461) : "r"(ring_pk_23[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_461;
        float _dot_462;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_462) : "r"(ring_pk_23[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_462;
        float _shfl_xor_sync_463 = __shfl_xor_sync(0xFFFFFFFF, hist[22], 4);
        float s_54;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_54) : "f"(hist[22]), "f"(_shfl_xor_sync_463));
        float _shfl_xor_sync_465 = __shfl_xor_sync(0xFFFFFFFF, s_54, 24);
        float _add_464;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_464) : "f"(s_54), "f"(_shfl_xor_sync_465));
        float v_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_15) : "f"(_add_464), "f"(hc_eps));
        float m_15;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_15) : "f"(g_14), "f"(hist[23]));
        float _shfl_xor_sync_466 = __shfl_xor_sync(0xFFFFFFFF, m_15, 4);
        float ip_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_15) : "f"(m_15), "f"(_shfl_xor_sync_466));
        float _shfl_xor_sync_469 = __shfl_xor_sync(0xFFFFFFFF, ip_15, 24);
        float _add_468;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_468) : "f"(ip_15), "f"(_shfl_xor_sync_469));
        float _sub_467;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_467) : "f"(g_14), "f"(_add_468));
        float g_15 = __fdiv_rn(_sub_467, v_15);
        int upper_4 = lane_1 & 16;
        float send_5 = ((upper_4 != 0) ? dots_p_3[0] : dots_p_3[2]);
        float keep_5 = ((upper_4 != 0) ? dots_p_3[2] : dots_p_3[0]);
        float _shfl_xor_sync_470 = __shfl_xor_sync(0xFFFFFFFF, send_5, 16);
        float add_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_10) : "f"(keep_5), "f"(_shfl_xor_sync_470));
        float send_1_3 = ((upper_4 != 0) ? dots_p_3[1] : dots_p_3[3]);
        float keep_1_3 = ((upper_4 != 0) ? dots_p_3[3] : dots_p_3[1]);
        float _shfl_xor_sync_471 = __shfl_xor_sync(0xFFFFFFFF, send_1_3, 16);
        float add_1_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_7) : "f"(keep_1_3), "f"(_shfl_xor_sync_471));
        int upper_1_3 = lane_1 & 8;
        float send_2_3 = ((upper_1_3 != 0) ? add_10 : add_1_7);
        float keep_2_3 = ((upper_1_3 != 0) ? add_1_7 : add_10);
        float _shfl_xor_sync_472 = __shfl_xor_sync(0xFFFFFFFF, send_2_3, 8);
        float add_2_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_3) : "f"(keep_2_3), "f"(_shfl_xor_sync_472));
        float _shfl_xor_sync_473 = __shfl_xor_sync(0xFFFFFFFF, hist[21], 1);
        float s_55;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_55) : "f"(hist[21]), "f"(_shfl_xor_sync_473));
        float _shfl_xor_sync_475 = __shfl_xor_sync(0xFFFFFFFF, s_55, 2);
        float _add_474;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_474) : "f"(s_55), "f"(_shfl_xor_sync_475));
        float v_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_16) : "f"(_add_474), "f"(hc_eps));
        float m_16;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_16) : "f"(g_15), "f"(hist[22]));
        float _shfl_xor_sync_476 = __shfl_xor_sync(0xFFFFFFFF, m_16, 1);
        float ip_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_16) : "f"(m_16), "f"(_shfl_xor_sync_476));
        float _shfl_xor_sync_479 = __shfl_xor_sync(0xFFFFFFFF, ip_16, 2);
        float _add_478;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_478) : "f"(ip_16), "f"(_shfl_xor_sync_479));
        float _sub_477;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_477) : "f"(g_15), "f"(_add_478));
        float g_16 = __fdiv_rn(_sub_477, v_16);
        float _shfl_xor_sync_480 = __shfl_xor_sync(0xFFFFFFFF, add_2_3, 4);
        float add_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_11) : "f"(add_2_3), "f"(_shfl_xor_sync_480));
        float _shfl_xor_sync_481 = __shfl_xor_sync(0xFFFFFFFF, add_11, 2);
        float add_1_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_8) : "f"(add_11), "f"(_shfl_xor_sync_481));
        float _shfl_xor_sync_483 = __shfl_xor_sync(0xFFFFFFFF, add_1_8, 1);
        float _add_482;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_482) : "f"(add_1_8), "f"(_shfl_xor_sync_483));
        dpre_part[own_base + 128] = _add_482;
        float _shfl_xor_sync_484 = __shfl_xor_sync(0xFFFFFFFF, hist[20], 4);
        float s_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_56) : "f"(hist[20]), "f"(_shfl_xor_sync_484));
        float _shfl_xor_sync_486 = __shfl_xor_sync(0xFFFFFFFF, s_56, 24);
        float _add_485;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_485) : "f"(s_56), "f"(_shfl_xor_sync_486));
        float v_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_17) : "f"(_add_485), "f"(hc_eps));
        float m_17;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_17) : "f"(g_16), "f"(hist[21]));
        float _shfl_xor_sync_487 = __shfl_xor_sync(0xFFFFFFFF, m_17, 4);
        float ip_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_17) : "f"(m_17), "f"(_shfl_xor_sync_487));
        float _shfl_xor_sync_490 = __shfl_xor_sync(0xFFFFFFFF, ip_17, 24);
        float _add_489;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_489) : "f"(ip_17), "f"(_shfl_xor_sync_490));
        float _sub_488;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_488) : "f"(g_16), "f"(_add_489));
        float g_17 = __fdiv_rn(_sub_488, v_17);
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
        float _shfl_xor_sync_491 = __shfl_xor_sync(0xFFFFFFFF, hist[19], 1);
        float s_57;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_57) : "f"(hist[19]), "f"(_shfl_xor_sync_491));
        float _shfl_xor_sync_493 = __shfl_xor_sync(0xFFFFFFFF, s_57, 2);
        float _add_492;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_492) : "f"(s_57), "f"(_shfl_xor_sync_493));
        float v_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_18) : "f"(_add_492), "f"(hc_eps));
        float m_18;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_18) : "f"(g_17), "f"(hist[20]));
        float _shfl_xor_sync_494 = __shfl_xor_sync(0xFFFFFFFF, m_18, 1);
        float ip_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_18) : "f"(m_18), "f"(_shfl_xor_sync_494));
        float _shfl_xor_sync_497 = __shfl_xor_sync(0xFFFFFFFF, ip_18, 2);
        float _add_496;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_496) : "f"(ip_18), "f"(_shfl_xor_sync_497));
        float _sub_495;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_495) : "f"(g_17), "f"(_add_496));
        float g_18 = __fdiv_rn(_sub_495, v_18);
        float _dot_498;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_498) : "r"(ring_pk_25[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[0] = _dot_498;
        float _dot_499;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_499) : "r"(ring_pk_25[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_499;
        float _dot_500;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_500) : "r"(ring_pk_25[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_500;
        float _dot_501;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_501) : "r"(ring_pk_25[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_501;
        float _dot_502;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_502) : "r"(ring_pk_25[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_502;
        float _dot_503;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_503) : "r"(ring_pk_25[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_503;
        float _dot_504;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_504) : "r"(ring_pk_25[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_504;
        float _dot_505;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_505) : "r"(ring_pk_25[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_505;
        unsigned int ring_pk_26[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(0) + 3]))
            : "r"(sa__1_1 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(4) + 3]))
            : "r"(sa__1_1 + 2064 + lane32));
        float _shfl_xor_sync_506 = __shfl_xor_sync(0xFFFFFFFF, hist[18], 4);
        float s_58;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_58) : "f"(hist[18]), "f"(_shfl_xor_sync_506));
        float _shfl_xor_sync_508 = __shfl_xor_sync(0xFFFFFFFF, s_58, 24);
        float _add_507;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_507) : "f"(s_58), "f"(_shfl_xor_sync_508));
        float v_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_19) : "f"(_add_507), "f"(hc_eps));
        float m_19;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_19) : "f"(g_18), "f"(hist[19]));
        float _shfl_xor_sync_509 = __shfl_xor_sync(0xFFFFFFFF, m_19, 4);
        float ip_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_19) : "f"(m_19), "f"(_shfl_xor_sync_509));
        float _shfl_xor_sync_512 = __shfl_xor_sync(0xFFFFFFFF, ip_19, 24);
        float _add_511;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_511) : "f"(ip_19), "f"(_shfl_xor_sync_512));
        float _sub_510;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_510) : "f"(g_18), "f"(_add_511));
        float g_19 = __fdiv_rn(_sub_510, v_19);
        float _dot_513;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_513) : "r"(ring_pk_26[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[1] = _dot_513;
        float _dot_514;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_514) : "r"(ring_pk_26[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_514;
        float _dot_515;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_515) : "r"(ring_pk_26[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_515;
        float _dot_516;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_516) : "r"(ring_pk_26[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_516;
        float _dot_517;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_517) : "r"(ring_pk_26[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_517;
        float _dot_518;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_518) : "r"(ring_pk_26[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_518;
        float _dot_519;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_519) : "r"(ring_pk_26[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_519;
        float _dot_520;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_520) : "r"(ring_pk_26[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_520;
        unsigned int ring_pk_27[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(0) + 3]))
            : "r"(sa__1_1 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(4) + 3]))
            : "r"(sa__1_1 + 3088 + lane32));
        float _shfl_xor_sync_521 = __shfl_xor_sync(0xFFFFFFFF, hist[17], 1);
        float s_59;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_59) : "f"(hist[17]), "f"(_shfl_xor_sync_521));
        float _shfl_xor_sync_523 = __shfl_xor_sync(0xFFFFFFFF, s_59, 2);
        float _add_522;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_522) : "f"(s_59), "f"(_shfl_xor_sync_523));
        float v_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_20) : "f"(_add_522), "f"(hc_eps));
        float m_20;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_20) : "f"(g_19), "f"(hist[18]));
        float _shfl_xor_sync_524 = __shfl_xor_sync(0xFFFFFFFF, m_20, 1);
        float ip_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_20) : "f"(m_20), "f"(_shfl_xor_sync_524));
        float _shfl_xor_sync_527 = __shfl_xor_sync(0xFFFFFFFF, ip_20, 2);
        float _add_526;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_526) : "f"(ip_20), "f"(_shfl_xor_sync_527));
        float _sub_525;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_525) : "f"(g_19), "f"(_add_526));
        float g_20 = __fdiv_rn(_sub_525, v_20);
        float _dot_528;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_528) : "r"(ring_pk_27[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[2] = _dot_528;
        float _dot_529;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_529) : "r"(ring_pk_27[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_529;
        float _dot_530;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_530) : "r"(ring_pk_27[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_530;
        float _dot_531;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_531) : "r"(ring_pk_27[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_531;
        float _dot_532;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_532) : "r"(ring_pk_27[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_532;
        float _dot_533;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_533) : "r"(ring_pk_27[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_533;
        float _dot_534;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_534) : "r"(ring_pk_27[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_534;
        float _dot_535;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_535) : "r"(ring_pk_27[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_535;
        unsigned int ring_pk_28[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(0) + 3]))
            : "r"(sa__1_1 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(4) + 3]))
            : "r"(sa__1_1 + 4112 + lane32));
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
        float _shfl_xor_sync_536 = __shfl_xor_sync(0xFFFFFFFF, hist[16], 4);
        float s_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_60) : "f"(hist[16]), "f"(_shfl_xor_sync_536));
        float _shfl_xor_sync_538 = __shfl_xor_sync(0xFFFFFFFF, s_60, 24);
        float _add_537;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_537) : "f"(s_60), "f"(_shfl_xor_sync_538));
        float v_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_21) : "f"(_add_537), "f"(hc_eps));
        float m_21;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_21) : "f"(g_20), "f"(hist[17]));
        float _shfl_xor_sync_539 = __shfl_xor_sync(0xFFFFFFFF, m_21, 4);
        float ip_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_21) : "f"(m_21), "f"(_shfl_xor_sync_539));
        float _shfl_xor_sync_542 = __shfl_xor_sync(0xFFFFFFFF, ip_21, 24);
        float _add_541;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_541) : "f"(ip_21), "f"(_shfl_xor_sync_542));
        float _sub_540;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_540) : "f"(g_20), "f"(_add_541));
        float g_21 = __fdiv_rn(_sub_540, v_21);
        float _dot_543;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_543) : "r"(ring_pk_28[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[3] = _dot_543;
        float _dot_544;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_544) : "r"(ring_pk_28[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_544;
        float _dot_545;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_545) : "r"(ring_pk_28[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_545;
        float _dot_546;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_546) : "r"(ring_pk_28[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_546;
        float _dot_547;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_547) : "r"(ring_pk_28[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_547;
        float _dot_548;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_548) : "r"(ring_pk_28[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_548;
        float _dot_549;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_549) : "r"(ring_pk_28[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_549;
        float _dot_550;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_550) : "r"(ring_pk_28[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_550;
        float _shfl_xor_sync_551 = __shfl_xor_sync(0xFFFFFFFF, hist[15], 1);
        float s_61;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_61) : "f"(hist[15]), "f"(_shfl_xor_sync_551));
        float _shfl_xor_sync_553 = __shfl_xor_sync(0xFFFFFFFF, s_61, 2);
        float _add_552;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_552) : "f"(s_61), "f"(_shfl_xor_sync_553));
        float v_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_22) : "f"(_add_552), "f"(hc_eps));
        float m_22;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_22) : "f"(g_21), "f"(hist[16]));
        float _shfl_xor_sync_554 = __shfl_xor_sync(0xFFFFFFFF, m_22, 1);
        float ip_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_22) : "f"(m_22), "f"(_shfl_xor_sync_554));
        float _shfl_xor_sync_557 = __shfl_xor_sync(0xFFFFFFFF, ip_22, 2);
        float _add_556;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_556) : "f"(ip_22), "f"(_shfl_xor_sync_557));
        float _sub_555;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_555) : "f"(g_21), "f"(_add_556));
        float g_22 = __fdiv_rn(_sub_555, v_22);
        int upper_5 = lane_1 & 16;
        float send_6 = ((upper_5 != 0) ? dots_p_4[0] : dots_p_4[2]);
        float keep_6 = ((upper_5 != 0) ? dots_p_4[2] : dots_p_4[0]);
        float _shfl_xor_sync_558 = __shfl_xor_sync(0xFFFFFFFF, send_6, 16);
        float add_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_12) : "f"(keep_6), "f"(_shfl_xor_sync_558));
        float send_1_4 = ((upper_5 != 0) ? dots_p_4[1] : dots_p_4[3]);
        float keep_1_4 = ((upper_5 != 0) ? dots_p_4[3] : dots_p_4[1]);
        float _shfl_xor_sync_559 = __shfl_xor_sync(0xFFFFFFFF, send_1_4, 16);
        float add_1_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_9) : "f"(keep_1_4), "f"(_shfl_xor_sync_559));
        int upper_1_4 = lane_1 & 8;
        float send_2_4 = ((upper_1_4 != 0) ? add_12 : add_1_9);
        float keep_2_4 = ((upper_1_4 != 0) ? add_1_9 : add_12);
        float _shfl_xor_sync_560 = __shfl_xor_sync(0xFFFFFFFF, send_2_4, 8);
        float add_2_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_4) : "f"(keep_2_4), "f"(_shfl_xor_sync_560));
        float _shfl_xor_sync_561 = __shfl_xor_sync(0xFFFFFFFF, hist[14], 4);
        float s_62;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_62) : "f"(hist[14]), "f"(_shfl_xor_sync_561));
        float _shfl_xor_sync_563 = __shfl_xor_sync(0xFFFFFFFF, s_62, 24);
        float _add_562;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_562) : "f"(s_62), "f"(_shfl_xor_sync_563));
        float v_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_23) : "f"(_add_562), "f"(hc_eps));
        float m_23;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_23) : "f"(g_22), "f"(hist[15]));
        float _shfl_xor_sync_564 = __shfl_xor_sync(0xFFFFFFFF, m_23, 4);
        float ip_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_23) : "f"(m_23), "f"(_shfl_xor_sync_564));
        float _shfl_xor_sync_567 = __shfl_xor_sync(0xFFFFFFFF, ip_23, 24);
        float _add_566;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_566) : "f"(ip_23), "f"(_shfl_xor_sync_567));
        float _sub_565;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_565) : "f"(g_22), "f"(_add_566));
        float g_23 = __fdiv_rn(_sub_565, v_23);
        float _shfl_xor_sync_568 = __shfl_xor_sync(0xFFFFFFFF, add_2_4, 4);
        float add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_13) : "f"(add_2_4), "f"(_shfl_xor_sync_568));
        float _shfl_xor_sync_569 = __shfl_xor_sync(0xFFFFFFFF, add_13, 2);
        float add_1_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_10) : "f"(add_13), "f"(_shfl_xor_sync_569));
        float _shfl_xor_sync_571 = __shfl_xor_sync(0xFFFFFFFF, add_1_10, 1);
        float _add_570;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_570) : "f"(add_1_10), "f"(_shfl_xor_sync_571));
        dpre_part[own_base + 160] = _add_570;
        float _shfl_xor_sync_572 = __shfl_xor_sync(0xFFFFFFFF, hist[13], 1);
        float s_63;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_63) : "f"(hist[13]), "f"(_shfl_xor_sync_572));
        float _shfl_xor_sync_574 = __shfl_xor_sync(0xFFFFFFFF, s_63, 2);
        float _add_573;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_573) : "f"(s_63), "f"(_shfl_xor_sync_574));
        float v_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_24) : "f"(_add_573), "f"(hc_eps));
        float m_24;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_24) : "f"(g_23), "f"(hist[14]));
        float _shfl_xor_sync_575 = __shfl_xor_sync(0xFFFFFFFF, m_24, 1);
        float ip_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_24) : "f"(m_24), "f"(_shfl_xor_sync_575));
        float _shfl_xor_sync_578 = __shfl_xor_sync(0xFFFFFFFF, ip_24, 2);
        float _add_577;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_577) : "f"(ip_24), "f"(_shfl_xor_sync_578));
        float _sub_576;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_576) : "f"(g_23), "f"(_add_577));
        float g_24 = __fdiv_rn(_sub_576, v_24);
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
        float _shfl_xor_sync_579 = __shfl_xor_sync(0xFFFFFFFF, hist[12], 4);
        float s_64;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_64) : "f"(hist[12]), "f"(_shfl_xor_sync_579));
        float _shfl_xor_sync_581 = __shfl_xor_sync(0xFFFFFFFF, s_64, 24);
        float _add_580;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_580) : "f"(s_64), "f"(_shfl_xor_sync_581));
        float v_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_25) : "f"(_add_580), "f"(hc_eps));
        float m_25;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_25) : "f"(g_24), "f"(hist[13]));
        float _shfl_xor_sync_582 = __shfl_xor_sync(0xFFFFFFFF, m_25, 4);
        float ip_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_25) : "f"(m_25), "f"(_shfl_xor_sync_582));
        float _shfl_xor_sync_585 = __shfl_xor_sync(0xFFFFFFFF, ip_25, 24);
        float _add_584;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_584) : "f"(ip_25), "f"(_shfl_xor_sync_585));
        float _sub_583;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_583) : "f"(g_24), "f"(_add_584));
        float g_25 = __fdiv_rn(_sub_583, v_25);
        float _dot_586;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_586) : "r"(ring_pk_30[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[0] = _dot_586;
        float _dot_587;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_587) : "r"(ring_pk_30[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_587;
        float _dot_588;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_588) : "r"(ring_pk_30[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_588;
        float _dot_589;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_589) : "r"(ring_pk_30[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_589;
        float _dot_590;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_590) : "r"(ring_pk_30[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_590;
        float _dot_591;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_591) : "r"(ring_pk_30[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_591;
        float _dot_592;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_592) : "r"(ring_pk_30[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_592;
        float _dot_593;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_593) : "r"(ring_pk_30[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_593;
        unsigned int ring_pk_31[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(0) + 3]))
            : "r"(sa__1_0 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(4) + 3]))
            : "r"(sa__1_0 + 2064 + lane32));
        float _shfl_xor_sync_594 = __shfl_xor_sync(0xFFFFFFFF, hist[11], 1);
        float s_65;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_65) : "f"(hist[11]), "f"(_shfl_xor_sync_594));
        float _shfl_xor_sync_596 = __shfl_xor_sync(0xFFFFFFFF, s_65, 2);
        float _add_595;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_595) : "f"(s_65), "f"(_shfl_xor_sync_596));
        float v_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_26) : "f"(_add_595), "f"(hc_eps));
        float m_26;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_26) : "f"(g_25), "f"(hist[12]));
        float _shfl_xor_sync_597 = __shfl_xor_sync(0xFFFFFFFF, m_26, 1);
        float ip_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_26) : "f"(m_26), "f"(_shfl_xor_sync_597));
        float _shfl_xor_sync_600 = __shfl_xor_sync(0xFFFFFFFF, ip_26, 2);
        float _add_599;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_599) : "f"(ip_26), "f"(_shfl_xor_sync_600));
        float _sub_598;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_598) : "f"(g_25), "f"(_add_599));
        float g_26 = __fdiv_rn(_sub_598, v_26);
        float _dot_601;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_601) : "r"(ring_pk_31[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[1] = _dot_601;
        float _dot_602;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_602) : "r"(ring_pk_31[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_602;
        float _dot_603;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_603) : "r"(ring_pk_31[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_603;
        float _dot_604;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_604) : "r"(ring_pk_31[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_604;
        float _dot_605;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_605) : "r"(ring_pk_31[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_605;
        float _dot_606;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_606) : "r"(ring_pk_31[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_606;
        float _dot_607;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_607) : "r"(ring_pk_31[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_607;
        float _dot_608;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_608) : "r"(ring_pk_31[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_608;
        unsigned int ring_pk_32[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(0) + 3]))
            : "r"(sa__1_0 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(4) + 3]))
            : "r"(sa__1_0 + 3088 + lane32));
        float _shfl_xor_sync_609 = __shfl_xor_sync(0xFFFFFFFF, hist[10], 4);
        float s_66;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_66) : "f"(hist[10]), "f"(_shfl_xor_sync_609));
        float _shfl_xor_sync_611 = __shfl_xor_sync(0xFFFFFFFF, s_66, 24);
        float _add_610;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_610) : "f"(s_66), "f"(_shfl_xor_sync_611));
        float v_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_27) : "f"(_add_610), "f"(hc_eps));
        float m_27;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_27) : "f"(g_26), "f"(hist[11]));
        float _shfl_xor_sync_612 = __shfl_xor_sync(0xFFFFFFFF, m_27, 4);
        float ip_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_27) : "f"(m_27), "f"(_shfl_xor_sync_612));
        float _shfl_xor_sync_615 = __shfl_xor_sync(0xFFFFFFFF, ip_27, 24);
        float _add_614;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_614) : "f"(ip_27), "f"(_shfl_xor_sync_615));
        float _sub_613;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_613) : "f"(g_26), "f"(_add_614));
        float g_27 = __fdiv_rn(_sub_613, v_27);
        float _dot_616;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_616) : "r"(ring_pk_32[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[2] = _dot_616;
        float _dot_617;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_617) : "r"(ring_pk_32[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_617;
        float _dot_618;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_618) : "r"(ring_pk_32[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_618;
        float _dot_619;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_619) : "r"(ring_pk_32[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_619;
        float _dot_620;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_620) : "r"(ring_pk_32[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_620;
        float _dot_621;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_621) : "r"(ring_pk_32[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_621;
        float _dot_622;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_622) : "r"(ring_pk_32[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_622;
        float _dot_623;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_623) : "r"(ring_pk_32[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_623;
        unsigned int ring_pk_33[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(0) + 3]))
            : "r"(sa__1_0 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(4) + 3]))
            : "r"(sa__1_0 + 4112 + lane32));
        __syncwarp();
        bool elected_0 = elect_sync();
        bool defer_el_0 = elected_0 && tail_pred;
        if (defer_el_0) {
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
        float _shfl_xor_sync_624 = __shfl_xor_sync(0xFFFFFFFF, hist[9], 1);
        float s_67;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_67) : "f"(hist[9]), "f"(_shfl_xor_sync_624));
        float _shfl_xor_sync_626 = __shfl_xor_sync(0xFFFFFFFF, s_67, 2);
        float _add_625;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_625) : "f"(s_67), "f"(_shfl_xor_sync_626));
        float v_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_28) : "f"(_add_625), "f"(hc_eps));
        float m_28;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_28) : "f"(g_27), "f"(hist[10]));
        float _shfl_xor_sync_627 = __shfl_xor_sync(0xFFFFFFFF, m_28, 1);
        float ip_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_28) : "f"(m_28), "f"(_shfl_xor_sync_627));
        float _shfl_xor_sync_630 = __shfl_xor_sync(0xFFFFFFFF, ip_28, 2);
        float _add_629;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_629) : "f"(ip_28), "f"(_shfl_xor_sync_630));
        float _sub_628;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_628) : "f"(g_27), "f"(_add_629));
        float g_28 = __fdiv_rn(_sub_628, v_28);
        float _dot_631;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_631) : "r"(ring_pk_33[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[3] = _dot_631;
        float _dot_632;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_632) : "r"(ring_pk_33[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_632;
        float _dot_633;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_633) : "r"(ring_pk_33[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_633;
        float _dot_634;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_634) : "r"(ring_pk_33[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_634;
        float _dot_635;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_635) : "r"(ring_pk_33[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_635;
        float _dot_636;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_636) : "r"(ring_pk_33[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_636;
        float _dot_637;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_637) : "r"(ring_pk_33[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_637;
        float _dot_638;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_638) : "r"(ring_pk_33[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_638;
        float _shfl_xor_sync_639 = __shfl_xor_sync(0xFFFFFFFF, hist[8], 4);
        float s_68;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_68) : "f"(hist[8]), "f"(_shfl_xor_sync_639));
        float _shfl_xor_sync_641 = __shfl_xor_sync(0xFFFFFFFF, s_68, 24);
        float _add_640;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_640) : "f"(s_68), "f"(_shfl_xor_sync_641));
        float v_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_29) : "f"(_add_640), "f"(hc_eps));
        float m_29;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_29) : "f"(g_28), "f"(hist[9]));
        float _shfl_xor_sync_642 = __shfl_xor_sync(0xFFFFFFFF, m_29, 4);
        float ip_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_29) : "f"(m_29), "f"(_shfl_xor_sync_642));
        float _shfl_xor_sync_645 = __shfl_xor_sync(0xFFFFFFFF, ip_29, 24);
        float _add_644;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_644) : "f"(ip_29), "f"(_shfl_xor_sync_645));
        float _sub_643;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_643) : "f"(g_28), "f"(_add_644));
        float g_29 = __fdiv_rn(_sub_643, v_29);
        int upper_6 = lane_1 & 16;
        float send_7 = ((upper_6 != 0) ? dots_p_5[0] : dots_p_5[2]);
        float keep_7 = ((upper_6 != 0) ? dots_p_5[2] : dots_p_5[0]);
        float _shfl_xor_sync_646 = __shfl_xor_sync(0xFFFFFFFF, send_7, 16);
        float add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_14) : "f"(keep_7), "f"(_shfl_xor_sync_646));
        float send_1_5 = ((upper_6 != 0) ? dots_p_5[1] : dots_p_5[3]);
        float keep_1_5 = ((upper_6 != 0) ? dots_p_5[3] : dots_p_5[1]);
        float _shfl_xor_sync_647 = __shfl_xor_sync(0xFFFFFFFF, send_1_5, 16);
        float add_1_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_11) : "f"(keep_1_5), "f"(_shfl_xor_sync_647));
        int upper_1_5 = lane_1 & 8;
        float send_2_5 = ((upper_1_5 != 0) ? add_14 : add_1_11);
        float keep_2_5 = ((upper_1_5 != 0) ? add_1_11 : add_14);
        float _shfl_xor_sync_648 = __shfl_xor_sync(0xFFFFFFFF, send_2_5, 8);
        float add_2_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_5) : "f"(keep_2_5), "f"(_shfl_xor_sync_648));
        float _shfl_xor_sync_649 = __shfl_xor_sync(0xFFFFFFFF, hist[7], 1);
        float s_69;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_69) : "f"(hist[7]), "f"(_shfl_xor_sync_649));
        float _shfl_xor_sync_651 = __shfl_xor_sync(0xFFFFFFFF, s_69, 2);
        float _add_650;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_650) : "f"(s_69), "f"(_shfl_xor_sync_651));
        float v_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_30) : "f"(_add_650), "f"(hc_eps));
        float m_30;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_30) : "f"(g_29), "f"(hist[8]));
        float _shfl_xor_sync_652 = __shfl_xor_sync(0xFFFFFFFF, m_30, 1);
        float ip_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_30) : "f"(m_30), "f"(_shfl_xor_sync_652));
        float _shfl_xor_sync_655 = __shfl_xor_sync(0xFFFFFFFF, ip_30, 2);
        float _add_654;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_654) : "f"(ip_30), "f"(_shfl_xor_sync_655));
        float _sub_653;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_653) : "f"(g_29), "f"(_add_654));
        float g_30 = __fdiv_rn(_sub_653, v_30);
        float _shfl_xor_sync_656 = __shfl_xor_sync(0xFFFFFFFF, add_2_5, 4);
        float add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_15) : "f"(add_2_5), "f"(_shfl_xor_sync_656));
        float _shfl_xor_sync_657 = __shfl_xor_sync(0xFFFFFFFF, add_15, 2);
        float add_1_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_12) : "f"(add_15), "f"(_shfl_xor_sync_657));
        float _shfl_xor_sync_659 = __shfl_xor_sync(0xFFFFFFFF, add_1_12, 1);
        float _add_658;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_658) : "f"(add_1_12), "f"(_shfl_xor_sync_659));
        dpre_part[own_base + 192] = _add_658;
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
        __syncwarp();
        bool elected_1 = elect_sync();
        bool defer_el_1 = elected_1 && tail_pred;
        if (defer_el_1) {
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
        float _add_195;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(hist[0]), "f"(hc_eps));
        float _shfl_xor_sync_737 = __shfl_xor_sync(0xFFFFFFFF, _add_195, 4);
        float add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_17) : "f"(_add_195), "f"(_shfl_xor_sync_737));
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
