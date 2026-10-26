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
kernel_cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr(__nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ dU, float* __restrict__ stats, float* __restrict__ gstats, float* __restrict__ scale, float* __restrict__ base, __nv_bfloat16* __restrict__ dzp, float* __restrict__ tokp, float* __restrict__ param_partial, int T, int t_pad, int rows_per_cta, float hc_eps)
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
    int st_ = warp_0 * 2;
    unsigned int sa_ = (unsigned int)(ring_addr + (unsigned int)(st_ * 5120));
    int st__3 = warp_0 * 2 + 1;
    unsigned int sa__4 = (unsigned int)(ring_addr + (unsigned int)(st__3 * 5120));
    if (t0 < t1) {
        int tp = t0;
        int tpc = ((tp < t1) ? tp : t_last);
        long long tp64 = (long long)tpc;
        int tp_0 = t0 + 1;
        int tpc_1 = ((tp_0 < t1) ? tp_0 : t_last);
        long long tp64_2 = (long long)tpc_1;
        int tp_3 = t0 + 2;
        int tpc_4 = ((tp_3 < t1) ? tp_3 : t_last);
        long long tp64_5 = (long long)tpc_4;
        int tp_6 = t0 + 3;
        int tpc_7 = ((tp_6 < t1) ? tp_6 : t_last);
        long long tp64_8 = (long long)tpc_7;
        int tp_9 = t0 + 4;
        int tpc_10 = ((tp_9 < t1) ? tp_9 : t_last);
        long long tp64_11 = (long long)tpc_10;
        int tp_12 = t0 + 5;
        int tpc_13 = ((tp_12 < t1) ? tp_12 : t_last);
        long long tp64_14 = (long long)tpc_13;
        int tp_15 = t0 + 6;
        int tpc_16 = ((tp_15 < t1) ? tp_15 : t_last);
        long long tp64_17 = (long long)tpc_16;
        int tp_18 = t0 + 7;
        int tpc_19 = ((tp_18 < t1) ? tp_18 : t_last);
        long long tp64_20 = (long long)tpc_19;
        int tp_21 = t0 + 8;
        int tpc_22 = ((tp_21 < t1) ? tp_21 : t_last);
        long long tp64_23 = (long long)tpc_22;
        int tp_24 = t0 + 9;
        int tpc_25 = ((tp_24 < t1) ? tp_24 : t_last);
        long long tp64_26 = (long long)tpc_25;
        int pro_base = warp_0 * 4 + lane_hi;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_), "l"(dU + (tp64 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 1024), "l"(X + (tp64 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 2048), "l"(X + ((tp64 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 3072), "l"(X + ((tp64 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 4096), "l"(X + ((tp64 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    int gi[1];
    gi[0] = 0;
    #pragma unroll 1
    for (int g0 = t0; g0 < t1; g0 += 8) {
        int cur_base = (gi[0] & 1) * 256 + warp_0 * 32;
        int nxt_base = (gi[0] + 1 & 1) * 256 + warp_0 * 4 + lane_hi;
        int tok = g0 + warp_0;
        int tn = g0 + 8;
        int tnc = ((tn < t1) ? tn : t_last);
        long long tn64 = (long long)tnc;
        int th = g0;
        int thc = ((th < t1) ? th : t_last);
        long long th64 = (long long)thc;
        int tn_0 = g0 + 9;
        int tnc_1 = ((tn_0 < t1) ? tn_0 : t_last);
        long long tn64_2 = (long long)tnc_1;
        int th_3 = g0 + 1;
        int thc_4 = ((th_3 < t1) ? th_3 : t_last);
        long long th64_5 = (long long)thc_4;
        int tn_6 = g0 + 10;
        int tnc_7 = ((tn_6 < t1) ? tn_6 : t_last);
        long long tn64_8 = (long long)tnc_7;
        int th_9 = g0 + 2;
        int thc_10 = ((th_9 < t1) ? th_9 : t_last);
        long long th64_11 = (long long)thc_10;
        int tn_12 = g0 + 11;
        int tnc_13 = ((tn_12 < t1) ? tn_12 : t_last);
        long long tn64_14 = (long long)tnc_13;
        int th_15 = g0 + 3;
        int thc_16 = ((th_15 < t1) ? th_15 : t_last);
        long long th64_17 = (long long)thc_16;
        int tn_18 = g0 + 12;
        int tnc_19 = ((tn_18 < t1) ? tn_18 : t_last);
        long long tn64_20 = (long long)tnc_19;
        int th_21 = g0 + 4;
        int thc_22 = ((th_21 < t1) ? th_21 : t_last);
        long long th64_23 = (long long)thc_22;
        int tn_24 = g0 + 13;
        int tnc_25 = ((tn_24 < t1) ? tn_24 : t_last);
        long long tn64_26 = (long long)tnc_25;
        int th_27 = g0 + 5;
        int thc_28 = ((th_27 < t1) ? th_27 : t_last);
        long long th64_29 = (long long)thc_28;
        int tn_30 = g0 + 14;
        int tnc_31 = ((tn_30 < t1) ? tn_30 : t_last);
        long long tn64_32 = (long long)tnc_31;
        int th_33 = g0 + 6;
        int thc_34 = ((th_33 < t1) ? th_33 : t_last);
        long long th64_35 = (long long)thc_34;
        int tn_36 = g0 + 15;
        int tnc_37 = ((tn_36 < t1) ? tn_36 : t_last);
        long long tn64_38 = (long long)tnc_37;
        int th_39 = g0 + 7;
        int thc_40 = ((th_39 < t1) ? th_39 : t_last);
        long long th64_41 = (long long)thc_40;
        int tn_42 = g0 + 16;
        int tnc_43 = ((tn_42 < t1) ? tn_42 : t_last);
        long long tn64_44 = (long long)tnc_43;
        int th_45 = g0 + 8;
        int thc_46 = ((th_45 < t1) ? th_45 : t_last);
        long long th64_47 = (long long)thc_46;
        int tn_48 = g0 + 17;
        int tnc_49 = ((tn_48 < t1) ? tn_48 : t_last);
        long long tn64_50 = (long long)tnc_49;
        int th_51 = g0 + 9;
        int thc_52 = ((th_51 < t1) ? th_51 : t_last);
        long long th64_53 = (long long)thc_52;
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
        float _add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(_exp_1), "f"(_shfl_xor_sync_4));
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 2);
        float _add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(_add_2), "f"(_shfl_xor_sync_6));
        float _fdiv_rn_2 = __fdiv_rn(_exp_1, _add_5);
        hist[0] = _fdiv_rn_2;
        float _add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(_fdiv_rn_2), "f"(hc_eps));
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, _add_3, 4);
        float _add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(_add_3), "f"(_shfl_xor_sync_7));
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, _add_4, 24);
        float _add_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(_add_4), "f"(_shfl_xor_sync_10));
        float _add_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(_add_9), "f"(hc_eps));
        float _fdiv_rn_3 = __fdiv_rn(_add_3, _add_8);
        hist[1] = _fdiv_rn_3;
        __syncwarp();
        bool _elect_sync_0 = elect_sync();
        bool defer_el = _elect_sync_0 && first_grp;
        if (defer_el) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__3) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4), "l"(dU + (td64 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 1024), "l"(X + (td64 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 2048), "l"(X + ((td64 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 3072), "l"(X + ((td64 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 4096), "l"(X + ((td64 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 1);
        float _add_5_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5_1) : "f"(_fdiv_rn_3), "f"(_shfl_xor_sync_11));
        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, _add_5_1, 2);
        float _add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(_add_5_1), "f"(_shfl_xor_sync_14));
        float _add_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(_add_13), "f"(hc_eps));
        float _fdiv_rn_4 = __fdiv_rn(_fdiv_rn_3, _add_12);
        hist[2] = _fdiv_rn_4;
        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 4);
        float _add_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(_fdiv_rn_4), "f"(_shfl_xor_sync_15));
        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, _add_6, 24);
        float _add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(_add_6), "f"(_shfl_xor_sync_18));
        float _add_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(_add_17), "f"(hc_eps));
        float _fdiv_rn_5 = __fdiv_rn(_fdiv_rn_4, _add_16);
        hist[3] = _fdiv_rn_5;
        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 1);
        float _add_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(_fdiv_rn_5), "f"(_shfl_xor_sync_19));
        float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, _add_7, 2);
        float _add_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(_add_7), "f"(_shfl_xor_sync_22));
        float _add_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(_add_21), "f"(hc_eps));
        float _fdiv_rn_6 = __fdiv_rn(_fdiv_rn_5, _add_20);
        hist[4] = _fdiv_rn_6;
        float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 4);
        float _add_8_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8_1) : "f"(_fdiv_rn_6), "f"(_shfl_xor_sync_23));
        float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, _add_8_1, 24);
        float _add_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(_add_8_1), "f"(_shfl_xor_sync_26));
        float _add_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(_add_25), "f"(hc_eps));
        float _fdiv_rn_7 = __fdiv_rn(_fdiv_rn_6, _add_24);
        hist[5] = _fdiv_rn_7;
        float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 1);
        float _add_9_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9_1) : "f"(_fdiv_rn_7), "f"(_shfl_xor_sync_27));
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, _add_9_1, 2);
        float _add_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(_add_9_1), "f"(_shfl_xor_sync_30));
        float _add_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(_add_29), "f"(hc_eps));
        float _fdiv_rn_8 = __fdiv_rn(_fdiv_rn_7, _add_28);
        hist[6] = _fdiv_rn_8;
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 4);
        float _add_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(_fdiv_rn_8), "f"(_shfl_xor_sync_31));
        float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, _add_10, 24);
        float _add_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(_add_10), "f"(_shfl_xor_sync_34));
        float _add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(_add_33), "f"(hc_eps));
        float _fdiv_rn_9 = __fdiv_rn(_fdiv_rn_8, _add_32);
        hist[7] = _fdiv_rn_9;
        float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 1);
        float _add_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(_fdiv_rn_9), "f"(_shfl_xor_sync_35));
        float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, _add_11, 2);
        float _add_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(_add_11), "f"(_shfl_xor_sync_38));
        float _add_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(_add_37), "f"(hc_eps));
        float _fdiv_rn_10 = __fdiv_rn(_fdiv_rn_9, _add_36);
        hist[8] = _fdiv_rn_10;
        float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 4);
        float _add_12_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12_1) : "f"(_fdiv_rn_10), "f"(_shfl_xor_sync_39));
        float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, _add_12_1, 24);
        float _add_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(_add_12_1), "f"(_shfl_xor_sync_42));
        float _add_40;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(_add_41), "f"(hc_eps));
        float _fdiv_rn_11 = __fdiv_rn(_fdiv_rn_10, _add_40);
        hist[9] = _fdiv_rn_11;
        float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 1);
        float _add_13_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13_1) : "f"(_fdiv_rn_11), "f"(_shfl_xor_sync_43));
        float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, _add_13_1, 2);
        float _add_45;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(_add_13_1), "f"(_shfl_xor_sync_46));
        float _add_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(_add_45), "f"(hc_eps));
        float _fdiv_rn_12 = __fdiv_rn(_fdiv_rn_11, _add_44);
        hist[10] = _fdiv_rn_12;
        float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 4);
        float _add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(_fdiv_rn_12), "f"(_shfl_xor_sync_47));
        float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, _add_14, 24);
        float _add_49;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(_add_14), "f"(_shfl_xor_sync_50));
        float _add_48;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(_add_49), "f"(hc_eps));
        float _fdiv_rn_13 = __fdiv_rn(_fdiv_rn_12, _add_48);
        hist[11] = _fdiv_rn_13;
        float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 1);
        float _add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(_fdiv_rn_13), "f"(_shfl_xor_sync_51));
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, _add_15, 2);
        float _add_53;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(_add_15), "f"(_shfl_xor_sync_54));
        float _add_52;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(_add_53), "f"(hc_eps));
        float _fdiv_rn_14 = __fdiv_rn(_fdiv_rn_13, _add_52);
        hist[12] = _fdiv_rn_14;
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 4);
        float _add_16_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16_1) : "f"(_fdiv_rn_14), "f"(_shfl_xor_sync_55));
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, _add_16_1, 24);
        float _add_57;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(_add_16_1), "f"(_shfl_xor_sync_58));
        float _add_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(_add_57), "f"(hc_eps));
        float _fdiv_rn_15 = __fdiv_rn(_fdiv_rn_14, _add_56);
        hist[13] = _fdiv_rn_15;
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 1);
        float _add_17_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17_1) : "f"(_fdiv_rn_15), "f"(_shfl_xor_sync_59));
        float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, _add_17_1, 2);
        float _add_61;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(_add_17_1), "f"(_shfl_xor_sync_62));
        float _add_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(_add_61), "f"(hc_eps));
        float _fdiv_rn_16 = __fdiv_rn(_fdiv_rn_15, _add_60);
        hist[14] = _fdiv_rn_16;
        float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 4);
        float _add_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(_fdiv_rn_16), "f"(_shfl_xor_sync_63));
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, _add_18, 24);
        float _add_65;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(_add_18), "f"(_shfl_xor_sync_66));
        float _add_64;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(_add_65), "f"(hc_eps));
        float _fdiv_rn_17 = __fdiv_rn(_fdiv_rn_16, _add_64);
        hist[15] = _fdiv_rn_17;
        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 1);
        float _add_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(_fdiv_rn_17), "f"(_shfl_xor_sync_67));
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, _add_19, 2);
        float _add_69;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(_add_19), "f"(_shfl_xor_sync_70));
        float _add_68;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(_add_69), "f"(hc_eps));
        float _fdiv_rn_18 = __fdiv_rn(_fdiv_rn_17, _add_68);
        hist[16] = _fdiv_rn_18;
        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 4);
        float _add_20_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20_1) : "f"(_fdiv_rn_18), "f"(_shfl_xor_sync_71));
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, _add_20_1, 24);
        float _add_73;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(_add_20_1), "f"(_shfl_xor_sync_74));
        float _add_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(_add_73), "f"(hc_eps));
        float _fdiv_rn_19 = __fdiv_rn(_fdiv_rn_18, _add_72);
        hist[17] = _fdiv_rn_19;
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 1);
        float _add_21_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21_1) : "f"(_fdiv_rn_19), "f"(_shfl_xor_sync_75));
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, _add_21_1, 2);
        float _add_77;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(_add_21_1), "f"(_shfl_xor_sync_78));
        float _add_76;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(_add_77), "f"(hc_eps));
        float _fdiv_rn_20 = __fdiv_rn(_fdiv_rn_19, _add_76);
        hist[18] = _fdiv_rn_20;
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 4);
        float _add_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(_fdiv_rn_20), "f"(_shfl_xor_sync_79));
        float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, _add_22, 24);
        float _add_81;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(_add_22), "f"(_shfl_xor_sync_82));
        float _add_80;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(_add_81), "f"(hc_eps));
        float _fdiv_rn_21 = __fdiv_rn(_fdiv_rn_20, _add_80);
        hist[19] = _fdiv_rn_21;
        float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 1);
        float _add_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(_fdiv_rn_21), "f"(_shfl_xor_sync_83));
        float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, _add_23, 2);
        float _add_85;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(_add_23), "f"(_shfl_xor_sync_86));
        float _add_84;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(_add_85), "f"(hc_eps));
        float _fdiv_rn_22 = __fdiv_rn(_fdiv_rn_21, _add_84);
        hist[20] = _fdiv_rn_22;
        float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 4);
        float _add_24_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24_1) : "f"(_fdiv_rn_22), "f"(_shfl_xor_sync_87));
        float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, _add_24_1, 24);
        float _add_89;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(_add_24_1), "f"(_shfl_xor_sync_90));
        float _add_88;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(_add_89), "f"(hc_eps));
        float _fdiv_rn_23 = __fdiv_rn(_fdiv_rn_22, _add_88);
        hist[21] = _fdiv_rn_23;
        float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 1);
        float _add_25_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25_1) : "f"(_fdiv_rn_23), "f"(_shfl_xor_sync_91));
        float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, _add_25_1, 2);
        float _add_93;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(_add_25_1), "f"(_shfl_xor_sync_94));
        float _add_92;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(_add_93), "f"(hc_eps));
        float _fdiv_rn_24 = __fdiv_rn(_fdiv_rn_23, _add_92);
        hist[22] = _fdiv_rn_24;
        float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 4);
        float _add_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(_fdiv_rn_24), "f"(_shfl_xor_sync_95));
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, _add_26, 24);
        float _add_97;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(_add_26), "f"(_shfl_xor_sync_98));
        float _add_96;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(_add_97), "f"(hc_eps));
        float _fdiv_rn_25 = __fdiv_rn(_fdiv_rn_24, _add_96);
        hist[23] = _fdiv_rn_25;
        mbarrier_wait(ring_bar_addr + (st_) * 8, 0);
        unsigned int ring_pk[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(0) + 3]))
            : "r"(sa_ + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(4) + 3]))
            : "r"(sa_ + 16 + lane32));
        unsigned int ring_pk_54[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[(0) + 3]))
            : "r"(sa_ + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[(4) + 3]))
            : "r"(sa_ + 1040 + lane32));
        float dots_p[4];
        float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 1);
        float _add_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(_fdiv_rn_25), "f"(_shfl_xor_sync_99));
        float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, _add_27, 2);
        float _add_101;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(_add_27), "f"(_shfl_xor_sync_102));
        float _add_100;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(_add_101), "f"(hc_eps));
        float _fdiv_rn_26 = __fdiv_rn(_fdiv_rn_25, _add_100);
        hist[24] = _fdiv_rn_26;
        float _dot_103;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_103) : "r"(ring_pk_54[0]), "r"(ring_pk[0]), "f"(-0.0f));
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
            : "=f"(_dot_104) : "r"(ring_pk_54[1]), "r"(ring_pk[1]), "f"(dots_p[0]));
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
            : "=f"(_dot_105) : "r"(ring_pk_54[2]), "r"(ring_pk[2]), "f"(dots_p[0]));
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
            : "=f"(_dot_106) : "r"(ring_pk_54[3]), "r"(ring_pk[3]), "f"(dots_p[0]));
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
            : "=f"(_dot_107) : "r"(ring_pk_54[4]), "r"(ring_pk[4]), "f"(dots_p[0]));
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
            : "=f"(_dot_108) : "r"(ring_pk_54[5]), "r"(ring_pk[5]), "f"(dots_p[0]));
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
            : "=f"(_dot_109) : "r"(ring_pk_54[6]), "r"(ring_pk[6]), "f"(dots_p[0]));
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
            : "=f"(_dot_110) : "r"(ring_pk_54[7]), "r"(ring_pk[7]), "f"(dots_p[0]));
        dots_p[0] = _dot_110;
        unsigned int ring_pk_55[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[(0) + 3]))
            : "r"(sa_ + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[(4) + 3]))
            : "r"(sa_ + 2064 + lane32));
        float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 4);
        float _add_28_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28_1) : "f"(_fdiv_rn_26), "f"(_shfl_xor_sync_111));
        float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, _add_28_1, 24);
        float _add_113;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(_add_28_1), "f"(_shfl_xor_sync_114));
        float _add_112;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(_add_113), "f"(hc_eps));
        float _fdiv_rn_27 = __fdiv_rn(_fdiv_rn_26, _add_112);
        hist[25] = _fdiv_rn_27;
        float _dot_115;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_115) : "r"(ring_pk_55[0]), "r"(ring_pk[0]), "f"(-0.0f));
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
            : "=f"(_dot_116) : "r"(ring_pk_55[1]), "r"(ring_pk[1]), "f"(dots_p[1]));
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
            : "=f"(_dot_117) : "r"(ring_pk_55[2]), "r"(ring_pk[2]), "f"(dots_p[1]));
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
            : "=f"(_dot_118) : "r"(ring_pk_55[3]), "r"(ring_pk[3]), "f"(dots_p[1]));
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
            : "=f"(_dot_119) : "r"(ring_pk_55[4]), "r"(ring_pk[4]), "f"(dots_p[1]));
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
            : "=f"(_dot_120) : "r"(ring_pk_55[5]), "r"(ring_pk[5]), "f"(dots_p[1]));
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
            : "=f"(_dot_121) : "r"(ring_pk_55[6]), "r"(ring_pk[6]), "f"(dots_p[1]));
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
            : "=f"(_dot_122) : "r"(ring_pk_55[7]), "r"(ring_pk[7]), "f"(dots_p[1]));
        dots_p[1] = _dot_122;
        unsigned int ring_pk_56[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[(0) + 3]))
            : "r"(sa_ + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[(4) + 3]))
            : "r"(sa_ + 3088 + lane32));
        float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 1);
        float _add_29_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29_1) : "f"(_fdiv_rn_27), "f"(_shfl_xor_sync_123));
        float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, _add_29_1, 2);
        float _add_125;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(_add_29_1), "f"(_shfl_xor_sync_126));
        float _add_124;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(_add_125), "f"(hc_eps));
        float _fdiv_rn_28 = __fdiv_rn(_fdiv_rn_27, _add_124);
        hist[26] = _fdiv_rn_28;
        float _dot_127;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_127) : "r"(ring_pk_56[0]), "r"(ring_pk[0]), "f"(-0.0f));
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
            : "=f"(_dot_128) : "r"(ring_pk_56[1]), "r"(ring_pk[1]), "f"(dots_p[2]));
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
            : "=f"(_dot_129) : "r"(ring_pk_56[2]), "r"(ring_pk[2]), "f"(dots_p[2]));
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
            : "=f"(_dot_130) : "r"(ring_pk_56[3]), "r"(ring_pk[3]), "f"(dots_p[2]));
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
            : "=f"(_dot_131) : "r"(ring_pk_56[4]), "r"(ring_pk[4]), "f"(dots_p[2]));
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
            : "=f"(_dot_132) : "r"(ring_pk_56[5]), "r"(ring_pk[5]), "f"(dots_p[2]));
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
            : "=f"(_dot_133) : "r"(ring_pk_56[6]), "r"(ring_pk[6]), "f"(dots_p[2]));
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
            : "=f"(_dot_134) : "r"(ring_pk_56[7]), "r"(ring_pk[7]), "f"(dots_p[2]));
        dots_p[2] = _dot_134;
        unsigned int ring_pk_57[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[(0) + 3]))
            : "r"(sa_ + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[(4) + 3]))
            : "r"(sa_ + 4112 + lane32));
        float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 4);
        float _add_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(_fdiv_rn_28), "f"(_shfl_xor_sync_135));
        float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, _add_30, 24);
        float _add_137;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(_add_30), "f"(_shfl_xor_sync_138));
        float _add_136;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(_add_137), "f"(hc_eps));
        float _fdiv_rn_29 = __fdiv_rn(_fdiv_rn_28, _add_136);
        hist[27] = _fdiv_rn_29;
        float _dot_139;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_139) : "r"(ring_pk_57[0]), "r"(ring_pk[0]), "f"(-0.0f));
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
            : "=f"(_dot_140) : "r"(ring_pk_57[1]), "r"(ring_pk[1]), "f"(dots_p[3]));
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
            : "=f"(_dot_141) : "r"(ring_pk_57[2]), "r"(ring_pk[2]), "f"(dots_p[3]));
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
            : "=f"(_dot_142) : "r"(ring_pk_57[3]), "r"(ring_pk[3]), "f"(dots_p[3]));
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
            : "=f"(_dot_143) : "r"(ring_pk_57[4]), "r"(ring_pk[4]), "f"(dots_p[3]));
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
            : "=f"(_dot_144) : "r"(ring_pk_57[5]), "r"(ring_pk[5]), "f"(dots_p[3]));
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
            : "=f"(_dot_145) : "r"(ring_pk_57[6]), "r"(ring_pk[6]), "f"(dots_p[3]));
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
            : "=f"(_dot_146) : "r"(ring_pk_57[7]), "r"(ring_pk[7]), "f"(dots_p[3]));
        dots_p[3] = _dot_146;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_), "l"(dU + (th64_11 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 1024), "l"(X + (th64_11 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 2048), "l"(X + ((th64_11 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 3072), "l"(X + ((th64_11 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 4096), "l"(X + ((th64_11 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 1);
        float _add_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(_fdiv_rn_29), "f"(_shfl_xor_sync_147));
        float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, _add_31, 2);
        float _add_149;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_149) : "f"(_add_31), "f"(_shfl_xor_sync_150));
        float _add_148;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(_add_149), "f"(hc_eps));
        float _fdiv_rn_30 = __fdiv_rn(_fdiv_rn_29, _add_148);
        hist[28] = _fdiv_rn_30;
        int upper = lane_1 & 16;
        float send = ((upper != 0) ? dots_p[0] : dots_p[2]);
        float keep = ((upper != 0) ? dots_p[2] : dots_p[0]);
        float _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
        float _add_32_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32_1) : "f"(keep), "f"(_shfl_xor_sync_151));
        float send_58 = ((upper != 0) ? dots_p[1] : dots_p[3]);
        float keep_59 = ((upper != 0) ? dots_p[3] : dots_p[1]);
        float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, send_58, 16);
        float _add_33_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33_1) : "f"(keep_59), "f"(_shfl_xor_sync_152));
        int upper_60 = lane_1 & 8;
        float send_61 = ((upper_60 != 0) ? _add_32_1 : _add_33_1);
        float keep_62 = ((upper_60 != 0) ? _add_33_1 : _add_32_1);
        float _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, send_61, 8);
        float _add_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(keep_62), "f"(_shfl_xor_sync_153));
        float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 4);
        float _add_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(_fdiv_rn_30), "f"(_shfl_xor_sync_154));
        float _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, _add_35, 24);
        float _add_156;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(_add_35), "f"(_shfl_xor_sync_157));
        float _add_155;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155) : "f"(_add_156), "f"(hc_eps));
        float _fdiv_rn_31 = __fdiv_rn(_fdiv_rn_30, _add_155);
        hist[29] = _fdiv_rn_31;
        float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, _add_34, 4);
        float _add_36_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36_1) : "f"(_add_34), "f"(_shfl_xor_sync_158));
        float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, _add_36_1, 2);
        float _add_37_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37_1) : "f"(_add_36_1), "f"(_shfl_xor_sync_159));
        float _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, _add_37_1, 1);
        float _add_160;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(_add_37_1), "f"(_shfl_xor_sync_161));
        dpre_part[own_base] = _add_160;
        float _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 1);
        float _add_38;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(_fdiv_rn_31), "f"(_shfl_xor_sync_162));
        float _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, _add_38, 2);
        float _add_164;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_164) : "f"(_add_38), "f"(_shfl_xor_sync_165));
        float _add_163;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(_add_164), "f"(hc_eps));
        float _fdiv_rn_32 = __fdiv_rn(_fdiv_rn_31, _add_163);
        hist[30] = _fdiv_rn_32;
        mbarrier_wait(ring_bar_addr + (st__3) * 8, 0);
        unsigned int ring_pk_63[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[(0) + 3]))
            : "r"(sa__4 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[(4) + 3]))
            : "r"(sa__4 + 16 + lane32));
        unsigned int ring_pk_64[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[(0) + 3]))
            : "r"(sa__4 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[(4) + 3]))
            : "r"(sa__4 + 1040 + lane32));
        float dots_p_65[4];
        float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 4);
        float _add_39;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(_fdiv_rn_32), "f"(_shfl_xor_sync_166));
        float _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, _add_39, 24);
        float _add_168;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(_add_39), "f"(_shfl_xor_sync_169));
        float _add_167;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(_add_168), "f"(hc_eps));
        float _fdiv_rn_33 = __fdiv_rn(_fdiv_rn_32, _add_167);
        hist[31] = _fdiv_rn_33;
        float _dot_170;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_170) : "r"(ring_pk_64[0]), "r"(ring_pk_63[0]), "f"(-0.0f));
        dots_p_65[0] = _dot_170;
        float _dot_171;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_171) : "r"(ring_pk_64[1]), "r"(ring_pk_63[1]), "f"(dots_p_65[0]));
        dots_p_65[0] = _dot_171;
        float _dot_172;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_172) : "r"(ring_pk_64[2]), "r"(ring_pk_63[2]), "f"(dots_p_65[0]));
        dots_p_65[0] = _dot_172;
        float _dot_173;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_173) : "r"(ring_pk_64[3]), "r"(ring_pk_63[3]), "f"(dots_p_65[0]));
        dots_p_65[0] = _dot_173;
        float _dot_174;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_174) : "r"(ring_pk_64[4]), "r"(ring_pk_63[4]), "f"(dots_p_65[0]));
        dots_p_65[0] = _dot_174;
        float _dot_175;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_175) : "r"(ring_pk_64[5]), "r"(ring_pk_63[5]), "f"(dots_p_65[0]));
        dots_p_65[0] = _dot_175;
        float _dot_176;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_176) : "r"(ring_pk_64[6]), "r"(ring_pk_63[6]), "f"(dots_p_65[0]));
        dots_p_65[0] = _dot_176;
        float _dot_177;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_177) : "r"(ring_pk_64[7]), "r"(ring_pk_63[7]), "f"(dots_p_65[0]));
        dots_p_65[0] = _dot_177;
        unsigned int ring_pk_66[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[(0) + 3]))
            : "r"(sa__4 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[(4) + 3]))
            : "r"(sa__4 + 2064 + lane32));
        float _shfl_xor_sync_178 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 1);
        float _add_40_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40_1) : "f"(_fdiv_rn_33), "f"(_shfl_xor_sync_178));
        float _shfl_xor_sync_181 = __shfl_xor_sync(0xFFFFFFFF, _add_40_1, 2);
        float _add_180;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_180) : "f"(_add_40_1), "f"(_shfl_xor_sync_181));
        float _add_179;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_179) : "f"(_add_180), "f"(hc_eps));
        float _fdiv_rn_34 = __fdiv_rn(_fdiv_rn_33, _add_179);
        hist[32] = _fdiv_rn_34;
        float _dot_182;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_182) : "r"(ring_pk_66[0]), "r"(ring_pk_63[0]), "f"(-0.0f));
        dots_p_65[1] = _dot_182;
        float _dot_183;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_183) : "r"(ring_pk_66[1]), "r"(ring_pk_63[1]), "f"(dots_p_65[1]));
        dots_p_65[1] = _dot_183;
        float _dot_184;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_184) : "r"(ring_pk_66[2]), "r"(ring_pk_63[2]), "f"(dots_p_65[1]));
        dots_p_65[1] = _dot_184;
        float _dot_185;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_185) : "r"(ring_pk_66[3]), "r"(ring_pk_63[3]), "f"(dots_p_65[1]));
        dots_p_65[1] = _dot_185;
        float _dot_186;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_186) : "r"(ring_pk_66[4]), "r"(ring_pk_63[4]), "f"(dots_p_65[1]));
        dots_p_65[1] = _dot_186;
        float _dot_187;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_187) : "r"(ring_pk_66[5]), "r"(ring_pk_63[5]), "f"(dots_p_65[1]));
        dots_p_65[1] = _dot_187;
        float _dot_188;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_188) : "r"(ring_pk_66[6]), "r"(ring_pk_63[6]), "f"(dots_p_65[1]));
        dots_p_65[1] = _dot_188;
        float _dot_189;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_189) : "r"(ring_pk_66[7]), "r"(ring_pk_63[7]), "f"(dots_p_65[1]));
        dots_p_65[1] = _dot_189;
        unsigned int ring_pk_67[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[(0) + 3]))
            : "r"(sa__4 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[(4) + 3]))
            : "r"(sa__4 + 3088 + lane32));
        float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 4);
        float _add_41_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41_1) : "f"(_fdiv_rn_34), "f"(_shfl_xor_sync_190));
        float _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, _add_41_1, 24);
        float _add_192;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_192) : "f"(_add_41_1), "f"(_shfl_xor_sync_193));
        float _add_191;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191) : "f"(_add_192), "f"(hc_eps));
        float _fdiv_rn_35 = __fdiv_rn(_fdiv_rn_34, _add_191);
        hist[33] = _fdiv_rn_35;
        float _dot_194;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_194) : "r"(ring_pk_67[0]), "r"(ring_pk_63[0]), "f"(-0.0f));
        dots_p_65[2] = _dot_194;
        float _dot_195;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_195) : "r"(ring_pk_67[1]), "r"(ring_pk_63[1]), "f"(dots_p_65[2]));
        dots_p_65[2] = _dot_195;
        float _dot_196;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_196) : "r"(ring_pk_67[2]), "r"(ring_pk_63[2]), "f"(dots_p_65[2]));
        dots_p_65[2] = _dot_196;
        float _dot_197;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_197) : "r"(ring_pk_67[3]), "r"(ring_pk_63[3]), "f"(dots_p_65[2]));
        dots_p_65[2] = _dot_197;
        float _dot_198;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_198) : "r"(ring_pk_67[4]), "r"(ring_pk_63[4]), "f"(dots_p_65[2]));
        dots_p_65[2] = _dot_198;
        float _dot_199;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_199) : "r"(ring_pk_67[5]), "r"(ring_pk_63[5]), "f"(dots_p_65[2]));
        dots_p_65[2] = _dot_199;
        float _dot_200;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_200) : "r"(ring_pk_67[6]), "r"(ring_pk_63[6]), "f"(dots_p_65[2]));
        dots_p_65[2] = _dot_200;
        float _dot_201;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_201) : "r"(ring_pk_67[7]), "r"(ring_pk_63[7]), "f"(dots_p_65[2]));
        dots_p_65[2] = _dot_201;
        unsigned int ring_pk_68[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[(0) + 3]))
            : "r"(sa__4 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[(4) + 3]))
            : "r"(sa__4 + 4112 + lane32));
        float _shfl_xor_sync_202 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 1);
        float _add_42;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(_fdiv_rn_35), "f"(_shfl_xor_sync_202));
        float _shfl_xor_sync_205 = __shfl_xor_sync(0xFFFFFFFF, _add_42, 2);
        float _add_204;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_204) : "f"(_add_42), "f"(_shfl_xor_sync_205));
        float _add_203;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_203) : "f"(_add_204), "f"(hc_eps));
        float _fdiv_rn_36 = __fdiv_rn(_fdiv_rn_35, _add_203);
        hist[34] = _fdiv_rn_36;
        float _dot_206;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_206) : "r"(ring_pk_68[0]), "r"(ring_pk_63[0]), "f"(-0.0f));
        dots_p_65[3] = _dot_206;
        float _dot_207;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_207) : "r"(ring_pk_68[1]), "r"(ring_pk_63[1]), "f"(dots_p_65[3]));
        dots_p_65[3] = _dot_207;
        float _dot_208;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_208) : "r"(ring_pk_68[2]), "r"(ring_pk_63[2]), "f"(dots_p_65[3]));
        dots_p_65[3] = _dot_208;
        float _dot_209;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_209) : "r"(ring_pk_68[3]), "r"(ring_pk_63[3]), "f"(dots_p_65[3]));
        dots_p_65[3] = _dot_209;
        float _dot_210;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_210) : "r"(ring_pk_68[4]), "r"(ring_pk_63[4]), "f"(dots_p_65[3]));
        dots_p_65[3] = _dot_210;
        float _dot_211;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_211) : "r"(ring_pk_68[5]), "r"(ring_pk_63[5]), "f"(dots_p_65[3]));
        dots_p_65[3] = _dot_211;
        float _dot_212;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_212) : "r"(ring_pk_68[6]), "r"(ring_pk_63[6]), "f"(dots_p_65[3]));
        dots_p_65[3] = _dot_212;
        float _dot_213;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_213) : "r"(ring_pk_68[7]), "r"(ring_pk_63[7]), "f"(dots_p_65[3]));
        dots_p_65[3] = _dot_213;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__3) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4), "l"(dU + (th64_17 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 1024), "l"(X + (th64_17 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 2048), "l"(X + ((th64_17 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 3072), "l"(X + ((th64_17 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 4096), "l"(X + ((th64_17 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_214 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 4);
        float _add_43;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(_fdiv_rn_36), "f"(_shfl_xor_sync_214));
        float _shfl_xor_sync_217 = __shfl_xor_sync(0xFFFFFFFF, _add_43, 24);
        float _add_216;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_216) : "f"(_add_43), "f"(_shfl_xor_sync_217));
        float _add_215;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_215) : "f"(_add_216), "f"(hc_eps));
        float _fdiv_rn_37 = __fdiv_rn(_fdiv_rn_36, _add_215);
        hist[35] = _fdiv_rn_37;
        int upper_69 = lane_1 & 16;
        float send_70 = ((upper_69 != 0) ? dots_p_65[0] : dots_p_65[2]);
        float keep_71 = ((upper_69 != 0) ? dots_p_65[2] : dots_p_65[0]);
        float _shfl_xor_sync_218 = __shfl_xor_sync(0xFFFFFFFF, send_70, 16);
        float _add_44_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44_1) : "f"(keep_71), "f"(_shfl_xor_sync_218));
        float send_72 = ((upper_69 != 0) ? dots_p_65[1] : dots_p_65[3]);
        float keep_73 = ((upper_69 != 0) ? dots_p_65[3] : dots_p_65[1]);
        float _shfl_xor_sync_219 = __shfl_xor_sync(0xFFFFFFFF, send_72, 16);
        float _add_45_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45_1) : "f"(keep_73), "f"(_shfl_xor_sync_219));
        int upper_74 = lane_1 & 8;
        float send_75 = ((upper_74 != 0) ? _add_44_1 : _add_45_1);
        float keep_76 = ((upper_74 != 0) ? _add_45_1 : _add_44_1);
        float _shfl_xor_sync_220 = __shfl_xor_sync(0xFFFFFFFF, send_75, 8);
        float _add_46;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(keep_76), "f"(_shfl_xor_sync_220));
        float _shfl_xor_sync_221 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 1);
        float _add_47;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(_fdiv_rn_37), "f"(_shfl_xor_sync_221));
        float _shfl_xor_sync_224 = __shfl_xor_sync(0xFFFFFFFF, _add_47, 2);
        float _add_223;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_223) : "f"(_add_47), "f"(_shfl_xor_sync_224));
        float _add_222;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_222) : "f"(_add_223), "f"(hc_eps));
        float _fdiv_rn_38 = __fdiv_rn(_fdiv_rn_37, _add_222);
        hist[36] = _fdiv_rn_38;
        float _shfl_xor_sync_225 = __shfl_xor_sync(0xFFFFFFFF, _add_46, 4);
        float _add_48_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48_1) : "f"(_add_46), "f"(_shfl_xor_sync_225));
        float _shfl_xor_sync_226 = __shfl_xor_sync(0xFFFFFFFF, _add_48_1, 2);
        float _add_49_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49_1) : "f"(_add_48_1), "f"(_shfl_xor_sync_226));
        float _shfl_xor_sync_228 = __shfl_xor_sync(0xFFFFFFFF, _add_49_1, 1);
        float _add_227;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_227) : "f"(_add_49_1), "f"(_shfl_xor_sync_228));
        dpre_part[own_base + 32] = _add_227;
        float _shfl_xor_sync_229 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 4);
        float _add_50;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(_fdiv_rn_38), "f"(_shfl_xor_sync_229));
        float _shfl_xor_sync_232 = __shfl_xor_sync(0xFFFFFFFF, _add_50, 24);
        float _add_231;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_231) : "f"(_add_50), "f"(_shfl_xor_sync_232));
        float _add_230;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_230) : "f"(_add_231), "f"(hc_eps));
        float _fdiv_rn_39 = __fdiv_rn(_fdiv_rn_38, _add_230);
        hist[37] = _fdiv_rn_39;
        mbarrier_wait(ring_bar_addr + (st_) * 8, 1);
        unsigned int ring_pk_77[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[(0) + 3]))
            : "r"(sa_ + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[(4) + 3]))
            : "r"(sa_ + 16 + lane32));
        unsigned int ring_pk_78[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[(0) + 3]))
            : "r"(sa_ + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[(4) + 3]))
            : "r"(sa_ + 1040 + lane32));
        float dots_p_79[4];
        float _shfl_xor_sync_233 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 1);
        float _add_51;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(_fdiv_rn_39), "f"(_shfl_xor_sync_233));
        float _shfl_xor_sync_236 = __shfl_xor_sync(0xFFFFFFFF, _add_51, 2);
        float _add_235;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_235) : "f"(_add_51), "f"(_shfl_xor_sync_236));
        float _add_234;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_234) : "f"(_add_235), "f"(hc_eps));
        float _fdiv_rn_40 = __fdiv_rn(_fdiv_rn_39, _add_234);
        hist[38] = _fdiv_rn_40;
        float _dot_237;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_237) : "r"(ring_pk_78[0]), "r"(ring_pk_77[0]), "f"(-0.0f));
        dots_p_79[0] = _dot_237;
        float _dot_238;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_238) : "r"(ring_pk_78[1]), "r"(ring_pk_77[1]), "f"(dots_p_79[0]));
        dots_p_79[0] = _dot_238;
        float _dot_239;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_239) : "r"(ring_pk_78[2]), "r"(ring_pk_77[2]), "f"(dots_p_79[0]));
        dots_p_79[0] = _dot_239;
        float _dot_240;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_240) : "r"(ring_pk_78[3]), "r"(ring_pk_77[3]), "f"(dots_p_79[0]));
        dots_p_79[0] = _dot_240;
        float _dot_241;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_241) : "r"(ring_pk_78[4]), "r"(ring_pk_77[4]), "f"(dots_p_79[0]));
        dots_p_79[0] = _dot_241;
        float _dot_242;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_242) : "r"(ring_pk_78[5]), "r"(ring_pk_77[5]), "f"(dots_p_79[0]));
        dots_p_79[0] = _dot_242;
        float _dot_243;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_243) : "r"(ring_pk_78[6]), "r"(ring_pk_77[6]), "f"(dots_p_79[0]));
        dots_p_79[0] = _dot_243;
        float _dot_244;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_244) : "r"(ring_pk_78[7]), "r"(ring_pk_77[7]), "f"(dots_p_79[0]));
        dots_p_79[0] = _dot_244;
        unsigned int ring_pk_80[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[(0) + 3]))
            : "r"(sa_ + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[(4) + 3]))
            : "r"(sa_ + 2064 + lane32));
        float _shfl_xor_sync_245 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_40, 4);
        float _add_52_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52_1) : "f"(_fdiv_rn_40), "f"(_shfl_xor_sync_245));
        float _shfl_xor_sync_248 = __shfl_xor_sync(0xFFFFFFFF, _add_52_1, 24);
        float _add_247;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_247) : "f"(_add_52_1), "f"(_shfl_xor_sync_248));
        float _add_246;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(_add_247), "f"(hc_eps));
        float _fdiv_rn_41 = __fdiv_rn(_fdiv_rn_40, _add_246);
        hist[39] = _fdiv_rn_41;
        float _dot_249;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_249) : "r"(ring_pk_80[0]), "r"(ring_pk_77[0]), "f"(-0.0f));
        dots_p_79[1] = _dot_249;
        float _dot_250;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_250) : "r"(ring_pk_80[1]), "r"(ring_pk_77[1]), "f"(dots_p_79[1]));
        dots_p_79[1] = _dot_250;
        float _dot_251;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_251) : "r"(ring_pk_80[2]), "r"(ring_pk_77[2]), "f"(dots_p_79[1]));
        dots_p_79[1] = _dot_251;
        float _dot_252;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_252) : "r"(ring_pk_80[3]), "r"(ring_pk_77[3]), "f"(dots_p_79[1]));
        dots_p_79[1] = _dot_252;
        float _dot_253;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_253) : "r"(ring_pk_80[4]), "r"(ring_pk_77[4]), "f"(dots_p_79[1]));
        dots_p_79[1] = _dot_253;
        float _dot_254;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_254) : "r"(ring_pk_80[5]), "r"(ring_pk_77[5]), "f"(dots_p_79[1]));
        dots_p_79[1] = _dot_254;
        float _dot_255;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_255) : "r"(ring_pk_80[6]), "r"(ring_pk_77[6]), "f"(dots_p_79[1]));
        dots_p_79[1] = _dot_255;
        float _dot_256;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_256) : "r"(ring_pk_80[7]), "r"(ring_pk_77[7]), "f"(dots_p_79[1]));
        dots_p_79[1] = _dot_256;
        unsigned int ring_pk_81[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[(0) + 3]))
            : "r"(sa_ + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[(4) + 3]))
            : "r"(sa_ + 3088 + lane32));
        float dsig;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dsig) : "f"(sig), "f"(sneg));
        float G = ((lane_1 >= 8 && lane_1 < 24) ? gval : 0.0f);
        float _shfl_xor_sync_257 = __shfl_xor_sync(0xFFFFFFFF, hist[38], 4);
        float _add_53_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53_1) : "f"(hist[38]), "f"(_shfl_xor_sync_257));
        float _shfl_xor_sync_259 = __shfl_xor_sync(0xFFFFFFFF, _add_53_1, 24);
        float _add_258;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_258) : "f"(_add_53_1), "f"(_shfl_xor_sync_259));
        float _add_54;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(_add_258), "f"(hc_eps));
        float _mul_1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(G), "f"(hist[39]));
        float _shfl_xor_sync_260 = __shfl_xor_sync(0xFFFFFFFF, _mul_1, 4);
        float _add_55;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(_mul_1), "f"(_shfl_xor_sync_260));
        float _shfl_xor_sync_263 = __shfl_xor_sync(0xFFFFFFFF, _add_55, 24);
        float _add_262;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_262) : "f"(_add_55), "f"(_shfl_xor_sync_263));
        float _sub_261;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_261) : "f"(G), "f"(_add_262));
        float _fdiv_rn_42 = __fdiv_rn(_sub_261, _add_54);
        float _dot_264;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_264) : "r"(ring_pk_81[0]), "r"(ring_pk_77[0]), "f"(-0.0f));
        dots_p_79[2] = _dot_264;
        float _dot_265;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_265) : "r"(ring_pk_81[1]), "r"(ring_pk_77[1]), "f"(dots_p_79[2]));
        dots_p_79[2] = _dot_265;
        float _dot_266;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_266) : "r"(ring_pk_81[2]), "r"(ring_pk_77[2]), "f"(dots_p_79[2]));
        dots_p_79[2] = _dot_266;
        float _dot_267;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_267) : "r"(ring_pk_81[3]), "r"(ring_pk_77[3]), "f"(dots_p_79[2]));
        dots_p_79[2] = _dot_267;
        float _dot_268;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_268) : "r"(ring_pk_81[4]), "r"(ring_pk_77[4]), "f"(dots_p_79[2]));
        dots_p_79[2] = _dot_268;
        float _dot_269;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_269) : "r"(ring_pk_81[5]), "r"(ring_pk_77[5]), "f"(dots_p_79[2]));
        dots_p_79[2] = _dot_269;
        float _dot_270;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_270) : "r"(ring_pk_81[6]), "r"(ring_pk_77[6]), "f"(dots_p_79[2]));
        dots_p_79[2] = _dot_270;
        float _dot_271;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_271) : "r"(ring_pk_81[7]), "r"(ring_pk_77[7]), "f"(dots_p_79[2]));
        dots_p_79[2] = _dot_271;
        unsigned int ring_pk_82[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[(0) + 3]))
            : "r"(sa_ + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[(4) + 3]))
            : "r"(sa_ + 4112 + lane32));
        float _shfl_xor_sync_272 = __shfl_xor_sync(0xFFFFFFFF, hist[37], 1);
        float _add_56_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56_1) : "f"(hist[37]), "f"(_shfl_xor_sync_272));
        float _shfl_xor_sync_274 = __shfl_xor_sync(0xFFFFFFFF, _add_56_1, 2);
        float _add_273;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_273) : "f"(_add_56_1), "f"(_shfl_xor_sync_274));
        float _add_57_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57_1) : "f"(_add_273), "f"(hc_eps));
        float _mul_2;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_fdiv_rn_42), "f"(hist[38]));
        float _shfl_xor_sync_275 = __shfl_xor_sync(0xFFFFFFFF, _mul_2, 1);
        float _add_58;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(_mul_2), "f"(_shfl_xor_sync_275));
        float _shfl_xor_sync_278 = __shfl_xor_sync(0xFFFFFFFF, _add_58, 2);
        float _add_277;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_277) : "f"(_add_58), "f"(_shfl_xor_sync_278));
        float _sub_276;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_276) : "f"(_fdiv_rn_42), "f"(_add_277));
        float _fdiv_rn_43 = __fdiv_rn(_sub_276, _add_57_1);
        float _dot_279;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_279) : "r"(ring_pk_82[0]), "r"(ring_pk_77[0]), "f"(-0.0f));
        dots_p_79[3] = _dot_279;
        float _dot_280;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_280) : "r"(ring_pk_82[1]), "r"(ring_pk_77[1]), "f"(dots_p_79[3]));
        dots_p_79[3] = _dot_280;
        float _dot_281;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_281) : "r"(ring_pk_82[2]), "r"(ring_pk_77[2]), "f"(dots_p_79[3]));
        dots_p_79[3] = _dot_281;
        float _dot_282;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_282) : "r"(ring_pk_82[3]), "r"(ring_pk_77[3]), "f"(dots_p_79[3]));
        dots_p_79[3] = _dot_282;
        float _dot_283;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_283) : "r"(ring_pk_82[4]), "r"(ring_pk_77[4]), "f"(dots_p_79[3]));
        dots_p_79[3] = _dot_283;
        float _dot_284;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_284) : "r"(ring_pk_82[5]), "r"(ring_pk_77[5]), "f"(dots_p_79[3]));
        dots_p_79[3] = _dot_284;
        float _dot_285;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_285) : "r"(ring_pk_82[6]), "r"(ring_pk_77[6]), "f"(dots_p_79[3]));
        dots_p_79[3] = _dot_285;
        float _dot_286;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_286) : "r"(ring_pk_82[7]), "r"(ring_pk_77[7]), "f"(dots_p_79[3]));
        dots_p_79[3] = _dot_286;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_), "l"(dU + (th64_23 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 1024), "l"(X + (th64_23 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 2048), "l"(X + ((th64_23 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 3072), "l"(X + ((th64_23 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 4096), "l"(X + ((th64_23 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_287 = __shfl_xor_sync(0xFFFFFFFF, hist[36], 4);
        float _add_59;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(hist[36]), "f"(_shfl_xor_sync_287));
        float _shfl_xor_sync_289 = __shfl_xor_sync(0xFFFFFFFF, _add_59, 24);
        float _add_288;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_288) : "f"(_add_59), "f"(_shfl_xor_sync_289));
        float _add_60_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60_1) : "f"(_add_288), "f"(hc_eps));
        float _mul_3;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_fdiv_rn_43), "f"(hist[37]));
        float _shfl_xor_sync_290 = __shfl_xor_sync(0xFFFFFFFF, _mul_3, 4);
        float _add_61_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61_1) : "f"(_mul_3), "f"(_shfl_xor_sync_290));
        float _shfl_xor_sync_293 = __shfl_xor_sync(0xFFFFFFFF, _add_61_1, 24);
        float _add_292;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_292) : "f"(_add_61_1), "f"(_shfl_xor_sync_293));
        float _sub_291;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_291) : "f"(_fdiv_rn_43), "f"(_add_292));
        float _fdiv_rn_44 = __fdiv_rn(_sub_291, _add_60_1);
        int upper_83 = lane_1 & 16;
        float send_84 = ((upper_83 != 0) ? dots_p_79[0] : dots_p_79[2]);
        float keep_85 = ((upper_83 != 0) ? dots_p_79[2] : dots_p_79[0]);
        float _shfl_xor_sync_294 = __shfl_xor_sync(0xFFFFFFFF, send_84, 16);
        float _add_62;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(keep_85), "f"(_shfl_xor_sync_294));
        float send_86 = ((upper_83 != 0) ? dots_p_79[1] : dots_p_79[3]);
        float keep_87 = ((upper_83 != 0) ? dots_p_79[3] : dots_p_79[1]);
        float _shfl_xor_sync_295 = __shfl_xor_sync(0xFFFFFFFF, send_86, 16);
        float _add_63;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(keep_87), "f"(_shfl_xor_sync_295));
        int upper_88 = lane_1 & 8;
        float send_89 = ((upper_88 != 0) ? _add_62 : _add_63);
        float keep_90 = ((upper_88 != 0) ? _add_63 : _add_62);
        float _shfl_xor_sync_296 = __shfl_xor_sync(0xFFFFFFFF, send_89, 8);
        float _add_64_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64_1) : "f"(keep_90), "f"(_shfl_xor_sync_296));
        float _shfl_xor_sync_297 = __shfl_xor_sync(0xFFFFFFFF, hist[35], 1);
        float _add_65_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65_1) : "f"(hist[35]), "f"(_shfl_xor_sync_297));
        float _shfl_xor_sync_299 = __shfl_xor_sync(0xFFFFFFFF, _add_65_1, 2);
        float _add_298;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_298) : "f"(_add_65_1), "f"(_shfl_xor_sync_299));
        float _add_66;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(_add_298), "f"(hc_eps));
        float _mul_4;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_fdiv_rn_44), "f"(hist[36]));
        float _shfl_xor_sync_300 = __shfl_xor_sync(0xFFFFFFFF, _mul_4, 1);
        float _add_67;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(_mul_4), "f"(_shfl_xor_sync_300));
        float _shfl_xor_sync_303 = __shfl_xor_sync(0xFFFFFFFF, _add_67, 2);
        float _add_302;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_302) : "f"(_add_67), "f"(_shfl_xor_sync_303));
        float _sub_301;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_301) : "f"(_fdiv_rn_44), "f"(_add_302));
        float _fdiv_rn_45 = __fdiv_rn(_sub_301, _add_66);
        float _shfl_xor_sync_304 = __shfl_xor_sync(0xFFFFFFFF, _add_64_1, 4);
        float _add_68_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68_1) : "f"(_add_64_1), "f"(_shfl_xor_sync_304));
        float _shfl_xor_sync_305 = __shfl_xor_sync(0xFFFFFFFF, _add_68_1, 2);
        float _add_69_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69_1) : "f"(_add_68_1), "f"(_shfl_xor_sync_305));
        float _shfl_xor_sync_307 = __shfl_xor_sync(0xFFFFFFFF, _add_69_1, 1);
        float _add_306;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_306) : "f"(_add_69_1), "f"(_shfl_xor_sync_307));
        dpre_part[own_base + 64] = _add_306;
        float _shfl_xor_sync_308 = __shfl_xor_sync(0xFFFFFFFF, hist[34], 4);
        float _add_70;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(hist[34]), "f"(_shfl_xor_sync_308));
        float _shfl_xor_sync_310 = __shfl_xor_sync(0xFFFFFFFF, _add_70, 24);
        float _add_309;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_309) : "f"(_add_70), "f"(_shfl_xor_sync_310));
        float _add_71;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(_add_309), "f"(hc_eps));
        float _mul_5;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_fdiv_rn_45), "f"(hist[35]));
        float _shfl_xor_sync_311 = __shfl_xor_sync(0xFFFFFFFF, _mul_5, 4);
        float _add_72_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72_1) : "f"(_mul_5), "f"(_shfl_xor_sync_311));
        float _shfl_xor_sync_314 = __shfl_xor_sync(0xFFFFFFFF, _add_72_1, 24);
        float _add_313;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_313) : "f"(_add_72_1), "f"(_shfl_xor_sync_314));
        float _sub_312;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_312) : "f"(_fdiv_rn_45), "f"(_add_313));
        float _fdiv_rn_46 = __fdiv_rn(_sub_312, _add_71);
        mbarrier_wait(ring_bar_addr + (st__3) * 8, 1);
        unsigned int ring_pk_91[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[(0) + 3]))
            : "r"(sa__4 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[(4) + 3]))
            : "r"(sa__4 + 16 + lane32));
        unsigned int ring_pk_92[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[(0) + 3]))
            : "r"(sa__4 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[(4) + 3]))
            : "r"(sa__4 + 1040 + lane32));
        float dots_p_93[4];
        float _shfl_xor_sync_315 = __shfl_xor_sync(0xFFFFFFFF, hist[33], 1);
        float _add_73_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73_1) : "f"(hist[33]), "f"(_shfl_xor_sync_315));
        float _shfl_xor_sync_317 = __shfl_xor_sync(0xFFFFFFFF, _add_73_1, 2);
        float _add_316;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_316) : "f"(_add_73_1), "f"(_shfl_xor_sync_317));
        float _add_74;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(_add_316), "f"(hc_eps));
        float _mul_6;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_fdiv_rn_46), "f"(hist[34]));
        float _shfl_xor_sync_318 = __shfl_xor_sync(0xFFFFFFFF, _mul_6, 1);
        float _add_75;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(_mul_6), "f"(_shfl_xor_sync_318));
        float _shfl_xor_sync_321 = __shfl_xor_sync(0xFFFFFFFF, _add_75, 2);
        float _add_320;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_320) : "f"(_add_75), "f"(_shfl_xor_sync_321));
        float _sub_319;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_319) : "f"(_fdiv_rn_46), "f"(_add_320));
        float _fdiv_rn_47 = __fdiv_rn(_sub_319, _add_74);
        float _dot_322;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_322) : "r"(ring_pk_92[0]), "r"(ring_pk_91[0]), "f"(-0.0f));
        dots_p_93[0] = _dot_322;
        float _dot_323;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_323) : "r"(ring_pk_92[1]), "r"(ring_pk_91[1]), "f"(dots_p_93[0]));
        dots_p_93[0] = _dot_323;
        float _dot_324;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_324) : "r"(ring_pk_92[2]), "r"(ring_pk_91[2]), "f"(dots_p_93[0]));
        dots_p_93[0] = _dot_324;
        float _dot_325;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_325) : "r"(ring_pk_92[3]), "r"(ring_pk_91[3]), "f"(dots_p_93[0]));
        dots_p_93[0] = _dot_325;
        float _dot_326;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_326) : "r"(ring_pk_92[4]), "r"(ring_pk_91[4]), "f"(dots_p_93[0]));
        dots_p_93[0] = _dot_326;
        float _dot_327;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_327) : "r"(ring_pk_92[5]), "r"(ring_pk_91[5]), "f"(dots_p_93[0]));
        dots_p_93[0] = _dot_327;
        float _dot_328;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_328) : "r"(ring_pk_92[6]), "r"(ring_pk_91[6]), "f"(dots_p_93[0]));
        dots_p_93[0] = _dot_328;
        float _dot_329;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_329) : "r"(ring_pk_92[7]), "r"(ring_pk_91[7]), "f"(dots_p_93[0]));
        dots_p_93[0] = _dot_329;
        unsigned int ring_pk_94[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[(0) + 3]))
            : "r"(sa__4 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[(4) + 3]))
            : "r"(sa__4 + 2064 + lane32));
        float _shfl_xor_sync_330 = __shfl_xor_sync(0xFFFFFFFF, hist[32], 4);
        float _add_76_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76_1) : "f"(hist[32]), "f"(_shfl_xor_sync_330));
        float _shfl_xor_sync_332 = __shfl_xor_sync(0xFFFFFFFF, _add_76_1, 24);
        float _add_331;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_331) : "f"(_add_76_1), "f"(_shfl_xor_sync_332));
        float _add_77_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77_1) : "f"(_add_331), "f"(hc_eps));
        float _mul_7;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_fdiv_rn_47), "f"(hist[33]));
        float _shfl_xor_sync_333 = __shfl_xor_sync(0xFFFFFFFF, _mul_7, 4);
        float _add_78;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(_mul_7), "f"(_shfl_xor_sync_333));
        float _shfl_xor_sync_336 = __shfl_xor_sync(0xFFFFFFFF, _add_78, 24);
        float _add_335;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_335) : "f"(_add_78), "f"(_shfl_xor_sync_336));
        float _sub_334;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_334) : "f"(_fdiv_rn_47), "f"(_add_335));
        float _fdiv_rn_48 = __fdiv_rn(_sub_334, _add_77_1);
        float _dot_337;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_337) : "r"(ring_pk_94[0]), "r"(ring_pk_91[0]), "f"(-0.0f));
        dots_p_93[1] = _dot_337;
        float _dot_338;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_338) : "r"(ring_pk_94[1]), "r"(ring_pk_91[1]), "f"(dots_p_93[1]));
        dots_p_93[1] = _dot_338;
        float _dot_339;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_339) : "r"(ring_pk_94[2]), "r"(ring_pk_91[2]), "f"(dots_p_93[1]));
        dots_p_93[1] = _dot_339;
        float _dot_340;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_340) : "r"(ring_pk_94[3]), "r"(ring_pk_91[3]), "f"(dots_p_93[1]));
        dots_p_93[1] = _dot_340;
        float _dot_341;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_341) : "r"(ring_pk_94[4]), "r"(ring_pk_91[4]), "f"(dots_p_93[1]));
        dots_p_93[1] = _dot_341;
        float _dot_342;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_342) : "r"(ring_pk_94[5]), "r"(ring_pk_91[5]), "f"(dots_p_93[1]));
        dots_p_93[1] = _dot_342;
        float _dot_343;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_343) : "r"(ring_pk_94[6]), "r"(ring_pk_91[6]), "f"(dots_p_93[1]));
        dots_p_93[1] = _dot_343;
        float _dot_344;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_344) : "r"(ring_pk_94[7]), "r"(ring_pk_91[7]), "f"(dots_p_93[1]));
        dots_p_93[1] = _dot_344;
        unsigned int ring_pk_95[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[(0) + 3]))
            : "r"(sa__4 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[(4) + 3]))
            : "r"(sa__4 + 3088 + lane32));
        float _shfl_xor_sync_345 = __shfl_xor_sync(0xFFFFFFFF, hist[31], 1);
        float _add_79;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(hist[31]), "f"(_shfl_xor_sync_345));
        float _shfl_xor_sync_347 = __shfl_xor_sync(0xFFFFFFFF, _add_79, 2);
        float _add_346;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_346) : "f"(_add_79), "f"(_shfl_xor_sync_347));
        float _add_80_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80_1) : "f"(_add_346), "f"(hc_eps));
        float _mul_8;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_fdiv_rn_48), "f"(hist[32]));
        float _shfl_xor_sync_348 = __shfl_xor_sync(0xFFFFFFFF, _mul_8, 1);
        float _add_81_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81_1) : "f"(_mul_8), "f"(_shfl_xor_sync_348));
        float _shfl_xor_sync_351 = __shfl_xor_sync(0xFFFFFFFF, _add_81_1, 2);
        float _add_350;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_350) : "f"(_add_81_1), "f"(_shfl_xor_sync_351));
        float _sub_349;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_349) : "f"(_fdiv_rn_48), "f"(_add_350));
        float _fdiv_rn_49 = __fdiv_rn(_sub_349, _add_80_1);
        float _dot_352;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_352) : "r"(ring_pk_95[0]), "r"(ring_pk_91[0]), "f"(-0.0f));
        dots_p_93[2] = _dot_352;
        float _dot_353;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_353) : "r"(ring_pk_95[1]), "r"(ring_pk_91[1]), "f"(dots_p_93[2]));
        dots_p_93[2] = _dot_353;
        float _dot_354;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_354) : "r"(ring_pk_95[2]), "r"(ring_pk_91[2]), "f"(dots_p_93[2]));
        dots_p_93[2] = _dot_354;
        float _dot_355;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_355) : "r"(ring_pk_95[3]), "r"(ring_pk_91[3]), "f"(dots_p_93[2]));
        dots_p_93[2] = _dot_355;
        float _dot_356;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_356) : "r"(ring_pk_95[4]), "r"(ring_pk_91[4]), "f"(dots_p_93[2]));
        dots_p_93[2] = _dot_356;
        float _dot_357;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_357) : "r"(ring_pk_95[5]), "r"(ring_pk_91[5]), "f"(dots_p_93[2]));
        dots_p_93[2] = _dot_357;
        float _dot_358;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_358) : "r"(ring_pk_95[6]), "r"(ring_pk_91[6]), "f"(dots_p_93[2]));
        dots_p_93[2] = _dot_358;
        float _dot_359;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_359) : "r"(ring_pk_95[7]), "r"(ring_pk_91[7]), "f"(dots_p_93[2]));
        dots_p_93[2] = _dot_359;
        unsigned int ring_pk_96[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[(0) + 3]))
            : "r"(sa__4 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[(4) + 3]))
            : "r"(sa__4 + 4112 + lane32));
        float _shfl_xor_sync_360 = __shfl_xor_sync(0xFFFFFFFF, hist[30], 4);
        float _add_82;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(hist[30]), "f"(_shfl_xor_sync_360));
        float _shfl_xor_sync_362 = __shfl_xor_sync(0xFFFFFFFF, _add_82, 24);
        float _add_361;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_361) : "f"(_add_82), "f"(_shfl_xor_sync_362));
        float _add_83;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(_add_361), "f"(hc_eps));
        float _mul_9;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_fdiv_rn_49), "f"(hist[31]));
        float _shfl_xor_sync_363 = __shfl_xor_sync(0xFFFFFFFF, _mul_9, 4);
        float _add_84_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84_1) : "f"(_mul_9), "f"(_shfl_xor_sync_363));
        float _shfl_xor_sync_366 = __shfl_xor_sync(0xFFFFFFFF, _add_84_1, 24);
        float _add_365;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_365) : "f"(_add_84_1), "f"(_shfl_xor_sync_366));
        float _sub_364;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_364) : "f"(_fdiv_rn_49), "f"(_add_365));
        float _fdiv_rn_50 = __fdiv_rn(_sub_364, _add_83);
        float _dot_367;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_367) : "r"(ring_pk_96[0]), "r"(ring_pk_91[0]), "f"(-0.0f));
        dots_p_93[3] = _dot_367;
        float _dot_368;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_368) : "r"(ring_pk_96[1]), "r"(ring_pk_91[1]), "f"(dots_p_93[3]));
        dots_p_93[3] = _dot_368;
        float _dot_369;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_369) : "r"(ring_pk_96[2]), "r"(ring_pk_91[2]), "f"(dots_p_93[3]));
        dots_p_93[3] = _dot_369;
        float _dot_370;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_370) : "r"(ring_pk_96[3]), "r"(ring_pk_91[3]), "f"(dots_p_93[3]));
        dots_p_93[3] = _dot_370;
        float _dot_371;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_371) : "r"(ring_pk_96[4]), "r"(ring_pk_91[4]), "f"(dots_p_93[3]));
        dots_p_93[3] = _dot_371;
        float _dot_372;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_372) : "r"(ring_pk_96[5]), "r"(ring_pk_91[5]), "f"(dots_p_93[3]));
        dots_p_93[3] = _dot_372;
        float _dot_373;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_373) : "r"(ring_pk_96[6]), "r"(ring_pk_91[6]), "f"(dots_p_93[3]));
        dots_p_93[3] = _dot_373;
        float _dot_374;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_374) : "r"(ring_pk_96[7]), "r"(ring_pk_91[7]), "f"(dots_p_93[3]));
        dots_p_93[3] = _dot_374;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__3) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4), "l"(dU + (th64_29 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 1024), "l"(X + (th64_29 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 2048), "l"(X + ((th64_29 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 3072), "l"(X + ((th64_29 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 4096), "l"(X + ((th64_29 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_375 = __shfl_xor_sync(0xFFFFFFFF, hist[29], 1);
        float _add_85_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85_1) : "f"(hist[29]), "f"(_shfl_xor_sync_375));
        float _shfl_xor_sync_377 = __shfl_xor_sync(0xFFFFFFFF, _add_85_1, 2);
        float _add_376;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_376) : "f"(_add_85_1), "f"(_shfl_xor_sync_377));
        float _add_86;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(_add_376), "f"(hc_eps));
        float _mul_10;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(_fdiv_rn_50), "f"(hist[30]));
        float _shfl_xor_sync_378 = __shfl_xor_sync(0xFFFFFFFF, _mul_10, 1);
        float _add_87;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(_mul_10), "f"(_shfl_xor_sync_378));
        float _shfl_xor_sync_381 = __shfl_xor_sync(0xFFFFFFFF, _add_87, 2);
        float _add_380;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_380) : "f"(_add_87), "f"(_shfl_xor_sync_381));
        float _sub_379;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_379) : "f"(_fdiv_rn_50), "f"(_add_380));
        float _fdiv_rn_51 = __fdiv_rn(_sub_379, _add_86);
        int upper_97 = lane_1 & 16;
        float send_98 = ((upper_97 != 0) ? dots_p_93[0] : dots_p_93[2]);
        float keep_99 = ((upper_97 != 0) ? dots_p_93[2] : dots_p_93[0]);
        float _shfl_xor_sync_382 = __shfl_xor_sync(0xFFFFFFFF, send_98, 16);
        float _add_88_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88_1) : "f"(keep_99), "f"(_shfl_xor_sync_382));
        float send_100 = ((upper_97 != 0) ? dots_p_93[1] : dots_p_93[3]);
        float keep_101 = ((upper_97 != 0) ? dots_p_93[3] : dots_p_93[1]);
        float _shfl_xor_sync_383 = __shfl_xor_sync(0xFFFFFFFF, send_100, 16);
        float _add_89_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89_1) : "f"(keep_101), "f"(_shfl_xor_sync_383));
        int upper_102 = lane_1 & 8;
        float send_103 = ((upper_102 != 0) ? _add_88_1 : _add_89_1);
        float keep_104 = ((upper_102 != 0) ? _add_89_1 : _add_88_1);
        float _shfl_xor_sync_384 = __shfl_xor_sync(0xFFFFFFFF, send_103, 8);
        float _add_90;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_90) : "f"(keep_104), "f"(_shfl_xor_sync_384));
        float _shfl_xor_sync_385 = __shfl_xor_sync(0xFFFFFFFF, hist[28], 4);
        float _add_91;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(hist[28]), "f"(_shfl_xor_sync_385));
        float _shfl_xor_sync_387 = __shfl_xor_sync(0xFFFFFFFF, _add_91, 24);
        float _add_386;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_386) : "f"(_add_91), "f"(_shfl_xor_sync_387));
        float _add_92_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92_1) : "f"(_add_386), "f"(hc_eps));
        float _mul_11;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_fdiv_rn_51), "f"(hist[29]));
        float _shfl_xor_sync_388 = __shfl_xor_sync(0xFFFFFFFF, _mul_11, 4);
        float _add_93_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93_1) : "f"(_mul_11), "f"(_shfl_xor_sync_388));
        float _shfl_xor_sync_391 = __shfl_xor_sync(0xFFFFFFFF, _add_93_1, 24);
        float _add_390;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_390) : "f"(_add_93_1), "f"(_shfl_xor_sync_391));
        float _sub_389;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_389) : "f"(_fdiv_rn_51), "f"(_add_390));
        float _fdiv_rn_52 = __fdiv_rn(_sub_389, _add_92_1);
        float _shfl_xor_sync_392 = __shfl_xor_sync(0xFFFFFFFF, _add_90, 4);
        float _add_94;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(_add_90), "f"(_shfl_xor_sync_392));
        float _shfl_xor_sync_393 = __shfl_xor_sync(0xFFFFFFFF, _add_94, 2);
        float _add_95;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(_add_94), "f"(_shfl_xor_sync_393));
        float _shfl_xor_sync_395 = __shfl_xor_sync(0xFFFFFFFF, _add_95, 1);
        float _add_394;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_394) : "f"(_add_95), "f"(_shfl_xor_sync_395));
        dpre_part[own_base + 96] = _add_394;
        float _shfl_xor_sync_396 = __shfl_xor_sync(0xFFFFFFFF, hist[27], 1);
        float _add_96_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96_1) : "f"(hist[27]), "f"(_shfl_xor_sync_396));
        float _shfl_xor_sync_398 = __shfl_xor_sync(0xFFFFFFFF, _add_96_1, 2);
        float _add_397;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_397) : "f"(_add_96_1), "f"(_shfl_xor_sync_398));
        float _add_97_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97_1) : "f"(_add_397), "f"(hc_eps));
        float _mul_12;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(_fdiv_rn_52), "f"(hist[28]));
        float _shfl_xor_sync_399 = __shfl_xor_sync(0xFFFFFFFF, _mul_12, 1);
        float _add_98;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(_mul_12), "f"(_shfl_xor_sync_399));
        float _shfl_xor_sync_402 = __shfl_xor_sync(0xFFFFFFFF, _add_98, 2);
        float _add_401;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_401) : "f"(_add_98), "f"(_shfl_xor_sync_402));
        float _sub_400;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_400) : "f"(_fdiv_rn_52), "f"(_add_401));
        float _fdiv_rn_53 = __fdiv_rn(_sub_400, _add_97_1);
        mbarrier_wait(ring_bar_addr + (st_) * 8, 0);
        unsigned int ring_pk_105[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[(0) + 3]))
            : "r"(sa_ + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[(4) + 3]))
            : "r"(sa_ + 16 + lane32));
        unsigned int ring_pk_106[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[(0) + 3]))
            : "r"(sa_ + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[(4) + 3]))
            : "r"(sa_ + 1040 + lane32));
        float dots_p_107[4];
        float _shfl_xor_sync_403 = __shfl_xor_sync(0xFFFFFFFF, hist[26], 4);
        float _add_99;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(hist[26]), "f"(_shfl_xor_sync_403));
        float _shfl_xor_sync_405 = __shfl_xor_sync(0xFFFFFFFF, _add_99, 24);
        float _add_404;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_404) : "f"(_add_99), "f"(_shfl_xor_sync_405));
        float _add_100_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100_1) : "f"(_add_404), "f"(hc_eps));
        float _mul_13;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(_fdiv_rn_53), "f"(hist[27]));
        float _shfl_xor_sync_406 = __shfl_xor_sync(0xFFFFFFFF, _mul_13, 4);
        float _add_101_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101_1) : "f"(_mul_13), "f"(_shfl_xor_sync_406));
        float _shfl_xor_sync_409 = __shfl_xor_sync(0xFFFFFFFF, _add_101_1, 24);
        float _add_408;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_408) : "f"(_add_101_1), "f"(_shfl_xor_sync_409));
        float _sub_407;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_407) : "f"(_fdiv_rn_53), "f"(_add_408));
        float _fdiv_rn_54 = __fdiv_rn(_sub_407, _add_100_1);
        float _dot_410;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_410) : "r"(ring_pk_106[0]), "r"(ring_pk_105[0]), "f"(-0.0f));
        dots_p_107[0] = _dot_410;
        float _dot_411;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_411) : "r"(ring_pk_106[1]), "r"(ring_pk_105[1]), "f"(dots_p_107[0]));
        dots_p_107[0] = _dot_411;
        float _dot_412;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_412) : "r"(ring_pk_106[2]), "r"(ring_pk_105[2]), "f"(dots_p_107[0]));
        dots_p_107[0] = _dot_412;
        float _dot_413;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_413) : "r"(ring_pk_106[3]), "r"(ring_pk_105[3]), "f"(dots_p_107[0]));
        dots_p_107[0] = _dot_413;
        float _dot_414;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_414) : "r"(ring_pk_106[4]), "r"(ring_pk_105[4]), "f"(dots_p_107[0]));
        dots_p_107[0] = _dot_414;
        float _dot_415;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_415) : "r"(ring_pk_106[5]), "r"(ring_pk_105[5]), "f"(dots_p_107[0]));
        dots_p_107[0] = _dot_415;
        float _dot_416;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_416) : "r"(ring_pk_106[6]), "r"(ring_pk_105[6]), "f"(dots_p_107[0]));
        dots_p_107[0] = _dot_416;
        float _dot_417;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_417) : "r"(ring_pk_106[7]), "r"(ring_pk_105[7]), "f"(dots_p_107[0]));
        dots_p_107[0] = _dot_417;
        unsigned int ring_pk_108[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[(0) + 3]))
            : "r"(sa_ + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[(4) + 3]))
            : "r"(sa_ + 2064 + lane32));
        float _shfl_xor_sync_418 = __shfl_xor_sync(0xFFFFFFFF, hist[25], 1);
        float _add_102;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(hist[25]), "f"(_shfl_xor_sync_418));
        float _shfl_xor_sync_420 = __shfl_xor_sync(0xFFFFFFFF, _add_102, 2);
        float _add_419;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_419) : "f"(_add_102), "f"(_shfl_xor_sync_420));
        float _add_103;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(_add_419), "f"(hc_eps));
        float _mul_14;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(_fdiv_rn_54), "f"(hist[26]));
        float _shfl_xor_sync_421 = __shfl_xor_sync(0xFFFFFFFF, _mul_14, 1);
        float _add_104;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(_mul_14), "f"(_shfl_xor_sync_421));
        float _shfl_xor_sync_424 = __shfl_xor_sync(0xFFFFFFFF, _add_104, 2);
        float _add_423;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_423) : "f"(_add_104), "f"(_shfl_xor_sync_424));
        float _sub_422;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_422) : "f"(_fdiv_rn_54), "f"(_add_423));
        float _fdiv_rn_55 = __fdiv_rn(_sub_422, _add_103);
        float _dot_425;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_425) : "r"(ring_pk_108[0]), "r"(ring_pk_105[0]), "f"(-0.0f));
        dots_p_107[1] = _dot_425;
        float _dot_426;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_426) : "r"(ring_pk_108[1]), "r"(ring_pk_105[1]), "f"(dots_p_107[1]));
        dots_p_107[1] = _dot_426;
        float _dot_427;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_427) : "r"(ring_pk_108[2]), "r"(ring_pk_105[2]), "f"(dots_p_107[1]));
        dots_p_107[1] = _dot_427;
        float _dot_428;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_428) : "r"(ring_pk_108[3]), "r"(ring_pk_105[3]), "f"(dots_p_107[1]));
        dots_p_107[1] = _dot_428;
        float _dot_429;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_429) : "r"(ring_pk_108[4]), "r"(ring_pk_105[4]), "f"(dots_p_107[1]));
        dots_p_107[1] = _dot_429;
        float _dot_430;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_430) : "r"(ring_pk_108[5]), "r"(ring_pk_105[5]), "f"(dots_p_107[1]));
        dots_p_107[1] = _dot_430;
        float _dot_431;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_431) : "r"(ring_pk_108[6]), "r"(ring_pk_105[6]), "f"(dots_p_107[1]));
        dots_p_107[1] = _dot_431;
        float _dot_432;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_432) : "r"(ring_pk_108[7]), "r"(ring_pk_105[7]), "f"(dots_p_107[1]));
        dots_p_107[1] = _dot_432;
        unsigned int ring_pk_109[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[(0) + 3]))
            : "r"(sa_ + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[(4) + 3]))
            : "r"(sa_ + 3088 + lane32));
        float _shfl_xor_sync_433 = __shfl_xor_sync(0xFFFFFFFF, hist[24], 4);
        float _add_105;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_105) : "f"(hist[24]), "f"(_shfl_xor_sync_433));
        float _shfl_xor_sync_435 = __shfl_xor_sync(0xFFFFFFFF, _add_105, 24);
        float _add_434;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_434) : "f"(_add_105), "f"(_shfl_xor_sync_435));
        float _add_106;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(_add_434), "f"(hc_eps));
        float _mul_15;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(_fdiv_rn_55), "f"(hist[25]));
        float _shfl_xor_sync_436 = __shfl_xor_sync(0xFFFFFFFF, _mul_15, 4);
        float _add_107;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(_mul_15), "f"(_shfl_xor_sync_436));
        float _shfl_xor_sync_439 = __shfl_xor_sync(0xFFFFFFFF, _add_107, 24);
        float _add_438;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_438) : "f"(_add_107), "f"(_shfl_xor_sync_439));
        float _sub_437;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_437) : "f"(_fdiv_rn_55), "f"(_add_438));
        float _fdiv_rn_56 = __fdiv_rn(_sub_437, _add_106);
        float _dot_440;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_440) : "r"(ring_pk_109[0]), "r"(ring_pk_105[0]), "f"(-0.0f));
        dots_p_107[2] = _dot_440;
        float _dot_441;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_441) : "r"(ring_pk_109[1]), "r"(ring_pk_105[1]), "f"(dots_p_107[2]));
        dots_p_107[2] = _dot_441;
        float _dot_442;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_442) : "r"(ring_pk_109[2]), "r"(ring_pk_105[2]), "f"(dots_p_107[2]));
        dots_p_107[2] = _dot_442;
        float _dot_443;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_443) : "r"(ring_pk_109[3]), "r"(ring_pk_105[3]), "f"(dots_p_107[2]));
        dots_p_107[2] = _dot_443;
        float _dot_444;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_444) : "r"(ring_pk_109[4]), "r"(ring_pk_105[4]), "f"(dots_p_107[2]));
        dots_p_107[2] = _dot_444;
        float _dot_445;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_445) : "r"(ring_pk_109[5]), "r"(ring_pk_105[5]), "f"(dots_p_107[2]));
        dots_p_107[2] = _dot_445;
        float _dot_446;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_446) : "r"(ring_pk_109[6]), "r"(ring_pk_105[6]), "f"(dots_p_107[2]));
        dots_p_107[2] = _dot_446;
        float _dot_447;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_447) : "r"(ring_pk_109[7]), "r"(ring_pk_105[7]), "f"(dots_p_107[2]));
        dots_p_107[2] = _dot_447;
        unsigned int ring_pk_110[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[(0) + 3]))
            : "r"(sa_ + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[(4) + 3]))
            : "r"(sa_ + 4112 + lane32));
        float _shfl_xor_sync_448 = __shfl_xor_sync(0xFFFFFFFF, hist[23], 1);
        float _add_108;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(hist[23]), "f"(_shfl_xor_sync_448));
        float _shfl_xor_sync_450 = __shfl_xor_sync(0xFFFFFFFF, _add_108, 2);
        float _add_449;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_449) : "f"(_add_108), "f"(_shfl_xor_sync_450));
        float _add_109;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(_add_449), "f"(hc_eps));
        float _mul_16;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_fdiv_rn_56), "f"(hist[24]));
        float _shfl_xor_sync_451 = __shfl_xor_sync(0xFFFFFFFF, _mul_16, 1);
        float _add_110;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(_mul_16), "f"(_shfl_xor_sync_451));
        float _shfl_xor_sync_454 = __shfl_xor_sync(0xFFFFFFFF, _add_110, 2);
        float _add_453;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_453) : "f"(_add_110), "f"(_shfl_xor_sync_454));
        float _sub_452;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_452) : "f"(_fdiv_rn_56), "f"(_add_453));
        float _fdiv_rn_57 = __fdiv_rn(_sub_452, _add_109);
        float _dot_455;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_455) : "r"(ring_pk_110[0]), "r"(ring_pk_105[0]), "f"(-0.0f));
        dots_p_107[3] = _dot_455;
        float _dot_456;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_456) : "r"(ring_pk_110[1]), "r"(ring_pk_105[1]), "f"(dots_p_107[3]));
        dots_p_107[3] = _dot_456;
        float _dot_457;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_457) : "r"(ring_pk_110[2]), "r"(ring_pk_105[2]), "f"(dots_p_107[3]));
        dots_p_107[3] = _dot_457;
        float _dot_458;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_458) : "r"(ring_pk_110[3]), "r"(ring_pk_105[3]), "f"(dots_p_107[3]));
        dots_p_107[3] = _dot_458;
        float _dot_459;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_459) : "r"(ring_pk_110[4]), "r"(ring_pk_105[4]), "f"(dots_p_107[3]));
        dots_p_107[3] = _dot_459;
        float _dot_460;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_460) : "r"(ring_pk_110[5]), "r"(ring_pk_105[5]), "f"(dots_p_107[3]));
        dots_p_107[3] = _dot_460;
        float _dot_461;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_461) : "r"(ring_pk_110[6]), "r"(ring_pk_105[6]), "f"(dots_p_107[3]));
        dots_p_107[3] = _dot_461;
        float _dot_462;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_462) : "r"(ring_pk_110[7]), "r"(ring_pk_105[7]), "f"(dots_p_107[3]));
        dots_p_107[3] = _dot_462;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_), "l"(dU + (th64_35 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 1024), "l"(X + (th64_35 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 2048), "l"(X + ((th64_35 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 3072), "l"(X + ((th64_35 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 4096), "l"(X + ((th64_35 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_463 = __shfl_xor_sync(0xFFFFFFFF, hist[22], 4);
        float _add_111;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(hist[22]), "f"(_shfl_xor_sync_463));
        float _shfl_xor_sync_465 = __shfl_xor_sync(0xFFFFFFFF, _add_111, 24);
        float _add_464;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_464) : "f"(_add_111), "f"(_shfl_xor_sync_465));
        float _add_112_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112_1) : "f"(_add_464), "f"(hc_eps));
        float _mul_17;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_fdiv_rn_57), "f"(hist[23]));
        float _shfl_xor_sync_466 = __shfl_xor_sync(0xFFFFFFFF, _mul_17, 4);
        float _add_113_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113_1) : "f"(_mul_17), "f"(_shfl_xor_sync_466));
        float _shfl_xor_sync_469 = __shfl_xor_sync(0xFFFFFFFF, _add_113_1, 24);
        float _add_468;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_468) : "f"(_add_113_1), "f"(_shfl_xor_sync_469));
        float _sub_467;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_467) : "f"(_fdiv_rn_57), "f"(_add_468));
        float _fdiv_rn_58 = __fdiv_rn(_sub_467, _add_112_1);
        int upper_111 = lane_1 & 16;
        float send_112 = ((upper_111 != 0) ? dots_p_107[0] : dots_p_107[2]);
        float keep_113 = ((upper_111 != 0) ? dots_p_107[2] : dots_p_107[0]);
        float _shfl_xor_sync_470 = __shfl_xor_sync(0xFFFFFFFF, send_112, 16);
        float _add_114;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(keep_113), "f"(_shfl_xor_sync_470));
        float send_114 = ((upper_111 != 0) ? dots_p_107[1] : dots_p_107[3]);
        float keep_115 = ((upper_111 != 0) ? dots_p_107[3] : dots_p_107[1]);
        float _shfl_xor_sync_471 = __shfl_xor_sync(0xFFFFFFFF, send_114, 16);
        float _add_115;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(keep_115), "f"(_shfl_xor_sync_471));
        int upper_116 = lane_1 & 8;
        float send_117 = ((upper_116 != 0) ? _add_114 : _add_115);
        float keep_118 = ((upper_116 != 0) ? _add_115 : _add_114);
        float _shfl_xor_sync_472 = __shfl_xor_sync(0xFFFFFFFF, send_117, 8);
        float _add_116;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(keep_118), "f"(_shfl_xor_sync_472));
        float _shfl_xor_sync_473 = __shfl_xor_sync(0xFFFFFFFF, hist[21], 1);
        float _add_117;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(hist[21]), "f"(_shfl_xor_sync_473));
        float _shfl_xor_sync_475 = __shfl_xor_sync(0xFFFFFFFF, _add_117, 2);
        float _add_474;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_474) : "f"(_add_117), "f"(_shfl_xor_sync_475));
        float _add_118;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(_add_474), "f"(hc_eps));
        float _mul_18;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(_fdiv_rn_58), "f"(hist[22]));
        float _shfl_xor_sync_476 = __shfl_xor_sync(0xFFFFFFFF, _mul_18, 1);
        float _add_119;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(_mul_18), "f"(_shfl_xor_sync_476));
        float _shfl_xor_sync_479 = __shfl_xor_sync(0xFFFFFFFF, _add_119, 2);
        float _add_478;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_478) : "f"(_add_119), "f"(_shfl_xor_sync_479));
        float _sub_477;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_477) : "f"(_fdiv_rn_58), "f"(_add_478));
        float _fdiv_rn_59 = __fdiv_rn(_sub_477, _add_118);
        float _shfl_xor_sync_480 = __shfl_xor_sync(0xFFFFFFFF, _add_116, 4);
        float _add_120;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(_add_116), "f"(_shfl_xor_sync_480));
        float _shfl_xor_sync_481 = __shfl_xor_sync(0xFFFFFFFF, _add_120, 2);
        float _add_121;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(_add_120), "f"(_shfl_xor_sync_481));
        float _shfl_xor_sync_483 = __shfl_xor_sync(0xFFFFFFFF, _add_121, 1);
        float _add_482;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_482) : "f"(_add_121), "f"(_shfl_xor_sync_483));
        dpre_part[own_base + 128] = _add_482;
        float _shfl_xor_sync_484 = __shfl_xor_sync(0xFFFFFFFF, hist[20], 4);
        float _add_122;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(hist[20]), "f"(_shfl_xor_sync_484));
        float _shfl_xor_sync_486 = __shfl_xor_sync(0xFFFFFFFF, _add_122, 24);
        float _add_485;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_485) : "f"(_add_122), "f"(_shfl_xor_sync_486));
        float _add_123;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(_add_485), "f"(hc_eps));
        float _mul_19;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_fdiv_rn_59), "f"(hist[21]));
        float _shfl_xor_sync_487 = __shfl_xor_sync(0xFFFFFFFF, _mul_19, 4);
        float _add_124_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124_1) : "f"(_mul_19), "f"(_shfl_xor_sync_487));
        float _shfl_xor_sync_490 = __shfl_xor_sync(0xFFFFFFFF, _add_124_1, 24);
        float _add_489;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_489) : "f"(_add_124_1), "f"(_shfl_xor_sync_490));
        float _sub_488;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_488) : "f"(_fdiv_rn_59), "f"(_add_489));
        float _fdiv_rn_60 = __fdiv_rn(_sub_488, _add_123);
        mbarrier_wait(ring_bar_addr + (st__3) * 8, 0);
        unsigned int ring_pk_119[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[(0) + 3]))
            : "r"(sa__4 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[(4) + 3]))
            : "r"(sa__4 + 16 + lane32));
        unsigned int ring_pk_120[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[(0) + 3]))
            : "r"(sa__4 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[(4) + 3]))
            : "r"(sa__4 + 1040 + lane32));
        float dots_p_121[4];
        float _shfl_xor_sync_491 = __shfl_xor_sync(0xFFFFFFFF, hist[19], 1);
        float _add_125_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125_1) : "f"(hist[19]), "f"(_shfl_xor_sync_491));
        float _shfl_xor_sync_493 = __shfl_xor_sync(0xFFFFFFFF, _add_125_1, 2);
        float _add_492;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_492) : "f"(_add_125_1), "f"(_shfl_xor_sync_493));
        float _add_126;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(_add_492), "f"(hc_eps));
        float _mul_20;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(_fdiv_rn_60), "f"(hist[20]));
        float _shfl_xor_sync_494 = __shfl_xor_sync(0xFFFFFFFF, _mul_20, 1);
        float _add_127;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(_mul_20), "f"(_shfl_xor_sync_494));
        float _shfl_xor_sync_497 = __shfl_xor_sync(0xFFFFFFFF, _add_127, 2);
        float _add_496;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_496) : "f"(_add_127), "f"(_shfl_xor_sync_497));
        float _sub_495;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_495) : "f"(_fdiv_rn_60), "f"(_add_496));
        float _fdiv_rn_61 = __fdiv_rn(_sub_495, _add_126);
        float _dot_498;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_498) : "r"(ring_pk_120[0]), "r"(ring_pk_119[0]), "f"(-0.0f));
        dots_p_121[0] = _dot_498;
        float _dot_499;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_499) : "r"(ring_pk_120[1]), "r"(ring_pk_119[1]), "f"(dots_p_121[0]));
        dots_p_121[0] = _dot_499;
        float _dot_500;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_500) : "r"(ring_pk_120[2]), "r"(ring_pk_119[2]), "f"(dots_p_121[0]));
        dots_p_121[0] = _dot_500;
        float _dot_501;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_501) : "r"(ring_pk_120[3]), "r"(ring_pk_119[3]), "f"(dots_p_121[0]));
        dots_p_121[0] = _dot_501;
        float _dot_502;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_502) : "r"(ring_pk_120[4]), "r"(ring_pk_119[4]), "f"(dots_p_121[0]));
        dots_p_121[0] = _dot_502;
        float _dot_503;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_503) : "r"(ring_pk_120[5]), "r"(ring_pk_119[5]), "f"(dots_p_121[0]));
        dots_p_121[0] = _dot_503;
        float _dot_504;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_504) : "r"(ring_pk_120[6]), "r"(ring_pk_119[6]), "f"(dots_p_121[0]));
        dots_p_121[0] = _dot_504;
        float _dot_505;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_505) : "r"(ring_pk_120[7]), "r"(ring_pk_119[7]), "f"(dots_p_121[0]));
        dots_p_121[0] = _dot_505;
        unsigned int ring_pk_122[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[(0) + 3]))
            : "r"(sa__4 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[(4) + 3]))
            : "r"(sa__4 + 2064 + lane32));
        float _shfl_xor_sync_506 = __shfl_xor_sync(0xFFFFFFFF, hist[18], 4);
        float _add_128;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(hist[18]), "f"(_shfl_xor_sync_506));
        float _shfl_xor_sync_508 = __shfl_xor_sync(0xFFFFFFFF, _add_128, 24);
        float _add_507;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_507) : "f"(_add_128), "f"(_shfl_xor_sync_508));
        float _add_129;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(_add_507), "f"(hc_eps));
        float _mul_21;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(_fdiv_rn_61), "f"(hist[19]));
        float _shfl_xor_sync_509 = __shfl_xor_sync(0xFFFFFFFF, _mul_21, 4);
        float _add_130;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(_mul_21), "f"(_shfl_xor_sync_509));
        float _shfl_xor_sync_512 = __shfl_xor_sync(0xFFFFFFFF, _add_130, 24);
        float _add_511;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_511) : "f"(_add_130), "f"(_shfl_xor_sync_512));
        float _sub_510;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_510) : "f"(_fdiv_rn_61), "f"(_add_511));
        float _fdiv_rn_62 = __fdiv_rn(_sub_510, _add_129);
        float _dot_513;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_513) : "r"(ring_pk_122[0]), "r"(ring_pk_119[0]), "f"(-0.0f));
        dots_p_121[1] = _dot_513;
        float _dot_514;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_514) : "r"(ring_pk_122[1]), "r"(ring_pk_119[1]), "f"(dots_p_121[1]));
        dots_p_121[1] = _dot_514;
        float _dot_515;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_515) : "r"(ring_pk_122[2]), "r"(ring_pk_119[2]), "f"(dots_p_121[1]));
        dots_p_121[1] = _dot_515;
        float _dot_516;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_516) : "r"(ring_pk_122[3]), "r"(ring_pk_119[3]), "f"(dots_p_121[1]));
        dots_p_121[1] = _dot_516;
        float _dot_517;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_517) : "r"(ring_pk_122[4]), "r"(ring_pk_119[4]), "f"(dots_p_121[1]));
        dots_p_121[1] = _dot_517;
        float _dot_518;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_518) : "r"(ring_pk_122[5]), "r"(ring_pk_119[5]), "f"(dots_p_121[1]));
        dots_p_121[1] = _dot_518;
        float _dot_519;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_519) : "r"(ring_pk_122[6]), "r"(ring_pk_119[6]), "f"(dots_p_121[1]));
        dots_p_121[1] = _dot_519;
        float _dot_520;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_520) : "r"(ring_pk_122[7]), "r"(ring_pk_119[7]), "f"(dots_p_121[1]));
        dots_p_121[1] = _dot_520;
        unsigned int ring_pk_123[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[(0) + 3]))
            : "r"(sa__4 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[(4) + 3]))
            : "r"(sa__4 + 3088 + lane32));
        float _shfl_xor_sync_521 = __shfl_xor_sync(0xFFFFFFFF, hist[17], 1);
        float _add_131;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(hist[17]), "f"(_shfl_xor_sync_521));
        float _shfl_xor_sync_523 = __shfl_xor_sync(0xFFFFFFFF, _add_131, 2);
        float _add_522;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_522) : "f"(_add_131), "f"(_shfl_xor_sync_523));
        float _add_132;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(_add_522), "f"(hc_eps));
        float _mul_22;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(_fdiv_rn_62), "f"(hist[18]));
        float _shfl_xor_sync_524 = __shfl_xor_sync(0xFFFFFFFF, _mul_22, 1);
        float _add_133;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(_mul_22), "f"(_shfl_xor_sync_524));
        float _shfl_xor_sync_527 = __shfl_xor_sync(0xFFFFFFFF, _add_133, 2);
        float _add_526;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_526) : "f"(_add_133), "f"(_shfl_xor_sync_527));
        float _sub_525;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_525) : "f"(_fdiv_rn_62), "f"(_add_526));
        float _fdiv_rn_63 = __fdiv_rn(_sub_525, _add_132);
        float _dot_528;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_528) : "r"(ring_pk_123[0]), "r"(ring_pk_119[0]), "f"(-0.0f));
        dots_p_121[2] = _dot_528;
        float _dot_529;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_529) : "r"(ring_pk_123[1]), "r"(ring_pk_119[1]), "f"(dots_p_121[2]));
        dots_p_121[2] = _dot_529;
        float _dot_530;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_530) : "r"(ring_pk_123[2]), "r"(ring_pk_119[2]), "f"(dots_p_121[2]));
        dots_p_121[2] = _dot_530;
        float _dot_531;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_531) : "r"(ring_pk_123[3]), "r"(ring_pk_119[3]), "f"(dots_p_121[2]));
        dots_p_121[2] = _dot_531;
        float _dot_532;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_532) : "r"(ring_pk_123[4]), "r"(ring_pk_119[4]), "f"(dots_p_121[2]));
        dots_p_121[2] = _dot_532;
        float _dot_533;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_533) : "r"(ring_pk_123[5]), "r"(ring_pk_119[5]), "f"(dots_p_121[2]));
        dots_p_121[2] = _dot_533;
        float _dot_534;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_534) : "r"(ring_pk_123[6]), "r"(ring_pk_119[6]), "f"(dots_p_121[2]));
        dots_p_121[2] = _dot_534;
        float _dot_535;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_535) : "r"(ring_pk_123[7]), "r"(ring_pk_119[7]), "f"(dots_p_121[2]));
        dots_p_121[2] = _dot_535;
        unsigned int ring_pk_124[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[(0) + 3]))
            : "r"(sa__4 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[(4) + 3]))
            : "r"(sa__4 + 4112 + lane32));
        float _shfl_xor_sync_536 = __shfl_xor_sync(0xFFFFFFFF, hist[16], 4);
        float _add_134;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(hist[16]), "f"(_shfl_xor_sync_536));
        float _shfl_xor_sync_538 = __shfl_xor_sync(0xFFFFFFFF, _add_134, 24);
        float _add_537;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_537) : "f"(_add_134), "f"(_shfl_xor_sync_538));
        float _add_135;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_135) : "f"(_add_537), "f"(hc_eps));
        float _mul_23;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_fdiv_rn_63), "f"(hist[17]));
        float _shfl_xor_sync_539 = __shfl_xor_sync(0xFFFFFFFF, _mul_23, 4);
        float _add_136_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136_1) : "f"(_mul_23), "f"(_shfl_xor_sync_539));
        float _shfl_xor_sync_542 = __shfl_xor_sync(0xFFFFFFFF, _add_136_1, 24);
        float _add_541;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_541) : "f"(_add_136_1), "f"(_shfl_xor_sync_542));
        float _sub_540;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_540) : "f"(_fdiv_rn_63), "f"(_add_541));
        float _fdiv_rn_64 = __fdiv_rn(_sub_540, _add_135);
        float _dot_543;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_543) : "r"(ring_pk_124[0]), "r"(ring_pk_119[0]), "f"(-0.0f));
        dots_p_121[3] = _dot_543;
        float _dot_544;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_544) : "r"(ring_pk_124[1]), "r"(ring_pk_119[1]), "f"(dots_p_121[3]));
        dots_p_121[3] = _dot_544;
        float _dot_545;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_545) : "r"(ring_pk_124[2]), "r"(ring_pk_119[2]), "f"(dots_p_121[3]));
        dots_p_121[3] = _dot_545;
        float _dot_546;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_546) : "r"(ring_pk_124[3]), "r"(ring_pk_119[3]), "f"(dots_p_121[3]));
        dots_p_121[3] = _dot_546;
        float _dot_547;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_547) : "r"(ring_pk_124[4]), "r"(ring_pk_119[4]), "f"(dots_p_121[3]));
        dots_p_121[3] = _dot_547;
        float _dot_548;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_548) : "r"(ring_pk_124[5]), "r"(ring_pk_119[5]), "f"(dots_p_121[3]));
        dots_p_121[3] = _dot_548;
        float _dot_549;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_549) : "r"(ring_pk_124[6]), "r"(ring_pk_119[6]), "f"(dots_p_121[3]));
        dots_p_121[3] = _dot_549;
        float _dot_550;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_550) : "r"(ring_pk_124[7]), "r"(ring_pk_119[7]), "f"(dots_p_121[3]));
        dots_p_121[3] = _dot_550;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__3) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4), "l"(dU + (th64_41 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 1024), "l"(X + (th64_41 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 2048), "l"(X + ((th64_41 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 3072), "l"(X + ((th64_41 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 4096), "l"(X + ((th64_41 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_551 = __shfl_xor_sync(0xFFFFFFFF, hist[15], 1);
        float _add_137_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137_1) : "f"(hist[15]), "f"(_shfl_xor_sync_551));
        float _shfl_xor_sync_553 = __shfl_xor_sync(0xFFFFFFFF, _add_137_1, 2);
        float _add_552;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_552) : "f"(_add_137_1), "f"(_shfl_xor_sync_553));
        float _add_138;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_138) : "f"(_add_552), "f"(hc_eps));
        float _mul_24;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(_fdiv_rn_64), "f"(hist[16]));
        float _shfl_xor_sync_554 = __shfl_xor_sync(0xFFFFFFFF, _mul_24, 1);
        float _add_139;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(_mul_24), "f"(_shfl_xor_sync_554));
        float _shfl_xor_sync_557 = __shfl_xor_sync(0xFFFFFFFF, _add_139, 2);
        float _add_556;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_556) : "f"(_add_139), "f"(_shfl_xor_sync_557));
        float _sub_555;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_555) : "f"(_fdiv_rn_64), "f"(_add_556));
        float _fdiv_rn_65 = __fdiv_rn(_sub_555, _add_138);
        int upper_125 = lane_1 & 16;
        float send_126 = ((upper_125 != 0) ? dots_p_121[0] : dots_p_121[2]);
        float keep_127 = ((upper_125 != 0) ? dots_p_121[2] : dots_p_121[0]);
        float _shfl_xor_sync_558 = __shfl_xor_sync(0xFFFFFFFF, send_126, 16);
        float _add_140;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_140) : "f"(keep_127), "f"(_shfl_xor_sync_558));
        float send_128 = ((upper_125 != 0) ? dots_p_121[1] : dots_p_121[3]);
        float keep_129 = ((upper_125 != 0) ? dots_p_121[3] : dots_p_121[1]);
        float _shfl_xor_sync_559 = __shfl_xor_sync(0xFFFFFFFF, send_128, 16);
        float _add_141;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_141) : "f"(keep_129), "f"(_shfl_xor_sync_559));
        int upper_130 = lane_1 & 8;
        float send_131 = ((upper_130 != 0) ? _add_140 : _add_141);
        float keep_132 = ((upper_130 != 0) ? _add_141 : _add_140);
        float _shfl_xor_sync_560 = __shfl_xor_sync(0xFFFFFFFF, send_131, 8);
        float _add_142;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(keep_132), "f"(_shfl_xor_sync_560));
        float _shfl_xor_sync_561 = __shfl_xor_sync(0xFFFFFFFF, hist[14], 4);
        float _add_143;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(hist[14]), "f"(_shfl_xor_sync_561));
        float _shfl_xor_sync_563 = __shfl_xor_sync(0xFFFFFFFF, _add_143, 24);
        float _add_562;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_562) : "f"(_add_143), "f"(_shfl_xor_sync_563));
        float _add_144;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(_add_562), "f"(hc_eps));
        float _mul_25;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(_fdiv_rn_65), "f"(hist[15]));
        float _shfl_xor_sync_564 = __shfl_xor_sync(0xFFFFFFFF, _mul_25, 4);
        float _add_145;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_145) : "f"(_mul_25), "f"(_shfl_xor_sync_564));
        float _shfl_xor_sync_567 = __shfl_xor_sync(0xFFFFFFFF, _add_145, 24);
        float _add_566;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_566) : "f"(_add_145), "f"(_shfl_xor_sync_567));
        float _sub_565;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_565) : "f"(_fdiv_rn_65), "f"(_add_566));
        float _fdiv_rn_66 = __fdiv_rn(_sub_565, _add_144);
        float _shfl_xor_sync_568 = __shfl_xor_sync(0xFFFFFFFF, _add_142, 4);
        float _add_146;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_146) : "f"(_add_142), "f"(_shfl_xor_sync_568));
        float _shfl_xor_sync_569 = __shfl_xor_sync(0xFFFFFFFF, _add_146, 2);
        float _add_147;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(_add_146), "f"(_shfl_xor_sync_569));
        float _shfl_xor_sync_571 = __shfl_xor_sync(0xFFFFFFFF, _add_147, 1);
        float _add_570;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_570) : "f"(_add_147), "f"(_shfl_xor_sync_571));
        dpre_part[own_base + 160] = _add_570;
        float _shfl_xor_sync_572 = __shfl_xor_sync(0xFFFFFFFF, hist[13], 1);
        float _add_148_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148_1) : "f"(hist[13]), "f"(_shfl_xor_sync_572));
        float _shfl_xor_sync_574 = __shfl_xor_sync(0xFFFFFFFF, _add_148_1, 2);
        float _add_573;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_573) : "f"(_add_148_1), "f"(_shfl_xor_sync_574));
        float _add_149_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_149_1) : "f"(_add_573), "f"(hc_eps));
        float _mul_26;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(_fdiv_rn_66), "f"(hist[14]));
        float _shfl_xor_sync_575 = __shfl_xor_sync(0xFFFFFFFF, _mul_26, 1);
        float _add_150;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_150) : "f"(_mul_26), "f"(_shfl_xor_sync_575));
        float _shfl_xor_sync_578 = __shfl_xor_sync(0xFFFFFFFF, _add_150, 2);
        float _add_577;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_577) : "f"(_add_150), "f"(_shfl_xor_sync_578));
        float _sub_576;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_576) : "f"(_fdiv_rn_66), "f"(_add_577));
        float _fdiv_rn_67 = __fdiv_rn(_sub_576, _add_149_1);
        mbarrier_wait(ring_bar_addr + (st_) * 8, 1);
        unsigned int ring_pk_133[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[(0) + 3]))
            : "r"(sa_ + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[(4) + 3]))
            : "r"(sa_ + 16 + lane32));
        unsigned int ring_pk_134[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[(0) + 3]))
            : "r"(sa_ + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[(4) + 3]))
            : "r"(sa_ + 1040 + lane32));
        float dots_p_135[4];
        float _shfl_xor_sync_579 = __shfl_xor_sync(0xFFFFFFFF, hist[12], 4);
        float _add_151;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(hist[12]), "f"(_shfl_xor_sync_579));
        float _shfl_xor_sync_581 = __shfl_xor_sync(0xFFFFFFFF, _add_151, 24);
        float _add_580;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_580) : "f"(_add_151), "f"(_shfl_xor_sync_581));
        float _add_152;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_152) : "f"(_add_580), "f"(hc_eps));
        float _mul_27;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(_fdiv_rn_67), "f"(hist[13]));
        float _shfl_xor_sync_582 = __shfl_xor_sync(0xFFFFFFFF, _mul_27, 4);
        float _add_153;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(_mul_27), "f"(_shfl_xor_sync_582));
        float _shfl_xor_sync_585 = __shfl_xor_sync(0xFFFFFFFF, _add_153, 24);
        float _add_584;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_584) : "f"(_add_153), "f"(_shfl_xor_sync_585));
        float _sub_583;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_583) : "f"(_fdiv_rn_67), "f"(_add_584));
        float _fdiv_rn_68 = __fdiv_rn(_sub_583, _add_152);
        float _dot_586;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_586) : "r"(ring_pk_134[0]), "r"(ring_pk_133[0]), "f"(-0.0f));
        dots_p_135[0] = _dot_586;
        float _dot_587;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_587) : "r"(ring_pk_134[1]), "r"(ring_pk_133[1]), "f"(dots_p_135[0]));
        dots_p_135[0] = _dot_587;
        float _dot_588;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_588) : "r"(ring_pk_134[2]), "r"(ring_pk_133[2]), "f"(dots_p_135[0]));
        dots_p_135[0] = _dot_588;
        float _dot_589;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_589) : "r"(ring_pk_134[3]), "r"(ring_pk_133[3]), "f"(dots_p_135[0]));
        dots_p_135[0] = _dot_589;
        float _dot_590;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_590) : "r"(ring_pk_134[4]), "r"(ring_pk_133[4]), "f"(dots_p_135[0]));
        dots_p_135[0] = _dot_590;
        float _dot_591;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_591) : "r"(ring_pk_134[5]), "r"(ring_pk_133[5]), "f"(dots_p_135[0]));
        dots_p_135[0] = _dot_591;
        float _dot_592;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_592) : "r"(ring_pk_134[6]), "r"(ring_pk_133[6]), "f"(dots_p_135[0]));
        dots_p_135[0] = _dot_592;
        float _dot_593;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_593) : "r"(ring_pk_134[7]), "r"(ring_pk_133[7]), "f"(dots_p_135[0]));
        dots_p_135[0] = _dot_593;
        unsigned int ring_pk_136[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[(0) + 3]))
            : "r"(sa_ + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[(4) + 3]))
            : "r"(sa_ + 2064 + lane32));
        float _shfl_xor_sync_594 = __shfl_xor_sync(0xFFFFFFFF, hist[11], 1);
        float _add_154;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_154) : "f"(hist[11]), "f"(_shfl_xor_sync_594));
        float _shfl_xor_sync_596 = __shfl_xor_sync(0xFFFFFFFF, _add_154, 2);
        float _add_595;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_595) : "f"(_add_154), "f"(_shfl_xor_sync_596));
        float _add_155_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155_1) : "f"(_add_595), "f"(hc_eps));
        float _mul_28;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(_fdiv_rn_68), "f"(hist[12]));
        float _shfl_xor_sync_597 = __shfl_xor_sync(0xFFFFFFFF, _mul_28, 1);
        float _add_156_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156_1) : "f"(_mul_28), "f"(_shfl_xor_sync_597));
        float _shfl_xor_sync_600 = __shfl_xor_sync(0xFFFFFFFF, _add_156_1, 2);
        float _add_599;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_599) : "f"(_add_156_1), "f"(_shfl_xor_sync_600));
        float _sub_598;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_598) : "f"(_fdiv_rn_68), "f"(_add_599));
        float _fdiv_rn_69 = __fdiv_rn(_sub_598, _add_155_1);
        float _dot_601;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_601) : "r"(ring_pk_136[0]), "r"(ring_pk_133[0]), "f"(-0.0f));
        dots_p_135[1] = _dot_601;
        float _dot_602;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_602) : "r"(ring_pk_136[1]), "r"(ring_pk_133[1]), "f"(dots_p_135[1]));
        dots_p_135[1] = _dot_602;
        float _dot_603;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_603) : "r"(ring_pk_136[2]), "r"(ring_pk_133[2]), "f"(dots_p_135[1]));
        dots_p_135[1] = _dot_603;
        float _dot_604;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_604) : "r"(ring_pk_136[3]), "r"(ring_pk_133[3]), "f"(dots_p_135[1]));
        dots_p_135[1] = _dot_604;
        float _dot_605;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_605) : "r"(ring_pk_136[4]), "r"(ring_pk_133[4]), "f"(dots_p_135[1]));
        dots_p_135[1] = _dot_605;
        float _dot_606;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_606) : "r"(ring_pk_136[5]), "r"(ring_pk_133[5]), "f"(dots_p_135[1]));
        dots_p_135[1] = _dot_606;
        float _dot_607;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_607) : "r"(ring_pk_136[6]), "r"(ring_pk_133[6]), "f"(dots_p_135[1]));
        dots_p_135[1] = _dot_607;
        float _dot_608;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_608) : "r"(ring_pk_136[7]), "r"(ring_pk_133[7]), "f"(dots_p_135[1]));
        dots_p_135[1] = _dot_608;
        unsigned int ring_pk_137[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[(0) + 3]))
            : "r"(sa_ + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[(4) + 3]))
            : "r"(sa_ + 3088 + lane32));
        float _shfl_xor_sync_609 = __shfl_xor_sync(0xFFFFFFFF, hist[10], 4);
        float _add_157;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(hist[10]), "f"(_shfl_xor_sync_609));
        float _shfl_xor_sync_611 = __shfl_xor_sync(0xFFFFFFFF, _add_157, 24);
        float _add_610;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_610) : "f"(_add_157), "f"(_shfl_xor_sync_611));
        float _add_158;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_158) : "f"(_add_610), "f"(hc_eps));
        float _mul_29;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(_fdiv_rn_69), "f"(hist[11]));
        float _shfl_xor_sync_612 = __shfl_xor_sync(0xFFFFFFFF, _mul_29, 4);
        float _add_159;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_159) : "f"(_mul_29), "f"(_shfl_xor_sync_612));
        float _shfl_xor_sync_615 = __shfl_xor_sync(0xFFFFFFFF, _add_159, 24);
        float _add_614;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_614) : "f"(_add_159), "f"(_shfl_xor_sync_615));
        float _sub_613;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_613) : "f"(_fdiv_rn_69), "f"(_add_614));
        float _fdiv_rn_70 = __fdiv_rn(_sub_613, _add_158);
        float _dot_616;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_616) : "r"(ring_pk_137[0]), "r"(ring_pk_133[0]), "f"(-0.0f));
        dots_p_135[2] = _dot_616;
        float _dot_617;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_617) : "r"(ring_pk_137[1]), "r"(ring_pk_133[1]), "f"(dots_p_135[2]));
        dots_p_135[2] = _dot_617;
        float _dot_618;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_618) : "r"(ring_pk_137[2]), "r"(ring_pk_133[2]), "f"(dots_p_135[2]));
        dots_p_135[2] = _dot_618;
        float _dot_619;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_619) : "r"(ring_pk_137[3]), "r"(ring_pk_133[3]), "f"(dots_p_135[2]));
        dots_p_135[2] = _dot_619;
        float _dot_620;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_620) : "r"(ring_pk_137[4]), "r"(ring_pk_133[4]), "f"(dots_p_135[2]));
        dots_p_135[2] = _dot_620;
        float _dot_621;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_621) : "r"(ring_pk_137[5]), "r"(ring_pk_133[5]), "f"(dots_p_135[2]));
        dots_p_135[2] = _dot_621;
        float _dot_622;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_622) : "r"(ring_pk_137[6]), "r"(ring_pk_133[6]), "f"(dots_p_135[2]));
        dots_p_135[2] = _dot_622;
        float _dot_623;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_623) : "r"(ring_pk_137[7]), "r"(ring_pk_133[7]), "f"(dots_p_135[2]));
        dots_p_135[2] = _dot_623;
        unsigned int ring_pk_138[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[(0) + 3]))
            : "r"(sa_ + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[(4) + 3]))
            : "r"(sa_ + 4112 + lane32));
        float _shfl_xor_sync_624 = __shfl_xor_sync(0xFFFFFFFF, hist[9], 1);
        float _add_160_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160_1) : "f"(hist[9]), "f"(_shfl_xor_sync_624));
        float _shfl_xor_sync_626 = __shfl_xor_sync(0xFFFFFFFF, _add_160_1, 2);
        float _add_625;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_625) : "f"(_add_160_1), "f"(_shfl_xor_sync_626));
        float _add_161;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(_add_625), "f"(hc_eps));
        float _mul_30;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(_fdiv_rn_70), "f"(hist[10]));
        float _shfl_xor_sync_627 = __shfl_xor_sync(0xFFFFFFFF, _mul_30, 1);
        float _add_162;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_162) : "f"(_mul_30), "f"(_shfl_xor_sync_627));
        float _shfl_xor_sync_630 = __shfl_xor_sync(0xFFFFFFFF, _add_162, 2);
        float _add_629;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_629) : "f"(_add_162), "f"(_shfl_xor_sync_630));
        float _sub_628;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_628) : "f"(_fdiv_rn_70), "f"(_add_629));
        float _fdiv_rn_71 = __fdiv_rn(_sub_628, _add_161);
        float _dot_631;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_631) : "r"(ring_pk_138[0]), "r"(ring_pk_133[0]), "f"(-0.0f));
        dots_p_135[3] = _dot_631;
        float _dot_632;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_632) : "r"(ring_pk_138[1]), "r"(ring_pk_133[1]), "f"(dots_p_135[3]));
        dots_p_135[3] = _dot_632;
        float _dot_633;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_633) : "r"(ring_pk_138[2]), "r"(ring_pk_133[2]), "f"(dots_p_135[3]));
        dots_p_135[3] = _dot_633;
        float _dot_634;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_634) : "r"(ring_pk_138[3]), "r"(ring_pk_133[3]), "f"(dots_p_135[3]));
        dots_p_135[3] = _dot_634;
        float _dot_635;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_635) : "r"(ring_pk_138[4]), "r"(ring_pk_133[4]), "f"(dots_p_135[3]));
        dots_p_135[3] = _dot_635;
        float _dot_636;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_636) : "r"(ring_pk_138[5]), "r"(ring_pk_133[5]), "f"(dots_p_135[3]));
        dots_p_135[3] = _dot_636;
        float _dot_637;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_637) : "r"(ring_pk_138[6]), "r"(ring_pk_133[6]), "f"(dots_p_135[3]));
        dots_p_135[3] = _dot_637;
        float _dot_638;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_638) : "r"(ring_pk_138[7]), "r"(ring_pk_133[7]), "f"(dots_p_135[3]));
        dots_p_135[3] = _dot_638;
        __syncwarp();
        bool _elect_sync_1 = elect_sync();
        bool tail_el = _elect_sync_1 && tail_pred;
        if (tail_el) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_), "l"(dU + (th64_47 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 1024), "l"(X + (th64_47 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 2048), "l"(X + ((th64_47 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 3072), "l"(X + ((th64_47 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_ + 4096), "l"(X + ((th64_47 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st_) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_639 = __shfl_xor_sync(0xFFFFFFFF, hist[8], 4);
        float _add_163_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163_1) : "f"(hist[8]), "f"(_shfl_xor_sync_639));
        float _shfl_xor_sync_641 = __shfl_xor_sync(0xFFFFFFFF, _add_163_1, 24);
        float _add_640;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_640) : "f"(_add_163_1), "f"(_shfl_xor_sync_641));
        float _add_164_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_164_1) : "f"(_add_640), "f"(hc_eps));
        float _mul_31;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(_fdiv_rn_71), "f"(hist[9]));
        float _shfl_xor_sync_642 = __shfl_xor_sync(0xFFFFFFFF, _mul_31, 4);
        float _add_165;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(_mul_31), "f"(_shfl_xor_sync_642));
        float _shfl_xor_sync_645 = __shfl_xor_sync(0xFFFFFFFF, _add_165, 24);
        float _add_644;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_644) : "f"(_add_165), "f"(_shfl_xor_sync_645));
        float _sub_643;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_643) : "f"(_fdiv_rn_71), "f"(_add_644));
        float _fdiv_rn_72 = __fdiv_rn(_sub_643, _add_164_1);
        int upper_139 = lane_1 & 16;
        float send_140 = ((upper_139 != 0) ? dots_p_135[0] : dots_p_135[2]);
        float keep_141 = ((upper_139 != 0) ? dots_p_135[2] : dots_p_135[0]);
        float _shfl_xor_sync_646 = __shfl_xor_sync(0xFFFFFFFF, send_140, 16);
        float _add_166;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_166) : "f"(keep_141), "f"(_shfl_xor_sync_646));
        float send_142 = ((upper_139 != 0) ? dots_p_135[1] : dots_p_135[3]);
        float keep_143 = ((upper_139 != 0) ? dots_p_135[3] : dots_p_135[1]);
        float _shfl_xor_sync_647 = __shfl_xor_sync(0xFFFFFFFF, send_142, 16);
        float _add_167_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167_1) : "f"(keep_143), "f"(_shfl_xor_sync_647));
        int upper_144 = lane_1 & 8;
        float send_145 = ((upper_144 != 0) ? _add_166 : _add_167_1);
        float keep_146 = ((upper_144 != 0) ? _add_167_1 : _add_166);
        float _shfl_xor_sync_648 = __shfl_xor_sync(0xFFFFFFFF, send_145, 8);
        float _add_168_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168_1) : "f"(keep_146), "f"(_shfl_xor_sync_648));
        float _shfl_xor_sync_649 = __shfl_xor_sync(0xFFFFFFFF, hist[7], 1);
        float _add_169;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(hist[7]), "f"(_shfl_xor_sync_649));
        float _shfl_xor_sync_651 = __shfl_xor_sync(0xFFFFFFFF, _add_169, 2);
        float _add_650;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_650) : "f"(_add_169), "f"(_shfl_xor_sync_651));
        float _add_170;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_170) : "f"(_add_650), "f"(hc_eps));
        float _mul_32;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(_fdiv_rn_72), "f"(hist[8]));
        float _shfl_xor_sync_652 = __shfl_xor_sync(0xFFFFFFFF, _mul_32, 1);
        float _add_171;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171) : "f"(_mul_32), "f"(_shfl_xor_sync_652));
        float _shfl_xor_sync_655 = __shfl_xor_sync(0xFFFFFFFF, _add_171, 2);
        float _add_654;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_654) : "f"(_add_171), "f"(_shfl_xor_sync_655));
        float _sub_653;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_653) : "f"(_fdiv_rn_72), "f"(_add_654));
        float _fdiv_rn_73 = __fdiv_rn(_sub_653, _add_170);
        float _shfl_xor_sync_656 = __shfl_xor_sync(0xFFFFFFFF, _add_168_1, 4);
        float _add_172;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(_add_168_1), "f"(_shfl_xor_sync_656));
        float _shfl_xor_sync_657 = __shfl_xor_sync(0xFFFFFFFF, _add_172, 2);
        float _add_173;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_173) : "f"(_add_172), "f"(_shfl_xor_sync_657));
        float _shfl_xor_sync_659 = __shfl_xor_sync(0xFFFFFFFF, _add_173, 1);
        float _add_658;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_658) : "f"(_add_173), "f"(_shfl_xor_sync_659));
        dpre_part[own_base + 192] = _add_658;
        float _shfl_xor_sync_660 = __shfl_xor_sync(0xFFFFFFFF, hist[6], 4);
        float _add_174;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_174) : "f"(hist[6]), "f"(_shfl_xor_sync_660));
        float _shfl_xor_sync_662 = __shfl_xor_sync(0xFFFFFFFF, _add_174, 24);
        float _add_661;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_661) : "f"(_add_174), "f"(_shfl_xor_sync_662));
        float _add_175;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_175) : "f"(_add_661), "f"(hc_eps));
        float _mul_33;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(_fdiv_rn_73), "f"(hist[7]));
        float _shfl_xor_sync_663 = __shfl_xor_sync(0xFFFFFFFF, _mul_33, 4);
        float _add_176;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_176) : "f"(_mul_33), "f"(_shfl_xor_sync_663));
        float _shfl_xor_sync_666 = __shfl_xor_sync(0xFFFFFFFF, _add_176, 24);
        float _add_665;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_665) : "f"(_add_176), "f"(_shfl_xor_sync_666));
        float _sub_664;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_664) : "f"(_fdiv_rn_73), "f"(_add_665));
        float _fdiv_rn_74 = __fdiv_rn(_sub_664, _add_175);
        mbarrier_wait(ring_bar_addr + (st__3) * 8, 1);
        unsigned int ring_pk_147[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[(0) + 3]))
            : "r"(sa__4 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[(4) + 3]))
            : "r"(sa__4 + 16 + lane32));
        unsigned int ring_pk_148[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[(0) + 3]))
            : "r"(sa__4 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[(4) + 3]))
            : "r"(sa__4 + 1040 + lane32));
        float dots_p_149[4];
        float _shfl_xor_sync_667 = __shfl_xor_sync(0xFFFFFFFF, hist[5], 1);
        float _add_177;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_177) : "f"(hist[5]), "f"(_shfl_xor_sync_667));
        float _shfl_xor_sync_669 = __shfl_xor_sync(0xFFFFFFFF, _add_177, 2);
        float _add_668;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_668) : "f"(_add_177), "f"(_shfl_xor_sync_669));
        float _add_178;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_178) : "f"(_add_668), "f"(hc_eps));
        float _mul_34;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(_fdiv_rn_74), "f"(hist[6]));
        float _shfl_xor_sync_670 = __shfl_xor_sync(0xFFFFFFFF, _mul_34, 1);
        float _add_179_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_179_1) : "f"(_mul_34), "f"(_shfl_xor_sync_670));
        float _shfl_xor_sync_673 = __shfl_xor_sync(0xFFFFFFFF, _add_179_1, 2);
        float _add_672;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_672) : "f"(_add_179_1), "f"(_shfl_xor_sync_673));
        float _sub_671;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_671) : "f"(_fdiv_rn_74), "f"(_add_672));
        float _fdiv_rn_75 = __fdiv_rn(_sub_671, _add_178);
        float _dot_674;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_674) : "r"(ring_pk_148[0]), "r"(ring_pk_147[0]), "f"(-0.0f));
        dots_p_149[0] = _dot_674;
        float _dot_675;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_675) : "r"(ring_pk_148[1]), "r"(ring_pk_147[1]), "f"(dots_p_149[0]));
        dots_p_149[0] = _dot_675;
        float _dot_676;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_676) : "r"(ring_pk_148[2]), "r"(ring_pk_147[2]), "f"(dots_p_149[0]));
        dots_p_149[0] = _dot_676;
        float _dot_677;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_677) : "r"(ring_pk_148[3]), "r"(ring_pk_147[3]), "f"(dots_p_149[0]));
        dots_p_149[0] = _dot_677;
        float _dot_678;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_678) : "r"(ring_pk_148[4]), "r"(ring_pk_147[4]), "f"(dots_p_149[0]));
        dots_p_149[0] = _dot_678;
        float _dot_679;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_679) : "r"(ring_pk_148[5]), "r"(ring_pk_147[5]), "f"(dots_p_149[0]));
        dots_p_149[0] = _dot_679;
        float _dot_680;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_680) : "r"(ring_pk_148[6]), "r"(ring_pk_147[6]), "f"(dots_p_149[0]));
        dots_p_149[0] = _dot_680;
        float _dot_681;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_681) : "r"(ring_pk_148[7]), "r"(ring_pk_147[7]), "f"(dots_p_149[0]));
        dots_p_149[0] = _dot_681;
        unsigned int ring_pk_150[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[(0) + 3]))
            : "r"(sa__4 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[(4) + 3]))
            : "r"(sa__4 + 2064 + lane32));
        float _shfl_xor_sync_682 = __shfl_xor_sync(0xFFFFFFFF, hist[4], 4);
        float _add_180_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_180_1) : "f"(hist[4]), "f"(_shfl_xor_sync_682));
        float _shfl_xor_sync_684 = __shfl_xor_sync(0xFFFFFFFF, _add_180_1, 24);
        float _add_683;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_683) : "f"(_add_180_1), "f"(_shfl_xor_sync_684));
        float _add_181;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181) : "f"(_add_683), "f"(hc_eps));
        float _mul_35;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(_fdiv_rn_75), "f"(hist[5]));
        float _shfl_xor_sync_685 = __shfl_xor_sync(0xFFFFFFFF, _mul_35, 4);
        float _add_182;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_182) : "f"(_mul_35), "f"(_shfl_xor_sync_685));
        float _shfl_xor_sync_688 = __shfl_xor_sync(0xFFFFFFFF, _add_182, 24);
        float _add_687;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_687) : "f"(_add_182), "f"(_shfl_xor_sync_688));
        float _sub_686;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_686) : "f"(_fdiv_rn_75), "f"(_add_687));
        float _fdiv_rn_76 = __fdiv_rn(_sub_686, _add_181);
        float _dot_689;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_689) : "r"(ring_pk_150[0]), "r"(ring_pk_147[0]), "f"(-0.0f));
        dots_p_149[1] = _dot_689;
        float _dot_690;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_690) : "r"(ring_pk_150[1]), "r"(ring_pk_147[1]), "f"(dots_p_149[1]));
        dots_p_149[1] = _dot_690;
        float _dot_691;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_691) : "r"(ring_pk_150[2]), "r"(ring_pk_147[2]), "f"(dots_p_149[1]));
        dots_p_149[1] = _dot_691;
        float _dot_692;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_692) : "r"(ring_pk_150[3]), "r"(ring_pk_147[3]), "f"(dots_p_149[1]));
        dots_p_149[1] = _dot_692;
        float _dot_693;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_693) : "r"(ring_pk_150[4]), "r"(ring_pk_147[4]), "f"(dots_p_149[1]));
        dots_p_149[1] = _dot_693;
        float _dot_694;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_694) : "r"(ring_pk_150[5]), "r"(ring_pk_147[5]), "f"(dots_p_149[1]));
        dots_p_149[1] = _dot_694;
        float _dot_695;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_695) : "r"(ring_pk_150[6]), "r"(ring_pk_147[6]), "f"(dots_p_149[1]));
        dots_p_149[1] = _dot_695;
        float _dot_696;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_696) : "r"(ring_pk_150[7]), "r"(ring_pk_147[7]), "f"(dots_p_149[1]));
        dots_p_149[1] = _dot_696;
        unsigned int ring_pk_151[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[(0) + 3]))
            : "r"(sa__4 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[(4) + 3]))
            : "r"(sa__4 + 3088 + lane32));
        float _shfl_xor_sync_697 = __shfl_xor_sync(0xFFFFFFFF, hist[3], 1);
        float _add_183;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183) : "f"(hist[3]), "f"(_shfl_xor_sync_697));
        float _shfl_xor_sync_699 = __shfl_xor_sync(0xFFFFFFFF, _add_183, 2);
        float _add_698;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_698) : "f"(_add_183), "f"(_shfl_xor_sync_699));
        float _add_184;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_184) : "f"(_add_698), "f"(hc_eps));
        float _mul_36;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(_fdiv_rn_76), "f"(hist[4]));
        float _shfl_xor_sync_700 = __shfl_xor_sync(0xFFFFFFFF, _mul_36, 1);
        float _add_185;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_185) : "f"(_mul_36), "f"(_shfl_xor_sync_700));
        float _shfl_xor_sync_703 = __shfl_xor_sync(0xFFFFFFFF, _add_185, 2);
        float _add_702;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_702) : "f"(_add_185), "f"(_shfl_xor_sync_703));
        float _sub_701;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_701) : "f"(_fdiv_rn_76), "f"(_add_702));
        float _fdiv_rn_77 = __fdiv_rn(_sub_701, _add_184);
        float _dot_704;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_704) : "r"(ring_pk_151[0]), "r"(ring_pk_147[0]), "f"(-0.0f));
        dots_p_149[2] = _dot_704;
        float _dot_705;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_705) : "r"(ring_pk_151[1]), "r"(ring_pk_147[1]), "f"(dots_p_149[2]));
        dots_p_149[2] = _dot_705;
        float _dot_706;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_706) : "r"(ring_pk_151[2]), "r"(ring_pk_147[2]), "f"(dots_p_149[2]));
        dots_p_149[2] = _dot_706;
        float _dot_707;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_707) : "r"(ring_pk_151[3]), "r"(ring_pk_147[3]), "f"(dots_p_149[2]));
        dots_p_149[2] = _dot_707;
        float _dot_708;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_708) : "r"(ring_pk_151[4]), "r"(ring_pk_147[4]), "f"(dots_p_149[2]));
        dots_p_149[2] = _dot_708;
        float _dot_709;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_709) : "r"(ring_pk_151[5]), "r"(ring_pk_147[5]), "f"(dots_p_149[2]));
        dots_p_149[2] = _dot_709;
        float _dot_710;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_710) : "r"(ring_pk_151[6]), "r"(ring_pk_147[6]), "f"(dots_p_149[2]));
        dots_p_149[2] = _dot_710;
        float _dot_711;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_711) : "r"(ring_pk_151[7]), "r"(ring_pk_147[7]), "f"(dots_p_149[2]));
        dots_p_149[2] = _dot_711;
        unsigned int ring_pk_152[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[(0) + 3]))
            : "r"(sa__4 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[(4) + 3]))
            : "r"(sa__4 + 4112 + lane32));
        float _shfl_xor_sync_712 = __shfl_xor_sync(0xFFFFFFFF, hist[2], 4);
        float _add_186;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_186) : "f"(hist[2]), "f"(_shfl_xor_sync_712));
        float _shfl_xor_sync_714 = __shfl_xor_sync(0xFFFFFFFF, _add_186, 24);
        float _add_713;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_713) : "f"(_add_186), "f"(_shfl_xor_sync_714));
        float _add_187;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_187) : "f"(_add_713), "f"(hc_eps));
        float _mul_37;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(_fdiv_rn_77), "f"(hist[3]));
        float _shfl_xor_sync_715 = __shfl_xor_sync(0xFFFFFFFF, _mul_37, 4);
        float _add_188;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_188) : "f"(_mul_37), "f"(_shfl_xor_sync_715));
        float _shfl_xor_sync_718 = __shfl_xor_sync(0xFFFFFFFF, _add_188, 24);
        float _add_717;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_717) : "f"(_add_188), "f"(_shfl_xor_sync_718));
        float _sub_716;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_716) : "f"(_fdiv_rn_77), "f"(_add_717));
        float _fdiv_rn_78 = __fdiv_rn(_sub_716, _add_187);
        float _dot_719;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_719) : "r"(ring_pk_152[0]), "r"(ring_pk_147[0]), "f"(-0.0f));
        dots_p_149[3] = _dot_719;
        float _dot_720;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_720) : "r"(ring_pk_152[1]), "r"(ring_pk_147[1]), "f"(dots_p_149[3]));
        dots_p_149[3] = _dot_720;
        float _dot_721;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_721) : "r"(ring_pk_152[2]), "r"(ring_pk_147[2]), "f"(dots_p_149[3]));
        dots_p_149[3] = _dot_721;
        float _dot_722;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_722) : "r"(ring_pk_152[3]), "r"(ring_pk_147[3]), "f"(dots_p_149[3]));
        dots_p_149[3] = _dot_722;
        float _dot_723;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_723) : "r"(ring_pk_152[4]), "r"(ring_pk_147[4]), "f"(dots_p_149[3]));
        dots_p_149[3] = _dot_723;
        float _dot_724;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_724) : "r"(ring_pk_152[5]), "r"(ring_pk_147[5]), "f"(dots_p_149[3]));
        dots_p_149[3] = _dot_724;
        float _dot_725;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_725) : "r"(ring_pk_152[6]), "r"(ring_pk_147[6]), "f"(dots_p_149[3]));
        dots_p_149[3] = _dot_725;
        float _dot_726;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_726) : "r"(ring_pk_152[7]), "r"(ring_pk_147[7]), "f"(dots_p_149[3]));
        dots_p_149[3] = _dot_726;
        __syncwarp();
        bool _elect_sync_2 = elect_sync();
        bool tail_el_153 = _elect_sync_2 && tail_pred;
        if (tail_el_153) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__3) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4), "l"(dU + (th64_53 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 1024), "l"(X + (th64_53 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 2048), "l"(X + ((th64_53 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 3072), "l"(X + ((th64_53 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 4096), "l"(X + ((th64_53 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_727 = __shfl_xor_sync(0xFFFFFFFF, hist[1], 1);
        float _add_189;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_189) : "f"(hist[1]), "f"(_shfl_xor_sync_727));
        float _shfl_xor_sync_729 = __shfl_xor_sync(0xFFFFFFFF, _add_189, 2);
        float _add_728;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_728) : "f"(_add_189), "f"(_shfl_xor_sync_729));
        float _add_190;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_190) : "f"(_add_728), "f"(hc_eps));
        float _mul_38;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(_fdiv_rn_78), "f"(hist[2]));
        float _shfl_xor_sync_730 = __shfl_xor_sync(0xFFFFFFFF, _mul_38, 1);
        float _add_191_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191_1) : "f"(_mul_38), "f"(_shfl_xor_sync_730));
        float _shfl_xor_sync_733 = __shfl_xor_sync(0xFFFFFFFF, _add_191_1, 2);
        float _add_732;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_732) : "f"(_add_191_1), "f"(_shfl_xor_sync_733));
        float _sub_731;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_731) : "f"(_fdiv_rn_78), "f"(_add_732));
        float _fdiv_rn_79 = __fdiv_rn(_sub_731, _add_190);
        int upper_154 = lane_1 & 16;
        float send_155 = ((upper_154 != 0) ? dots_p_149[0] : dots_p_149[2]);
        float keep_156 = ((upper_154 != 0) ? dots_p_149[2] : dots_p_149[0]);
        float _shfl_xor_sync_734 = __shfl_xor_sync(0xFFFFFFFF, send_155, 16);
        float _add_192_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_192_1) : "f"(keep_156), "f"(_shfl_xor_sync_734));
        float send_157 = ((upper_154 != 0) ? dots_p_149[1] : dots_p_149[3]);
        float keep_158 = ((upper_154 != 0) ? dots_p_149[3] : dots_p_149[1]);
        float _shfl_xor_sync_735 = __shfl_xor_sync(0xFFFFFFFF, send_157, 16);
        float _add_193;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_193) : "f"(keep_158), "f"(_shfl_xor_sync_735));
        int upper_159 = lane_1 & 8;
        float send_160 = ((upper_159 != 0) ? _add_192_1 : _add_193);
        float keep_161 = ((upper_159 != 0) ? _add_193 : _add_192_1);
        float _shfl_xor_sync_736 = __shfl_xor_sync(0xFFFFFFFF, send_160, 8);
        float _add_194;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_194) : "f"(keep_161), "f"(_shfl_xor_sync_736));
        float _add_195;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(hist[0]), "f"(hc_eps));
        float _shfl_xor_sync_737 = __shfl_xor_sync(0xFFFFFFFF, _add_195, 4);
        float _add_196;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_196) : "f"(_add_195), "f"(_shfl_xor_sync_737));
        float _shfl_xor_sync_739 = __shfl_xor_sync(0xFFFFFFFF, _add_196, 24);
        float _add_738;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_738) : "f"(_add_196), "f"(_shfl_xor_sync_739));
        float _add_197;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(_add_738), "f"(hc_eps));
        float _mul_39;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(_fdiv_rn_79), "f"(hist[1]));
        float _shfl_xor_sync_740 = __shfl_xor_sync(0xFFFFFFFF, _mul_39, 4);
        float _add_198;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_198) : "f"(_mul_39), "f"(_shfl_xor_sync_740));
        float _shfl_xor_sync_743 = __shfl_xor_sync(0xFFFFFFFF, _add_198, 24);
        float _add_742;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_742) : "f"(_add_198), "f"(_shfl_xor_sync_743));
        float _sub_741;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_741) : "f"(_fdiv_rn_79), "f"(_add_742));
        float _fdiv_rn_80 = __fdiv_rn(_sub_741, _add_197);
        float _shfl_xor_sync_744 = __shfl_xor_sync(0xFFFFFFFF, _add_194, 4);
        float _add_199;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_199) : "f"(_add_194), "f"(_shfl_xor_sync_744));
        float _shfl_xor_sync_745 = __shfl_xor_sync(0xFFFFFFFF, _add_199, 2);
        float _add_200;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_200) : "f"(_add_199), "f"(_shfl_xor_sync_745));
        float _shfl_xor_sync_747 = __shfl_xor_sync(0xFFFFFFFF, _add_200, 1);
        float _add_746;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_746) : "f"(_add_200), "f"(_shfl_xor_sync_747));
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
        float da_163 = ((lane_1 < 24) ? da_late : 0.0f);
        float dz;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz) : "f"(da_163), "f"(sc));
        float daz;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(daz) : "f"(da_163), "f"(z));
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
        float _cvt_f32_0 = (float)(__nv_bfloat16)dzr;
        float _sub_0_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0_1) : "f"(dzr), "f"(_cvt_f32_0));
        float _cvt_f32_1 = (float)(__nv_bfloat16)_sub_0_1;
        float _sub_764;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_764) : "f"(_sub_0_1), "f"(_cvt_f32_1));
        float _cvt_f32_2 = (float)(__nv_bfloat16)_sub_764;
        float p1 = ((lane_1 < 24) ? _cvt_f32_0 : 0.0f);
        float p2 = ((lane_1 < 24) ? _cvt_f32_1 : 0.0f);
        float p3 = ((lane_1 < 24) ? _cvt_f32_2 : 0.0f);
        long long drow = tok64 * 96 + lane64;
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow])), "h"(__bfloat16_as_ushort((__nv_bfloat16)p1)), "r"((unsigned int)(tok < t1)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow + 32])), "h"(__bfloat16_as_ushort((__nv_bfloat16)p2)), "r"((unsigned int)(tok < t1)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow + 64])), "h"(__bfloat16_as_ushort((__nv_bfloat16)p3)), "r"((unsigned int)(tok < t1)) : "memory");
        long long trow = tok64 * 24;
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow])), "f"(r), "r"((unsigned int)(tok < t1 && lane_1 == 0)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + 1])), "f"(c_t), "r"((unsigned int)(tok < t1 && lane_1 == 0)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + 2 + lane64])), "f"(_add_1), "r"((unsigned int)(tok < t1 && lane_1 < 4)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + lane64 - 2])), "f"(_fdiv_rn_41), "r"((unsigned int)(tok < t1 && (lane_1 >= 8 && lane_1 < 24))) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + lane64 - 2])), "f"(0.0f), "r"((unsigned int)(tok < t1 && (lane_1 >= 24 && lane_1 < 26))) : "memory");
        float _sub_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(da_163), "f"(comp[0]));
        float _add_201;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_201) : "f"(acc[0]), "f"(_sub_1));
        float _sub_766;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_766) : "f"(_add_201), "f"(acc[0]));
        float _sub_765;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_765) : "f"(_sub_766), "f"(_sub_1));
        comp[0] = ((tok < t1) ? _sub_765 : comp[0]);
        acc[0] = ((tok < t1) ? _add_201 : acc[0]);
        float _sub_2_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2_1) : "f"(daz), "f"(comp[1]));
        float _add_202;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(acc[1]), "f"(_sub_2_1));
        float _sub_768;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_768) : "f"(_add_202), "f"(acc[1]));
        float _sub_767;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_767) : "f"(_sub_768), "f"(_sub_2_1));
        comp[1] = ((tok < t1) ? _sub_767 : comp[1]);
        acc[1] = ((tok < t1) ? _add_202 : acc[1]);
        gi[0] = gi[0] + 1;
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
            float _sub_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(fold[src]), "f"(oc[0]));
            float _add_203_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_203_1) : "f"(o[0]), "f"(_sub_3));
            float _sub_770;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_770) : "f"(_add_203_1), "f"(o[0]));
            float _sub_769;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_769) : "f"(_sub_770), "f"(_sub_3));
            oc[0] = _sub_769;
            o[0] = _add_203_1;
            float _sub_771;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_771) : "f"(0.0f), "f"(fold[src + 1]));
            float _sub_4;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4) : "f"(_sub_771), "f"(oc[0]));
            float _add_204_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_204_1) : "f"(o[0]), "f"(_sub_4));
            float _sub_773;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_773) : "f"(_add_204_1), "f"(o[0]));
            float _sub_772;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_772) : "f"(_sub_773), "f"(_sub_4));
            oc[0] = _sub_772;
            o[0] = _add_204_1;
            float _sub_5;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(fold[src + 2]), "f"(oc[1]));
            float _add_205;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_205) : "f"(o[1]), "f"(_sub_5));
            float _sub_775;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_775) : "f"(_add_205), "f"(o[1]));
            float _sub_774;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_774) : "f"(_sub_775), "f"(_sub_5));
            oc[1] = _sub_774;
            o[1] = _add_205;
            float _sub_776;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_776) : "f"(0.0f), "f"(fold[src + 3]));
            float _sub_6;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(_sub_776), "f"(oc[1]));
            float _add_206;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_206) : "f"(o[1]), "f"(_sub_6));
            float _sub_778;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_778) : "f"(_add_206), "f"(o[1]));
            float _sub_777;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_777) : "f"(_sub_778), "f"(_sub_6));
            oc[1] = _sub_777;
            o[1] = _add_206;
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
                float _sub_7;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(gsum[mrow]), "f"(gc[0]));
                float _add_207;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_207) : "f"(ga[0]), "f"(_sub_7));
                float _sub_780;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_780) : "f"(_add_207), "f"(ga[0]));
                float _sub_779;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_779) : "f"(_sub_780), "f"(_sub_7));
                gc[0] = _sub_779;
                ga[0] = _add_207;
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
