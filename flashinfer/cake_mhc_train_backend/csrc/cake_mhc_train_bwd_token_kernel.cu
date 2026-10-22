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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


extern "C" __device__ float __nv_expf(float);



extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_mhc_train_bwd_token(__nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ dU, float* __restrict__ stats, float* __restrict__ gstats, float* __restrict__ scale, float* __restrict__ base, __nv_bfloat16* __restrict__ dzp, float* __restrict__ tokp, float* __restrict__ param_partial, int T, int t_pad, int rows_per_cta, float hc_eps)
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

    // === Task calls (dependency order) ===
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
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__3) * 8, 5120);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4), "l"(dU + (tp64_2 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 1024), "l"(X + (tp64_2 * 4 * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 2048), "l"(X + ((tp64_2 * 4 + 1) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 3072), "l"(X + ((tp64_2 * 4 + 2) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__4 + 4096), "l"(X + ((tp64_2 * 4 + 3) * 4096 + wcol)), "r"((unsigned)(1024)), "r"(ring_bar_addr + (st__3) * 8),
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
        int tokc = ((tok < t1) ? tok : t_last);
        long long tok64 = (long long)tokc;
        float dp[1];
        long long srow = tok64 * 64;
        float r = stats[srow];
        float z_load = stats[srow + 1 + n64];
        float z = ((lane_1 < 24) ? z_load : 0.0f);
        float gval = gstats[tok64 * 32 + (long long)gcol];
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(z), "f"(sc), "f"(bs));
        float a_map = _fma_0;
        float a = ((lane_1 < 24) ? a_map : 0.0f);
        float _mul_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(gval), "f"(2.0f));
        float gtwo = _mul_0;
        float hist[40];
        float _fabs_0;
        asm("abs.f32 %0, %1;" : "=f"(_fabs_0) : "f"(a));
        float _sub_0;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(0.0f), "f"(_fabs_0));
        float _exp_0 = __nv_expf(_sub_0);
        float _add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(1.0f), "f"(_exp_0));
        float _fdiv_rn_0 = __fdiv_rn(1.0f, _add_0);
        float _fdiv_rn_1 = __fdiv_rn(_exp_0, _add_0);
        float sig = ((a >= 0.0f) ? _fdiv_rn_0 : _fdiv_rn_1);
        float sneg = ((a >= 0.0f) ? _fdiv_rn_1 : _fdiv_rn_0);
        float _add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(sig), "f"(hc_eps));
        float _mul_1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(2.0f), "f"(sig));
        float logit = ((lane_1 >= 8 && lane_1 < 24) ? a : 0.0f);
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
        float _max_0 = max_noftz(logit, _shfl_xor_0);
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, _max_0, 2);
        float _max_1 = max_noftz(_max_0, _shfl_xor_1);
        float _sub_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(logit), "f"(_max_1));
        float _exp_1 = __nv_expf(_sub_1);
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
        float _add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(_exp_1), "f"(_shfl_xor_2));
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 2);
        float _add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(_add_2), "f"(_shfl_xor_3));
        float _fdiv_rn_2 = __fdiv_rn(_exp_1, _add_3);
        hist[0] = _fdiv_rn_2;
        float _add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(_fdiv_rn_2), "f"(hc_eps));
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, _add_4, 4);
        float _add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(_add_4), "f"(_shfl_xor_4));
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, _add_5, 24);
        float _add_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(_add_5), "f"(_shfl_xor_5));
        float _add_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(_add_6), "f"(hc_eps));
        float _fdiv_rn_3 = __fdiv_rn(_add_4, _add_7);
        hist[1] = _fdiv_rn_3;
        mbarrier_wait(ring_bar_addr + (st_) * 8, 0);
        unsigned int ring_pk[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(0) + 3]))
            : "r"(sa_ + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(4) + 3]))
            : "r"(sa_ + 16 + lane32));
        float _unpack_bf16_0[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk[_pair]) << 16);
            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk[_pair]) & 0xffff0000u);
        }
        unsigned int ring_pk_54[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[(0) + 3]))
            : "r"(sa_ + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_54[(4) + 3]))
            : "r"(sa_ + 1040 + lane32));
        float _unpack_bf16_1[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_54[_pair]) << 16);
            (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_54[_pair]) & 0xffff0000u);
        }
        float dots_p[4];
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 1);
        float _add_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(_fdiv_rn_3), "f"(_shfl_xor_6));
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, _add_8, 2);
        float _add_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(_add_8), "f"(_shfl_xor_7));
        float _add_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(_add_9), "f"(hc_eps));
        float _fdiv_rn_4 = __fdiv_rn(_fdiv_rn_3, _add_10);
        hist[2] = _fdiv_rn_4;
        float _mul_2;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_unpack_bf16_1[0]), "f"(_unpack_bf16_0[0]));
        dots_p[0] = _mul_2;
        float _fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(_unpack_bf16_1[1]), "f"(_unpack_bf16_0[1]), "f"(dots_p[0]));
        dots_p[0] = _fma_1;
        float _fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(_unpack_bf16_1[2]), "f"(_unpack_bf16_0[2]), "f"(dots_p[0]));
        dots_p[0] = _fma_2;
        float _fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(_unpack_bf16_1[3]), "f"(_unpack_bf16_0[3]), "f"(dots_p[0]));
        dots_p[0] = _fma_3;
        float _fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(_unpack_bf16_1[4]), "f"(_unpack_bf16_0[4]), "f"(dots_p[0]));
        dots_p[0] = _fma_4;
        float _fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(_unpack_bf16_1[5]), "f"(_unpack_bf16_0[5]), "f"(dots_p[0]));
        dots_p[0] = _fma_5;
        float _fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(_unpack_bf16_1[6]), "f"(_unpack_bf16_0[6]), "f"(dots_p[0]));
        dots_p[0] = _fma_6;
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(_unpack_bf16_1[7]), "f"(_unpack_bf16_0[7]), "f"(dots_p[0]));
        dots_p[0] = _fma_7;
        float _fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(_unpack_bf16_1[8]), "f"(_unpack_bf16_0[8]), "f"(dots_p[0]));
        dots_p[0] = _fma_8;
        float _fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(_unpack_bf16_1[9]), "f"(_unpack_bf16_0[9]), "f"(dots_p[0]));
        dots_p[0] = _fma_9;
        float _fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(_unpack_bf16_1[10]), "f"(_unpack_bf16_0[10]), "f"(dots_p[0]));
        dots_p[0] = _fma_10;
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(_unpack_bf16_1[11]), "f"(_unpack_bf16_0[11]), "f"(dots_p[0]));
        dots_p[0] = _fma_11;
        float _fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(_unpack_bf16_1[12]), "f"(_unpack_bf16_0[12]), "f"(dots_p[0]));
        dots_p[0] = _fma_12;
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(_unpack_bf16_1[13]), "f"(_unpack_bf16_0[13]), "f"(dots_p[0]));
        dots_p[0] = _fma_13;
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(_unpack_bf16_1[14]), "f"(_unpack_bf16_0[14]), "f"(dots_p[0]));
        dots_p[0] = _fma_14;
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(_unpack_bf16_1[15]), "f"(_unpack_bf16_0[15]), "f"(dots_p[0]));
        dots_p[0] = _fma_15;
        unsigned int ring_pk_55[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[(0) + 3]))
            : "r"(sa_ + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_55[(4) + 3]))
            : "r"(sa_ + 2064 + lane32));
        float _unpack_bf16_2[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_55[_pair]) << 16);
            (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_55[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 4);
        float _add_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(_fdiv_rn_4), "f"(_shfl_xor_8));
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, _add_11, 24);
        float _add_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(_add_11), "f"(_shfl_xor_9));
        float _add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(_add_12), "f"(hc_eps));
        float _fdiv_rn_5 = __fdiv_rn(_fdiv_rn_4, _add_13);
        hist[3] = _fdiv_rn_5;
        float _mul_3;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_unpack_bf16_2[0]), "f"(_unpack_bf16_0[0]));
        dots_p[1] = _mul_3;
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(_unpack_bf16_2[1]), "f"(_unpack_bf16_0[1]), "f"(dots_p[1]));
        dots_p[1] = _fma_16;
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(_unpack_bf16_2[2]), "f"(_unpack_bf16_0[2]), "f"(dots_p[1]));
        dots_p[1] = _fma_17;
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(_unpack_bf16_2[3]), "f"(_unpack_bf16_0[3]), "f"(dots_p[1]));
        dots_p[1] = _fma_18;
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(_unpack_bf16_2[4]), "f"(_unpack_bf16_0[4]), "f"(dots_p[1]));
        dots_p[1] = _fma_19;
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(_unpack_bf16_2[5]), "f"(_unpack_bf16_0[5]), "f"(dots_p[1]));
        dots_p[1] = _fma_20;
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(_unpack_bf16_2[6]), "f"(_unpack_bf16_0[6]), "f"(dots_p[1]));
        dots_p[1] = _fma_21;
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(_unpack_bf16_2[7]), "f"(_unpack_bf16_0[7]), "f"(dots_p[1]));
        dots_p[1] = _fma_22;
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(_unpack_bf16_2[8]), "f"(_unpack_bf16_0[8]), "f"(dots_p[1]));
        dots_p[1] = _fma_23;
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(_unpack_bf16_2[9]), "f"(_unpack_bf16_0[9]), "f"(dots_p[1]));
        dots_p[1] = _fma_24;
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(_unpack_bf16_2[10]), "f"(_unpack_bf16_0[10]), "f"(dots_p[1]));
        dots_p[1] = _fma_25;
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(_unpack_bf16_2[11]), "f"(_unpack_bf16_0[11]), "f"(dots_p[1]));
        dots_p[1] = _fma_26;
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(_unpack_bf16_2[12]), "f"(_unpack_bf16_0[12]), "f"(dots_p[1]));
        dots_p[1] = _fma_27;
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(_unpack_bf16_2[13]), "f"(_unpack_bf16_0[13]), "f"(dots_p[1]));
        dots_p[1] = _fma_28;
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(_unpack_bf16_2[14]), "f"(_unpack_bf16_0[14]), "f"(dots_p[1]));
        dots_p[1] = _fma_29;
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(_unpack_bf16_2[15]), "f"(_unpack_bf16_0[15]), "f"(dots_p[1]));
        dots_p[1] = _fma_30;
        unsigned int ring_pk_56[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[(0) + 3]))
            : "r"(sa_ + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_56[(4) + 3]))
            : "r"(sa_ + 3088 + lane32));
        float _unpack_bf16_3[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_56[_pair]) << 16);
            (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_56[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 1);
        float _add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(_fdiv_rn_5), "f"(_shfl_xor_10));
        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, _add_14, 2);
        float _add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(_add_14), "f"(_shfl_xor_11));
        float _add_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(_add_15), "f"(hc_eps));
        float _fdiv_rn_6 = __fdiv_rn(_fdiv_rn_5, _add_16);
        hist[4] = _fdiv_rn_6;
        float _mul_4;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_unpack_bf16_3[0]), "f"(_unpack_bf16_0[0]));
        dots_p[2] = _mul_4;
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(_unpack_bf16_3[1]), "f"(_unpack_bf16_0[1]), "f"(dots_p[2]));
        dots_p[2] = _fma_31;
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(_unpack_bf16_3[2]), "f"(_unpack_bf16_0[2]), "f"(dots_p[2]));
        dots_p[2] = _fma_32;
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(_unpack_bf16_3[3]), "f"(_unpack_bf16_0[3]), "f"(dots_p[2]));
        dots_p[2] = _fma_33;
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(_unpack_bf16_3[4]), "f"(_unpack_bf16_0[4]), "f"(dots_p[2]));
        dots_p[2] = _fma_34;
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(_unpack_bf16_3[5]), "f"(_unpack_bf16_0[5]), "f"(dots_p[2]));
        dots_p[2] = _fma_35;
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(_unpack_bf16_3[6]), "f"(_unpack_bf16_0[6]), "f"(dots_p[2]));
        dots_p[2] = _fma_36;
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(_unpack_bf16_3[7]), "f"(_unpack_bf16_0[7]), "f"(dots_p[2]));
        dots_p[2] = _fma_37;
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(_unpack_bf16_3[8]), "f"(_unpack_bf16_0[8]), "f"(dots_p[2]));
        dots_p[2] = _fma_38;
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(_unpack_bf16_3[9]), "f"(_unpack_bf16_0[9]), "f"(dots_p[2]));
        dots_p[2] = _fma_39;
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(_unpack_bf16_3[10]), "f"(_unpack_bf16_0[10]), "f"(dots_p[2]));
        dots_p[2] = _fma_40;
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(_unpack_bf16_3[11]), "f"(_unpack_bf16_0[11]), "f"(dots_p[2]));
        dots_p[2] = _fma_41;
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(_unpack_bf16_3[12]), "f"(_unpack_bf16_0[12]), "f"(dots_p[2]));
        dots_p[2] = _fma_42;
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(_unpack_bf16_3[13]), "f"(_unpack_bf16_0[13]), "f"(dots_p[2]));
        dots_p[2] = _fma_43;
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(_unpack_bf16_3[14]), "f"(_unpack_bf16_0[14]), "f"(dots_p[2]));
        dots_p[2] = _fma_44;
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(_unpack_bf16_3[15]), "f"(_unpack_bf16_0[15]), "f"(dots_p[2]));
        dots_p[2] = _fma_45;
        unsigned int ring_pk_57[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[(0) + 3]))
            : "r"(sa_ + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_57[(4) + 3]))
            : "r"(sa_ + 4112 + lane32));
        float _unpack_bf16_4[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_57[_pair]) << 16);
            (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_57[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 4);
        float _add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(_fdiv_rn_6), "f"(_shfl_xor_12));
        float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, _add_17, 24);
        float _add_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(_add_17), "f"(_shfl_xor_13));
        float _add_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(_add_18), "f"(hc_eps));
        float _fdiv_rn_7 = __fdiv_rn(_fdiv_rn_6, _add_19);
        hist[5] = _fdiv_rn_7;
        float _mul_5;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_unpack_bf16_4[0]), "f"(_unpack_bf16_0[0]));
        dots_p[3] = _mul_5;
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(_unpack_bf16_4[1]), "f"(_unpack_bf16_0[1]), "f"(dots_p[3]));
        dots_p[3] = _fma_46;
        float _fma_47;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(_unpack_bf16_4[2]), "f"(_unpack_bf16_0[2]), "f"(dots_p[3]));
        dots_p[3] = _fma_47;
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(_unpack_bf16_4[3]), "f"(_unpack_bf16_0[3]), "f"(dots_p[3]));
        dots_p[3] = _fma_48;
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(_unpack_bf16_4[4]), "f"(_unpack_bf16_0[4]), "f"(dots_p[3]));
        dots_p[3] = _fma_49;
        float _fma_50;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(_unpack_bf16_4[5]), "f"(_unpack_bf16_0[5]), "f"(dots_p[3]));
        dots_p[3] = _fma_50;
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(_unpack_bf16_4[6]), "f"(_unpack_bf16_0[6]), "f"(dots_p[3]));
        dots_p[3] = _fma_51;
        float _fma_52;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(_unpack_bf16_4[7]), "f"(_unpack_bf16_0[7]), "f"(dots_p[3]));
        dots_p[3] = _fma_52;
        float _fma_53;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(_unpack_bf16_4[8]), "f"(_unpack_bf16_0[8]), "f"(dots_p[3]));
        dots_p[3] = _fma_53;
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(_unpack_bf16_4[9]), "f"(_unpack_bf16_0[9]), "f"(dots_p[3]));
        dots_p[3] = _fma_54;
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(_unpack_bf16_4[10]), "f"(_unpack_bf16_0[10]), "f"(dots_p[3]));
        dots_p[3] = _fma_55;
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(_unpack_bf16_4[11]), "f"(_unpack_bf16_0[11]), "f"(dots_p[3]));
        dots_p[3] = _fma_56;
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(_unpack_bf16_4[12]), "f"(_unpack_bf16_0[12]), "f"(dots_p[3]));
        dots_p[3] = _fma_57;
        float _fma_58;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(_unpack_bf16_4[13]), "f"(_unpack_bf16_0[13]), "f"(dots_p[3]));
        dots_p[3] = _fma_58;
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(_unpack_bf16_4[14]), "f"(_unpack_bf16_0[14]), "f"(dots_p[3]));
        dots_p[3] = _fma_59;
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(_unpack_bf16_4[15]), "f"(_unpack_bf16_0[15]), "f"(dots_p[3]));
        dots_p[3] = _fma_60;
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
        float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 1);
        float _add_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(_fdiv_rn_7), "f"(_shfl_xor_14));
        float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, _add_20, 2);
        float _add_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(_add_20), "f"(_shfl_xor_15));
        float _add_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(_add_21), "f"(hc_eps));
        float _fdiv_rn_8 = __fdiv_rn(_fdiv_rn_7, _add_22);
        hist[6] = _fdiv_rn_8;
        int upper = lane_1 & 16;
        float send = ((upper != 0) ? dots_p[0] : dots_p[2]);
        float keep = ((upper != 0) ? dots_p[2] : dots_p[0]);
        float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
        float _add_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(keep), "f"(_shfl_xor_16));
        float send_58 = ((upper != 0) ? dots_p[1] : dots_p[3]);
        float keep_59 = ((upper != 0) ? dots_p[3] : dots_p[1]);
        float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, send_58, 16);
        float _add_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(keep_59), "f"(_shfl_xor_17));
        int upper_60 = lane_1 & 8;
        float send_61 = ((upper_60 != 0) ? _add_23 : _add_24);
        float keep_62 = ((upper_60 != 0) ? _add_24 : _add_23);
        float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, send_61, 8);
        float _add_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(keep_62), "f"(_shfl_xor_18));
        float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 4);
        float _add_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(_fdiv_rn_8), "f"(_shfl_xor_19));
        float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, _add_26, 24);
        float _add_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(_add_26), "f"(_shfl_xor_20));
        float _add_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(_add_27), "f"(hc_eps));
        float _fdiv_rn_9 = __fdiv_rn(_fdiv_rn_8, _add_28);
        hist[7] = _fdiv_rn_9;
        float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, _add_25, 4);
        float _add_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(_add_25), "f"(_shfl_xor_21));
        float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, _add_29, 2);
        float _add_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(_add_29), "f"(_shfl_xor_22));
        float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, _add_30, 1);
        float _add_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(_add_30), "f"(_shfl_xor_23));
        dpre_part[own_base] = _add_31;
        float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 1);
        float _add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(_fdiv_rn_9), "f"(_shfl_xor_24));
        float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, _add_32, 2);
        float _add_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(_add_32), "f"(_shfl_xor_25));
        float _add_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(_add_33), "f"(hc_eps));
        float _fdiv_rn_10 = __fdiv_rn(_fdiv_rn_9, _add_34);
        hist[8] = _fdiv_rn_10;
        float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 4);
        float _add_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(_fdiv_rn_10), "f"(_shfl_xor_26));
        float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, _add_35, 24);
        float _add_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(_add_35), "f"(_shfl_xor_27));
        float _add_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(_add_36), "f"(hc_eps));
        float _fdiv_rn_11 = __fdiv_rn(_fdiv_rn_10, _add_37);
        hist[9] = _fdiv_rn_11;
        float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 1);
        float _add_38;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(_fdiv_rn_11), "f"(_shfl_xor_28));
        float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, _add_38, 2);
        float _add_39;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(_add_38), "f"(_shfl_xor_29));
        float _add_40;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(_add_39), "f"(hc_eps));
        float _fdiv_rn_12 = __fdiv_rn(_fdiv_rn_11, _add_40);
        hist[10] = _fdiv_rn_12;
        mbarrier_wait(ring_bar_addr + (st__3) * 8, 0);
        unsigned int ring_pk_63[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[(0) + 3]))
            : "r"(sa__4 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_63[(4) + 3]))
            : "r"(sa__4 + 16 + lane32));
        float _unpack_bf16_5[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_63[_pair]) << 16);
            (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_63[_pair]) & 0xffff0000u);
        }
        unsigned int ring_pk_64[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[(0) + 3]))
            : "r"(sa__4 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_64[(4) + 3]))
            : "r"(sa__4 + 1040 + lane32));
        float _unpack_bf16_6[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_64[_pair]) << 16);
            (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_64[_pair]) & 0xffff0000u);
        }
        float dots_p_65[4];
        float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 4);
        float _add_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(_fdiv_rn_12), "f"(_shfl_xor_30));
        float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, _add_41, 24);
        float _add_42;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(_add_41), "f"(_shfl_xor_31));
        float _add_43;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(_add_42), "f"(hc_eps));
        float _fdiv_rn_13 = __fdiv_rn(_fdiv_rn_12, _add_43);
        hist[11] = _fdiv_rn_13;
        float _mul_6;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_unpack_bf16_6[0]), "f"(_unpack_bf16_5[0]));
        dots_p_65[0] = _mul_6;
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(_unpack_bf16_6[1]), "f"(_unpack_bf16_5[1]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_61;
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(_unpack_bf16_6[2]), "f"(_unpack_bf16_5[2]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_62;
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(_unpack_bf16_6[3]), "f"(_unpack_bf16_5[3]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_63;
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(_unpack_bf16_6[4]), "f"(_unpack_bf16_5[4]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_64;
        float _fma_65;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(_unpack_bf16_6[5]), "f"(_unpack_bf16_5[5]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_65;
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(_unpack_bf16_6[6]), "f"(_unpack_bf16_5[6]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_66;
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(_unpack_bf16_6[7]), "f"(_unpack_bf16_5[7]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_67;
        float _fma_68;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(_unpack_bf16_6[8]), "f"(_unpack_bf16_5[8]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_68;
        float _fma_69;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(_unpack_bf16_6[9]), "f"(_unpack_bf16_5[9]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_69;
        float _fma_70;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(_unpack_bf16_6[10]), "f"(_unpack_bf16_5[10]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_70;
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(_unpack_bf16_6[11]), "f"(_unpack_bf16_5[11]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_71;
        float _fma_72;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(_unpack_bf16_6[12]), "f"(_unpack_bf16_5[12]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_72;
        float _fma_73;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(_unpack_bf16_6[13]), "f"(_unpack_bf16_5[13]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_73;
        float _fma_74;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(_unpack_bf16_6[14]), "f"(_unpack_bf16_5[14]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_74;
        float _fma_75;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(_unpack_bf16_6[15]), "f"(_unpack_bf16_5[15]), "f"(dots_p_65[0]));
        dots_p_65[0] = _fma_75;
        unsigned int ring_pk_66[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[(0) + 3]))
            : "r"(sa__4 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_66[(4) + 3]))
            : "r"(sa__4 + 2064 + lane32));
        float _unpack_bf16_7[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_66[_pair]) << 16);
            (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_66[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 1);
        float _add_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(_fdiv_rn_13), "f"(_shfl_xor_32));
        float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, _add_44, 2);
        float _add_45;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(_add_44), "f"(_shfl_xor_33));
        float _add_46;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(_add_45), "f"(hc_eps));
        float _fdiv_rn_14 = __fdiv_rn(_fdiv_rn_13, _add_46);
        hist[12] = _fdiv_rn_14;
        float _mul_7;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_unpack_bf16_7[0]), "f"(_unpack_bf16_5[0]));
        dots_p_65[1] = _mul_7;
        float _fma_76;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(_unpack_bf16_7[1]), "f"(_unpack_bf16_5[1]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_76;
        float _fma_77;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(_unpack_bf16_7[2]), "f"(_unpack_bf16_5[2]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_77;
        float _fma_78;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(_unpack_bf16_7[3]), "f"(_unpack_bf16_5[3]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_78;
        float _fma_79;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(_unpack_bf16_7[4]), "f"(_unpack_bf16_5[4]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_79;
        float _fma_80;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(_unpack_bf16_7[5]), "f"(_unpack_bf16_5[5]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_80;
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(_unpack_bf16_7[6]), "f"(_unpack_bf16_5[6]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_81;
        float _fma_82;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(_unpack_bf16_7[7]), "f"(_unpack_bf16_5[7]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_82;
        float _fma_83;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(_unpack_bf16_7[8]), "f"(_unpack_bf16_5[8]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_83;
        float _fma_84;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(_unpack_bf16_7[9]), "f"(_unpack_bf16_5[9]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_84;
        float _fma_85;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(_unpack_bf16_7[10]), "f"(_unpack_bf16_5[10]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_85;
        float _fma_86;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(_unpack_bf16_7[11]), "f"(_unpack_bf16_5[11]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_86;
        float _fma_87;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(_unpack_bf16_7[12]), "f"(_unpack_bf16_5[12]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_87;
        float _fma_88;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(_unpack_bf16_7[13]), "f"(_unpack_bf16_5[13]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_88;
        float _fma_89;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(_unpack_bf16_7[14]), "f"(_unpack_bf16_5[14]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_89;
        float _fma_90;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(_unpack_bf16_7[15]), "f"(_unpack_bf16_5[15]), "f"(dots_p_65[1]));
        dots_p_65[1] = _fma_90;
        unsigned int ring_pk_67[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[(0) + 3]))
            : "r"(sa__4 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_67[(4) + 3]))
            : "r"(sa__4 + 3088 + lane32));
        float _unpack_bf16_8[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_67[_pair]) << 16);
            (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_67[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 4);
        float _add_47;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(_fdiv_rn_14), "f"(_shfl_xor_34));
        float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, _add_47, 24);
        float _add_48;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(_add_47), "f"(_shfl_xor_35));
        float _add_49;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(_add_48), "f"(hc_eps));
        float _fdiv_rn_15 = __fdiv_rn(_fdiv_rn_14, _add_49);
        hist[13] = _fdiv_rn_15;
        float _mul_8;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_unpack_bf16_8[0]), "f"(_unpack_bf16_5[0]));
        dots_p_65[2] = _mul_8;
        float _fma_91;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(_unpack_bf16_8[1]), "f"(_unpack_bf16_5[1]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_91;
        float _fma_92;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(_unpack_bf16_8[2]), "f"(_unpack_bf16_5[2]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_92;
        float _fma_93;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(_unpack_bf16_8[3]), "f"(_unpack_bf16_5[3]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_93;
        float _fma_94;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(_unpack_bf16_8[4]), "f"(_unpack_bf16_5[4]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_94;
        float _fma_95;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(_unpack_bf16_8[5]), "f"(_unpack_bf16_5[5]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_95;
        float _fma_96;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(_unpack_bf16_8[6]), "f"(_unpack_bf16_5[6]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_96;
        float _fma_97;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(_unpack_bf16_8[7]), "f"(_unpack_bf16_5[7]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_97;
        float _fma_98;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(_unpack_bf16_8[8]), "f"(_unpack_bf16_5[8]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_98;
        float _fma_99;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(_unpack_bf16_8[9]), "f"(_unpack_bf16_5[9]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_99;
        float _fma_100;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(_unpack_bf16_8[10]), "f"(_unpack_bf16_5[10]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_100;
        float _fma_101;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(_unpack_bf16_8[11]), "f"(_unpack_bf16_5[11]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_101;
        float _fma_102;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(_unpack_bf16_8[12]), "f"(_unpack_bf16_5[12]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_102;
        float _fma_103;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(_unpack_bf16_8[13]), "f"(_unpack_bf16_5[13]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_103;
        float _fma_104;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(_unpack_bf16_8[14]), "f"(_unpack_bf16_5[14]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_104;
        float _fma_105;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(_unpack_bf16_8[15]), "f"(_unpack_bf16_5[15]), "f"(dots_p_65[2]));
        dots_p_65[2] = _fma_105;
        unsigned int ring_pk_68[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[(0) + 3]))
            : "r"(sa__4 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_68[(4) + 3]))
            : "r"(sa__4 + 4112 + lane32));
        float _unpack_bf16_9[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_68[_pair]) << 16);
            (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_68[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 1);
        float _add_50;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(_fdiv_rn_15), "f"(_shfl_xor_36));
        float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, _add_50, 2);
        float _add_51;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(_add_50), "f"(_shfl_xor_37));
        float _add_52;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(_add_51), "f"(hc_eps));
        float _fdiv_rn_16 = __fdiv_rn(_fdiv_rn_15, _add_52);
        hist[14] = _fdiv_rn_16;
        float _mul_9;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_unpack_bf16_9[0]), "f"(_unpack_bf16_5[0]));
        dots_p_65[3] = _mul_9;
        float _fma_106;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(_unpack_bf16_9[1]), "f"(_unpack_bf16_5[1]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_106;
        float _fma_107;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(_unpack_bf16_9[2]), "f"(_unpack_bf16_5[2]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_107;
        float _fma_108;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(_unpack_bf16_9[3]), "f"(_unpack_bf16_5[3]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_108;
        float _fma_109;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(_unpack_bf16_9[4]), "f"(_unpack_bf16_5[4]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_109;
        float _fma_110;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(_unpack_bf16_9[5]), "f"(_unpack_bf16_5[5]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_110;
        float _fma_111;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(_unpack_bf16_9[6]), "f"(_unpack_bf16_5[6]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_111;
        float _fma_112;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(_unpack_bf16_9[7]), "f"(_unpack_bf16_5[7]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_112;
        float _fma_113;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(_unpack_bf16_9[8]), "f"(_unpack_bf16_5[8]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_113;
        float _fma_114;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(_unpack_bf16_9[9]), "f"(_unpack_bf16_5[9]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_114;
        float _fma_115;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(_unpack_bf16_9[10]), "f"(_unpack_bf16_5[10]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_115;
        float _fma_116;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(_unpack_bf16_9[11]), "f"(_unpack_bf16_5[11]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_116;
        float _fma_117;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(_unpack_bf16_9[12]), "f"(_unpack_bf16_5[12]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_117;
        float _fma_118;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(_unpack_bf16_9[13]), "f"(_unpack_bf16_5[13]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_118;
        float _fma_119;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(_unpack_bf16_9[14]), "f"(_unpack_bf16_5[14]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_119;
        float _fma_120;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(_unpack_bf16_9[15]), "f"(_unpack_bf16_5[15]), "f"(dots_p_65[3]));
        dots_p_65[3] = _fma_120;
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
        float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 4);
        float _add_53;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(_fdiv_rn_16), "f"(_shfl_xor_38));
        float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, _add_53, 24);
        float _add_54;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(_add_53), "f"(_shfl_xor_39));
        float _add_55;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(_add_54), "f"(hc_eps));
        float _fdiv_rn_17 = __fdiv_rn(_fdiv_rn_16, _add_55);
        hist[15] = _fdiv_rn_17;
        int upper_69 = lane_1 & 16;
        float send_70 = ((upper_69 != 0) ? dots_p_65[0] : dots_p_65[2]);
        float keep_71 = ((upper_69 != 0) ? dots_p_65[2] : dots_p_65[0]);
        float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, send_70, 16);
        float _add_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(keep_71), "f"(_shfl_xor_40));
        float send_72 = ((upper_69 != 0) ? dots_p_65[1] : dots_p_65[3]);
        float keep_73 = ((upper_69 != 0) ? dots_p_65[3] : dots_p_65[1]);
        float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, send_72, 16);
        float _add_57;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(keep_73), "f"(_shfl_xor_41));
        int upper_74 = lane_1 & 8;
        float send_75 = ((upper_74 != 0) ? _add_56 : _add_57);
        float keep_76 = ((upper_74 != 0) ? _add_57 : _add_56);
        float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, send_75, 8);
        float _add_58;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(keep_76), "f"(_shfl_xor_42));
        float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 1);
        float _add_59;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(_fdiv_rn_17), "f"(_shfl_xor_43));
        float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, _add_59, 2);
        float _add_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(_add_59), "f"(_shfl_xor_44));
        float _add_61;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(_add_60), "f"(hc_eps));
        float _fdiv_rn_18 = __fdiv_rn(_fdiv_rn_17, _add_61);
        hist[16] = _fdiv_rn_18;
        float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, _add_58, 4);
        float _add_62;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(_add_58), "f"(_shfl_xor_45));
        float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, _add_62, 2);
        float _add_63;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(_add_62), "f"(_shfl_xor_46));
        float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, _add_63, 1);
        float _add_64;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(_add_63), "f"(_shfl_xor_47));
        dpre_part[own_base + 32] = _add_64;
        float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 4);
        float _add_65;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(_fdiv_rn_18), "f"(_shfl_xor_48));
        float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, _add_65, 24);
        float _add_66;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(_add_65), "f"(_shfl_xor_49));
        float _add_67;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(_add_66), "f"(hc_eps));
        float _fdiv_rn_19 = __fdiv_rn(_fdiv_rn_18, _add_67);
        hist[17] = _fdiv_rn_19;
        float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 1);
        float _add_68;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(_fdiv_rn_19), "f"(_shfl_xor_50));
        float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, _add_68, 2);
        float _add_69;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(_add_68), "f"(_shfl_xor_51));
        float _add_70;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(_add_69), "f"(hc_eps));
        float _fdiv_rn_20 = __fdiv_rn(_fdiv_rn_19, _add_70);
        hist[18] = _fdiv_rn_20;
        float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 4);
        float _add_71;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(_fdiv_rn_20), "f"(_shfl_xor_52));
        float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, _add_71, 24);
        float _add_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(_add_71), "f"(_shfl_xor_53));
        float _add_73;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(_add_72), "f"(hc_eps));
        float _fdiv_rn_21 = __fdiv_rn(_fdiv_rn_20, _add_73);
        hist[19] = _fdiv_rn_21;
        mbarrier_wait(ring_bar_addr + (st_) * 8, 1);
        unsigned int ring_pk_77[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[(0) + 3]))
            : "r"(sa_ + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_77[(4) + 3]))
            : "r"(sa_ + 16 + lane32));
        float _unpack_bf16_10[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_77[_pair]) << 16);
            (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_77[_pair]) & 0xffff0000u);
        }
        unsigned int ring_pk_78[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[(0) + 3]))
            : "r"(sa_ + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_78[(4) + 3]))
            : "r"(sa_ + 1040 + lane32));
        float _unpack_bf16_11[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_78[_pair]) << 16);
            (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_78[_pair]) & 0xffff0000u);
        }
        float dots_p_79[4];
        float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 1);
        float _add_74;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(_fdiv_rn_21), "f"(_shfl_xor_54));
        float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, _add_74, 2);
        float _add_75;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(_add_74), "f"(_shfl_xor_55));
        float _add_76;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(_add_75), "f"(hc_eps));
        float _fdiv_rn_22 = __fdiv_rn(_fdiv_rn_21, _add_76);
        hist[20] = _fdiv_rn_22;
        float _mul_10;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(_unpack_bf16_11[0]), "f"(_unpack_bf16_10[0]));
        dots_p_79[0] = _mul_10;
        float _fma_121;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(_unpack_bf16_11[1]), "f"(_unpack_bf16_10[1]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_121;
        float _fma_122;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(_unpack_bf16_11[2]), "f"(_unpack_bf16_10[2]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_122;
        float _fma_123;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(_unpack_bf16_11[3]), "f"(_unpack_bf16_10[3]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_123;
        float _fma_124;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(_unpack_bf16_11[4]), "f"(_unpack_bf16_10[4]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_124;
        float _fma_125;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(_unpack_bf16_11[5]), "f"(_unpack_bf16_10[5]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_125;
        float _fma_126;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(_unpack_bf16_11[6]), "f"(_unpack_bf16_10[6]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_126;
        float _fma_127;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(_unpack_bf16_11[7]), "f"(_unpack_bf16_10[7]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_127;
        float _fma_128;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(_unpack_bf16_11[8]), "f"(_unpack_bf16_10[8]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_128;
        float _fma_129;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(_unpack_bf16_11[9]), "f"(_unpack_bf16_10[9]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_129;
        float _fma_130;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(_unpack_bf16_11[10]), "f"(_unpack_bf16_10[10]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_130;
        float _fma_131;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(_unpack_bf16_11[11]), "f"(_unpack_bf16_10[11]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_131;
        float _fma_132;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(_unpack_bf16_11[12]), "f"(_unpack_bf16_10[12]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_132;
        float _fma_133;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(_unpack_bf16_11[13]), "f"(_unpack_bf16_10[13]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_133;
        float _fma_134;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(_unpack_bf16_11[14]), "f"(_unpack_bf16_10[14]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_134;
        float _fma_135;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(_unpack_bf16_11[15]), "f"(_unpack_bf16_10[15]), "f"(dots_p_79[0]));
        dots_p_79[0] = _fma_135;
        unsigned int ring_pk_80[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[(0) + 3]))
            : "r"(sa_ + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_80[(4) + 3]))
            : "r"(sa_ + 2064 + lane32));
        float _unpack_bf16_12[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_80[_pair]) << 16);
            (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_80[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 4);
        float _add_77;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(_fdiv_rn_22), "f"(_shfl_xor_56));
        float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, _add_77, 24);
        float _add_78;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(_add_77), "f"(_shfl_xor_57));
        float _add_79;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(_add_78), "f"(hc_eps));
        float _fdiv_rn_23 = __fdiv_rn(_fdiv_rn_22, _add_79);
        hist[21] = _fdiv_rn_23;
        float _mul_11;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_unpack_bf16_12[0]), "f"(_unpack_bf16_10[0]));
        dots_p_79[1] = _mul_11;
        float _fma_136;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(_unpack_bf16_12[1]), "f"(_unpack_bf16_10[1]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_136;
        float _fma_137;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(_unpack_bf16_12[2]), "f"(_unpack_bf16_10[2]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_137;
        float _fma_138;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(_unpack_bf16_12[3]), "f"(_unpack_bf16_10[3]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_138;
        float _fma_139;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(_unpack_bf16_12[4]), "f"(_unpack_bf16_10[4]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_139;
        float _fma_140;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(_unpack_bf16_12[5]), "f"(_unpack_bf16_10[5]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_140;
        float _fma_141;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(_unpack_bf16_12[6]), "f"(_unpack_bf16_10[6]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_141;
        float _fma_142;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(_unpack_bf16_12[7]), "f"(_unpack_bf16_10[7]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_142;
        float _fma_143;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(_unpack_bf16_12[8]), "f"(_unpack_bf16_10[8]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_143;
        float _fma_144;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(_unpack_bf16_12[9]), "f"(_unpack_bf16_10[9]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_144;
        float _fma_145;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(_unpack_bf16_12[10]), "f"(_unpack_bf16_10[10]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_145;
        float _fma_146;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(_unpack_bf16_12[11]), "f"(_unpack_bf16_10[11]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_146;
        float _fma_147;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(_unpack_bf16_12[12]), "f"(_unpack_bf16_10[12]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_147;
        float _fma_148;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(_unpack_bf16_12[13]), "f"(_unpack_bf16_10[13]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_148;
        float _fma_149;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(_unpack_bf16_12[14]), "f"(_unpack_bf16_10[14]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_149;
        float _fma_150;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(_unpack_bf16_12[15]), "f"(_unpack_bf16_10[15]), "f"(dots_p_79[1]));
        dots_p_79[1] = _fma_150;
        unsigned int ring_pk_81[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[(0) + 3]))
            : "r"(sa_ + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_81[(4) + 3]))
            : "r"(sa_ + 3088 + lane32));
        float _unpack_bf16_13[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_81[_pair]) << 16);
            (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_81[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 1);
        float _add_80;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(_fdiv_rn_23), "f"(_shfl_xor_58));
        float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, _add_80, 2);
        float _add_81;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(_add_80), "f"(_shfl_xor_59));
        float _add_82;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(_add_81), "f"(hc_eps));
        float _fdiv_rn_24 = __fdiv_rn(_fdiv_rn_23, _add_82);
        hist[22] = _fdiv_rn_24;
        float _mul_12;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(_unpack_bf16_13[0]), "f"(_unpack_bf16_10[0]));
        dots_p_79[2] = _mul_12;
        float _fma_151;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(_unpack_bf16_13[1]), "f"(_unpack_bf16_10[1]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_151;
        float _fma_152;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(_unpack_bf16_13[2]), "f"(_unpack_bf16_10[2]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_152;
        float _fma_153;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(_unpack_bf16_13[3]), "f"(_unpack_bf16_10[3]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_153;
        float _fma_154;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_154) : "f"(_unpack_bf16_13[4]), "f"(_unpack_bf16_10[4]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_154;
        float _fma_155;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(_unpack_bf16_13[5]), "f"(_unpack_bf16_10[5]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_155;
        float _fma_156;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(_unpack_bf16_13[6]), "f"(_unpack_bf16_10[6]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_156;
        float _fma_157;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(_unpack_bf16_13[7]), "f"(_unpack_bf16_10[7]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_157;
        float _fma_158;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(_unpack_bf16_13[8]), "f"(_unpack_bf16_10[8]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_158;
        float _fma_159;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(_unpack_bf16_13[9]), "f"(_unpack_bf16_10[9]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_159;
        float _fma_160;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(_unpack_bf16_13[10]), "f"(_unpack_bf16_10[10]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_160;
        float _fma_161;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(_unpack_bf16_13[11]), "f"(_unpack_bf16_10[11]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_161;
        float _fma_162;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(_unpack_bf16_13[12]), "f"(_unpack_bf16_10[12]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_162;
        float _fma_163;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(_unpack_bf16_13[13]), "f"(_unpack_bf16_10[13]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_163;
        float _fma_164;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(_unpack_bf16_13[14]), "f"(_unpack_bf16_10[14]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_164;
        float _fma_165;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(_unpack_bf16_13[15]), "f"(_unpack_bf16_10[15]), "f"(dots_p_79[2]));
        dots_p_79[2] = _fma_165;
        unsigned int ring_pk_82[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[(0) + 3]))
            : "r"(sa_ + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_82[(4) + 3]))
            : "r"(sa_ + 4112 + lane32));
        float _unpack_bf16_14[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_82[_pair]) << 16);
            (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_82[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 4);
        float _add_83;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(_fdiv_rn_24), "f"(_shfl_xor_60));
        float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, _add_83, 24);
        float _add_84;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(_add_83), "f"(_shfl_xor_61));
        float _add_85;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(_add_84), "f"(hc_eps));
        float _fdiv_rn_25 = __fdiv_rn(_fdiv_rn_24, _add_85);
        hist[23] = _fdiv_rn_25;
        float _mul_13;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(_unpack_bf16_14[0]), "f"(_unpack_bf16_10[0]));
        dots_p_79[3] = _mul_13;
        float _fma_166;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(_unpack_bf16_14[1]), "f"(_unpack_bf16_10[1]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_166;
        float _fma_167;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(_unpack_bf16_14[2]), "f"(_unpack_bf16_10[2]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_167;
        float _fma_168;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(_unpack_bf16_14[3]), "f"(_unpack_bf16_10[3]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_168;
        float _fma_169;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(_unpack_bf16_14[4]), "f"(_unpack_bf16_10[4]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_169;
        float _fma_170;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(_unpack_bf16_14[5]), "f"(_unpack_bf16_10[5]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_170;
        float _fma_171;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(_unpack_bf16_14[6]), "f"(_unpack_bf16_10[6]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_171;
        float _fma_172;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(_unpack_bf16_14[7]), "f"(_unpack_bf16_10[7]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_172;
        float _fma_173;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(_unpack_bf16_14[8]), "f"(_unpack_bf16_10[8]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_173;
        float _fma_174;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(_unpack_bf16_14[9]), "f"(_unpack_bf16_10[9]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_174;
        float _fma_175;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(_unpack_bf16_14[10]), "f"(_unpack_bf16_10[10]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_175;
        float _fma_176;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(_unpack_bf16_14[11]), "f"(_unpack_bf16_10[11]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_176;
        float _fma_177;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(_unpack_bf16_14[12]), "f"(_unpack_bf16_10[12]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_177;
        float _fma_178;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(_unpack_bf16_14[13]), "f"(_unpack_bf16_10[13]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_178;
        float _fma_179;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(_unpack_bf16_14[14]), "f"(_unpack_bf16_10[14]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_179;
        float _fma_180;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(_unpack_bf16_14[15]), "f"(_unpack_bf16_10[15]), "f"(dots_p_79[3]));
        dots_p_79[3] = _fma_180;
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
        float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 1);
        float _add_86;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(_fdiv_rn_25), "f"(_shfl_xor_62));
        float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, _add_86, 2);
        float _add_87;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(_add_86), "f"(_shfl_xor_63));
        float _add_88;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(_add_87), "f"(hc_eps));
        float _fdiv_rn_26 = __fdiv_rn(_fdiv_rn_25, _add_88);
        hist[24] = _fdiv_rn_26;
        int upper_83 = lane_1 & 16;
        float send_84 = ((upper_83 != 0) ? dots_p_79[0] : dots_p_79[2]);
        float keep_85 = ((upper_83 != 0) ? dots_p_79[2] : dots_p_79[0]);
        float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, send_84, 16);
        float _add_89;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(keep_85), "f"(_shfl_xor_64));
        float send_86 = ((upper_83 != 0) ? dots_p_79[1] : dots_p_79[3]);
        float keep_87 = ((upper_83 != 0) ? dots_p_79[3] : dots_p_79[1]);
        float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, send_86, 16);
        float _add_90;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_90) : "f"(keep_87), "f"(_shfl_xor_65));
        int upper_88 = lane_1 & 8;
        float send_89 = ((upper_88 != 0) ? _add_89 : _add_90);
        float keep_90 = ((upper_88 != 0) ? _add_90 : _add_89);
        float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, send_89, 8);
        float _add_91;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(keep_90), "f"(_shfl_xor_66));
        float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 4);
        float _add_92;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(_fdiv_rn_26), "f"(_shfl_xor_67));
        float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, _add_92, 24);
        float _add_93;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(_add_92), "f"(_shfl_xor_68));
        float _add_94;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(_add_93), "f"(hc_eps));
        float _fdiv_rn_27 = __fdiv_rn(_fdiv_rn_26, _add_94);
        hist[25] = _fdiv_rn_27;
        float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, _add_91, 4);
        float _add_95;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(_add_91), "f"(_shfl_xor_69));
        float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, _add_95, 2);
        float _add_96;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(_add_95), "f"(_shfl_xor_70));
        float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, _add_96, 1);
        float _add_97;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(_add_96), "f"(_shfl_xor_71));
        dpre_part[own_base + 64] = _add_97;
        float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 1);
        float _add_98;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(_fdiv_rn_27), "f"(_shfl_xor_72));
        float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, _add_98, 2);
        float _add_99;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(_add_98), "f"(_shfl_xor_73));
        float _add_100;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(_add_99), "f"(hc_eps));
        float _fdiv_rn_28 = __fdiv_rn(_fdiv_rn_27, _add_100);
        hist[26] = _fdiv_rn_28;
        float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 4);
        float _add_101;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(_fdiv_rn_28), "f"(_shfl_xor_74));
        float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, _add_101, 24);
        float _add_102;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(_add_101), "f"(_shfl_xor_75));
        float _add_103;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(_add_102), "f"(hc_eps));
        float _fdiv_rn_29 = __fdiv_rn(_fdiv_rn_28, _add_103);
        hist[27] = _fdiv_rn_29;
        float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 1);
        float _add_104;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(_fdiv_rn_29), "f"(_shfl_xor_76));
        float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, _add_104, 2);
        float _add_105;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_105) : "f"(_add_104), "f"(_shfl_xor_77));
        float _add_106;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(_add_105), "f"(hc_eps));
        float _fdiv_rn_30 = __fdiv_rn(_fdiv_rn_29, _add_106);
        hist[28] = _fdiv_rn_30;
        mbarrier_wait(ring_bar_addr + (st__3) * 8, 1);
        unsigned int ring_pk_91[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[(0) + 3]))
            : "r"(sa__4 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_91[(4) + 3]))
            : "r"(sa__4 + 16 + lane32));
        float _unpack_bf16_15[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_91[_pair]) << 16);
            (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_91[_pair]) & 0xffff0000u);
        }
        unsigned int ring_pk_92[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[(0) + 3]))
            : "r"(sa__4 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_92[(4) + 3]))
            : "r"(sa__4 + 1040 + lane32));
        float _unpack_bf16_16[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_92[_pair]) << 16);
            (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_92[_pair]) & 0xffff0000u);
        }
        float dots_p_93[4];
        float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 4);
        float _add_107;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(_fdiv_rn_30), "f"(_shfl_xor_78));
        float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, _add_107, 24);
        float _add_108;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(_add_107), "f"(_shfl_xor_79));
        float _add_109;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(_add_108), "f"(hc_eps));
        float _fdiv_rn_31 = __fdiv_rn(_fdiv_rn_30, _add_109);
        hist[29] = _fdiv_rn_31;
        float _mul_14;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(_unpack_bf16_16[0]), "f"(_unpack_bf16_15[0]));
        dots_p_93[0] = _mul_14;
        float _fma_181;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(_unpack_bf16_16[1]), "f"(_unpack_bf16_15[1]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_181;
        float _fma_182;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(_unpack_bf16_16[2]), "f"(_unpack_bf16_15[2]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_182;
        float _fma_183;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(_unpack_bf16_16[3]), "f"(_unpack_bf16_15[3]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_183;
        float _fma_184;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(_unpack_bf16_16[4]), "f"(_unpack_bf16_15[4]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_184;
        float _fma_185;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_185) : "f"(_unpack_bf16_16[5]), "f"(_unpack_bf16_15[5]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_185;
        float _fma_186;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(_unpack_bf16_16[6]), "f"(_unpack_bf16_15[6]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_186;
        float _fma_187;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_187) : "f"(_unpack_bf16_16[7]), "f"(_unpack_bf16_15[7]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_187;
        float _fma_188;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_188) : "f"(_unpack_bf16_16[8]), "f"(_unpack_bf16_15[8]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_188;
        float _fma_189;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(_unpack_bf16_16[9]), "f"(_unpack_bf16_15[9]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_189;
        float _fma_190;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_190) : "f"(_unpack_bf16_16[10]), "f"(_unpack_bf16_15[10]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_190;
        float _fma_191;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_191) : "f"(_unpack_bf16_16[11]), "f"(_unpack_bf16_15[11]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_191;
        float _fma_192;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(_unpack_bf16_16[12]), "f"(_unpack_bf16_15[12]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_192;
        float _fma_193;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_193) : "f"(_unpack_bf16_16[13]), "f"(_unpack_bf16_15[13]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_193;
        float _fma_194;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(_unpack_bf16_16[14]), "f"(_unpack_bf16_15[14]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_194;
        float _fma_195;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(_unpack_bf16_16[15]), "f"(_unpack_bf16_15[15]), "f"(dots_p_93[0]));
        dots_p_93[0] = _fma_195;
        unsigned int ring_pk_94[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[(0) + 3]))
            : "r"(sa__4 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_94[(4) + 3]))
            : "r"(sa__4 + 2064 + lane32));
        float _unpack_bf16_17[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_94[_pair]) << 16);
            (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_94[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 1);
        float _add_110;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(_fdiv_rn_31), "f"(_shfl_xor_80));
        float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, _add_110, 2);
        float _add_111;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(_add_110), "f"(_shfl_xor_81));
        float _add_112;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(_add_111), "f"(hc_eps));
        float _fdiv_rn_32 = __fdiv_rn(_fdiv_rn_31, _add_112);
        hist[30] = _fdiv_rn_32;
        float _mul_15;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(_unpack_bf16_17[0]), "f"(_unpack_bf16_15[0]));
        dots_p_93[1] = _mul_15;
        float _fma_196;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(_unpack_bf16_17[1]), "f"(_unpack_bf16_15[1]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_196;
        float _fma_197;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(_unpack_bf16_17[2]), "f"(_unpack_bf16_15[2]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_197;
        float _fma_198;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_198) : "f"(_unpack_bf16_17[3]), "f"(_unpack_bf16_15[3]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_198;
        float _fma_199;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_199) : "f"(_unpack_bf16_17[4]), "f"(_unpack_bf16_15[4]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_199;
        float _fma_200;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(_unpack_bf16_17[5]), "f"(_unpack_bf16_15[5]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_200;
        float _fma_201;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_201) : "f"(_unpack_bf16_17[6]), "f"(_unpack_bf16_15[6]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_201;
        float _fma_202;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_202) : "f"(_unpack_bf16_17[7]), "f"(_unpack_bf16_15[7]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_202;
        float _fma_203;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_203) : "f"(_unpack_bf16_17[8]), "f"(_unpack_bf16_15[8]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_203;
        float _fma_204;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(_unpack_bf16_17[9]), "f"(_unpack_bf16_15[9]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_204;
        float _fma_205;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_205) : "f"(_unpack_bf16_17[10]), "f"(_unpack_bf16_15[10]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_205;
        float _fma_206;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_206) : "f"(_unpack_bf16_17[11]), "f"(_unpack_bf16_15[11]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_206;
        float _fma_207;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_207) : "f"(_unpack_bf16_17[12]), "f"(_unpack_bf16_15[12]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_207;
        float _fma_208;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_208) : "f"(_unpack_bf16_17[13]), "f"(_unpack_bf16_15[13]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_208;
        float _fma_209;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_209) : "f"(_unpack_bf16_17[14]), "f"(_unpack_bf16_15[14]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_209;
        float _fma_210;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_210) : "f"(_unpack_bf16_17[15]), "f"(_unpack_bf16_15[15]), "f"(dots_p_93[1]));
        dots_p_93[1] = _fma_210;
        unsigned int ring_pk_95[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[(0) + 3]))
            : "r"(sa__4 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_95[(4) + 3]))
            : "r"(sa__4 + 3088 + lane32));
        float _unpack_bf16_18[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_95[_pair]) << 16);
            (&_unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_95[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 4);
        float _add_113;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(_fdiv_rn_32), "f"(_shfl_xor_82));
        float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, _add_113, 24);
        float _add_114;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(_add_113), "f"(_shfl_xor_83));
        float _add_115;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(_add_114), "f"(hc_eps));
        float _fdiv_rn_33 = __fdiv_rn(_fdiv_rn_32, _add_115);
        hist[31] = _fdiv_rn_33;
        float _mul_16;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_unpack_bf16_18[0]), "f"(_unpack_bf16_15[0]));
        dots_p_93[2] = _mul_16;
        float _fma_211;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_211) : "f"(_unpack_bf16_18[1]), "f"(_unpack_bf16_15[1]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_211;
        float _fma_212;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(_unpack_bf16_18[2]), "f"(_unpack_bf16_15[2]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_212;
        float _fma_213;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_213) : "f"(_unpack_bf16_18[3]), "f"(_unpack_bf16_15[3]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_213;
        float _fma_214;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_214) : "f"(_unpack_bf16_18[4]), "f"(_unpack_bf16_15[4]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_214;
        float _fma_215;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(_unpack_bf16_18[5]), "f"(_unpack_bf16_15[5]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_215;
        float _fma_216;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_216) : "f"(_unpack_bf16_18[6]), "f"(_unpack_bf16_15[6]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_216;
        float _fma_217;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(_unpack_bf16_18[7]), "f"(_unpack_bf16_15[7]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_217;
        float _fma_218;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(_unpack_bf16_18[8]), "f"(_unpack_bf16_15[8]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_218;
        float _fma_219;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_219) : "f"(_unpack_bf16_18[9]), "f"(_unpack_bf16_15[9]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_219;
        float _fma_220;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(_unpack_bf16_18[10]), "f"(_unpack_bf16_15[10]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_220;
        float _fma_221;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(_unpack_bf16_18[11]), "f"(_unpack_bf16_15[11]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_221;
        float _fma_222;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(_unpack_bf16_18[12]), "f"(_unpack_bf16_15[12]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_222;
        float _fma_223;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(_unpack_bf16_18[13]), "f"(_unpack_bf16_15[13]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_223;
        float _fma_224;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(_unpack_bf16_18[14]), "f"(_unpack_bf16_15[14]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_224;
        float _fma_225;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_225) : "f"(_unpack_bf16_18[15]), "f"(_unpack_bf16_15[15]), "f"(dots_p_93[2]));
        dots_p_93[2] = _fma_225;
        unsigned int ring_pk_96[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[(0) + 3]))
            : "r"(sa__4 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_96[(4) + 3]))
            : "r"(sa__4 + 4112 + lane32));
        float _unpack_bf16_19[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_96[_pair]) << 16);
            (&_unpack_bf16_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_96[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 1);
        float _add_116;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(_fdiv_rn_33), "f"(_shfl_xor_84));
        float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, _add_116, 2);
        float _add_117;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(_add_116), "f"(_shfl_xor_85));
        float _add_118;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(_add_117), "f"(hc_eps));
        float _fdiv_rn_34 = __fdiv_rn(_fdiv_rn_33, _add_118);
        hist[32] = _fdiv_rn_34;
        float _mul_17;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_unpack_bf16_19[0]), "f"(_unpack_bf16_15[0]));
        dots_p_93[3] = _mul_17;
        float _fma_226;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_226) : "f"(_unpack_bf16_19[1]), "f"(_unpack_bf16_15[1]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_226;
        float _fma_227;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(_unpack_bf16_19[2]), "f"(_unpack_bf16_15[2]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_227;
        float _fma_228;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(_unpack_bf16_19[3]), "f"(_unpack_bf16_15[3]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_228;
        float _fma_229;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_229) : "f"(_unpack_bf16_19[4]), "f"(_unpack_bf16_15[4]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_229;
        float _fma_230;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(_unpack_bf16_19[5]), "f"(_unpack_bf16_15[5]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_230;
        float _fma_231;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(_unpack_bf16_19[6]), "f"(_unpack_bf16_15[6]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_231;
        float _fma_232;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(_unpack_bf16_19[7]), "f"(_unpack_bf16_15[7]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_232;
        float _fma_233;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(_unpack_bf16_19[8]), "f"(_unpack_bf16_15[8]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_233;
        float _fma_234;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(_unpack_bf16_19[9]), "f"(_unpack_bf16_15[9]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_234;
        float _fma_235;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(_unpack_bf16_19[10]), "f"(_unpack_bf16_15[10]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_235;
        float _fma_236;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(_unpack_bf16_19[11]), "f"(_unpack_bf16_15[11]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_236;
        float _fma_237;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(_unpack_bf16_19[12]), "f"(_unpack_bf16_15[12]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_237;
        float _fma_238;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_238) : "f"(_unpack_bf16_19[13]), "f"(_unpack_bf16_15[13]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_238;
        float _fma_239;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_239) : "f"(_unpack_bf16_19[14]), "f"(_unpack_bf16_15[14]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_239;
        float _fma_240;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(_unpack_bf16_19[15]), "f"(_unpack_bf16_15[15]), "f"(dots_p_93[3]));
        dots_p_93[3] = _fma_240;
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
        float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 4);
        float _add_119;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(_fdiv_rn_34), "f"(_shfl_xor_86));
        float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, _add_119, 24);
        float _add_120;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(_add_119), "f"(_shfl_xor_87));
        float _add_121;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(_add_120), "f"(hc_eps));
        float _fdiv_rn_35 = __fdiv_rn(_fdiv_rn_34, _add_121);
        hist[33] = _fdiv_rn_35;
        int upper_97 = lane_1 & 16;
        float send_98 = ((upper_97 != 0) ? dots_p_93[0] : dots_p_93[2]);
        float keep_99 = ((upper_97 != 0) ? dots_p_93[2] : dots_p_93[0]);
        float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, send_98, 16);
        float _add_122;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(keep_99), "f"(_shfl_xor_88));
        float send_100 = ((upper_97 != 0) ? dots_p_93[1] : dots_p_93[3]);
        float keep_101 = ((upper_97 != 0) ? dots_p_93[3] : dots_p_93[1]);
        float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, send_100, 16);
        float _add_123;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(keep_101), "f"(_shfl_xor_89));
        int upper_102 = lane_1 & 8;
        float send_103 = ((upper_102 != 0) ? _add_122 : _add_123);
        float keep_104 = ((upper_102 != 0) ? _add_123 : _add_122);
        float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, send_103, 8);
        float _add_124;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(keep_104), "f"(_shfl_xor_90));
        float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 1);
        float _add_125;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(_fdiv_rn_35), "f"(_shfl_xor_91));
        float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, _add_125, 2);
        float _add_126;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(_add_125), "f"(_shfl_xor_92));
        float _add_127;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(_add_126), "f"(hc_eps));
        float _fdiv_rn_36 = __fdiv_rn(_fdiv_rn_35, _add_127);
        hist[34] = _fdiv_rn_36;
        float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, _add_124, 4);
        float _add_128;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(_add_124), "f"(_shfl_xor_93));
        float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, _add_128, 2);
        float _add_129;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(_add_128), "f"(_shfl_xor_94));
        float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, _add_129, 1);
        float _add_130;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(_add_129), "f"(_shfl_xor_95));
        dpre_part[own_base + 96] = _add_130;
        float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 4);
        float _add_131;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(_fdiv_rn_36), "f"(_shfl_xor_96));
        float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, _add_131, 24);
        float _add_132;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(_add_131), "f"(_shfl_xor_97));
        float _add_133;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(_add_132), "f"(hc_eps));
        float _fdiv_rn_37 = __fdiv_rn(_fdiv_rn_36, _add_133);
        hist[35] = _fdiv_rn_37;
        float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 1);
        float _add_134;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(_fdiv_rn_37), "f"(_shfl_xor_98));
        float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, _add_134, 2);
        float _add_135;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_135) : "f"(_add_134), "f"(_shfl_xor_99));
        float _add_136;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(_add_135), "f"(hc_eps));
        float _fdiv_rn_38 = __fdiv_rn(_fdiv_rn_37, _add_136);
        hist[36] = _fdiv_rn_38;
        float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 4);
        float _add_137;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(_fdiv_rn_38), "f"(_shfl_xor_100));
        float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, _add_137, 24);
        float _add_138;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_138) : "f"(_add_137), "f"(_shfl_xor_101));
        float _add_139;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(_add_138), "f"(hc_eps));
        float _fdiv_rn_39 = __fdiv_rn(_fdiv_rn_38, _add_139);
        hist[37] = _fdiv_rn_39;
        mbarrier_wait(ring_bar_addr + (st_) * 8, 0);
        unsigned int ring_pk_105[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[(0) + 3]))
            : "r"(sa_ + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_105[(4) + 3]))
            : "r"(sa_ + 16 + lane32));
        float _unpack_bf16_20[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_105[_pair]) << 16);
            (&_unpack_bf16_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_105[_pair]) & 0xffff0000u);
        }
        unsigned int ring_pk_106[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[(0) + 3]))
            : "r"(sa_ + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_106[(4) + 3]))
            : "r"(sa_ + 1040 + lane32));
        float _unpack_bf16_21[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_106[_pair]) << 16);
            (&_unpack_bf16_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_106[_pair]) & 0xffff0000u);
        }
        float dots_p_107[4];
        float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 1);
        float _add_140;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_140) : "f"(_fdiv_rn_39), "f"(_shfl_xor_102));
        float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, _add_140, 2);
        float _add_141;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_141) : "f"(_add_140), "f"(_shfl_xor_103));
        float _add_142;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(_add_141), "f"(hc_eps));
        float _fdiv_rn_40 = __fdiv_rn(_fdiv_rn_39, _add_142);
        hist[38] = _fdiv_rn_40;
        float _mul_18;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(_unpack_bf16_21[0]), "f"(_unpack_bf16_20[0]));
        dots_p_107[0] = _mul_18;
        float _fma_241;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(_unpack_bf16_21[1]), "f"(_unpack_bf16_20[1]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_241;
        float _fma_242;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(_unpack_bf16_21[2]), "f"(_unpack_bf16_20[2]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_242;
        float _fma_243;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_243) : "f"(_unpack_bf16_21[3]), "f"(_unpack_bf16_20[3]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_243;
        float _fma_244;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_244) : "f"(_unpack_bf16_21[4]), "f"(_unpack_bf16_20[4]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_244;
        float _fma_245;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(_unpack_bf16_21[5]), "f"(_unpack_bf16_20[5]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_245;
        float _fma_246;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_246) : "f"(_unpack_bf16_21[6]), "f"(_unpack_bf16_20[6]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_246;
        float _fma_247;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_247) : "f"(_unpack_bf16_21[7]), "f"(_unpack_bf16_20[7]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_247;
        float _fma_248;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(_unpack_bf16_21[8]), "f"(_unpack_bf16_20[8]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_248;
        float _fma_249;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(_unpack_bf16_21[9]), "f"(_unpack_bf16_20[9]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_249;
        float _fma_250;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_250) : "f"(_unpack_bf16_21[10]), "f"(_unpack_bf16_20[10]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_250;
        float _fma_251;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(_unpack_bf16_21[11]), "f"(_unpack_bf16_20[11]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_251;
        float _fma_252;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(_unpack_bf16_21[12]), "f"(_unpack_bf16_20[12]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_252;
        float _fma_253;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(_unpack_bf16_21[13]), "f"(_unpack_bf16_20[13]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_253;
        float _fma_254;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_254) : "f"(_unpack_bf16_21[14]), "f"(_unpack_bf16_20[14]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_254;
        float _fma_255;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_255) : "f"(_unpack_bf16_21[15]), "f"(_unpack_bf16_20[15]), "f"(dots_p_107[0]));
        dots_p_107[0] = _fma_255;
        unsigned int ring_pk_108[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[(0) + 3]))
            : "r"(sa_ + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_108[(4) + 3]))
            : "r"(sa_ + 2064 + lane32));
        float _unpack_bf16_22[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_22[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_108[_pair]) << 16);
            (&_unpack_bf16_22[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_108[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_40, 4);
        float _add_143;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(_fdiv_rn_40), "f"(_shfl_xor_104));
        float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, _add_143, 24);
        float _add_144;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(_add_143), "f"(_shfl_xor_105));
        float _add_145;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_145) : "f"(_add_144), "f"(hc_eps));
        float _fdiv_rn_41 = __fdiv_rn(_fdiv_rn_40, _add_145);
        hist[39] = _fdiv_rn_41;
        float _mul_19;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_unpack_bf16_22[0]), "f"(_unpack_bf16_20[0]));
        dots_p_107[1] = _mul_19;
        float _fma_256;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(_unpack_bf16_22[1]), "f"(_unpack_bf16_20[1]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_256;
        float _fma_257;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_257) : "f"(_unpack_bf16_22[2]), "f"(_unpack_bf16_20[2]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_257;
        float _fma_258;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(_unpack_bf16_22[3]), "f"(_unpack_bf16_20[3]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_258;
        float _fma_259;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(_unpack_bf16_22[4]), "f"(_unpack_bf16_20[4]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_259;
        float _fma_260;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_260) : "f"(_unpack_bf16_22[5]), "f"(_unpack_bf16_20[5]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_260;
        float _fma_261;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(_unpack_bf16_22[6]), "f"(_unpack_bf16_20[6]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_261;
        float _fma_262;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(_unpack_bf16_22[7]), "f"(_unpack_bf16_20[7]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_262;
        float _fma_263;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_263) : "f"(_unpack_bf16_22[8]), "f"(_unpack_bf16_20[8]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_263;
        float _fma_264;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_264) : "f"(_unpack_bf16_22[9]), "f"(_unpack_bf16_20[9]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_264;
        float _fma_265;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(_unpack_bf16_22[10]), "f"(_unpack_bf16_20[10]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_265;
        float _fma_266;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(_unpack_bf16_22[11]), "f"(_unpack_bf16_20[11]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_266;
        float _fma_267;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(_unpack_bf16_22[12]), "f"(_unpack_bf16_20[12]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_267;
        float _fma_268;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(_unpack_bf16_22[13]), "f"(_unpack_bf16_20[13]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_268;
        float _fma_269;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(_unpack_bf16_22[14]), "f"(_unpack_bf16_20[14]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_269;
        float _fma_270;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(_unpack_bf16_22[15]), "f"(_unpack_bf16_20[15]), "f"(dots_p_107[1]));
        dots_p_107[1] = _fma_270;
        unsigned int ring_pk_109[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[(0) + 3]))
            : "r"(sa_ + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_109[(4) + 3]))
            : "r"(sa_ + 3088 + lane32));
        float _unpack_bf16_23[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_23[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_109[_pair]) << 16);
            (&_unpack_bf16_23[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_109[_pair]) & 0xffff0000u);
        }
        float _mul_20;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(sig), "f"(sneg));
        float dsig = _mul_20;
        float G = ((lane_1 >= 8 && lane_1 < 24) ? gval : 0.0f);
        float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, hist[38], 4);
        float _add_146;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_146) : "f"(hist[38]), "f"(_shfl_xor_106));
        float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, _add_146, 24);
        float _add_147;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(_add_146), "f"(_shfl_xor_107));
        float _add_148;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(_add_147), "f"(hc_eps));
        float _mul_21;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(G), "f"(hist[39]));
        float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, _mul_21, 4);
        float _add_149;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_149) : "f"(_mul_21), "f"(_shfl_xor_108));
        float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, _add_149, 24);
        float _add_150;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_150) : "f"(_add_149), "f"(_shfl_xor_109));
        float _sub_2;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2) : "f"(G), "f"(_add_150));
        float _fdiv_rn_42 = __fdiv_rn(_sub_2, _add_148);
        float _mul_22;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(_unpack_bf16_23[0]), "f"(_unpack_bf16_20[0]));
        dots_p_107[2] = _mul_22;
        float _fma_271;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(_unpack_bf16_23[1]), "f"(_unpack_bf16_20[1]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_271;
        float _fma_272;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_272) : "f"(_unpack_bf16_23[2]), "f"(_unpack_bf16_20[2]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_272;
        float _fma_273;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_273) : "f"(_unpack_bf16_23[3]), "f"(_unpack_bf16_20[3]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_273;
        float _fma_274;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(_unpack_bf16_23[4]), "f"(_unpack_bf16_20[4]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_274;
        float _fma_275;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_275) : "f"(_unpack_bf16_23[5]), "f"(_unpack_bf16_20[5]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_275;
        float _fma_276;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_276) : "f"(_unpack_bf16_23[6]), "f"(_unpack_bf16_20[6]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_276;
        float _fma_277;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(_unpack_bf16_23[7]), "f"(_unpack_bf16_20[7]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_277;
        float _fma_278;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(_unpack_bf16_23[8]), "f"(_unpack_bf16_20[8]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_278;
        float _fma_279;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(_unpack_bf16_23[9]), "f"(_unpack_bf16_20[9]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_279;
        float _fma_280;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_280) : "f"(_unpack_bf16_23[10]), "f"(_unpack_bf16_20[10]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_280;
        float _fma_281;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_281) : "f"(_unpack_bf16_23[11]), "f"(_unpack_bf16_20[11]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_281;
        float _fma_282;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_282) : "f"(_unpack_bf16_23[12]), "f"(_unpack_bf16_20[12]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_282;
        float _fma_283;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_283) : "f"(_unpack_bf16_23[13]), "f"(_unpack_bf16_20[13]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_283;
        float _fma_284;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_284) : "f"(_unpack_bf16_23[14]), "f"(_unpack_bf16_20[14]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_284;
        float _fma_285;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(_unpack_bf16_23[15]), "f"(_unpack_bf16_20[15]), "f"(dots_p_107[2]));
        dots_p_107[2] = _fma_285;
        unsigned int ring_pk_110[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[(0) + 3]))
            : "r"(sa_ + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_110[(4) + 3]))
            : "r"(sa_ + 4112 + lane32));
        float _unpack_bf16_24[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_24[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_110[_pair]) << 16);
            (&_unpack_bf16_24[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_110[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, hist[37], 1);
        float _add_151;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(hist[37]), "f"(_shfl_xor_110));
        float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, _add_151, 2);
        float _add_152;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_152) : "f"(_add_151), "f"(_shfl_xor_111));
        float _add_153;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(_add_152), "f"(hc_eps));
        float _mul_23;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_fdiv_rn_42), "f"(hist[38]));
        float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, _mul_23, 1);
        float _add_154;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_154) : "f"(_mul_23), "f"(_shfl_xor_112));
        float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, _add_154, 2);
        float _add_155;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155) : "f"(_add_154), "f"(_shfl_xor_113));
        float _sub_3;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(_fdiv_rn_42), "f"(_add_155));
        float _fdiv_rn_43 = __fdiv_rn(_sub_3, _add_153);
        float _mul_24;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(_unpack_bf16_24[0]), "f"(_unpack_bf16_20[0]));
        dots_p_107[3] = _mul_24;
        float _fma_286;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(_unpack_bf16_24[1]), "f"(_unpack_bf16_20[1]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_286;
        float _fma_287;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(_unpack_bf16_24[2]), "f"(_unpack_bf16_20[2]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_287;
        float _fma_288;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_288) : "f"(_unpack_bf16_24[3]), "f"(_unpack_bf16_20[3]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_288;
        float _fma_289;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(_unpack_bf16_24[4]), "f"(_unpack_bf16_20[4]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_289;
        float _fma_290;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(_unpack_bf16_24[5]), "f"(_unpack_bf16_20[5]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_290;
        float _fma_291;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_291) : "f"(_unpack_bf16_24[6]), "f"(_unpack_bf16_20[6]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_291;
        float _fma_292;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_292) : "f"(_unpack_bf16_24[7]), "f"(_unpack_bf16_20[7]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_292;
        float _fma_293;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_293) : "f"(_unpack_bf16_24[8]), "f"(_unpack_bf16_20[8]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_293;
        float _fma_294;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_294) : "f"(_unpack_bf16_24[9]), "f"(_unpack_bf16_20[9]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_294;
        float _fma_295;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_295) : "f"(_unpack_bf16_24[10]), "f"(_unpack_bf16_20[10]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_295;
        float _fma_296;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_296) : "f"(_unpack_bf16_24[11]), "f"(_unpack_bf16_20[11]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_296;
        float _fma_297;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(_unpack_bf16_24[12]), "f"(_unpack_bf16_20[12]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_297;
        float _fma_298;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_298) : "f"(_unpack_bf16_24[13]), "f"(_unpack_bf16_20[13]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_298;
        float _fma_299;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_299) : "f"(_unpack_bf16_24[14]), "f"(_unpack_bf16_20[14]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_299;
        float _fma_300;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(_unpack_bf16_24[15]), "f"(_unpack_bf16_20[15]), "f"(dots_p_107[3]));
        dots_p_107[3] = _fma_300;
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
        float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, hist[36], 4);
        float _add_156;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(hist[36]), "f"(_shfl_xor_114));
        float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, _add_156, 24);
        float _add_157;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(_add_156), "f"(_shfl_xor_115));
        float _add_158;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_158) : "f"(_add_157), "f"(hc_eps));
        float _mul_25;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(_fdiv_rn_43), "f"(hist[37]));
        float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, _mul_25, 4);
        float _add_159;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_159) : "f"(_mul_25), "f"(_shfl_xor_116));
        float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, _add_159, 24);
        float _add_160;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(_add_159), "f"(_shfl_xor_117));
        float _sub_4;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4) : "f"(_fdiv_rn_43), "f"(_add_160));
        float _fdiv_rn_44 = __fdiv_rn(_sub_4, _add_158);
        int upper_111 = lane_1 & 16;
        float send_112 = ((upper_111 != 0) ? dots_p_107[0] : dots_p_107[2]);
        float keep_113 = ((upper_111 != 0) ? dots_p_107[2] : dots_p_107[0]);
        float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, send_112, 16);
        float _add_161;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(keep_113), "f"(_shfl_xor_118));
        float send_114 = ((upper_111 != 0) ? dots_p_107[1] : dots_p_107[3]);
        float keep_115 = ((upper_111 != 0) ? dots_p_107[3] : dots_p_107[1]);
        float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, send_114, 16);
        float _add_162;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_162) : "f"(keep_115), "f"(_shfl_xor_119));
        int upper_116 = lane_1 & 8;
        float send_117 = ((upper_116 != 0) ? _add_161 : _add_162);
        float keep_118 = ((upper_116 != 0) ? _add_162 : _add_161);
        float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, send_117, 8);
        float _add_163;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(keep_118), "f"(_shfl_xor_120));
        float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, hist[35], 1);
        float _add_164;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_164) : "f"(hist[35]), "f"(_shfl_xor_121));
        float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, _add_164, 2);
        float _add_165;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(_add_164), "f"(_shfl_xor_122));
        float _add_166;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_166) : "f"(_add_165), "f"(hc_eps));
        float _mul_26;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(_fdiv_rn_44), "f"(hist[36]));
        float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, _mul_26, 1);
        float _add_167;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(_mul_26), "f"(_shfl_xor_123));
        float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, _add_167, 2);
        float _add_168;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(_add_167), "f"(_shfl_xor_124));
        float _sub_5;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(_fdiv_rn_44), "f"(_add_168));
        float _fdiv_rn_45 = __fdiv_rn(_sub_5, _add_166);
        float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, _add_163, 4);
        float _add_169;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(_add_163), "f"(_shfl_xor_125));
        float _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, _add_169, 2);
        float _add_170;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_170) : "f"(_add_169), "f"(_shfl_xor_126));
        float _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, _add_170, 1);
        float _add_171;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171) : "f"(_add_170), "f"(_shfl_xor_127));
        dpre_part[own_base + 128] = _add_171;
        float _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, hist[34], 4);
        float _add_172;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(hist[34]), "f"(_shfl_xor_128));
        float _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, _add_172, 24);
        float _add_173;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_173) : "f"(_add_172), "f"(_shfl_xor_129));
        float _add_174;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_174) : "f"(_add_173), "f"(hc_eps));
        float _mul_27;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(_fdiv_rn_45), "f"(hist[35]));
        float _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, _mul_27, 4);
        float _add_175;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_175) : "f"(_mul_27), "f"(_shfl_xor_130));
        float _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, _add_175, 24);
        float _add_176;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_176) : "f"(_add_175), "f"(_shfl_xor_131));
        float _sub_6;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(_fdiv_rn_45), "f"(_add_176));
        float _fdiv_rn_46 = __fdiv_rn(_sub_6, _add_174);
        float _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, hist[33], 1);
        float _add_177;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_177) : "f"(hist[33]), "f"(_shfl_xor_132));
        float _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, _add_177, 2);
        float _add_178;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_178) : "f"(_add_177), "f"(_shfl_xor_133));
        float _add_179;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_179) : "f"(_add_178), "f"(hc_eps));
        float _mul_28;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(_fdiv_rn_46), "f"(hist[34]));
        float _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, _mul_28, 1);
        float _add_180;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_180) : "f"(_mul_28), "f"(_shfl_xor_134));
        float _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, _add_180, 2);
        float _add_181;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181) : "f"(_add_180), "f"(_shfl_xor_135));
        float _sub_7;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(_fdiv_rn_46), "f"(_add_181));
        float _fdiv_rn_47 = __fdiv_rn(_sub_7, _add_179);
        float _shfl_xor_136 = __shfl_xor_sync(0xFFFFFFFF, hist[32], 4);
        float _add_182;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_182) : "f"(hist[32]), "f"(_shfl_xor_136));
        float _shfl_xor_137 = __shfl_xor_sync(0xFFFFFFFF, _add_182, 24);
        float _add_183;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183) : "f"(_add_182), "f"(_shfl_xor_137));
        float _add_184;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_184) : "f"(_add_183), "f"(hc_eps));
        float _mul_29;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(_fdiv_rn_47), "f"(hist[33]));
        float _shfl_xor_138 = __shfl_xor_sync(0xFFFFFFFF, _mul_29, 4);
        float _add_185;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_185) : "f"(_mul_29), "f"(_shfl_xor_138));
        float _shfl_xor_139 = __shfl_xor_sync(0xFFFFFFFF, _add_185, 24);
        float _add_186;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_186) : "f"(_add_185), "f"(_shfl_xor_139));
        float _sub_8;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(_fdiv_rn_47), "f"(_add_186));
        float _fdiv_rn_48 = __fdiv_rn(_sub_8, _add_184);
        mbarrier_wait(ring_bar_addr + (st__3) * 8, 0);
        unsigned int ring_pk_119[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[(0) + 3]))
            : "r"(sa__4 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_119[(4) + 3]))
            : "r"(sa__4 + 16 + lane32));
        float _unpack_bf16_25[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_25[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_119[_pair]) << 16);
            (&_unpack_bf16_25[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_119[_pair]) & 0xffff0000u);
        }
        unsigned int ring_pk_120[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[(0) + 3]))
            : "r"(sa__4 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_120[(4) + 3]))
            : "r"(sa__4 + 1040 + lane32));
        float _unpack_bf16_26[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_26[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_120[_pair]) << 16);
            (&_unpack_bf16_26[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_120[_pair]) & 0xffff0000u);
        }
        float dots_p_121[4];
        float _shfl_xor_140 = __shfl_xor_sync(0xFFFFFFFF, hist[31], 1);
        float _add_187;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_187) : "f"(hist[31]), "f"(_shfl_xor_140));
        float _shfl_xor_141 = __shfl_xor_sync(0xFFFFFFFF, _add_187, 2);
        float _add_188;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_188) : "f"(_add_187), "f"(_shfl_xor_141));
        float _add_189;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_189) : "f"(_add_188), "f"(hc_eps));
        float _mul_30;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(_fdiv_rn_48), "f"(hist[32]));
        float _shfl_xor_142 = __shfl_xor_sync(0xFFFFFFFF, _mul_30, 1);
        float _add_190;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_190) : "f"(_mul_30), "f"(_shfl_xor_142));
        float _shfl_xor_143 = __shfl_xor_sync(0xFFFFFFFF, _add_190, 2);
        float _add_191;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191) : "f"(_add_190), "f"(_shfl_xor_143));
        float _sub_9;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9) : "f"(_fdiv_rn_48), "f"(_add_191));
        float _fdiv_rn_49 = __fdiv_rn(_sub_9, _add_189);
        float _mul_31;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(_unpack_bf16_26[0]), "f"(_unpack_bf16_25[0]));
        dots_p_121[0] = _mul_31;
        float _fma_301;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_301) : "f"(_unpack_bf16_26[1]), "f"(_unpack_bf16_25[1]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_301;
        float _fma_302;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_302) : "f"(_unpack_bf16_26[2]), "f"(_unpack_bf16_25[2]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_302;
        float _fma_303;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(_unpack_bf16_26[3]), "f"(_unpack_bf16_25[3]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_303;
        float _fma_304;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_304) : "f"(_unpack_bf16_26[4]), "f"(_unpack_bf16_25[4]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_304;
        float _fma_305;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(_unpack_bf16_26[5]), "f"(_unpack_bf16_25[5]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_305;
        float _fma_306;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(_unpack_bf16_26[6]), "f"(_unpack_bf16_25[6]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_306;
        float _fma_307;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_307) : "f"(_unpack_bf16_26[7]), "f"(_unpack_bf16_25[7]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_307;
        float _fma_308;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_308) : "f"(_unpack_bf16_26[8]), "f"(_unpack_bf16_25[8]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_308;
        float _fma_309;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_309) : "f"(_unpack_bf16_26[9]), "f"(_unpack_bf16_25[9]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_309;
        float _fma_310;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_310) : "f"(_unpack_bf16_26[10]), "f"(_unpack_bf16_25[10]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_310;
        float _fma_311;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_311) : "f"(_unpack_bf16_26[11]), "f"(_unpack_bf16_25[11]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_311;
        float _fma_312;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(_unpack_bf16_26[12]), "f"(_unpack_bf16_25[12]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_312;
        float _fma_313;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_313) : "f"(_unpack_bf16_26[13]), "f"(_unpack_bf16_25[13]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_313;
        float _fma_314;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_314) : "f"(_unpack_bf16_26[14]), "f"(_unpack_bf16_25[14]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_314;
        float _fma_315;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_315) : "f"(_unpack_bf16_26[15]), "f"(_unpack_bf16_25[15]), "f"(dots_p_121[0]));
        dots_p_121[0] = _fma_315;
        unsigned int ring_pk_122[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[(0) + 3]))
            : "r"(sa__4 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_122[(4) + 3]))
            : "r"(sa__4 + 2064 + lane32));
        float _unpack_bf16_27[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_27[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_122[_pair]) << 16);
            (&_unpack_bf16_27[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_122[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_144 = __shfl_xor_sync(0xFFFFFFFF, hist[30], 4);
        float _add_192;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_192) : "f"(hist[30]), "f"(_shfl_xor_144));
        float _shfl_xor_145 = __shfl_xor_sync(0xFFFFFFFF, _add_192, 24);
        float _add_193;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_193) : "f"(_add_192), "f"(_shfl_xor_145));
        float _add_194;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_194) : "f"(_add_193), "f"(hc_eps));
        float _mul_32;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(_fdiv_rn_49), "f"(hist[31]));
        float _shfl_xor_146 = __shfl_xor_sync(0xFFFFFFFF, _mul_32, 4);
        float _add_195;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(_mul_32), "f"(_shfl_xor_146));
        float _shfl_xor_147 = __shfl_xor_sync(0xFFFFFFFF, _add_195, 24);
        float _add_196;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_196) : "f"(_add_195), "f"(_shfl_xor_147));
        float _sub_10;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(_fdiv_rn_49), "f"(_add_196));
        float _fdiv_rn_50 = __fdiv_rn(_sub_10, _add_194);
        float _mul_33;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(_unpack_bf16_27[0]), "f"(_unpack_bf16_25[0]));
        dots_p_121[1] = _mul_33;
        float _fma_316;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_316) : "f"(_unpack_bf16_27[1]), "f"(_unpack_bf16_25[1]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_316;
        float _fma_317;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_317) : "f"(_unpack_bf16_27[2]), "f"(_unpack_bf16_25[2]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_317;
        float _fma_318;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(_unpack_bf16_27[3]), "f"(_unpack_bf16_25[3]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_318;
        float _fma_319;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_319) : "f"(_unpack_bf16_27[4]), "f"(_unpack_bf16_25[4]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_319;
        float _fma_320;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_320) : "f"(_unpack_bf16_27[5]), "f"(_unpack_bf16_25[5]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_320;
        float _fma_321;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_321) : "f"(_unpack_bf16_27[6]), "f"(_unpack_bf16_25[6]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_321;
        float _fma_322;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_322) : "f"(_unpack_bf16_27[7]), "f"(_unpack_bf16_25[7]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_322;
        float _fma_323;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_323) : "f"(_unpack_bf16_27[8]), "f"(_unpack_bf16_25[8]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_323;
        float _fma_324;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_324) : "f"(_unpack_bf16_27[9]), "f"(_unpack_bf16_25[9]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_324;
        float _fma_325;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_325) : "f"(_unpack_bf16_27[10]), "f"(_unpack_bf16_25[10]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_325;
        float _fma_326;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_326) : "f"(_unpack_bf16_27[11]), "f"(_unpack_bf16_25[11]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_326;
        float _fma_327;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_327) : "f"(_unpack_bf16_27[12]), "f"(_unpack_bf16_25[12]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_327;
        float _fma_328;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_328) : "f"(_unpack_bf16_27[13]), "f"(_unpack_bf16_25[13]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_328;
        float _fma_329;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_329) : "f"(_unpack_bf16_27[14]), "f"(_unpack_bf16_25[14]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_329;
        float _fma_330;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_330) : "f"(_unpack_bf16_27[15]), "f"(_unpack_bf16_25[15]), "f"(dots_p_121[1]));
        dots_p_121[1] = _fma_330;
        unsigned int ring_pk_123[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[(0) + 3]))
            : "r"(sa__4 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_123[(4) + 3]))
            : "r"(sa__4 + 3088 + lane32));
        float _unpack_bf16_28[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_28[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_123[_pair]) << 16);
            (&_unpack_bf16_28[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_123[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_148 = __shfl_xor_sync(0xFFFFFFFF, hist[29], 1);
        float _add_197;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(hist[29]), "f"(_shfl_xor_148));
        float _shfl_xor_149 = __shfl_xor_sync(0xFFFFFFFF, _add_197, 2);
        float _add_198;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_198) : "f"(_add_197), "f"(_shfl_xor_149));
        float _add_199;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_199) : "f"(_add_198), "f"(hc_eps));
        float _mul_34;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(_fdiv_rn_50), "f"(hist[30]));
        float _shfl_xor_150 = __shfl_xor_sync(0xFFFFFFFF, _mul_34, 1);
        float _add_200;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_200) : "f"(_mul_34), "f"(_shfl_xor_150));
        float _shfl_xor_151 = __shfl_xor_sync(0xFFFFFFFF, _add_200, 2);
        float _add_201;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_201) : "f"(_add_200), "f"(_shfl_xor_151));
        float _sub_11;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(_fdiv_rn_50), "f"(_add_201));
        float _fdiv_rn_51 = __fdiv_rn(_sub_11, _add_199);
        float _mul_35;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(_unpack_bf16_28[0]), "f"(_unpack_bf16_25[0]));
        dots_p_121[2] = _mul_35;
        float _fma_331;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_331) : "f"(_unpack_bf16_28[1]), "f"(_unpack_bf16_25[1]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_331;
        float _fma_332;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_332) : "f"(_unpack_bf16_28[2]), "f"(_unpack_bf16_25[2]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_332;
        float _fma_333;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_333) : "f"(_unpack_bf16_28[3]), "f"(_unpack_bf16_25[3]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_333;
        float _fma_334;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_334) : "f"(_unpack_bf16_28[4]), "f"(_unpack_bf16_25[4]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_334;
        float _fma_335;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_335) : "f"(_unpack_bf16_28[5]), "f"(_unpack_bf16_25[5]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_335;
        float _fma_336;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_336) : "f"(_unpack_bf16_28[6]), "f"(_unpack_bf16_25[6]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_336;
        float _fma_337;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_337) : "f"(_unpack_bf16_28[7]), "f"(_unpack_bf16_25[7]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_337;
        float _fma_338;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_338) : "f"(_unpack_bf16_28[8]), "f"(_unpack_bf16_25[8]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_338;
        float _fma_339;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_339) : "f"(_unpack_bf16_28[9]), "f"(_unpack_bf16_25[9]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_339;
        float _fma_340;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_340) : "f"(_unpack_bf16_28[10]), "f"(_unpack_bf16_25[10]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_340;
        float _fma_341;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_341) : "f"(_unpack_bf16_28[11]), "f"(_unpack_bf16_25[11]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_341;
        float _fma_342;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_342) : "f"(_unpack_bf16_28[12]), "f"(_unpack_bf16_25[12]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_342;
        float _fma_343;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_343) : "f"(_unpack_bf16_28[13]), "f"(_unpack_bf16_25[13]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_343;
        float _fma_344;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_344) : "f"(_unpack_bf16_28[14]), "f"(_unpack_bf16_25[14]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_344;
        float _fma_345;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_345) : "f"(_unpack_bf16_28[15]), "f"(_unpack_bf16_25[15]), "f"(dots_p_121[2]));
        dots_p_121[2] = _fma_345;
        unsigned int ring_pk_124[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[(0) + 3]))
            : "r"(sa__4 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_124[(4) + 3]))
            : "r"(sa__4 + 4112 + lane32));
        float _unpack_bf16_29[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_29[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_124[_pair]) << 16);
            (&_unpack_bf16_29[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_124[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_152 = __shfl_xor_sync(0xFFFFFFFF, hist[28], 4);
        float _add_202;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(hist[28]), "f"(_shfl_xor_152));
        float _shfl_xor_153 = __shfl_xor_sync(0xFFFFFFFF, _add_202, 24);
        float _add_203;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_203) : "f"(_add_202), "f"(_shfl_xor_153));
        float _add_204;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_204) : "f"(_add_203), "f"(hc_eps));
        float _mul_36;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(_fdiv_rn_51), "f"(hist[29]));
        float _shfl_xor_154 = __shfl_xor_sync(0xFFFFFFFF, _mul_36, 4);
        float _add_205;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_205) : "f"(_mul_36), "f"(_shfl_xor_154));
        float _shfl_xor_155 = __shfl_xor_sync(0xFFFFFFFF, _add_205, 24);
        float _add_206;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_206) : "f"(_add_205), "f"(_shfl_xor_155));
        float _sub_12;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(_fdiv_rn_51), "f"(_add_206));
        float _fdiv_rn_52 = __fdiv_rn(_sub_12, _add_204);
        float _mul_37;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(_unpack_bf16_29[0]), "f"(_unpack_bf16_25[0]));
        dots_p_121[3] = _mul_37;
        float _fma_346;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_346) : "f"(_unpack_bf16_29[1]), "f"(_unpack_bf16_25[1]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_346;
        float _fma_347;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_347) : "f"(_unpack_bf16_29[2]), "f"(_unpack_bf16_25[2]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_347;
        float _fma_348;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_348) : "f"(_unpack_bf16_29[3]), "f"(_unpack_bf16_25[3]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_348;
        float _fma_349;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_349) : "f"(_unpack_bf16_29[4]), "f"(_unpack_bf16_25[4]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_349;
        float _fma_350;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_350) : "f"(_unpack_bf16_29[5]), "f"(_unpack_bf16_25[5]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_350;
        float _fma_351;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_351) : "f"(_unpack_bf16_29[6]), "f"(_unpack_bf16_25[6]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_351;
        float _fma_352;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_352) : "f"(_unpack_bf16_29[7]), "f"(_unpack_bf16_25[7]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_352;
        float _fma_353;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_353) : "f"(_unpack_bf16_29[8]), "f"(_unpack_bf16_25[8]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_353;
        float _fma_354;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_354) : "f"(_unpack_bf16_29[9]), "f"(_unpack_bf16_25[9]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_354;
        float _fma_355;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_355) : "f"(_unpack_bf16_29[10]), "f"(_unpack_bf16_25[10]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_355;
        float _fma_356;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_356) : "f"(_unpack_bf16_29[11]), "f"(_unpack_bf16_25[11]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_356;
        float _fma_357;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_357) : "f"(_unpack_bf16_29[12]), "f"(_unpack_bf16_25[12]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_357;
        float _fma_358;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_358) : "f"(_unpack_bf16_29[13]), "f"(_unpack_bf16_25[13]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_358;
        float _fma_359;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_359) : "f"(_unpack_bf16_29[14]), "f"(_unpack_bf16_25[14]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_359;
        float _fma_360;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_360) : "f"(_unpack_bf16_29[15]), "f"(_unpack_bf16_25[15]), "f"(dots_p_121[3]));
        dots_p_121[3] = _fma_360;
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
        float _shfl_xor_156 = __shfl_xor_sync(0xFFFFFFFF, hist[27], 1);
        float _add_207;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_207) : "f"(hist[27]), "f"(_shfl_xor_156));
        float _shfl_xor_157 = __shfl_xor_sync(0xFFFFFFFF, _add_207, 2);
        float _add_208;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(_add_207), "f"(_shfl_xor_157));
        float _add_209;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_209) : "f"(_add_208), "f"(hc_eps));
        float _mul_38;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(_fdiv_rn_52), "f"(hist[28]));
        float _shfl_xor_158 = __shfl_xor_sync(0xFFFFFFFF, _mul_38, 1);
        float _add_210;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_210) : "f"(_mul_38), "f"(_shfl_xor_158));
        float _shfl_xor_159 = __shfl_xor_sync(0xFFFFFFFF, _add_210, 2);
        float _add_211;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_211) : "f"(_add_210), "f"(_shfl_xor_159));
        float _sub_13;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(_fdiv_rn_52), "f"(_add_211));
        float _fdiv_rn_53 = __fdiv_rn(_sub_13, _add_209);
        int upper_125 = lane_1 & 16;
        float send_126 = ((upper_125 != 0) ? dots_p_121[0] : dots_p_121[2]);
        float keep_127 = ((upper_125 != 0) ? dots_p_121[2] : dots_p_121[0]);
        float _shfl_xor_160 = __shfl_xor_sync(0xFFFFFFFF, send_126, 16);
        float _add_212;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_212) : "f"(keep_127), "f"(_shfl_xor_160));
        float send_128 = ((upper_125 != 0) ? dots_p_121[1] : dots_p_121[3]);
        float keep_129 = ((upper_125 != 0) ? dots_p_121[3] : dots_p_121[1]);
        float _shfl_xor_161 = __shfl_xor_sync(0xFFFFFFFF, send_128, 16);
        float _add_213;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_213) : "f"(keep_129), "f"(_shfl_xor_161));
        int upper_130 = lane_1 & 8;
        float send_131 = ((upper_130 != 0) ? _add_212 : _add_213);
        float keep_132 = ((upper_130 != 0) ? _add_213 : _add_212);
        float _shfl_xor_162 = __shfl_xor_sync(0xFFFFFFFF, send_131, 8);
        float _add_214;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_214) : "f"(keep_132), "f"(_shfl_xor_162));
        float _shfl_xor_163 = __shfl_xor_sync(0xFFFFFFFF, hist[26], 4);
        float _add_215;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_215) : "f"(hist[26]), "f"(_shfl_xor_163));
        float _shfl_xor_164 = __shfl_xor_sync(0xFFFFFFFF, _add_215, 24);
        float _add_216;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_216) : "f"(_add_215), "f"(_shfl_xor_164));
        float _add_217;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_217) : "f"(_add_216), "f"(hc_eps));
        float _mul_39;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(_fdiv_rn_53), "f"(hist[27]));
        float _shfl_xor_165 = __shfl_xor_sync(0xFFFFFFFF, _mul_39, 4);
        float _add_218;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_218) : "f"(_mul_39), "f"(_shfl_xor_165));
        float _shfl_xor_166 = __shfl_xor_sync(0xFFFFFFFF, _add_218, 24);
        float _add_219;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_219) : "f"(_add_218), "f"(_shfl_xor_166));
        float _sub_14;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(_fdiv_rn_53), "f"(_add_219));
        float _fdiv_rn_54 = __fdiv_rn(_sub_14, _add_217);
        float _shfl_xor_167 = __shfl_xor_sync(0xFFFFFFFF, _add_214, 4);
        float _add_220;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_220) : "f"(_add_214), "f"(_shfl_xor_167));
        float _shfl_xor_168 = __shfl_xor_sync(0xFFFFFFFF, _add_220, 2);
        float _add_221;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_221) : "f"(_add_220), "f"(_shfl_xor_168));
        float _shfl_xor_169 = __shfl_xor_sync(0xFFFFFFFF, _add_221, 1);
        float _add_222;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_222) : "f"(_add_221), "f"(_shfl_xor_169));
        dpre_part[own_base + 160] = _add_222;
        float _shfl_xor_170 = __shfl_xor_sync(0xFFFFFFFF, hist[25], 1);
        float _add_223;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_223) : "f"(hist[25]), "f"(_shfl_xor_170));
        float _shfl_xor_171 = __shfl_xor_sync(0xFFFFFFFF, _add_223, 2);
        float _add_224;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_224) : "f"(_add_223), "f"(_shfl_xor_171));
        float _add_225;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_225) : "f"(_add_224), "f"(hc_eps));
        float _mul_40;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_40) : "f"(_fdiv_rn_54), "f"(hist[26]));
        float _shfl_xor_172 = __shfl_xor_sync(0xFFFFFFFF, _mul_40, 1);
        float _add_226;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_226) : "f"(_mul_40), "f"(_shfl_xor_172));
        float _shfl_xor_173 = __shfl_xor_sync(0xFFFFFFFF, _add_226, 2);
        float _add_227;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_227) : "f"(_add_226), "f"(_shfl_xor_173));
        float _sub_15;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(_fdiv_rn_54), "f"(_add_227));
        float _fdiv_rn_55 = __fdiv_rn(_sub_15, _add_225);
        float _shfl_xor_174 = __shfl_xor_sync(0xFFFFFFFF, hist[24], 4);
        float _add_228;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_228) : "f"(hist[24]), "f"(_shfl_xor_174));
        float _shfl_xor_175 = __shfl_xor_sync(0xFFFFFFFF, _add_228, 24);
        float _add_229;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_229) : "f"(_add_228), "f"(_shfl_xor_175));
        float _add_230;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_230) : "f"(_add_229), "f"(hc_eps));
        float _mul_41;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_41) : "f"(_fdiv_rn_55), "f"(hist[25]));
        float _shfl_xor_176 = __shfl_xor_sync(0xFFFFFFFF, _mul_41, 4);
        float _add_231;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_231) : "f"(_mul_41), "f"(_shfl_xor_176));
        float _shfl_xor_177 = __shfl_xor_sync(0xFFFFFFFF, _add_231, 24);
        float _add_232;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_232) : "f"(_add_231), "f"(_shfl_xor_177));
        float _sub_16;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(_fdiv_rn_55), "f"(_add_232));
        float _fdiv_rn_56 = __fdiv_rn(_sub_16, _add_230);
        float _shfl_xor_178 = __shfl_xor_sync(0xFFFFFFFF, hist[23], 1);
        float _add_233;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_233) : "f"(hist[23]), "f"(_shfl_xor_178));
        float _shfl_xor_179 = __shfl_xor_sync(0xFFFFFFFF, _add_233, 2);
        float _add_234;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_234) : "f"(_add_233), "f"(_shfl_xor_179));
        float _add_235;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_235) : "f"(_add_234), "f"(hc_eps));
        float _mul_42;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_42) : "f"(_fdiv_rn_56), "f"(hist[24]));
        float _shfl_xor_180 = __shfl_xor_sync(0xFFFFFFFF, _mul_42, 1);
        float _add_236;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_236) : "f"(_mul_42), "f"(_shfl_xor_180));
        float _shfl_xor_181 = __shfl_xor_sync(0xFFFFFFFF, _add_236, 2);
        float _add_237;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_237) : "f"(_add_236), "f"(_shfl_xor_181));
        float _sub_17;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(_fdiv_rn_56), "f"(_add_237));
        float _fdiv_rn_57 = __fdiv_rn(_sub_17, _add_235);
        mbarrier_wait(ring_bar_addr + (st_) * 8, 1);
        unsigned int ring_pk_133[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[(0) + 3]))
            : "r"(sa_ + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_133[(4) + 3]))
            : "r"(sa_ + 16 + lane32));
        float _unpack_bf16_30[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_30[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_133[_pair]) << 16);
            (&_unpack_bf16_30[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_133[_pair]) & 0xffff0000u);
        }
        unsigned int ring_pk_134[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[(0) + 3]))
            : "r"(sa_ + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_134[(4) + 3]))
            : "r"(sa_ + 1040 + lane32));
        float _unpack_bf16_31[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_31[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_134[_pair]) << 16);
            (&_unpack_bf16_31[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_134[_pair]) & 0xffff0000u);
        }
        float dots_p_135[4];
        float _shfl_xor_182 = __shfl_xor_sync(0xFFFFFFFF, hist[22], 4);
        float _add_238;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_238) : "f"(hist[22]), "f"(_shfl_xor_182));
        float _shfl_xor_183 = __shfl_xor_sync(0xFFFFFFFF, _add_238, 24);
        float _add_239;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_239) : "f"(_add_238), "f"(_shfl_xor_183));
        float _add_240;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_240) : "f"(_add_239), "f"(hc_eps));
        float _mul_43;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_43) : "f"(_fdiv_rn_57), "f"(hist[23]));
        float _shfl_xor_184 = __shfl_xor_sync(0xFFFFFFFF, _mul_43, 4);
        float _add_241;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_241) : "f"(_mul_43), "f"(_shfl_xor_184));
        float _shfl_xor_185 = __shfl_xor_sync(0xFFFFFFFF, _add_241, 24);
        float _add_242;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_242) : "f"(_add_241), "f"(_shfl_xor_185));
        float _sub_18;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(_fdiv_rn_57), "f"(_add_242));
        float _fdiv_rn_58 = __fdiv_rn(_sub_18, _add_240);
        float _mul_44;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_44) : "f"(_unpack_bf16_31[0]), "f"(_unpack_bf16_30[0]));
        dots_p_135[0] = _mul_44;
        float _fma_361;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_361) : "f"(_unpack_bf16_31[1]), "f"(_unpack_bf16_30[1]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_361;
        float _fma_362;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_362) : "f"(_unpack_bf16_31[2]), "f"(_unpack_bf16_30[2]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_362;
        float _fma_363;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_363) : "f"(_unpack_bf16_31[3]), "f"(_unpack_bf16_30[3]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_363;
        float _fma_364;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_364) : "f"(_unpack_bf16_31[4]), "f"(_unpack_bf16_30[4]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_364;
        float _fma_365;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_365) : "f"(_unpack_bf16_31[5]), "f"(_unpack_bf16_30[5]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_365;
        float _fma_366;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_366) : "f"(_unpack_bf16_31[6]), "f"(_unpack_bf16_30[6]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_366;
        float _fma_367;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_367) : "f"(_unpack_bf16_31[7]), "f"(_unpack_bf16_30[7]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_367;
        float _fma_368;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_368) : "f"(_unpack_bf16_31[8]), "f"(_unpack_bf16_30[8]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_368;
        float _fma_369;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_369) : "f"(_unpack_bf16_31[9]), "f"(_unpack_bf16_30[9]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_369;
        float _fma_370;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_370) : "f"(_unpack_bf16_31[10]), "f"(_unpack_bf16_30[10]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_370;
        float _fma_371;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_371) : "f"(_unpack_bf16_31[11]), "f"(_unpack_bf16_30[11]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_371;
        float _fma_372;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_372) : "f"(_unpack_bf16_31[12]), "f"(_unpack_bf16_30[12]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_372;
        float _fma_373;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_373) : "f"(_unpack_bf16_31[13]), "f"(_unpack_bf16_30[13]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_373;
        float _fma_374;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_374) : "f"(_unpack_bf16_31[14]), "f"(_unpack_bf16_30[14]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_374;
        float _fma_375;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_375) : "f"(_unpack_bf16_31[15]), "f"(_unpack_bf16_30[15]), "f"(dots_p_135[0]));
        dots_p_135[0] = _fma_375;
        unsigned int ring_pk_136[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[(0) + 3]))
            : "r"(sa_ + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_136[(4) + 3]))
            : "r"(sa_ + 2064 + lane32));
        float _unpack_bf16_32[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_136[_pair]) << 16);
            (&_unpack_bf16_32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_136[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_186 = __shfl_xor_sync(0xFFFFFFFF, hist[21], 1);
        float _add_243;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_243) : "f"(hist[21]), "f"(_shfl_xor_186));
        float _shfl_xor_187 = __shfl_xor_sync(0xFFFFFFFF, _add_243, 2);
        float _add_244;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_244) : "f"(_add_243), "f"(_shfl_xor_187));
        float _add_245;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_245) : "f"(_add_244), "f"(hc_eps));
        float _mul_45;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_45) : "f"(_fdiv_rn_58), "f"(hist[22]));
        float _shfl_xor_188 = __shfl_xor_sync(0xFFFFFFFF, _mul_45, 1);
        float _add_246;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(_mul_45), "f"(_shfl_xor_188));
        float _shfl_xor_189 = __shfl_xor_sync(0xFFFFFFFF, _add_246, 2);
        float _add_247;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_247) : "f"(_add_246), "f"(_shfl_xor_189));
        float _sub_19;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(_fdiv_rn_58), "f"(_add_247));
        float _fdiv_rn_59 = __fdiv_rn(_sub_19, _add_245);
        float _mul_46;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_46) : "f"(_unpack_bf16_32[0]), "f"(_unpack_bf16_30[0]));
        dots_p_135[1] = _mul_46;
        float _fma_376;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_376) : "f"(_unpack_bf16_32[1]), "f"(_unpack_bf16_30[1]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_376;
        float _fma_377;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_377) : "f"(_unpack_bf16_32[2]), "f"(_unpack_bf16_30[2]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_377;
        float _fma_378;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_378) : "f"(_unpack_bf16_32[3]), "f"(_unpack_bf16_30[3]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_378;
        float _fma_379;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_379) : "f"(_unpack_bf16_32[4]), "f"(_unpack_bf16_30[4]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_379;
        float _fma_380;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_380) : "f"(_unpack_bf16_32[5]), "f"(_unpack_bf16_30[5]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_380;
        float _fma_381;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_381) : "f"(_unpack_bf16_32[6]), "f"(_unpack_bf16_30[6]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_381;
        float _fma_382;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_382) : "f"(_unpack_bf16_32[7]), "f"(_unpack_bf16_30[7]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_382;
        float _fma_383;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_383) : "f"(_unpack_bf16_32[8]), "f"(_unpack_bf16_30[8]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_383;
        float _fma_384;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_384) : "f"(_unpack_bf16_32[9]), "f"(_unpack_bf16_30[9]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_384;
        float _fma_385;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_385) : "f"(_unpack_bf16_32[10]), "f"(_unpack_bf16_30[10]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_385;
        float _fma_386;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_386) : "f"(_unpack_bf16_32[11]), "f"(_unpack_bf16_30[11]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_386;
        float _fma_387;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_387) : "f"(_unpack_bf16_32[12]), "f"(_unpack_bf16_30[12]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_387;
        float _fma_388;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_388) : "f"(_unpack_bf16_32[13]), "f"(_unpack_bf16_30[13]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_388;
        float _fma_389;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_389) : "f"(_unpack_bf16_32[14]), "f"(_unpack_bf16_30[14]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_389;
        float _fma_390;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_390) : "f"(_unpack_bf16_32[15]), "f"(_unpack_bf16_30[15]), "f"(dots_p_135[1]));
        dots_p_135[1] = _fma_390;
        unsigned int ring_pk_137[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[(0) + 3]))
            : "r"(sa_ + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_137[(4) + 3]))
            : "r"(sa_ + 3088 + lane32));
        float _unpack_bf16_33[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_33[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_137[_pair]) << 16);
            (&_unpack_bf16_33[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_137[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_190 = __shfl_xor_sync(0xFFFFFFFF, hist[20], 4);
        float _add_248;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_248) : "f"(hist[20]), "f"(_shfl_xor_190));
        float _shfl_xor_191 = __shfl_xor_sync(0xFFFFFFFF, _add_248, 24);
        float _add_249;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_249) : "f"(_add_248), "f"(_shfl_xor_191));
        float _add_250;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_250) : "f"(_add_249), "f"(hc_eps));
        float _mul_47;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_47) : "f"(_fdiv_rn_59), "f"(hist[21]));
        float _shfl_xor_192 = __shfl_xor_sync(0xFFFFFFFF, _mul_47, 4);
        float _add_251;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_251) : "f"(_mul_47), "f"(_shfl_xor_192));
        float _shfl_xor_193 = __shfl_xor_sync(0xFFFFFFFF, _add_251, 24);
        float _add_252;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_252) : "f"(_add_251), "f"(_shfl_xor_193));
        float _sub_20;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(_fdiv_rn_59), "f"(_add_252));
        float _fdiv_rn_60 = __fdiv_rn(_sub_20, _add_250);
        float _mul_48;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_48) : "f"(_unpack_bf16_33[0]), "f"(_unpack_bf16_30[0]));
        dots_p_135[2] = _mul_48;
        float _fma_391;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_391) : "f"(_unpack_bf16_33[1]), "f"(_unpack_bf16_30[1]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_391;
        float _fma_392;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_392) : "f"(_unpack_bf16_33[2]), "f"(_unpack_bf16_30[2]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_392;
        float _fma_393;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_393) : "f"(_unpack_bf16_33[3]), "f"(_unpack_bf16_30[3]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_393;
        float _fma_394;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_394) : "f"(_unpack_bf16_33[4]), "f"(_unpack_bf16_30[4]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_394;
        float _fma_395;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_395) : "f"(_unpack_bf16_33[5]), "f"(_unpack_bf16_30[5]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_395;
        float _fma_396;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_396) : "f"(_unpack_bf16_33[6]), "f"(_unpack_bf16_30[6]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_396;
        float _fma_397;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_397) : "f"(_unpack_bf16_33[7]), "f"(_unpack_bf16_30[7]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_397;
        float _fma_398;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_398) : "f"(_unpack_bf16_33[8]), "f"(_unpack_bf16_30[8]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_398;
        float _fma_399;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_399) : "f"(_unpack_bf16_33[9]), "f"(_unpack_bf16_30[9]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_399;
        float _fma_400;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_400) : "f"(_unpack_bf16_33[10]), "f"(_unpack_bf16_30[10]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_400;
        float _fma_401;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_401) : "f"(_unpack_bf16_33[11]), "f"(_unpack_bf16_30[11]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_401;
        float _fma_402;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_402) : "f"(_unpack_bf16_33[12]), "f"(_unpack_bf16_30[12]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_402;
        float _fma_403;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_403) : "f"(_unpack_bf16_33[13]), "f"(_unpack_bf16_30[13]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_403;
        float _fma_404;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_404) : "f"(_unpack_bf16_33[14]), "f"(_unpack_bf16_30[14]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_404;
        float _fma_405;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_405) : "f"(_unpack_bf16_33[15]), "f"(_unpack_bf16_30[15]), "f"(dots_p_135[2]));
        dots_p_135[2] = _fma_405;
        unsigned int ring_pk_138[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[(0) + 3]))
            : "r"(sa_ + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_138[(4) + 3]))
            : "r"(sa_ + 4112 + lane32));
        float _unpack_bf16_34[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_34[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_138[_pair]) << 16);
            (&_unpack_bf16_34[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_138[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_194 = __shfl_xor_sync(0xFFFFFFFF, hist[19], 1);
        float _add_253;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_253) : "f"(hist[19]), "f"(_shfl_xor_194));
        float _shfl_xor_195 = __shfl_xor_sync(0xFFFFFFFF, _add_253, 2);
        float _add_254;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_254) : "f"(_add_253), "f"(_shfl_xor_195));
        float _add_255;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_255) : "f"(_add_254), "f"(hc_eps));
        float _mul_49;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_49) : "f"(_fdiv_rn_60), "f"(hist[20]));
        float _shfl_xor_196 = __shfl_xor_sync(0xFFFFFFFF, _mul_49, 1);
        float _add_256;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_256) : "f"(_mul_49), "f"(_shfl_xor_196));
        float _shfl_xor_197 = __shfl_xor_sync(0xFFFFFFFF, _add_256, 2);
        float _add_257;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_257) : "f"(_add_256), "f"(_shfl_xor_197));
        float _sub_21;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(_fdiv_rn_60), "f"(_add_257));
        float _fdiv_rn_61 = __fdiv_rn(_sub_21, _add_255);
        float _mul_50;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_50) : "f"(_unpack_bf16_34[0]), "f"(_unpack_bf16_30[0]));
        dots_p_135[3] = _mul_50;
        float _fma_406;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_406) : "f"(_unpack_bf16_34[1]), "f"(_unpack_bf16_30[1]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_406;
        float _fma_407;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_407) : "f"(_unpack_bf16_34[2]), "f"(_unpack_bf16_30[2]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_407;
        float _fma_408;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_408) : "f"(_unpack_bf16_34[3]), "f"(_unpack_bf16_30[3]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_408;
        float _fma_409;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_409) : "f"(_unpack_bf16_34[4]), "f"(_unpack_bf16_30[4]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_409;
        float _fma_410;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_410) : "f"(_unpack_bf16_34[5]), "f"(_unpack_bf16_30[5]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_410;
        float _fma_411;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_411) : "f"(_unpack_bf16_34[6]), "f"(_unpack_bf16_30[6]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_411;
        float _fma_412;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_412) : "f"(_unpack_bf16_34[7]), "f"(_unpack_bf16_30[7]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_412;
        float _fma_413;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_413) : "f"(_unpack_bf16_34[8]), "f"(_unpack_bf16_30[8]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_413;
        float _fma_414;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_414) : "f"(_unpack_bf16_34[9]), "f"(_unpack_bf16_30[9]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_414;
        float _fma_415;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_415) : "f"(_unpack_bf16_34[10]), "f"(_unpack_bf16_30[10]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_415;
        float _fma_416;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_416) : "f"(_unpack_bf16_34[11]), "f"(_unpack_bf16_30[11]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_416;
        float _fma_417;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_417) : "f"(_unpack_bf16_34[12]), "f"(_unpack_bf16_30[12]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_417;
        float _fma_418;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_418) : "f"(_unpack_bf16_34[13]), "f"(_unpack_bf16_30[13]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_418;
        float _fma_419;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_419) : "f"(_unpack_bf16_34[14]), "f"(_unpack_bf16_30[14]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_419;
        float _fma_420;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_420) : "f"(_unpack_bf16_34[15]), "f"(_unpack_bf16_30[15]), "f"(dots_p_135[3]));
        dots_p_135[3] = _fma_420;
        __syncwarp();
        if (elect_sync()) {
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
        float _shfl_xor_198 = __shfl_xor_sync(0xFFFFFFFF, hist[18], 4);
        float _add_258;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_258) : "f"(hist[18]), "f"(_shfl_xor_198));
        float _shfl_xor_199 = __shfl_xor_sync(0xFFFFFFFF, _add_258, 24);
        float _add_259;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_259) : "f"(_add_258), "f"(_shfl_xor_199));
        float _add_260;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_260) : "f"(_add_259), "f"(hc_eps));
        float _mul_51;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_51) : "f"(_fdiv_rn_61), "f"(hist[19]));
        float _shfl_xor_200 = __shfl_xor_sync(0xFFFFFFFF, _mul_51, 4);
        float _add_261;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_261) : "f"(_mul_51), "f"(_shfl_xor_200));
        float _shfl_xor_201 = __shfl_xor_sync(0xFFFFFFFF, _add_261, 24);
        float _add_262;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_262) : "f"(_add_261), "f"(_shfl_xor_201));
        float _sub_22;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(_fdiv_rn_61), "f"(_add_262));
        float _fdiv_rn_62 = __fdiv_rn(_sub_22, _add_260);
        int upper_139 = lane_1 & 16;
        float send_140 = ((upper_139 != 0) ? dots_p_135[0] : dots_p_135[2]);
        float keep_141 = ((upper_139 != 0) ? dots_p_135[2] : dots_p_135[0]);
        float _shfl_xor_202 = __shfl_xor_sync(0xFFFFFFFF, send_140, 16);
        float _add_263;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_263) : "f"(keep_141), "f"(_shfl_xor_202));
        float send_142 = ((upper_139 != 0) ? dots_p_135[1] : dots_p_135[3]);
        float keep_143 = ((upper_139 != 0) ? dots_p_135[3] : dots_p_135[1]);
        float _shfl_xor_203 = __shfl_xor_sync(0xFFFFFFFF, send_142, 16);
        float _add_264;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_264) : "f"(keep_143), "f"(_shfl_xor_203));
        int upper_144 = lane_1 & 8;
        float send_145 = ((upper_144 != 0) ? _add_263 : _add_264);
        float keep_146 = ((upper_144 != 0) ? _add_264 : _add_263);
        float _shfl_xor_204 = __shfl_xor_sync(0xFFFFFFFF, send_145, 8);
        float _add_265;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_265) : "f"(keep_146), "f"(_shfl_xor_204));
        float _shfl_xor_205 = __shfl_xor_sync(0xFFFFFFFF, hist[17], 1);
        float _add_266;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_266) : "f"(hist[17]), "f"(_shfl_xor_205));
        float _shfl_xor_206 = __shfl_xor_sync(0xFFFFFFFF, _add_266, 2);
        float _add_267;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_267) : "f"(_add_266), "f"(_shfl_xor_206));
        float _add_268;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_268) : "f"(_add_267), "f"(hc_eps));
        float _mul_52;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_52) : "f"(_fdiv_rn_62), "f"(hist[18]));
        float _shfl_xor_207 = __shfl_xor_sync(0xFFFFFFFF, _mul_52, 1);
        float _add_269;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_269) : "f"(_mul_52), "f"(_shfl_xor_207));
        float _shfl_xor_208 = __shfl_xor_sync(0xFFFFFFFF, _add_269, 2);
        float _add_270;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_270) : "f"(_add_269), "f"(_shfl_xor_208));
        float _sub_23;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(_fdiv_rn_62), "f"(_add_270));
        float _fdiv_rn_63 = __fdiv_rn(_sub_23, _add_268);
        float _shfl_xor_209 = __shfl_xor_sync(0xFFFFFFFF, _add_265, 4);
        float _add_271;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_271) : "f"(_add_265), "f"(_shfl_xor_209));
        float _shfl_xor_210 = __shfl_xor_sync(0xFFFFFFFF, _add_271, 2);
        float _add_272;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_272) : "f"(_add_271), "f"(_shfl_xor_210));
        float _shfl_xor_211 = __shfl_xor_sync(0xFFFFFFFF, _add_272, 1);
        float _add_273;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_273) : "f"(_add_272), "f"(_shfl_xor_211));
        dpre_part[own_base + 192] = _add_273;
        float _shfl_xor_212 = __shfl_xor_sync(0xFFFFFFFF, hist[16], 4);
        float _add_274;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_274) : "f"(hist[16]), "f"(_shfl_xor_212));
        float _shfl_xor_213 = __shfl_xor_sync(0xFFFFFFFF, _add_274, 24);
        float _add_275;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_275) : "f"(_add_274), "f"(_shfl_xor_213));
        float _add_276;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_276) : "f"(_add_275), "f"(hc_eps));
        float _mul_53;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_53) : "f"(_fdiv_rn_63), "f"(hist[17]));
        float _shfl_xor_214 = __shfl_xor_sync(0xFFFFFFFF, _mul_53, 4);
        float _add_277;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_277) : "f"(_mul_53), "f"(_shfl_xor_214));
        float _shfl_xor_215 = __shfl_xor_sync(0xFFFFFFFF, _add_277, 24);
        float _add_278;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_278) : "f"(_add_277), "f"(_shfl_xor_215));
        float _sub_24;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(_fdiv_rn_63), "f"(_add_278));
        float _fdiv_rn_64 = __fdiv_rn(_sub_24, _add_276);
        float _shfl_xor_216 = __shfl_xor_sync(0xFFFFFFFF, hist[15], 1);
        float _add_279;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_279) : "f"(hist[15]), "f"(_shfl_xor_216));
        float _shfl_xor_217 = __shfl_xor_sync(0xFFFFFFFF, _add_279, 2);
        float _add_280;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_280) : "f"(_add_279), "f"(_shfl_xor_217));
        float _add_281;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_281) : "f"(_add_280), "f"(hc_eps));
        float _mul_54;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_54) : "f"(_fdiv_rn_64), "f"(hist[16]));
        float _shfl_xor_218 = __shfl_xor_sync(0xFFFFFFFF, _mul_54, 1);
        float _add_282;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_282) : "f"(_mul_54), "f"(_shfl_xor_218));
        float _shfl_xor_219 = __shfl_xor_sync(0xFFFFFFFF, _add_282, 2);
        float _add_283;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_283) : "f"(_add_282), "f"(_shfl_xor_219));
        float _sub_25;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(_fdiv_rn_64), "f"(_add_283));
        float _fdiv_rn_65 = __fdiv_rn(_sub_25, _add_281);
        float _shfl_xor_220 = __shfl_xor_sync(0xFFFFFFFF, hist[14], 4);
        float _add_284;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_284) : "f"(hist[14]), "f"(_shfl_xor_220));
        float _shfl_xor_221 = __shfl_xor_sync(0xFFFFFFFF, _add_284, 24);
        float _add_285;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_285) : "f"(_add_284), "f"(_shfl_xor_221));
        float _add_286;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_286) : "f"(_add_285), "f"(hc_eps));
        float _mul_55;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_55) : "f"(_fdiv_rn_65), "f"(hist[15]));
        float _shfl_xor_222 = __shfl_xor_sync(0xFFFFFFFF, _mul_55, 4);
        float _add_287;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_287) : "f"(_mul_55), "f"(_shfl_xor_222));
        float _shfl_xor_223 = __shfl_xor_sync(0xFFFFFFFF, _add_287, 24);
        float _add_288;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_288) : "f"(_add_287), "f"(_shfl_xor_223));
        float _sub_26;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(_fdiv_rn_65), "f"(_add_288));
        float _fdiv_rn_66 = __fdiv_rn(_sub_26, _add_286);
        mbarrier_wait(ring_bar_addr + (st__3) * 8, 1);
        unsigned int ring_pk_147[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[(0) + 3]))
            : "r"(sa__4 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_147[(4) + 3]))
            : "r"(sa__4 + 16 + lane32));
        float _unpack_bf16_35[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_35[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_147[_pair]) << 16);
            (&_unpack_bf16_35[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_147[_pair]) & 0xffff0000u);
        }
        unsigned int ring_pk_148[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[(0) + 3]))
            : "r"(sa__4 + 1024 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_148[(4) + 3]))
            : "r"(sa__4 + 1040 + lane32));
        float _unpack_bf16_36[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_36[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_148[_pair]) << 16);
            (&_unpack_bf16_36[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_148[_pair]) & 0xffff0000u);
        }
        float dots_p_149[4];
        float _shfl_xor_224 = __shfl_xor_sync(0xFFFFFFFF, hist[13], 1);
        float _add_289;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_289) : "f"(hist[13]), "f"(_shfl_xor_224));
        float _shfl_xor_225 = __shfl_xor_sync(0xFFFFFFFF, _add_289, 2);
        float _add_290;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_290) : "f"(_add_289), "f"(_shfl_xor_225));
        float _add_291;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_291) : "f"(_add_290), "f"(hc_eps));
        float _mul_56;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_56) : "f"(_fdiv_rn_66), "f"(hist[14]));
        float _shfl_xor_226 = __shfl_xor_sync(0xFFFFFFFF, _mul_56, 1);
        float _add_292;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_292) : "f"(_mul_56), "f"(_shfl_xor_226));
        float _shfl_xor_227 = __shfl_xor_sync(0xFFFFFFFF, _add_292, 2);
        float _add_293;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_293) : "f"(_add_292), "f"(_shfl_xor_227));
        float _sub_27;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(_fdiv_rn_66), "f"(_add_293));
        float _fdiv_rn_67 = __fdiv_rn(_sub_27, _add_291);
        float _mul_57;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_57) : "f"(_unpack_bf16_36[0]), "f"(_unpack_bf16_35[0]));
        dots_p_149[0] = _mul_57;
        float _fma_421;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_421) : "f"(_unpack_bf16_36[1]), "f"(_unpack_bf16_35[1]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_421;
        float _fma_422;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_422) : "f"(_unpack_bf16_36[2]), "f"(_unpack_bf16_35[2]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_422;
        float _fma_423;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_423) : "f"(_unpack_bf16_36[3]), "f"(_unpack_bf16_35[3]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_423;
        float _fma_424;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_424) : "f"(_unpack_bf16_36[4]), "f"(_unpack_bf16_35[4]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_424;
        float _fma_425;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_425) : "f"(_unpack_bf16_36[5]), "f"(_unpack_bf16_35[5]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_425;
        float _fma_426;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_426) : "f"(_unpack_bf16_36[6]), "f"(_unpack_bf16_35[6]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_426;
        float _fma_427;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_427) : "f"(_unpack_bf16_36[7]), "f"(_unpack_bf16_35[7]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_427;
        float _fma_428;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_428) : "f"(_unpack_bf16_36[8]), "f"(_unpack_bf16_35[8]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_428;
        float _fma_429;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_429) : "f"(_unpack_bf16_36[9]), "f"(_unpack_bf16_35[9]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_429;
        float _fma_430;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_430) : "f"(_unpack_bf16_36[10]), "f"(_unpack_bf16_35[10]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_430;
        float _fma_431;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_431) : "f"(_unpack_bf16_36[11]), "f"(_unpack_bf16_35[11]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_431;
        float _fma_432;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_432) : "f"(_unpack_bf16_36[12]), "f"(_unpack_bf16_35[12]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_432;
        float _fma_433;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_433) : "f"(_unpack_bf16_36[13]), "f"(_unpack_bf16_35[13]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_433;
        float _fma_434;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_434) : "f"(_unpack_bf16_36[14]), "f"(_unpack_bf16_35[14]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_434;
        float _fma_435;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_435) : "f"(_unpack_bf16_36[15]), "f"(_unpack_bf16_35[15]), "f"(dots_p_149[0]));
        dots_p_149[0] = _fma_435;
        unsigned int ring_pk_150[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[(0) + 3]))
            : "r"(sa__4 + 2048 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_150[(4) + 3]))
            : "r"(sa__4 + 2064 + lane32));
        float _unpack_bf16_37[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_37[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_150[_pair]) << 16);
            (&_unpack_bf16_37[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_150[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_228 = __shfl_xor_sync(0xFFFFFFFF, hist[12], 4);
        float _add_294;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_294) : "f"(hist[12]), "f"(_shfl_xor_228));
        float _shfl_xor_229 = __shfl_xor_sync(0xFFFFFFFF, _add_294, 24);
        float _add_295;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_295) : "f"(_add_294), "f"(_shfl_xor_229));
        float _add_296;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_296) : "f"(_add_295), "f"(hc_eps));
        float _mul_58;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_58) : "f"(_fdiv_rn_67), "f"(hist[13]));
        float _shfl_xor_230 = __shfl_xor_sync(0xFFFFFFFF, _mul_58, 4);
        float _add_297;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_297) : "f"(_mul_58), "f"(_shfl_xor_230));
        float _shfl_xor_231 = __shfl_xor_sync(0xFFFFFFFF, _add_297, 24);
        float _add_298;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_298) : "f"(_add_297), "f"(_shfl_xor_231));
        float _sub_28;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(_fdiv_rn_67), "f"(_add_298));
        float _fdiv_rn_68 = __fdiv_rn(_sub_28, _add_296);
        float _mul_59;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_59) : "f"(_unpack_bf16_37[0]), "f"(_unpack_bf16_35[0]));
        dots_p_149[1] = _mul_59;
        float _fma_436;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_436) : "f"(_unpack_bf16_37[1]), "f"(_unpack_bf16_35[1]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_436;
        float _fma_437;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_437) : "f"(_unpack_bf16_37[2]), "f"(_unpack_bf16_35[2]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_437;
        float _fma_438;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_438) : "f"(_unpack_bf16_37[3]), "f"(_unpack_bf16_35[3]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_438;
        float _fma_439;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_439) : "f"(_unpack_bf16_37[4]), "f"(_unpack_bf16_35[4]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_439;
        float _fma_440;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_440) : "f"(_unpack_bf16_37[5]), "f"(_unpack_bf16_35[5]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_440;
        float _fma_441;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_441) : "f"(_unpack_bf16_37[6]), "f"(_unpack_bf16_35[6]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_441;
        float _fma_442;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_442) : "f"(_unpack_bf16_37[7]), "f"(_unpack_bf16_35[7]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_442;
        float _fma_443;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_443) : "f"(_unpack_bf16_37[8]), "f"(_unpack_bf16_35[8]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_443;
        float _fma_444;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_444) : "f"(_unpack_bf16_37[9]), "f"(_unpack_bf16_35[9]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_444;
        float _fma_445;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_445) : "f"(_unpack_bf16_37[10]), "f"(_unpack_bf16_35[10]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_445;
        float _fma_446;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_446) : "f"(_unpack_bf16_37[11]), "f"(_unpack_bf16_35[11]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_446;
        float _fma_447;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_447) : "f"(_unpack_bf16_37[12]), "f"(_unpack_bf16_35[12]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_447;
        float _fma_448;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_448) : "f"(_unpack_bf16_37[13]), "f"(_unpack_bf16_35[13]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_448;
        float _fma_449;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_449) : "f"(_unpack_bf16_37[14]), "f"(_unpack_bf16_35[14]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_449;
        float _fma_450;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_450) : "f"(_unpack_bf16_37[15]), "f"(_unpack_bf16_35[15]), "f"(dots_p_149[1]));
        dots_p_149[1] = _fma_450;
        unsigned int ring_pk_151[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[(0) + 3]))
            : "r"(sa__4 + 3072 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_151[(4) + 3]))
            : "r"(sa__4 + 3088 + lane32));
        float _unpack_bf16_38[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_38[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_151[_pair]) << 16);
            (&_unpack_bf16_38[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_151[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_232 = __shfl_xor_sync(0xFFFFFFFF, hist[11], 1);
        float _add_299;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_299) : "f"(hist[11]), "f"(_shfl_xor_232));
        float _shfl_xor_233 = __shfl_xor_sync(0xFFFFFFFF, _add_299, 2);
        float _add_300;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_300) : "f"(_add_299), "f"(_shfl_xor_233));
        float _add_301;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_301) : "f"(_add_300), "f"(hc_eps));
        float _mul_60;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_60) : "f"(_fdiv_rn_68), "f"(hist[12]));
        float _shfl_xor_234 = __shfl_xor_sync(0xFFFFFFFF, _mul_60, 1);
        float _add_302;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_302) : "f"(_mul_60), "f"(_shfl_xor_234));
        float _shfl_xor_235 = __shfl_xor_sync(0xFFFFFFFF, _add_302, 2);
        float _add_303;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_303) : "f"(_add_302), "f"(_shfl_xor_235));
        float _sub_29;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(_fdiv_rn_68), "f"(_add_303));
        float _fdiv_rn_69 = __fdiv_rn(_sub_29, _add_301);
        float _mul_61;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_61) : "f"(_unpack_bf16_38[0]), "f"(_unpack_bf16_35[0]));
        dots_p_149[2] = _mul_61;
        float _fma_451;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_451) : "f"(_unpack_bf16_38[1]), "f"(_unpack_bf16_35[1]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_451;
        float _fma_452;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_452) : "f"(_unpack_bf16_38[2]), "f"(_unpack_bf16_35[2]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_452;
        float _fma_453;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_453) : "f"(_unpack_bf16_38[3]), "f"(_unpack_bf16_35[3]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_453;
        float _fma_454;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_454) : "f"(_unpack_bf16_38[4]), "f"(_unpack_bf16_35[4]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_454;
        float _fma_455;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_455) : "f"(_unpack_bf16_38[5]), "f"(_unpack_bf16_35[5]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_455;
        float _fma_456;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_456) : "f"(_unpack_bf16_38[6]), "f"(_unpack_bf16_35[6]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_456;
        float _fma_457;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_457) : "f"(_unpack_bf16_38[7]), "f"(_unpack_bf16_35[7]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_457;
        float _fma_458;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_458) : "f"(_unpack_bf16_38[8]), "f"(_unpack_bf16_35[8]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_458;
        float _fma_459;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_459) : "f"(_unpack_bf16_38[9]), "f"(_unpack_bf16_35[9]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_459;
        float _fma_460;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_460) : "f"(_unpack_bf16_38[10]), "f"(_unpack_bf16_35[10]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_460;
        float _fma_461;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_461) : "f"(_unpack_bf16_38[11]), "f"(_unpack_bf16_35[11]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_461;
        float _fma_462;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_462) : "f"(_unpack_bf16_38[12]), "f"(_unpack_bf16_35[12]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_462;
        float _fma_463;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_463) : "f"(_unpack_bf16_38[13]), "f"(_unpack_bf16_35[13]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_463;
        float _fma_464;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_464) : "f"(_unpack_bf16_38[14]), "f"(_unpack_bf16_35[14]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_464;
        float _fma_465;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_465) : "f"(_unpack_bf16_38[15]), "f"(_unpack_bf16_35[15]), "f"(dots_p_149[2]));
        dots_p_149[2] = _fma_465;
        unsigned int ring_pk_152[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[(0) + 3]))
            : "r"(sa__4 + 4096 + lane32));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_152[(4) + 3]))
            : "r"(sa__4 + 4112 + lane32));
        float _unpack_bf16_39[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_39[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_pk_152[_pair]) << 16);
            (&_unpack_bf16_39[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_pk_152[_pair]) & 0xffff0000u);
        }
        float _shfl_xor_236 = __shfl_xor_sync(0xFFFFFFFF, hist[10], 4);
        float _add_304;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_304) : "f"(hist[10]), "f"(_shfl_xor_236));
        float _shfl_xor_237 = __shfl_xor_sync(0xFFFFFFFF, _add_304, 24);
        float _add_305;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_305) : "f"(_add_304), "f"(_shfl_xor_237));
        float _add_306;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_306) : "f"(_add_305), "f"(hc_eps));
        float _mul_62;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_62) : "f"(_fdiv_rn_69), "f"(hist[11]));
        float _shfl_xor_238 = __shfl_xor_sync(0xFFFFFFFF, _mul_62, 4);
        float _add_307;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_307) : "f"(_mul_62), "f"(_shfl_xor_238));
        float _shfl_xor_239 = __shfl_xor_sync(0xFFFFFFFF, _add_307, 24);
        float _add_308;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_308) : "f"(_add_307), "f"(_shfl_xor_239));
        float _sub_30;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(_fdiv_rn_69), "f"(_add_308));
        float _fdiv_rn_70 = __fdiv_rn(_sub_30, _add_306);
        float _mul_63;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_63) : "f"(_unpack_bf16_39[0]), "f"(_unpack_bf16_35[0]));
        dots_p_149[3] = _mul_63;
        float _fma_466;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_466) : "f"(_unpack_bf16_39[1]), "f"(_unpack_bf16_35[1]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_466;
        float _fma_467;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_467) : "f"(_unpack_bf16_39[2]), "f"(_unpack_bf16_35[2]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_467;
        float _fma_468;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_468) : "f"(_unpack_bf16_39[3]), "f"(_unpack_bf16_35[3]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_468;
        float _fma_469;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_469) : "f"(_unpack_bf16_39[4]), "f"(_unpack_bf16_35[4]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_469;
        float _fma_470;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_470) : "f"(_unpack_bf16_39[5]), "f"(_unpack_bf16_35[5]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_470;
        float _fma_471;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_471) : "f"(_unpack_bf16_39[6]), "f"(_unpack_bf16_35[6]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_471;
        float _fma_472;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_472) : "f"(_unpack_bf16_39[7]), "f"(_unpack_bf16_35[7]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_472;
        float _fma_473;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_473) : "f"(_unpack_bf16_39[8]), "f"(_unpack_bf16_35[8]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_473;
        float _fma_474;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_474) : "f"(_unpack_bf16_39[9]), "f"(_unpack_bf16_35[9]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_474;
        float _fma_475;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_475) : "f"(_unpack_bf16_39[10]), "f"(_unpack_bf16_35[10]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_475;
        float _fma_476;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_476) : "f"(_unpack_bf16_39[11]), "f"(_unpack_bf16_35[11]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_476;
        float _fma_477;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_477) : "f"(_unpack_bf16_39[12]), "f"(_unpack_bf16_35[12]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_477;
        float _fma_478;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_478) : "f"(_unpack_bf16_39[13]), "f"(_unpack_bf16_35[13]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_478;
        float _fma_479;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_479) : "f"(_unpack_bf16_39[14]), "f"(_unpack_bf16_35[14]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_479;
        float _fma_480;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_480) : "f"(_unpack_bf16_39[15]), "f"(_unpack_bf16_35[15]), "f"(dots_p_149[3]));
        dots_p_149[3] = _fma_480;
        __syncwarp();
        if (elect_sync()) {
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
        float _shfl_xor_240 = __shfl_xor_sync(0xFFFFFFFF, hist[9], 1);
        float _add_309;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_309) : "f"(hist[9]), "f"(_shfl_xor_240));
        float _shfl_xor_241 = __shfl_xor_sync(0xFFFFFFFF, _add_309, 2);
        float _add_310;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_310) : "f"(_add_309), "f"(_shfl_xor_241));
        float _add_311;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_311) : "f"(_add_310), "f"(hc_eps));
        float _mul_64;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_64) : "f"(_fdiv_rn_70), "f"(hist[10]));
        float _shfl_xor_242 = __shfl_xor_sync(0xFFFFFFFF, _mul_64, 1);
        float _add_312;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_312) : "f"(_mul_64), "f"(_shfl_xor_242));
        float _shfl_xor_243 = __shfl_xor_sync(0xFFFFFFFF, _add_312, 2);
        float _add_313;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_313) : "f"(_add_312), "f"(_shfl_xor_243));
        float _sub_31;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(_fdiv_rn_70), "f"(_add_313));
        float _fdiv_rn_71 = __fdiv_rn(_sub_31, _add_311);
        int upper_153 = lane_1 & 16;
        float send_154 = ((upper_153 != 0) ? dots_p_149[0] : dots_p_149[2]);
        float keep_155 = ((upper_153 != 0) ? dots_p_149[2] : dots_p_149[0]);
        float _shfl_xor_244 = __shfl_xor_sync(0xFFFFFFFF, send_154, 16);
        float _add_314;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_314) : "f"(keep_155), "f"(_shfl_xor_244));
        float send_156 = ((upper_153 != 0) ? dots_p_149[1] : dots_p_149[3]);
        float keep_157 = ((upper_153 != 0) ? dots_p_149[3] : dots_p_149[1]);
        float _shfl_xor_245 = __shfl_xor_sync(0xFFFFFFFF, send_156, 16);
        float _add_315;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_315) : "f"(keep_157), "f"(_shfl_xor_245));
        int upper_158 = lane_1 & 8;
        float send_159 = ((upper_158 != 0) ? _add_314 : _add_315);
        float keep_160 = ((upper_158 != 0) ? _add_315 : _add_314);
        float _shfl_xor_246 = __shfl_xor_sync(0xFFFFFFFF, send_159, 8);
        float _add_316;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_316) : "f"(keep_160), "f"(_shfl_xor_246));
        float _shfl_xor_247 = __shfl_xor_sync(0xFFFFFFFF, hist[8], 4);
        float _add_317;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_317) : "f"(hist[8]), "f"(_shfl_xor_247));
        float _shfl_xor_248 = __shfl_xor_sync(0xFFFFFFFF, _add_317, 24);
        float _add_318;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_318) : "f"(_add_317), "f"(_shfl_xor_248));
        float _add_319;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_319) : "f"(_add_318), "f"(hc_eps));
        float _mul_65;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_65) : "f"(_fdiv_rn_71), "f"(hist[9]));
        float _shfl_xor_249 = __shfl_xor_sync(0xFFFFFFFF, _mul_65, 4);
        float _add_320;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_320) : "f"(_mul_65), "f"(_shfl_xor_249));
        float _shfl_xor_250 = __shfl_xor_sync(0xFFFFFFFF, _add_320, 24);
        float _add_321;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_321) : "f"(_add_320), "f"(_shfl_xor_250));
        float _sub_32;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(_fdiv_rn_71), "f"(_add_321));
        float _fdiv_rn_72 = __fdiv_rn(_sub_32, _add_319);
        float _shfl_xor_251 = __shfl_xor_sync(0xFFFFFFFF, _add_316, 4);
        float _add_322;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_322) : "f"(_add_316), "f"(_shfl_xor_251));
        float _shfl_xor_252 = __shfl_xor_sync(0xFFFFFFFF, _add_322, 2);
        float _add_323;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_323) : "f"(_add_322), "f"(_shfl_xor_252));
        float _shfl_xor_253 = __shfl_xor_sync(0xFFFFFFFF, _add_323, 1);
        float _add_324;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_324) : "f"(_add_323), "f"(_shfl_xor_253));
        dpre_part[own_base + 224] = _add_324;
        float _shfl_xor_254 = __shfl_xor_sync(0xFFFFFFFF, hist[7], 1);
        float _add_325;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_325) : "f"(hist[7]), "f"(_shfl_xor_254));
        float _shfl_xor_255 = __shfl_xor_sync(0xFFFFFFFF, _add_325, 2);
        float _add_326;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_326) : "f"(_add_325), "f"(_shfl_xor_255));
        float _add_327;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_327) : "f"(_add_326), "f"(hc_eps));
        float _mul_66;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_66) : "f"(_fdiv_rn_72), "f"(hist[8]));
        float _shfl_xor_256 = __shfl_xor_sync(0xFFFFFFFF, _mul_66, 1);
        float _add_328;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_328) : "f"(_mul_66), "f"(_shfl_xor_256));
        float _shfl_xor_257 = __shfl_xor_sync(0xFFFFFFFF, _add_328, 2);
        float _add_329;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_329) : "f"(_add_328), "f"(_shfl_xor_257));
        float _sub_33;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33) : "f"(_fdiv_rn_72), "f"(_add_329));
        float _fdiv_rn_73 = __fdiv_rn(_sub_33, _add_327);
        float _shfl_xor_258 = __shfl_xor_sync(0xFFFFFFFF, hist[6], 4);
        float _add_330;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_330) : "f"(hist[6]), "f"(_shfl_xor_258));
        float _shfl_xor_259 = __shfl_xor_sync(0xFFFFFFFF, _add_330, 24);
        float _add_331;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_331) : "f"(_add_330), "f"(_shfl_xor_259));
        float _add_332;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_332) : "f"(_add_331), "f"(hc_eps));
        float _mul_67;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_67) : "f"(_fdiv_rn_73), "f"(hist[7]));
        float _shfl_xor_260 = __shfl_xor_sync(0xFFFFFFFF, _mul_67, 4);
        float _add_333;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_333) : "f"(_mul_67), "f"(_shfl_xor_260));
        float _shfl_xor_261 = __shfl_xor_sync(0xFFFFFFFF, _add_333, 24);
        float _add_334;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_334) : "f"(_add_333), "f"(_shfl_xor_261));
        float _sub_34;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(_fdiv_rn_73), "f"(_add_334));
        float _fdiv_rn_74 = __fdiv_rn(_sub_34, _add_332);
        float _shfl_xor_262 = __shfl_xor_sync(0xFFFFFFFF, hist[5], 1);
        float _add_335;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_335) : "f"(hist[5]), "f"(_shfl_xor_262));
        float _shfl_xor_263 = __shfl_xor_sync(0xFFFFFFFF, _add_335, 2);
        float _add_336;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_336) : "f"(_add_335), "f"(_shfl_xor_263));
        float _add_337;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_337) : "f"(_add_336), "f"(hc_eps));
        float _mul_68;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_68) : "f"(_fdiv_rn_74), "f"(hist[6]));
        float _shfl_xor_264 = __shfl_xor_sync(0xFFFFFFFF, _mul_68, 1);
        float _add_338;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_338) : "f"(_mul_68), "f"(_shfl_xor_264));
        float _shfl_xor_265 = __shfl_xor_sync(0xFFFFFFFF, _add_338, 2);
        float _add_339;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_339) : "f"(_add_338), "f"(_shfl_xor_265));
        float _sub_35;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(_fdiv_rn_74), "f"(_add_339));
        float _fdiv_rn_75 = __fdiv_rn(_sub_35, _add_337);
        float _shfl_xor_266 = __shfl_xor_sync(0xFFFFFFFF, hist[4], 4);
        float _add_340;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_340) : "f"(hist[4]), "f"(_shfl_xor_266));
        float _shfl_xor_267 = __shfl_xor_sync(0xFFFFFFFF, _add_340, 24);
        float _add_341;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_341) : "f"(_add_340), "f"(_shfl_xor_267));
        float _add_342;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_342) : "f"(_add_341), "f"(hc_eps));
        float _mul_69;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_69) : "f"(_fdiv_rn_75), "f"(hist[5]));
        float _shfl_xor_268 = __shfl_xor_sync(0xFFFFFFFF, _mul_69, 4);
        float _add_343;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_343) : "f"(_mul_69), "f"(_shfl_xor_268));
        float _shfl_xor_269 = __shfl_xor_sync(0xFFFFFFFF, _add_343, 24);
        float _add_344;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_344) : "f"(_add_343), "f"(_shfl_xor_269));
        float _sub_36;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(_fdiv_rn_75), "f"(_add_344));
        float _fdiv_rn_76 = __fdiv_rn(_sub_36, _add_342);
        float _shfl_xor_270 = __shfl_xor_sync(0xFFFFFFFF, hist[3], 1);
        float _add_345;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_345) : "f"(hist[3]), "f"(_shfl_xor_270));
        float _shfl_xor_271 = __shfl_xor_sync(0xFFFFFFFF, _add_345, 2);
        float _add_346;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_346) : "f"(_add_345), "f"(_shfl_xor_271));
        float _add_347;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_347) : "f"(_add_346), "f"(hc_eps));
        float _mul_70;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_70) : "f"(_fdiv_rn_76), "f"(hist[4]));
        float _shfl_xor_272 = __shfl_xor_sync(0xFFFFFFFF, _mul_70, 1);
        float _add_348;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_348) : "f"(_mul_70), "f"(_shfl_xor_272));
        float _shfl_xor_273 = __shfl_xor_sync(0xFFFFFFFF, _add_348, 2);
        float _add_349;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_349) : "f"(_add_348), "f"(_shfl_xor_273));
        float _sub_37;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(_fdiv_rn_76), "f"(_add_349));
        float _fdiv_rn_77 = __fdiv_rn(_sub_37, _add_347);
        float _shfl_xor_274 = __shfl_xor_sync(0xFFFFFFFF, hist[2], 4);
        float _add_350;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_350) : "f"(hist[2]), "f"(_shfl_xor_274));
        float _shfl_xor_275 = __shfl_xor_sync(0xFFFFFFFF, _add_350, 24);
        float _add_351;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_351) : "f"(_add_350), "f"(_shfl_xor_275));
        float _add_352;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_352) : "f"(_add_351), "f"(hc_eps));
        float _mul_71;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_71) : "f"(_fdiv_rn_77), "f"(hist[3]));
        float _shfl_xor_276 = __shfl_xor_sync(0xFFFFFFFF, _mul_71, 4);
        float _add_353;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_353) : "f"(_mul_71), "f"(_shfl_xor_276));
        float _shfl_xor_277 = __shfl_xor_sync(0xFFFFFFFF, _add_353, 24);
        float _add_354;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_354) : "f"(_add_353), "f"(_shfl_xor_277));
        float _sub_38;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(_fdiv_rn_77), "f"(_add_354));
        float _fdiv_rn_78 = __fdiv_rn(_sub_38, _add_352);
        float _shfl_xor_278 = __shfl_xor_sync(0xFFFFFFFF, hist[1], 1);
        float _add_355;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_355) : "f"(hist[1]), "f"(_shfl_xor_278));
        float _shfl_xor_279 = __shfl_xor_sync(0xFFFFFFFF, _add_355, 2);
        float _add_356;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_356) : "f"(_add_355), "f"(_shfl_xor_279));
        float _add_357;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_357) : "f"(_add_356), "f"(hc_eps));
        float _mul_72;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_72) : "f"(_fdiv_rn_78), "f"(hist[2]));
        float _shfl_xor_280 = __shfl_xor_sync(0xFFFFFFFF, _mul_72, 1);
        float _add_358;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_358) : "f"(_mul_72), "f"(_shfl_xor_280));
        float _shfl_xor_281 = __shfl_xor_sync(0xFFFFFFFF, _add_358, 2);
        float _add_359;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_359) : "f"(_add_358), "f"(_shfl_xor_281));
        float _sub_39;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(_fdiv_rn_78), "f"(_add_359));
        float _fdiv_rn_79 = __fdiv_rn(_sub_39, _add_357);
        float _add_360;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_360) : "f"(hist[0]), "f"(hc_eps));
        float _shfl_xor_282 = __shfl_xor_sync(0xFFFFFFFF, _add_360, 4);
        float _add_361;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_361) : "f"(_add_360), "f"(_shfl_xor_282));
        float _shfl_xor_283 = __shfl_xor_sync(0xFFFFFFFF, _add_361, 24);
        float _add_362;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_362) : "f"(_add_361), "f"(_shfl_xor_283));
        float _add_363;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_363) : "f"(_add_362), "f"(hc_eps));
        float _mul_73;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_73) : "f"(_fdiv_rn_79), "f"(hist[1]));
        float _shfl_xor_284 = __shfl_xor_sync(0xFFFFFFFF, _mul_73, 4);
        float _add_364;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_364) : "f"(_mul_73), "f"(_shfl_xor_284));
        float _shfl_xor_285 = __shfl_xor_sync(0xFFFFFFFF, _add_364, 24);
        float _add_365;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_365) : "f"(_add_364), "f"(_shfl_xor_285));
        float _sub_40;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(_fdiv_rn_79), "f"(_add_365));
        float _fdiv_rn_80 = __fdiv_rn(_sub_40, _add_363);
        float acc_161 = 0.0f;
        float _shfl_xor_286 = __shfl_xor_sync(0xFFFFFFFF, hist[0], 1);
        float _shfl_xor_287 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_80, 1);
        float _sub_41;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(_fdiv_rn_80), "f"(_shfl_xor_287));
        float _fma_481;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_481) : "f"(_shfl_xor_286), "f"(_sub_41), "f"(acc_161));
        float _shfl_xor_288 = __shfl_xor_sync(0xFFFFFFFF, hist[0], 2);
        float _shfl_xor_289 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_80, 2);
        float _sub_42;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(_fdiv_rn_80), "f"(_shfl_xor_289));
        float _fma_482;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_482) : "f"(_shfl_xor_288), "f"(_sub_42), "f"(_fma_481));
        float _shfl_xor_290 = __shfl_xor_sync(0xFFFFFFFF, hist[0], 3);
        float _shfl_xor_291 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_80, 3);
        float _sub_43;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(_fdiv_rn_80), "f"(_shfl_xor_291));
        float _fma_483;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_483) : "f"(_shfl_xor_290), "f"(_sub_43), "f"(_fma_482));
        float _mul_74;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_74) : "f"(hist[0]), "f"(_fma_483));
        float _mul_75;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_75) : "f"(1.0f), "f"(dsig));
        float da = ((lane_1 >= 8 && lane_1 < 24) ? _mul_74 : _mul_75);
        __syncthreads();
        dp[0] = dpre_part[cur_base + lc4];
        float _add_366;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_366) : "f"(dp[0]), "f"(dpre_part[cur_base + 4 + lc4]));
        dp[0] = _add_366;
        float _add_367;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_367) : "f"(dp[0]), "f"(dpre_part[cur_base + 8 + lc4]));
        dp[0] = _add_367;
        float _add_368;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_368) : "f"(dp[0]), "f"(dpre_part[cur_base + 12 + lc4]));
        dp[0] = _add_368;
        float _add_369;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_369) : "f"(dp[0]), "f"(dpre_part[cur_base + 16 + lc4]));
        dp[0] = _add_369;
        float _add_370;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_370) : "f"(dp[0]), "f"(dpre_part[cur_base + 20 + lc4]));
        dp[0] = _add_370;
        float _add_371;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_371) : "f"(dp[0]), "f"(dpre_part[cur_base + 24 + lc4]));
        dp[0] = _add_371;
        float _add_372;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_372) : "f"(dp[0]), "f"(dpre_part[cur_base + 28 + lc4]));
        dp[0] = _add_372;
        float gsig = ((lane_1 < 4) ? dp[0] : gtwo);
        float _mul_76;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_76) : "f"(gsig), "f"(da));
        float da_sig = _mul_76;
        float da_late = ((lane_1 >= 8 && lane_1 < 24) ? da : da_sig);
        float da_162 = ((lane_1 < 24) ? da_late : 0.0f);
        float _mul_77;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_77) : "f"(da_162), "f"(sc));
        float dz = _mul_77;
        float _mul_78;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_78) : "f"(da_162), "f"(z));
        float daz = _mul_78;
        float _mul_79;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_79) : "f"(dz), "f"(z));
        float dzz = _mul_79;
        float _warp_reduce_0 = dzz;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        float _mul_80;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_80) : "f"(r), "f"(r));
        float rr = _mul_80;
        float _mul_81;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_81) : "f"(_warp_reduce_0), "f"(6.103515625e-05f));
        float _mul_82;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_82) : "f"(rr), "f"(_mul_81));
        float c_t = _mul_82;
        float _mul_83;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_83) : "f"(dz), "f"(r));
        float dzr = _mul_83;
        __nv_bfloat16 _cvt_bf16_0 = (__nv_bfloat16)dzr;
        float _cvt_f32_0 = (float)_cvt_bf16_0;
        float _sub_44;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(dzr), "f"(_cvt_f32_0));
        __nv_bfloat16 _cvt_bf16_1 = (__nv_bfloat16)_sub_44;
        float _cvt_f32_1 = (float)_cvt_bf16_1;
        float _sub_45;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(_sub_44), "f"(_cvt_f32_1));
        __nv_bfloat16 _cvt_bf16_2 = (__nv_bfloat16)_sub_45;
        float _cvt_f32_2 = (float)_cvt_bf16_2;
        float p1 = ((lane_1 < 24) ? _cvt_f32_0 : 0.0f);
        float p2 = ((lane_1 < 24) ? _cvt_f32_1 : 0.0f);
        float p3 = ((lane_1 < 24) ? _cvt_f32_2 : 0.0f);
        long long drow = tok64 * 96 + lane64;
        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow])), "f"(p1), "r"((unsigned int)(tok < t1)) : "memory");
        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow + 32])), "f"(p2), "r"((unsigned int)(tok < t1)) : "memory");
        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow + 64])), "f"(p3), "r"((unsigned int)(tok < t1)) : "memory");
        long long trow = tok64 * 24;
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow])), "f"(r), "r"((unsigned int)(tok < t1 && lane_1 == 0)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + 1])), "f"(c_t), "r"((unsigned int)(tok < t1 && lane_1 == 0)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + 2 + lane64])), "f"(_add_1), "r"((unsigned int)(tok < t1 && lane_1 < 4)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + lane64 - 2])), "f"(_fdiv_rn_41), "r"((unsigned int)(tok < t1 && (lane_1 >= 8 && lane_1 < 24))) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + lane64 - 2])), "f"(0.0f), "r"((unsigned int)(tok < t1 && (lane_1 >= 24 && lane_1 < 26))) : "memory");
        float _sub_46;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(da_162), "f"(comp[0]));
        float _add_373;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_373) : "f"(acc[0]), "f"(_sub_46));
        float _sub_47;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(_add_373), "f"(acc[0]));
        float _sub_48;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(_sub_47), "f"(_sub_46));
        comp[0] = ((tok < t1) ? _sub_48 : comp[0]);
        acc[0] = ((tok < t1) ? _add_373 : acc[0]);
        float _sub_49;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(daz), "f"(comp[1]));
        float _add_374;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_374) : "f"(acc[1]), "f"(_sub_49));
        float _sub_50;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(_add_374), "f"(acc[1]));
        float _sub_51;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(_sub_50), "f"(_sub_49));
        comp[1] = ((tok < t1) ? _sub_51 : comp[1]);
        acc[1] = ((tok < t1) ? _add_374 : acc[1]);
        gi[0] = gi[0] + 1;
    }
    if (t0 < t1) {
        mbarrier_wait(ring_bar_addr + (st_) * 8, 0);
        mbarrier_wait(ring_bar_addr + (st__3) * 8, 0);
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
            float _sub_52;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(fold[src]), "f"(oc[0]));
            float _add_375;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_375) : "f"(o[0]), "f"(_sub_52));
            float _sub_53;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(_add_375), "f"(o[0]));
            float _sub_54;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(_sub_53), "f"(_sub_52));
            oc[0] = _sub_54;
            o[0] = _add_375;
            float _sub_55;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(0.0f), "f"(fold[src + 1]));
            float _sub_56;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56) : "f"(_sub_55), "f"(oc[0]));
            float _add_376;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_376) : "f"(o[0]), "f"(_sub_56));
            float _sub_57;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57) : "f"(_add_376), "f"(o[0]));
            float _sub_58;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_58) : "f"(_sub_57), "f"(_sub_56));
            oc[0] = _sub_58;
            o[0] = _add_376;
            float _sub_59;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(fold[src + 2]), "f"(oc[1]));
            float _add_377;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_377) : "f"(o[1]), "f"(_sub_59));
            float _sub_60;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60) : "f"(_add_377), "f"(o[1]));
            float _sub_61;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(_sub_60), "f"(_sub_59));
            oc[1] = _sub_61;
            o[1] = _add_377;
            float _sub_62;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62) : "f"(0.0f), "f"(fold[src + 3]));
            float _sub_63;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(_sub_62), "f"(oc[1]));
            float _add_378;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_378) : "f"(o[1]), "f"(_sub_63));
            float _sub_64;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_64) : "f"(_add_378), "f"(o[1]));
            float _sub_65;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_65) : "f"(_sub_64), "f"(_sub_63));
            oc[1] = _sub_65;
            o[1] = _add_378;
        }
        float _sub_66;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(o[0]), "f"(oc[0]));
        float dbase_n = _sub_66;
        float _sub_67;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(o[1]), "f"(oc[1]));
        float dsz_n = _sub_67;
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
                float _sub_68;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(gsum[mrow]), "f"(gc[0]));
                float _add_379;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_379) : "f"(ga[0]), "f"(_sub_68));
                float _sub_69;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(_add_379), "f"(ga[0]));
                float _sub_70;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(_sub_69), "f"(_sub_68));
                gc[0] = _sub_70;
                ga[0] = _add_379;
            }
            float _sub_71;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(ga[0]), "f"(gc[0]));
            *(reinterpret_cast<float*>(param_partial + (prow + 24 + lane64)) + (0)) = _sub_71;
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
                *(reinterpret_cast<__nv_bfloat16*>(dzp + (t64 * 96 + (long long)ez)) + (0)) = __float2bfloat16_rn(zero);
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
