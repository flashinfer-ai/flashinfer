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
#define SMEM_RING_STAGE_BYTES 102400
#define SMEM_RING_STRIDE 102400
#define SMEM_TOTAL 111616
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
kernel_cake_mhc_det_train_bwd_token__ao22_rae7_pd1_ntr__h5120(__nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ dU, float* __restrict__ stats, float* __restrict__ gstats, float* __restrict__ scale, float* __restrict__ base, __nv_bfloat16* __restrict__ dzp, float* __restrict__ tokp, float* __restrict__ param_partial, int T, int t_pad, int rows_per_cta, float hc_eps)
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
    long long wcol = (long long)warp_0 * 640;
    unsigned int lane32 = (unsigned int)(lane_1 * 40);
    int st__1_0 = warp_0 * 2;
    unsigned int sa__1_0 = (unsigned int)(ring_addr + (unsigned int)(st__1_0 * 6400));
    int st__1_1 = warp_0 * 2 + 1;
    unsigned int sa__1_1 = (unsigned int)(ring_addr + (unsigned int)(st__1_1 * 6400));
    if (t0 < t1) {
        int tp_1_0 = t0;
        int tpc_2_0 = ((tp_1_0 < t1) ? tp_1_0 : t_last);
        long long tp2_0 = (long long)tpc_2_0;
        int tp_1_1 = t0 + 1;
        int tpc_2_1 = ((tp_1_1 < t1) ? tp_1_1 : t_last);
        long long tp2_1 = (long long)tpc_2_1;
        int tp_1_2 = t0 + 2;
        int tpc_2_2 = ((tp_1_2 < t1) ? tp_1_2 : t_last);
        long long tp2_2 = (long long)tpc_2_2;
        int tp_1_3 = t0 + 3;
        int tpc_2_3 = ((tp_1_3 < t1) ? tp_1_3 : t_last);
        long long tp2_3 = (long long)tpc_2_3;
        int tp_1_4 = t0 + 4;
        int tpc_2_4 = ((tp_1_4 < t1) ? tp_1_4 : t_last);
        long long tp2_4 = (long long)tpc_2_4;
        int tp_1_5 = t0 + 5;
        int tpc_2_5 = ((tp_1_5 < t1) ? tp_1_5 : t_last);
        long long tp2_5 = (long long)tpc_2_5;
        int tp_1_6 = t0 + 6;
        int tpc_2_6 = ((tp_1_6 < t1) ? tp_1_6 : t_last);
        long long tp2_6 = (long long)tpc_2_6;
        int tp_1_7 = t0 + 7;
        int tpc_2_7 = ((tp_1_7 < t1) ? tp_1_7 : t_last);
        long long tp2_7 = (long long)tpc_2_7;
        int tp_1_8 = t0 + 8;
        int tpc_2_8 = ((tp_1_8 < t1) ? tp_1_8 : t_last);
        long long tp2_8 = (long long)tpc_2_8;
        int tp_1_9 = t0 + 9;
        int tpc_2_9 = ((tp_1_9 < t1) ? tp_1_9 : t_last);
        long long tp2_9 = (long long)tpc_2_9;
        int pro_base = warp_0 * 4 + lane_hi;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_0) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0), "l"(dU + (tp2_0 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 1280), "l"(X + (tp2_0 * 4 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 2560), "l"(X + ((tp2_0 * 4 + 1) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 3840), "l"(X + ((tp2_0 * 4 + 2) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 5120), "l"(X + ((tp2_0 * 4 + 3) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
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
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_1) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1), "l"(dU + (td64 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 1280), "l"(X + (td64 * 4 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 2560), "l"(X + ((td64 * 4 + 1) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 3840), "l"(X + ((td64 * 4 + 2) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 5120), "l"(X + ((td64 * 4 + 3) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
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
        unsigned int ring_pk[10];
        unsigned int lane_row = sa__1_0 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(0) + 1]))
            : "r"(lane_row));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(2) + 1]))
            : "r"(lane_row + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(4) + 1]))
            : "r"(lane_row + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(6) + 1]))
            : "r"(lane_row + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk[(8) + 1]))
            : "r"(lane_row + 32));
        unsigned int ring_pk_0[10];
        unsigned int lane_row_0 = sa__1_0 + 1280 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[(0) + 1]))
            : "r"(lane_row_0));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[(2) + 1]))
            : "r"(lane_row_0 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[(4) + 1]))
            : "r"(lane_row_0 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[(6) + 1]))
            : "r"(lane_row_0 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_0[(8) + 1]))
            : "r"(lane_row_0 + 32));
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
        float _dot_111;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_111) : "r"(ring_pk_0[8]), "r"(ring_pk[8]), "f"(dots_p[0]));
        dots_p[0] = _dot_111;
        float _dot_112;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_112) : "r"(ring_pk_0[9]), "r"(ring_pk[9]), "f"(dots_p[0]));
        dots_p[0] = _dot_112;
        unsigned int ring_pk_1[10];
        unsigned int lane_row_1 = sa__1_0 + 2560 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(0) + 1]))
            : "r"(lane_row_1));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(2) + 1]))
            : "r"(lane_row_1 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(4) + 1]))
            : "r"(lane_row_1 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(6) + 1]))
            : "r"(lane_row_1 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_1[(8) + 1]))
            : "r"(lane_row_1 + 32));
        float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, c_22, 4);
        float s_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_23) : "f"(c_22), "f"(_shfl_xor_sync_113));
        float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, s_23, 24);
        float _add_115;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(s_23), "f"(_shfl_xor_sync_116));
        float _add_114;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(_add_115), "f"(hc_eps));
        float c_23 = __fdiv_rn(c_22, _add_114);
        hist[25] = c_23;
        float _dot_117;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_117) : "r"(ring_pk_1[0]), "r"(ring_pk[0]), "f"(-0.0f));
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
            : "=f"(_dot_118) : "r"(ring_pk_1[1]), "r"(ring_pk[1]), "f"(dots_p[1]));
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
            : "=f"(_dot_119) : "r"(ring_pk_1[2]), "r"(ring_pk[2]), "f"(dots_p[1]));
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
            : "=f"(_dot_120) : "r"(ring_pk_1[3]), "r"(ring_pk[3]), "f"(dots_p[1]));
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
            : "=f"(_dot_121) : "r"(ring_pk_1[4]), "r"(ring_pk[4]), "f"(dots_p[1]));
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
            : "=f"(_dot_122) : "r"(ring_pk_1[5]), "r"(ring_pk[5]), "f"(dots_p[1]));
        dots_p[1] = _dot_122;
        float _dot_123;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_123) : "r"(ring_pk_1[6]), "r"(ring_pk[6]), "f"(dots_p[1]));
        dots_p[1] = _dot_123;
        float _dot_124;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_124) : "r"(ring_pk_1[7]), "r"(ring_pk[7]), "f"(dots_p[1]));
        dots_p[1] = _dot_124;
        float _dot_125;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_125) : "r"(ring_pk_1[8]), "r"(ring_pk[8]), "f"(dots_p[1]));
        dots_p[1] = _dot_125;
        float _dot_126;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_126) : "r"(ring_pk_1[9]), "r"(ring_pk[9]), "f"(dots_p[1]));
        dots_p[1] = _dot_126;
        unsigned int ring_pk_2[10];
        unsigned int lane_row_2 = sa__1_0 + 3840 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(0) + 1]))
            : "r"(lane_row_2));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(2) + 1]))
            : "r"(lane_row_2 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(4) + 1]))
            : "r"(lane_row_2 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(6) + 1]))
            : "r"(lane_row_2 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_2[(8) + 1]))
            : "r"(lane_row_2 + 32));
        float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, c_23, 1);
        float s_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_24) : "f"(c_23), "f"(_shfl_xor_sync_127));
        float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, s_24, 2);
        float _add_129;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(s_24), "f"(_shfl_xor_sync_130));
        float _add_128;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(_add_129), "f"(hc_eps));
        float c_24 = __fdiv_rn(c_23, _add_128);
        hist[26] = c_24;
        float _dot_131;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_131) : "r"(ring_pk_2[0]), "r"(ring_pk[0]), "f"(-0.0f));
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
            : "=f"(_dot_132) : "r"(ring_pk_2[1]), "r"(ring_pk[1]), "f"(dots_p[2]));
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
            : "=f"(_dot_133) : "r"(ring_pk_2[2]), "r"(ring_pk[2]), "f"(dots_p[2]));
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
            : "=f"(_dot_134) : "r"(ring_pk_2[3]), "r"(ring_pk[3]), "f"(dots_p[2]));
        dots_p[2] = _dot_134;
        float _dot_135;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_135) : "r"(ring_pk_2[4]), "r"(ring_pk[4]), "f"(dots_p[2]));
        dots_p[2] = _dot_135;
        float _dot_136;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_136) : "r"(ring_pk_2[5]), "r"(ring_pk[5]), "f"(dots_p[2]));
        dots_p[2] = _dot_136;
        float _dot_137;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_137) : "r"(ring_pk_2[6]), "r"(ring_pk[6]), "f"(dots_p[2]));
        dots_p[2] = _dot_137;
        float _dot_138;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_138) : "r"(ring_pk_2[7]), "r"(ring_pk[7]), "f"(dots_p[2]));
        dots_p[2] = _dot_138;
        float _dot_139;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_139) : "r"(ring_pk_2[8]), "r"(ring_pk[8]), "f"(dots_p[2]));
        dots_p[2] = _dot_139;
        float _dot_140;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_140) : "r"(ring_pk_2[9]), "r"(ring_pk[9]), "f"(dots_p[2]));
        dots_p[2] = _dot_140;
        unsigned int ring_pk_3[10];
        unsigned int lane_row_3 = sa__1_0 + 5120 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(0) + 1]))
            : "r"(lane_row_3));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(2) + 1]))
            : "r"(lane_row_3 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(4) + 1]))
            : "r"(lane_row_3 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(6) + 1]))
            : "r"(lane_row_3 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_3[(8) + 1]))
            : "r"(lane_row_3 + 32));
        float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, c_24, 4);
        float s_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_25) : "f"(c_24), "f"(_shfl_xor_sync_141));
        float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, s_25, 24);
        float _add_143;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(s_25), "f"(_shfl_xor_sync_144));
        float _add_142;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(_add_143), "f"(hc_eps));
        float c_25 = __fdiv_rn(c_24, _add_142);
        hist[27] = c_25;
        float _dot_145;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_145) : "r"(ring_pk_3[0]), "r"(ring_pk[0]), "f"(-0.0f));
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
            : "=f"(_dot_146) : "r"(ring_pk_3[1]), "r"(ring_pk[1]), "f"(dots_p[3]));
        dots_p[3] = _dot_146;
        float _dot_147;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_147) : "r"(ring_pk_3[2]), "r"(ring_pk[2]), "f"(dots_p[3]));
        dots_p[3] = _dot_147;
        float _dot_148;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_148) : "r"(ring_pk_3[3]), "r"(ring_pk[3]), "f"(dots_p[3]));
        dots_p[3] = _dot_148;
        float _dot_149;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_149) : "r"(ring_pk_3[4]), "r"(ring_pk[4]), "f"(dots_p[3]));
        dots_p[3] = _dot_149;
        float _dot_150;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_150) : "r"(ring_pk_3[5]), "r"(ring_pk[5]), "f"(dots_p[3]));
        dots_p[3] = _dot_150;
        float _dot_151;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_151) : "r"(ring_pk_3[6]), "r"(ring_pk[6]), "f"(dots_p[3]));
        dots_p[3] = _dot_151;
        float _dot_152;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_152) : "r"(ring_pk_3[7]), "r"(ring_pk[7]), "f"(dots_p[3]));
        dots_p[3] = _dot_152;
        float _dot_153;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_153) : "r"(ring_pk_3[8]), "r"(ring_pk[8]), "f"(dots_p[3]));
        dots_p[3] = _dot_153;
        float _dot_154;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_154) : "r"(ring_pk_3[9]), "r"(ring_pk[9]), "f"(dots_p[3]));
        dots_p[3] = _dot_154;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_0) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0), "l"(dU + (th2_2 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 1280), "l"(X + (th2_2 * 4 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 2560), "l"(X + ((th2_2 * 4 + 1) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 3840), "l"(X + ((th2_2 * 4 + 2) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 5120), "l"(X + ((th2_2 * 4 + 3) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, c_25, 1);
        float s_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_26) : "f"(c_25), "f"(_shfl_xor_sync_155));
        float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, s_26, 2);
        float _add_157;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(s_26), "f"(_shfl_xor_sync_158));
        float _add_156;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(_add_157), "f"(hc_eps));
        float c_26 = __fdiv_rn(c_25, _add_156);
        hist[28] = c_26;
        int upper = lane_1 & 16;
        float send = ((upper != 0) ? dots_p[0] : dots_p[2]);
        float keep = ((upper != 0) ? dots_p[2] : dots_p[0]);
        float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
        float add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_0) : "f"(keep), "f"(_shfl_xor_sync_159));
        float send_1 = ((upper != 0) ? dots_p[1] : dots_p[3]);
        float keep_1 = ((upper != 0) ? dots_p[3] : dots_p[1]);
        float _shfl_xor_sync_160 = __shfl_xor_sync(0xFFFFFFFF, send_1, 16);
        float add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(keep_1), "f"(_shfl_xor_sync_160));
        int upper_1 = lane_1 & 8;
        float send_2 = ((upper_1 != 0) ? add_0 : add_1);
        float keep_2 = ((upper_1 != 0) ? add_1 : add_0);
        float _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, send_2, 8);
        float add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(keep_2), "f"(_shfl_xor_sync_161));
        float _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, c_26, 4);
        float s_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_27) : "f"(c_26), "f"(_shfl_xor_sync_162));
        float _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, s_27, 24);
        float _add_164;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_164) : "f"(s_27), "f"(_shfl_xor_sync_165));
        float _add_163;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(_add_164), "f"(hc_eps));
        float c_27 = __fdiv_rn(c_26, _add_163);
        hist[29] = c_27;
        float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, add_2, 4);
        float add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(add_2), "f"(_shfl_xor_sync_166));
        float _shfl_xor_sync_167 = __shfl_xor_sync(0xFFFFFFFF, add_5, 2);
        float add_1_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_0) : "f"(add_5), "f"(_shfl_xor_sync_167));
        float _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, add_1_0, 1);
        float _add_168;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(add_1_0), "f"(_shfl_xor_sync_169));
        dpre_part[own_base] = _add_168;
        float _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, c_27, 1);
        float s_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_28) : "f"(c_27), "f"(_shfl_xor_sync_170));
        float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, s_28, 2);
        float _add_172;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(s_28), "f"(_shfl_xor_sync_173));
        float _add_171;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171) : "f"(_add_172), "f"(hc_eps));
        float c_28 = __fdiv_rn(c_27, _add_171);
        hist[30] = c_28;
        mbarrier_wait(ring_bar_addr + (st__1_1) * 8, 0);
        unsigned int ring_pk_4[10];
        unsigned int lane_row_4 = sa__1_1 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[(0) + 1]))
            : "r"(lane_row_4));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[(2) + 1]))
            : "r"(lane_row_4 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[(4) + 1]))
            : "r"(lane_row_4 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[(6) + 1]))
            : "r"(lane_row_4 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_4[(8) + 1]))
            : "r"(lane_row_4 + 32));
        unsigned int ring_pk_5[10];
        unsigned int lane_row_5 = sa__1_1 + 1280 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[(0) + 1]))
            : "r"(lane_row_5));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[(2) + 1]))
            : "r"(lane_row_5 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[(4) + 1]))
            : "r"(lane_row_5 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[(6) + 1]))
            : "r"(lane_row_5 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_5[(8) + 1]))
            : "r"(lane_row_5 + 32));
        float dots_p_0[4];
        float _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, c_28, 4);
        float s_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_29) : "f"(c_28), "f"(_shfl_xor_sync_174));
        float _shfl_xor_sync_177 = __shfl_xor_sync(0xFFFFFFFF, s_29, 24);
        float _add_176;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_176) : "f"(s_29), "f"(_shfl_xor_sync_177));
        float _add_175;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_175) : "f"(_add_176), "f"(hc_eps));
        float c_29 = __fdiv_rn(c_28, _add_175);
        hist[31] = c_29;
        float _dot_178;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_178) : "r"(ring_pk_5[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[0] = _dot_178;
        float _dot_179;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_179) : "r"(ring_pk_5[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_179;
        float _dot_180;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_180) : "r"(ring_pk_5[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_180;
        float _dot_181;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_181) : "r"(ring_pk_5[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_181;
        float _dot_182;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_182) : "r"(ring_pk_5[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_182;
        float _dot_183;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_183) : "r"(ring_pk_5[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_183;
        float _dot_184;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_184) : "r"(ring_pk_5[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_184;
        float _dot_185;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_185) : "r"(ring_pk_5[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_185;
        float _dot_186;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_186) : "r"(ring_pk_5[8]), "r"(ring_pk_4[8]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_186;
        float _dot_187;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_187) : "r"(ring_pk_5[9]), "r"(ring_pk_4[9]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_187;
        unsigned int ring_pk_6[10];
        unsigned int lane_row_6 = sa__1_1 + 2560 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(0) + 1]))
            : "r"(lane_row_6));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(2) + 1]))
            : "r"(lane_row_6 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(4) + 1]))
            : "r"(lane_row_6 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(6) + 1]))
            : "r"(lane_row_6 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_6[(8) + 1]))
            : "r"(lane_row_6 + 32));
        float _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, c_29, 1);
        float s_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_30) : "f"(c_29), "f"(_shfl_xor_sync_188));
        float _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, s_30, 2);
        float _add_190;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_190) : "f"(s_30), "f"(_shfl_xor_sync_191));
        float _add_189;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_189) : "f"(_add_190), "f"(hc_eps));
        float c_30 = __fdiv_rn(c_29, _add_189);
        hist[32] = c_30;
        float _dot_192;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_192) : "r"(ring_pk_6[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[1] = _dot_192;
        float _dot_193;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_193) : "r"(ring_pk_6[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_193;
        float _dot_194;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_194) : "r"(ring_pk_6[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_194;
        float _dot_195;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_195) : "r"(ring_pk_6[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_195;
        float _dot_196;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_196) : "r"(ring_pk_6[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_196;
        float _dot_197;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_197) : "r"(ring_pk_6[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_197;
        float _dot_198;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_198) : "r"(ring_pk_6[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_198;
        float _dot_199;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_199) : "r"(ring_pk_6[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_199;
        float _dot_200;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_200) : "r"(ring_pk_6[8]), "r"(ring_pk_4[8]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_200;
        float _dot_201;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_201) : "r"(ring_pk_6[9]), "r"(ring_pk_4[9]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_201;
        unsigned int ring_pk_7[10];
        unsigned int lane_row_7 = sa__1_1 + 3840 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(0) + 1]))
            : "r"(lane_row_7));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(2) + 1]))
            : "r"(lane_row_7 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(4) + 1]))
            : "r"(lane_row_7 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(6) + 1]))
            : "r"(lane_row_7 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_7[(8) + 1]))
            : "r"(lane_row_7 + 32));
        float _shfl_xor_sync_202 = __shfl_xor_sync(0xFFFFFFFF, c_30, 4);
        float s_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_31) : "f"(c_30), "f"(_shfl_xor_sync_202));
        float _shfl_xor_sync_205 = __shfl_xor_sync(0xFFFFFFFF, s_31, 24);
        float _add_204;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_204) : "f"(s_31), "f"(_shfl_xor_sync_205));
        float _add_203;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_203) : "f"(_add_204), "f"(hc_eps));
        float c_31 = __fdiv_rn(c_30, _add_203);
        hist[33] = c_31;
        float _dot_206;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_206) : "r"(ring_pk_7[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[2] = _dot_206;
        float _dot_207;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_207) : "r"(ring_pk_7[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_207;
        float _dot_208;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_208) : "r"(ring_pk_7[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_208;
        float _dot_209;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_209) : "r"(ring_pk_7[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_209;
        float _dot_210;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_210) : "r"(ring_pk_7[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_210;
        float _dot_211;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_211) : "r"(ring_pk_7[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_211;
        float _dot_212;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_212) : "r"(ring_pk_7[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_212;
        float _dot_213;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_213) : "r"(ring_pk_7[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_213;
        float _dot_214;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_214) : "r"(ring_pk_7[8]), "r"(ring_pk_4[8]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_214;
        float _dot_215;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_215) : "r"(ring_pk_7[9]), "r"(ring_pk_4[9]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_215;
        unsigned int ring_pk_8[10];
        unsigned int lane_row_8 = sa__1_1 + 5120 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(0) + 1]))
            : "r"(lane_row_8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(2) + 1]))
            : "r"(lane_row_8 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(4) + 1]))
            : "r"(lane_row_8 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(6) + 1]))
            : "r"(lane_row_8 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_8[(8) + 1]))
            : "r"(lane_row_8 + 32));
        float _shfl_xor_sync_216 = __shfl_xor_sync(0xFFFFFFFF, c_31, 1);
        float s_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_32) : "f"(c_31), "f"(_shfl_xor_sync_216));
        float _shfl_xor_sync_219 = __shfl_xor_sync(0xFFFFFFFF, s_32, 2);
        float _add_218;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_218) : "f"(s_32), "f"(_shfl_xor_sync_219));
        float _add_217;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_217) : "f"(_add_218), "f"(hc_eps));
        float c_32 = __fdiv_rn(c_31, _add_217);
        hist[34] = c_32;
        float _dot_220;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_220) : "r"(ring_pk_8[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[3] = _dot_220;
        float _dot_221;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_221) : "r"(ring_pk_8[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_221;
        float _dot_222;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_222) : "r"(ring_pk_8[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_222;
        float _dot_223;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_223) : "r"(ring_pk_8[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_223;
        float _dot_224;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_224) : "r"(ring_pk_8[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_224;
        float _dot_225;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_225) : "r"(ring_pk_8[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_225;
        float _dot_226;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_226) : "r"(ring_pk_8[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_226;
        float _dot_227;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_227) : "r"(ring_pk_8[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_227;
        float _dot_228;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_228) : "r"(ring_pk_8[8]), "r"(ring_pk_4[8]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_228;
        float _dot_229;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_229) : "r"(ring_pk_8[9]), "r"(ring_pk_4[9]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_229;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_1) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1), "l"(dU + (th2_3 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 1280), "l"(X + (th2_3 * 4 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 2560), "l"(X + ((th2_3 * 4 + 1) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 3840), "l"(X + ((th2_3 * 4 + 2) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 5120), "l"(X + ((th2_3 * 4 + 3) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_230 = __shfl_xor_sync(0xFFFFFFFF, c_32, 4);
        float s_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_33) : "f"(c_32), "f"(_shfl_xor_sync_230));
        float _shfl_xor_sync_233 = __shfl_xor_sync(0xFFFFFFFF, s_33, 24);
        float _add_232;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_232) : "f"(s_33), "f"(_shfl_xor_sync_233));
        float _add_231;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_231) : "f"(_add_232), "f"(hc_eps));
        float c_33 = __fdiv_rn(c_32, _add_231);
        hist[35] = c_33;
        int upper_0 = lane_1 & 16;
        float send_0 = ((upper_0 != 0) ? dots_p_0[0] : dots_p_0[2]);
        float keep_0 = ((upper_0 != 0) ? dots_p_0[2] : dots_p_0[0]);
        float _shfl_xor_sync_234 = __shfl_xor_sync(0xFFFFFFFF, send_0, 16);
        float add_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_6) : "f"(keep_0), "f"(_shfl_xor_sync_234));
        float send_1_0 = ((upper_0 != 0) ? dots_p_0[1] : dots_p_0[3]);
        float keep_1_0 = ((upper_0 != 0) ? dots_p_0[3] : dots_p_0[1]);
        float _shfl_xor_sync_235 = __shfl_xor_sync(0xFFFFFFFF, send_1_0, 16);
        float add_1_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_1) : "f"(keep_1_0), "f"(_shfl_xor_sync_235));
        int upper_1_0 = lane_1 & 8;
        float send_2_0 = ((upper_1_0 != 0) ? add_6 : add_1_1);
        float keep_2_0 = ((upper_1_0 != 0) ? add_1_1 : add_6);
        float _shfl_xor_sync_236 = __shfl_xor_sync(0xFFFFFFFF, send_2_0, 8);
        float add_2_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_0) : "f"(keep_2_0), "f"(_shfl_xor_sync_236));
        float _shfl_xor_sync_237 = __shfl_xor_sync(0xFFFFFFFF, c_33, 1);
        float s_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_34) : "f"(c_33), "f"(_shfl_xor_sync_237));
        float _shfl_xor_sync_240 = __shfl_xor_sync(0xFFFFFFFF, s_34, 2);
        float _add_239;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_239) : "f"(s_34), "f"(_shfl_xor_sync_240));
        float _add_238;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_238) : "f"(_add_239), "f"(hc_eps));
        float c_34 = __fdiv_rn(c_33, _add_238);
        hist[36] = c_34;
        float _shfl_xor_sync_241 = __shfl_xor_sync(0xFFFFFFFF, add_2_0, 4);
        float add_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7) : "f"(add_2_0), "f"(_shfl_xor_sync_241));
        float _shfl_xor_sync_242 = __shfl_xor_sync(0xFFFFFFFF, add_7, 2);
        float add_1_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_2) : "f"(add_7), "f"(_shfl_xor_sync_242));
        float _shfl_xor_sync_244 = __shfl_xor_sync(0xFFFFFFFF, add_1_2, 1);
        float _add_243;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_243) : "f"(add_1_2), "f"(_shfl_xor_sync_244));
        dpre_part[own_base + 32] = _add_243;
        float _shfl_xor_sync_245 = __shfl_xor_sync(0xFFFFFFFF, c_34, 4);
        float s_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_35) : "f"(c_34), "f"(_shfl_xor_sync_245));
        float _shfl_xor_sync_248 = __shfl_xor_sync(0xFFFFFFFF, s_35, 24);
        float _add_247;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_247) : "f"(s_35), "f"(_shfl_xor_sync_248));
        float _add_246;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(_add_247), "f"(hc_eps));
        float c_35 = __fdiv_rn(c_34, _add_246);
        hist[37] = c_35;
        mbarrier_wait(ring_bar_addr + (st__1_0) * 8, 1);
        unsigned int ring_pk_9[10];
        unsigned int lane_row_9 = sa__1_0 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[(0) + 1]))
            : "r"(lane_row_9));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[(2) + 1]))
            : "r"(lane_row_9 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[(4) + 1]))
            : "r"(lane_row_9 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[(6) + 1]))
            : "r"(lane_row_9 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_9[(8) + 1]))
            : "r"(lane_row_9 + 32));
        unsigned int ring_pk_10[10];
        unsigned int lane_row_10 = sa__1_0 + 1280 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[(0) + 1]))
            : "r"(lane_row_10));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[(2) + 1]))
            : "r"(lane_row_10 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[(4) + 1]))
            : "r"(lane_row_10 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[(6) + 1]))
            : "r"(lane_row_10 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_10[(8) + 1]))
            : "r"(lane_row_10 + 32));
        float dots_p_1[4];
        float _shfl_xor_sync_249 = __shfl_xor_sync(0xFFFFFFFF, c_35, 1);
        float s_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_36) : "f"(c_35), "f"(_shfl_xor_sync_249));
        float _shfl_xor_sync_252 = __shfl_xor_sync(0xFFFFFFFF, s_36, 2);
        float _add_251;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_251) : "f"(s_36), "f"(_shfl_xor_sync_252));
        float _add_250;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_250) : "f"(_add_251), "f"(hc_eps));
        float c_36 = __fdiv_rn(c_35, _add_250);
        hist[38] = c_36;
        float _dot_253;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_253) : "r"(ring_pk_10[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[0] = _dot_253;
        float _dot_254;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_254) : "r"(ring_pk_10[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_254;
        float _dot_255;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_255) : "r"(ring_pk_10[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_255;
        float _dot_256;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_256) : "r"(ring_pk_10[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_256;
        float _dot_257;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_257) : "r"(ring_pk_10[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_257;
        float _dot_258;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_258) : "r"(ring_pk_10[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_258;
        float _dot_259;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_259) : "r"(ring_pk_10[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_259;
        float _dot_260;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_260) : "r"(ring_pk_10[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_260;
        float _dot_261;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_261) : "r"(ring_pk_10[8]), "r"(ring_pk_9[8]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_261;
        float _dot_262;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_262) : "r"(ring_pk_10[9]), "r"(ring_pk_9[9]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_262;
        unsigned int ring_pk_11[10];
        unsigned int lane_row_11 = sa__1_0 + 2560 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(0) + 1]))
            : "r"(lane_row_11));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(2) + 1]))
            : "r"(lane_row_11 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(4) + 1]))
            : "r"(lane_row_11 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(6) + 1]))
            : "r"(lane_row_11 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_11[(8) + 1]))
            : "r"(lane_row_11 + 32));
        float _shfl_xor_sync_263 = __shfl_xor_sync(0xFFFFFFFF, c_36, 4);
        float s_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_37) : "f"(c_36), "f"(_shfl_xor_sync_263));
        float _shfl_xor_sync_266 = __shfl_xor_sync(0xFFFFFFFF, s_37, 24);
        float _add_265;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_265) : "f"(s_37), "f"(_shfl_xor_sync_266));
        float _add_264;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_264) : "f"(_add_265), "f"(hc_eps));
        float c_37 = __fdiv_rn(c_36, _add_264);
        hist[39] = c_37;
        float _dot_267;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_267) : "r"(ring_pk_11[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[1] = _dot_267;
        float _dot_268;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_268) : "r"(ring_pk_11[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_268;
        float _dot_269;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_269) : "r"(ring_pk_11[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_269;
        float _dot_270;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_270) : "r"(ring_pk_11[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_270;
        float _dot_271;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_271) : "r"(ring_pk_11[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_271;
        float _dot_272;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_272) : "r"(ring_pk_11[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_272;
        float _dot_273;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_273) : "r"(ring_pk_11[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_273;
        float _dot_274;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_274) : "r"(ring_pk_11[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_274;
        float _dot_275;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_275) : "r"(ring_pk_11[8]), "r"(ring_pk_9[8]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_275;
        float _dot_276;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_276) : "r"(ring_pk_11[9]), "r"(ring_pk_9[9]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_276;
        unsigned int ring_pk_12[10];
        unsigned int lane_row_12 = sa__1_0 + 3840 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(0) + 1]))
            : "r"(lane_row_12));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(2) + 1]))
            : "r"(lane_row_12 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(4) + 1]))
            : "r"(lane_row_12 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(6) + 1]))
            : "r"(lane_row_12 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_12[(8) + 1]))
            : "r"(lane_row_12 + 32));
        float dsig;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dsig) : "f"(sig), "f"(sneg));
        float G = ((lane_1 >= 8 && lane_1 < 24) ? gval : 0.0f);
        float _shfl_xor_sync_277 = __shfl_xor_sync(0xFFFFFFFF, hist[38], 4);
        float s_38;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_38) : "f"(hist[38]), "f"(_shfl_xor_sync_277));
        float _shfl_xor_sync_279 = __shfl_xor_sync(0xFFFFFFFF, s_38, 24);
        float _add_278;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_278) : "f"(s_38), "f"(_shfl_xor_sync_279));
        float v;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(_add_278), "f"(hc_eps));
        float m;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m) : "f"(G), "f"(hist[39]));
        float _shfl_xor_sync_280 = __shfl_xor_sync(0xFFFFFFFF, m, 4);
        float ip;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip) : "f"(m), "f"(_shfl_xor_sync_280));
        float _shfl_xor_sync_283 = __shfl_xor_sync(0xFFFFFFFF, ip, 24);
        float _add_282;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_282) : "f"(ip), "f"(_shfl_xor_sync_283));
        float _sub_281;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_281) : "f"(G), "f"(_add_282));
        float g = __fdiv_rn(_sub_281, v);
        float _dot_284;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_284) : "r"(ring_pk_12[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[2] = _dot_284;
        float _dot_285;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_285) : "r"(ring_pk_12[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_285;
        float _dot_286;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_286) : "r"(ring_pk_12[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_286;
        float _dot_287;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_287) : "r"(ring_pk_12[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_287;
        float _dot_288;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_288) : "r"(ring_pk_12[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_288;
        float _dot_289;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_289) : "r"(ring_pk_12[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_289;
        float _dot_290;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_290) : "r"(ring_pk_12[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_290;
        float _dot_291;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_291) : "r"(ring_pk_12[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_291;
        float _dot_292;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_292) : "r"(ring_pk_12[8]), "r"(ring_pk_9[8]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_292;
        float _dot_293;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_293) : "r"(ring_pk_12[9]), "r"(ring_pk_9[9]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_293;
        unsigned int ring_pk_13[10];
        unsigned int lane_row_13 = sa__1_0 + 5120 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(0) + 1]))
            : "r"(lane_row_13));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(2) + 1]))
            : "r"(lane_row_13 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(4) + 1]))
            : "r"(lane_row_13 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(6) + 1]))
            : "r"(lane_row_13 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_13[(8) + 1]))
            : "r"(lane_row_13 + 32));
        float _shfl_xor_sync_294 = __shfl_xor_sync(0xFFFFFFFF, hist[37], 1);
        float s_39;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_39) : "f"(hist[37]), "f"(_shfl_xor_sync_294));
        float _shfl_xor_sync_296 = __shfl_xor_sync(0xFFFFFFFF, s_39, 2);
        float _add_295;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_295) : "f"(s_39), "f"(_shfl_xor_sync_296));
        float v_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_0) : "f"(_add_295), "f"(hc_eps));
        float m_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_0) : "f"(g), "f"(hist[38]));
        float _shfl_xor_sync_297 = __shfl_xor_sync(0xFFFFFFFF, m_0, 1);
        float ip_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_0) : "f"(m_0), "f"(_shfl_xor_sync_297));
        float _shfl_xor_sync_300 = __shfl_xor_sync(0xFFFFFFFF, ip_0, 2);
        float _add_299;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_299) : "f"(ip_0), "f"(_shfl_xor_sync_300));
        float _sub_298;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_298) : "f"(g), "f"(_add_299));
        float g_0 = __fdiv_rn(_sub_298, v_0);
        float _dot_301;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_301) : "r"(ring_pk_13[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[3] = _dot_301;
        float _dot_302;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_302) : "r"(ring_pk_13[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_302;
        float _dot_303;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_303) : "r"(ring_pk_13[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_303;
        float _dot_304;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_304) : "r"(ring_pk_13[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_304;
        float _dot_305;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_305) : "r"(ring_pk_13[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_305;
        float _dot_306;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_306) : "r"(ring_pk_13[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_306;
        float _dot_307;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_307) : "r"(ring_pk_13[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_307;
        float _dot_308;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_308) : "r"(ring_pk_13[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_308;
        float _dot_309;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_309) : "r"(ring_pk_13[8]), "r"(ring_pk_9[8]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_309;
        float _dot_310;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_310) : "r"(ring_pk_13[9]), "r"(ring_pk_9[9]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_310;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_0) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0), "l"(dU + (th2_4 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 1280), "l"(X + (th2_4 * 4 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 2560), "l"(X + ((th2_4 * 4 + 1) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 3840), "l"(X + ((th2_4 * 4 + 2) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 5120), "l"(X + ((th2_4 * 4 + 3) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_311 = __shfl_xor_sync(0xFFFFFFFF, hist[36], 4);
        float s_40;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_40) : "f"(hist[36]), "f"(_shfl_xor_sync_311));
        float _shfl_xor_sync_313 = __shfl_xor_sync(0xFFFFFFFF, s_40, 24);
        float _add_312;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_312) : "f"(s_40), "f"(_shfl_xor_sync_313));
        float v_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_1) : "f"(_add_312), "f"(hc_eps));
        float m_1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_1) : "f"(g_0), "f"(hist[37]));
        float _shfl_xor_sync_314 = __shfl_xor_sync(0xFFFFFFFF, m_1, 4);
        float ip_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_1) : "f"(m_1), "f"(_shfl_xor_sync_314));
        float _shfl_xor_sync_317 = __shfl_xor_sync(0xFFFFFFFF, ip_1, 24);
        float _add_316;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_316) : "f"(ip_1), "f"(_shfl_xor_sync_317));
        float _sub_315;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_315) : "f"(g_0), "f"(_add_316));
        float g_1 = __fdiv_rn(_sub_315, v_1);
        int upper_2 = lane_1 & 16;
        float send_3 = ((upper_2 != 0) ? dots_p_1[0] : dots_p_1[2]);
        float keep_3 = ((upper_2 != 0) ? dots_p_1[2] : dots_p_1[0]);
        float _shfl_xor_sync_318 = __shfl_xor_sync(0xFFFFFFFF, send_3, 16);
        float add_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8) : "f"(keep_3), "f"(_shfl_xor_sync_318));
        float send_1_1 = ((upper_2 != 0) ? dots_p_1[1] : dots_p_1[3]);
        float keep_1_1 = ((upper_2 != 0) ? dots_p_1[3] : dots_p_1[1]);
        float _shfl_xor_sync_319 = __shfl_xor_sync(0xFFFFFFFF, send_1_1, 16);
        float add_1_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_3) : "f"(keep_1_1), "f"(_shfl_xor_sync_319));
        int upper_1_1 = lane_1 & 8;
        float send_2_1 = ((upper_1_1 != 0) ? add_8 : add_1_3);
        float keep_2_1 = ((upper_1_1 != 0) ? add_1_3 : add_8);
        float _shfl_xor_sync_320 = __shfl_xor_sync(0xFFFFFFFF, send_2_1, 8);
        float add_2_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_1) : "f"(keep_2_1), "f"(_shfl_xor_sync_320));
        float _shfl_xor_sync_321 = __shfl_xor_sync(0xFFFFFFFF, hist[35], 1);
        float s_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_41) : "f"(hist[35]), "f"(_shfl_xor_sync_321));
        float _shfl_xor_sync_323 = __shfl_xor_sync(0xFFFFFFFF, s_41, 2);
        float _add_322;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_322) : "f"(s_41), "f"(_shfl_xor_sync_323));
        float v_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_2) : "f"(_add_322), "f"(hc_eps));
        float m_2;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_2) : "f"(g_1), "f"(hist[36]));
        float _shfl_xor_sync_324 = __shfl_xor_sync(0xFFFFFFFF, m_2, 1);
        float ip_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_2) : "f"(m_2), "f"(_shfl_xor_sync_324));
        float _shfl_xor_sync_327 = __shfl_xor_sync(0xFFFFFFFF, ip_2, 2);
        float _add_326;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_326) : "f"(ip_2), "f"(_shfl_xor_sync_327));
        float _sub_325;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_325) : "f"(g_1), "f"(_add_326));
        float g_2 = __fdiv_rn(_sub_325, v_2);
        float _shfl_xor_sync_328 = __shfl_xor_sync(0xFFFFFFFF, add_2_1, 4);
        float add_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_9) : "f"(add_2_1), "f"(_shfl_xor_sync_328));
        float _shfl_xor_sync_329 = __shfl_xor_sync(0xFFFFFFFF, add_9, 2);
        float add_1_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_4) : "f"(add_9), "f"(_shfl_xor_sync_329));
        float _shfl_xor_sync_331 = __shfl_xor_sync(0xFFFFFFFF, add_1_4, 1);
        float _add_330;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_330) : "f"(add_1_4), "f"(_shfl_xor_sync_331));
        dpre_part[own_base + 64] = _add_330;
        float _shfl_xor_sync_332 = __shfl_xor_sync(0xFFFFFFFF, hist[34], 4);
        float s_42;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_42) : "f"(hist[34]), "f"(_shfl_xor_sync_332));
        float _shfl_xor_sync_334 = __shfl_xor_sync(0xFFFFFFFF, s_42, 24);
        float _add_333;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_333) : "f"(s_42), "f"(_shfl_xor_sync_334));
        float v_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_3) : "f"(_add_333), "f"(hc_eps));
        float m_3;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_3) : "f"(g_2), "f"(hist[35]));
        float _shfl_xor_sync_335 = __shfl_xor_sync(0xFFFFFFFF, m_3, 4);
        float ip_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_3) : "f"(m_3), "f"(_shfl_xor_sync_335));
        float _shfl_xor_sync_338 = __shfl_xor_sync(0xFFFFFFFF, ip_3, 24);
        float _add_337;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_337) : "f"(ip_3), "f"(_shfl_xor_sync_338));
        float _sub_336;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_336) : "f"(g_2), "f"(_add_337));
        float g_3 = __fdiv_rn(_sub_336, v_3);
        mbarrier_wait(ring_bar_addr + (st__1_1) * 8, 1);
        unsigned int ring_pk_14[10];
        unsigned int lane_row_14 = sa__1_1 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[(0) + 1]))
            : "r"(lane_row_14));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[(2) + 1]))
            : "r"(lane_row_14 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[(4) + 1]))
            : "r"(lane_row_14 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[(6) + 1]))
            : "r"(lane_row_14 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_14[(8) + 1]))
            : "r"(lane_row_14 + 32));
        unsigned int ring_pk_15[10];
        unsigned int lane_row_15 = sa__1_1 + 1280 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[(0) + 1]))
            : "r"(lane_row_15));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[(2) + 1]))
            : "r"(lane_row_15 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[(4) + 1]))
            : "r"(lane_row_15 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[(6) + 1]))
            : "r"(lane_row_15 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_15[(8) + 1]))
            : "r"(lane_row_15 + 32));
        float dots_p_2[4];
        float _shfl_xor_sync_339 = __shfl_xor_sync(0xFFFFFFFF, hist[33], 1);
        float s_43;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_43) : "f"(hist[33]), "f"(_shfl_xor_sync_339));
        float _shfl_xor_sync_341 = __shfl_xor_sync(0xFFFFFFFF, s_43, 2);
        float _add_340;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_340) : "f"(s_43), "f"(_shfl_xor_sync_341));
        float v_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_4) : "f"(_add_340), "f"(hc_eps));
        float m_4;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_4) : "f"(g_3), "f"(hist[34]));
        float _shfl_xor_sync_342 = __shfl_xor_sync(0xFFFFFFFF, m_4, 1);
        float ip_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_4) : "f"(m_4), "f"(_shfl_xor_sync_342));
        float _shfl_xor_sync_345 = __shfl_xor_sync(0xFFFFFFFF, ip_4, 2);
        float _add_344;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_344) : "f"(ip_4), "f"(_shfl_xor_sync_345));
        float _sub_343;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_343) : "f"(g_3), "f"(_add_344));
        float g_4 = __fdiv_rn(_sub_343, v_4);
        float _dot_346;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_346) : "r"(ring_pk_15[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[0] = _dot_346;
        float _dot_347;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_347) : "r"(ring_pk_15[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_347;
        float _dot_348;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_348) : "r"(ring_pk_15[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_348;
        float _dot_349;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_349) : "r"(ring_pk_15[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_349;
        float _dot_350;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_350) : "r"(ring_pk_15[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_350;
        float _dot_351;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_351) : "r"(ring_pk_15[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_351;
        float _dot_352;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_352) : "r"(ring_pk_15[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_352;
        float _dot_353;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_353) : "r"(ring_pk_15[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_353;
        float _dot_354;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_354) : "r"(ring_pk_15[8]), "r"(ring_pk_14[8]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_354;
        float _dot_355;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_355) : "r"(ring_pk_15[9]), "r"(ring_pk_14[9]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_355;
        unsigned int ring_pk_16[10];
        unsigned int lane_row_16 = sa__1_1 + 2560 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(0) + 1]))
            : "r"(lane_row_16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(2) + 1]))
            : "r"(lane_row_16 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(4) + 1]))
            : "r"(lane_row_16 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(6) + 1]))
            : "r"(lane_row_16 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_16[(8) + 1]))
            : "r"(lane_row_16 + 32));
        float _shfl_xor_sync_356 = __shfl_xor_sync(0xFFFFFFFF, hist[32], 4);
        float s_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_44) : "f"(hist[32]), "f"(_shfl_xor_sync_356));
        float _shfl_xor_sync_358 = __shfl_xor_sync(0xFFFFFFFF, s_44, 24);
        float _add_357;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_357) : "f"(s_44), "f"(_shfl_xor_sync_358));
        float v_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_5) : "f"(_add_357), "f"(hc_eps));
        float m_5;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_5) : "f"(g_4), "f"(hist[33]));
        float _shfl_xor_sync_359 = __shfl_xor_sync(0xFFFFFFFF, m_5, 4);
        float ip_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_5) : "f"(m_5), "f"(_shfl_xor_sync_359));
        float _shfl_xor_sync_362 = __shfl_xor_sync(0xFFFFFFFF, ip_5, 24);
        float _add_361;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_361) : "f"(ip_5), "f"(_shfl_xor_sync_362));
        float _sub_360;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_360) : "f"(g_4), "f"(_add_361));
        float g_5 = __fdiv_rn(_sub_360, v_5);
        float _dot_363;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_363) : "r"(ring_pk_16[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[1] = _dot_363;
        float _dot_364;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_364) : "r"(ring_pk_16[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_364;
        float _dot_365;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_365) : "r"(ring_pk_16[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_365;
        float _dot_366;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_366) : "r"(ring_pk_16[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_366;
        float _dot_367;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_367) : "r"(ring_pk_16[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_367;
        float _dot_368;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_368) : "r"(ring_pk_16[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_368;
        float _dot_369;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_369) : "r"(ring_pk_16[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_369;
        float _dot_370;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_370) : "r"(ring_pk_16[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_370;
        float _dot_371;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_371) : "r"(ring_pk_16[8]), "r"(ring_pk_14[8]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_371;
        float _dot_372;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_372) : "r"(ring_pk_16[9]), "r"(ring_pk_14[9]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_372;
        unsigned int ring_pk_17[10];
        unsigned int lane_row_17 = sa__1_1 + 3840 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(0) + 1]))
            : "r"(lane_row_17));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(2) + 1]))
            : "r"(lane_row_17 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(4) + 1]))
            : "r"(lane_row_17 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(6) + 1]))
            : "r"(lane_row_17 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_17[(8) + 1]))
            : "r"(lane_row_17 + 32));
        float _shfl_xor_sync_373 = __shfl_xor_sync(0xFFFFFFFF, hist[31], 1);
        float s_45;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_45) : "f"(hist[31]), "f"(_shfl_xor_sync_373));
        float _shfl_xor_sync_375 = __shfl_xor_sync(0xFFFFFFFF, s_45, 2);
        float _add_374;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_374) : "f"(s_45), "f"(_shfl_xor_sync_375));
        float v_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_6) : "f"(_add_374), "f"(hc_eps));
        float m_6;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_6) : "f"(g_5), "f"(hist[32]));
        float _shfl_xor_sync_376 = __shfl_xor_sync(0xFFFFFFFF, m_6, 1);
        float ip_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_6) : "f"(m_6), "f"(_shfl_xor_sync_376));
        float _shfl_xor_sync_379 = __shfl_xor_sync(0xFFFFFFFF, ip_6, 2);
        float _add_378;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_378) : "f"(ip_6), "f"(_shfl_xor_sync_379));
        float _sub_377;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_377) : "f"(g_5), "f"(_add_378));
        float g_6 = __fdiv_rn(_sub_377, v_6);
        float _dot_380;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_380) : "r"(ring_pk_17[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[2] = _dot_380;
        float _dot_381;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_381) : "r"(ring_pk_17[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_381;
        float _dot_382;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_382) : "r"(ring_pk_17[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_382;
        float _dot_383;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_383) : "r"(ring_pk_17[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_383;
        float _dot_384;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_384) : "r"(ring_pk_17[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_384;
        float _dot_385;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_385) : "r"(ring_pk_17[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_385;
        float _dot_386;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_386) : "r"(ring_pk_17[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_386;
        float _dot_387;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_387) : "r"(ring_pk_17[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_387;
        float _dot_388;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_388) : "r"(ring_pk_17[8]), "r"(ring_pk_14[8]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_388;
        float _dot_389;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_389) : "r"(ring_pk_17[9]), "r"(ring_pk_14[9]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_389;
        unsigned int ring_pk_18[10];
        unsigned int lane_row_18 = sa__1_1 + 5120 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(0) + 1]))
            : "r"(lane_row_18));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(2) + 1]))
            : "r"(lane_row_18 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(4) + 1]))
            : "r"(lane_row_18 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(6) + 1]))
            : "r"(lane_row_18 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_18[(8) + 1]))
            : "r"(lane_row_18 + 32));
        float _shfl_xor_sync_390 = __shfl_xor_sync(0xFFFFFFFF, hist[30], 4);
        float s_46;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_46) : "f"(hist[30]), "f"(_shfl_xor_sync_390));
        float _shfl_xor_sync_392 = __shfl_xor_sync(0xFFFFFFFF, s_46, 24);
        float _add_391;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_391) : "f"(s_46), "f"(_shfl_xor_sync_392));
        float v_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_7) : "f"(_add_391), "f"(hc_eps));
        float m_7;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_7) : "f"(g_6), "f"(hist[31]));
        float _shfl_xor_sync_393 = __shfl_xor_sync(0xFFFFFFFF, m_7, 4);
        float ip_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_7) : "f"(m_7), "f"(_shfl_xor_sync_393));
        float _shfl_xor_sync_396 = __shfl_xor_sync(0xFFFFFFFF, ip_7, 24);
        float _add_395;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_395) : "f"(ip_7), "f"(_shfl_xor_sync_396));
        float _sub_394;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_394) : "f"(g_6), "f"(_add_395));
        float g_7 = __fdiv_rn(_sub_394, v_7);
        float _dot_397;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_397) : "r"(ring_pk_18[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[3] = _dot_397;
        float _dot_398;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_398) : "r"(ring_pk_18[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_398;
        float _dot_399;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_399) : "r"(ring_pk_18[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_399;
        float _dot_400;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_400) : "r"(ring_pk_18[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_400;
        float _dot_401;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_401) : "r"(ring_pk_18[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_401;
        float _dot_402;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_402) : "r"(ring_pk_18[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_402;
        float _dot_403;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_403) : "r"(ring_pk_18[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_403;
        float _dot_404;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_404) : "r"(ring_pk_18[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_404;
        float _dot_405;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_405) : "r"(ring_pk_18[8]), "r"(ring_pk_14[8]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_405;
        float _dot_406;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_406) : "r"(ring_pk_18[9]), "r"(ring_pk_14[9]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_406;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_1) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1), "l"(dU + (th2_5 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 1280), "l"(X + (th2_5 * 4 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 2560), "l"(X + ((th2_5 * 4 + 1) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 3840), "l"(X + ((th2_5 * 4 + 2) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 5120), "l"(X + ((th2_5 * 4 + 3) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_407 = __shfl_xor_sync(0xFFFFFFFF, hist[29], 1);
        float s_47;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_47) : "f"(hist[29]), "f"(_shfl_xor_sync_407));
        float _shfl_xor_sync_409 = __shfl_xor_sync(0xFFFFFFFF, s_47, 2);
        float _add_408;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_408) : "f"(s_47), "f"(_shfl_xor_sync_409));
        float v_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_8) : "f"(_add_408), "f"(hc_eps));
        float m_8;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_8) : "f"(g_7), "f"(hist[30]));
        float _shfl_xor_sync_410 = __shfl_xor_sync(0xFFFFFFFF, m_8, 1);
        float ip_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_8) : "f"(m_8), "f"(_shfl_xor_sync_410));
        float _shfl_xor_sync_413 = __shfl_xor_sync(0xFFFFFFFF, ip_8, 2);
        float _add_412;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_412) : "f"(ip_8), "f"(_shfl_xor_sync_413));
        float _sub_411;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_411) : "f"(g_7), "f"(_add_412));
        float g_8 = __fdiv_rn(_sub_411, v_8);
        int upper_3 = lane_1 & 16;
        float send_4 = ((upper_3 != 0) ? dots_p_2[0] : dots_p_2[2]);
        float keep_4 = ((upper_3 != 0) ? dots_p_2[2] : dots_p_2[0]);
        float _shfl_xor_sync_414 = __shfl_xor_sync(0xFFFFFFFF, send_4, 16);
        float add_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_10) : "f"(keep_4), "f"(_shfl_xor_sync_414));
        float send_1_2 = ((upper_3 != 0) ? dots_p_2[1] : dots_p_2[3]);
        float keep_1_2 = ((upper_3 != 0) ? dots_p_2[3] : dots_p_2[1]);
        float _shfl_xor_sync_415 = __shfl_xor_sync(0xFFFFFFFF, send_1_2, 16);
        float add_1_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_5) : "f"(keep_1_2), "f"(_shfl_xor_sync_415));
        int upper_1_2 = lane_1 & 8;
        float send_2_2 = ((upper_1_2 != 0) ? add_10 : add_1_5);
        float keep_2_2 = ((upper_1_2 != 0) ? add_1_5 : add_10);
        float _shfl_xor_sync_416 = __shfl_xor_sync(0xFFFFFFFF, send_2_2, 8);
        float add_2_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_2) : "f"(keep_2_2), "f"(_shfl_xor_sync_416));
        float _shfl_xor_sync_417 = __shfl_xor_sync(0xFFFFFFFF, hist[28], 4);
        float s_48;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_48) : "f"(hist[28]), "f"(_shfl_xor_sync_417));
        float _shfl_xor_sync_419 = __shfl_xor_sync(0xFFFFFFFF, s_48, 24);
        float _add_418;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_418) : "f"(s_48), "f"(_shfl_xor_sync_419));
        float v_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_9) : "f"(_add_418), "f"(hc_eps));
        float m_9;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_9) : "f"(g_8), "f"(hist[29]));
        float _shfl_xor_sync_420 = __shfl_xor_sync(0xFFFFFFFF, m_9, 4);
        float ip_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_9) : "f"(m_9), "f"(_shfl_xor_sync_420));
        float _shfl_xor_sync_423 = __shfl_xor_sync(0xFFFFFFFF, ip_9, 24);
        float _add_422;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_422) : "f"(ip_9), "f"(_shfl_xor_sync_423));
        float _sub_421;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_421) : "f"(g_8), "f"(_add_422));
        float g_9 = __fdiv_rn(_sub_421, v_9);
        float _shfl_xor_sync_424 = __shfl_xor_sync(0xFFFFFFFF, add_2_2, 4);
        float add_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_11) : "f"(add_2_2), "f"(_shfl_xor_sync_424));
        float _shfl_xor_sync_425 = __shfl_xor_sync(0xFFFFFFFF, add_11, 2);
        float add_1_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_6) : "f"(add_11), "f"(_shfl_xor_sync_425));
        float _shfl_xor_sync_427 = __shfl_xor_sync(0xFFFFFFFF, add_1_6, 1);
        float _add_426;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_426) : "f"(add_1_6), "f"(_shfl_xor_sync_427));
        dpre_part[own_base + 96] = _add_426;
        float _shfl_xor_sync_428 = __shfl_xor_sync(0xFFFFFFFF, hist[27], 1);
        float s_49;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_49) : "f"(hist[27]), "f"(_shfl_xor_sync_428));
        float _shfl_xor_sync_430 = __shfl_xor_sync(0xFFFFFFFF, s_49, 2);
        float _add_429;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_429) : "f"(s_49), "f"(_shfl_xor_sync_430));
        float v_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_10) : "f"(_add_429), "f"(hc_eps));
        float m_10;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_10) : "f"(g_9), "f"(hist[28]));
        float _shfl_xor_sync_431 = __shfl_xor_sync(0xFFFFFFFF, m_10, 1);
        float ip_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_10) : "f"(m_10), "f"(_shfl_xor_sync_431));
        float _shfl_xor_sync_434 = __shfl_xor_sync(0xFFFFFFFF, ip_10, 2);
        float _add_433;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_433) : "f"(ip_10), "f"(_shfl_xor_sync_434));
        float _sub_432;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_432) : "f"(g_9), "f"(_add_433));
        float g_10 = __fdiv_rn(_sub_432, v_10);
        mbarrier_wait(ring_bar_addr + (st__1_0) * 8, 0);
        unsigned int ring_pk_19[10];
        unsigned int lane_row_19 = sa__1_0 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[(0) + 1]))
            : "r"(lane_row_19));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[(2) + 1]))
            : "r"(lane_row_19 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[(4) + 1]))
            : "r"(lane_row_19 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[(6) + 1]))
            : "r"(lane_row_19 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_19[(8) + 1]))
            : "r"(lane_row_19 + 32));
        unsigned int ring_pk_20[10];
        unsigned int lane_row_20 = sa__1_0 + 1280 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[(0) + 1]))
            : "r"(lane_row_20));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[(2) + 1]))
            : "r"(lane_row_20 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[(4) + 1]))
            : "r"(lane_row_20 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[(6) + 1]))
            : "r"(lane_row_20 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_20[(8) + 1]))
            : "r"(lane_row_20 + 32));
        float dots_p_3[4];
        float _shfl_xor_sync_435 = __shfl_xor_sync(0xFFFFFFFF, hist[26], 4);
        float s_50;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_50) : "f"(hist[26]), "f"(_shfl_xor_sync_435));
        float _shfl_xor_sync_437 = __shfl_xor_sync(0xFFFFFFFF, s_50, 24);
        float _add_436;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_436) : "f"(s_50), "f"(_shfl_xor_sync_437));
        float v_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_11) : "f"(_add_436), "f"(hc_eps));
        float m_11;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_11) : "f"(g_10), "f"(hist[27]));
        float _shfl_xor_sync_438 = __shfl_xor_sync(0xFFFFFFFF, m_11, 4);
        float ip_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_11) : "f"(m_11), "f"(_shfl_xor_sync_438));
        float _shfl_xor_sync_441 = __shfl_xor_sync(0xFFFFFFFF, ip_11, 24);
        float _add_440;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_440) : "f"(ip_11), "f"(_shfl_xor_sync_441));
        float _sub_439;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_439) : "f"(g_10), "f"(_add_440));
        float g_11 = __fdiv_rn(_sub_439, v_11);
        float _dot_442;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_442) : "r"(ring_pk_20[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[0] = _dot_442;
        float _dot_443;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_443) : "r"(ring_pk_20[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_443;
        float _dot_444;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_444) : "r"(ring_pk_20[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_444;
        float _dot_445;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_445) : "r"(ring_pk_20[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_445;
        float _dot_446;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_446) : "r"(ring_pk_20[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_446;
        float _dot_447;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_447) : "r"(ring_pk_20[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_447;
        float _dot_448;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_448) : "r"(ring_pk_20[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_448;
        float _dot_449;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_449) : "r"(ring_pk_20[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_449;
        float _dot_450;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_450) : "r"(ring_pk_20[8]), "r"(ring_pk_19[8]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_450;
        float _dot_451;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_451) : "r"(ring_pk_20[9]), "r"(ring_pk_19[9]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_451;
        unsigned int ring_pk_21[10];
        unsigned int lane_row_21 = sa__1_0 + 2560 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(0) + 1]))
            : "r"(lane_row_21));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(2) + 1]))
            : "r"(lane_row_21 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(4) + 1]))
            : "r"(lane_row_21 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(6) + 1]))
            : "r"(lane_row_21 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_21[(8) + 1]))
            : "r"(lane_row_21 + 32));
        float _shfl_xor_sync_452 = __shfl_xor_sync(0xFFFFFFFF, hist[25], 1);
        float s_51;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_51) : "f"(hist[25]), "f"(_shfl_xor_sync_452));
        float _shfl_xor_sync_454 = __shfl_xor_sync(0xFFFFFFFF, s_51, 2);
        float _add_453;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_453) : "f"(s_51), "f"(_shfl_xor_sync_454));
        float v_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_12) : "f"(_add_453), "f"(hc_eps));
        float m_12;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_12) : "f"(g_11), "f"(hist[26]));
        float _shfl_xor_sync_455 = __shfl_xor_sync(0xFFFFFFFF, m_12, 1);
        float ip_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_12) : "f"(m_12), "f"(_shfl_xor_sync_455));
        float _shfl_xor_sync_458 = __shfl_xor_sync(0xFFFFFFFF, ip_12, 2);
        float _add_457;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_457) : "f"(ip_12), "f"(_shfl_xor_sync_458));
        float _sub_456;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_456) : "f"(g_11), "f"(_add_457));
        float g_12 = __fdiv_rn(_sub_456, v_12);
        float _dot_459;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_459) : "r"(ring_pk_21[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[1] = _dot_459;
        float _dot_460;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_460) : "r"(ring_pk_21[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_460;
        float _dot_461;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_461) : "r"(ring_pk_21[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_461;
        float _dot_462;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_462) : "r"(ring_pk_21[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_462;
        float _dot_463;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_463) : "r"(ring_pk_21[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_463;
        float _dot_464;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_464) : "r"(ring_pk_21[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_464;
        float _dot_465;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_465) : "r"(ring_pk_21[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_465;
        float _dot_466;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_466) : "r"(ring_pk_21[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_466;
        float _dot_467;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_467) : "r"(ring_pk_21[8]), "r"(ring_pk_19[8]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_467;
        float _dot_468;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_468) : "r"(ring_pk_21[9]), "r"(ring_pk_19[9]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_468;
        unsigned int ring_pk_22[10];
        unsigned int lane_row_22 = sa__1_0 + 3840 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(0) + 1]))
            : "r"(lane_row_22));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(2) + 1]))
            : "r"(lane_row_22 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(4) + 1]))
            : "r"(lane_row_22 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(6) + 1]))
            : "r"(lane_row_22 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_22[(8) + 1]))
            : "r"(lane_row_22 + 32));
        float _shfl_xor_sync_469 = __shfl_xor_sync(0xFFFFFFFF, hist[24], 4);
        float s_52;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_52) : "f"(hist[24]), "f"(_shfl_xor_sync_469));
        float _shfl_xor_sync_471 = __shfl_xor_sync(0xFFFFFFFF, s_52, 24);
        float _add_470;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_470) : "f"(s_52), "f"(_shfl_xor_sync_471));
        float v_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_13) : "f"(_add_470), "f"(hc_eps));
        float m_13;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_13) : "f"(g_12), "f"(hist[25]));
        float _shfl_xor_sync_472 = __shfl_xor_sync(0xFFFFFFFF, m_13, 4);
        float ip_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_13) : "f"(m_13), "f"(_shfl_xor_sync_472));
        float _shfl_xor_sync_475 = __shfl_xor_sync(0xFFFFFFFF, ip_13, 24);
        float _add_474;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_474) : "f"(ip_13), "f"(_shfl_xor_sync_475));
        float _sub_473;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_473) : "f"(g_12), "f"(_add_474));
        float g_13 = __fdiv_rn(_sub_473, v_13);
        float _dot_476;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_476) : "r"(ring_pk_22[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[2] = _dot_476;
        float _dot_477;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_477) : "r"(ring_pk_22[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_477;
        float _dot_478;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_478) : "r"(ring_pk_22[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_478;
        float _dot_479;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_479) : "r"(ring_pk_22[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_479;
        float _dot_480;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_480) : "r"(ring_pk_22[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_480;
        float _dot_481;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_481) : "r"(ring_pk_22[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_481;
        float _dot_482;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_482) : "r"(ring_pk_22[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_482;
        float _dot_483;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_483) : "r"(ring_pk_22[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_483;
        float _dot_484;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_484) : "r"(ring_pk_22[8]), "r"(ring_pk_19[8]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_484;
        float _dot_485;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_485) : "r"(ring_pk_22[9]), "r"(ring_pk_19[9]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_485;
        unsigned int ring_pk_23[10];
        unsigned int lane_row_23 = sa__1_0 + 5120 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(0) + 1]))
            : "r"(lane_row_23));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(2) + 1]))
            : "r"(lane_row_23 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(4) + 1]))
            : "r"(lane_row_23 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(6) + 1]))
            : "r"(lane_row_23 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_23[(8) + 1]))
            : "r"(lane_row_23 + 32));
        float _shfl_xor_sync_486 = __shfl_xor_sync(0xFFFFFFFF, hist[23], 1);
        float s_53;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_53) : "f"(hist[23]), "f"(_shfl_xor_sync_486));
        float _shfl_xor_sync_488 = __shfl_xor_sync(0xFFFFFFFF, s_53, 2);
        float _add_487;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_487) : "f"(s_53), "f"(_shfl_xor_sync_488));
        float v_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_14) : "f"(_add_487), "f"(hc_eps));
        float m_14;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_14) : "f"(g_13), "f"(hist[24]));
        float _shfl_xor_sync_489 = __shfl_xor_sync(0xFFFFFFFF, m_14, 1);
        float ip_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_14) : "f"(m_14), "f"(_shfl_xor_sync_489));
        float _shfl_xor_sync_492 = __shfl_xor_sync(0xFFFFFFFF, ip_14, 2);
        float _add_491;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_491) : "f"(ip_14), "f"(_shfl_xor_sync_492));
        float _sub_490;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_490) : "f"(g_13), "f"(_add_491));
        float g_14 = __fdiv_rn(_sub_490, v_14);
        float _dot_493;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_493) : "r"(ring_pk_23[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[3] = _dot_493;
        float _dot_494;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_494) : "r"(ring_pk_23[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_494;
        float _dot_495;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_495) : "r"(ring_pk_23[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_495;
        float _dot_496;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_496) : "r"(ring_pk_23[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_496;
        float _dot_497;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_497) : "r"(ring_pk_23[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_497;
        float _dot_498;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_498) : "r"(ring_pk_23[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_498;
        float _dot_499;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_499) : "r"(ring_pk_23[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_499;
        float _dot_500;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_500) : "r"(ring_pk_23[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_500;
        float _dot_501;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_501) : "r"(ring_pk_23[8]), "r"(ring_pk_19[8]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_501;
        float _dot_502;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_502) : "r"(ring_pk_23[9]), "r"(ring_pk_19[9]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_502;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_0) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0), "l"(dU + (th2_6 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 1280), "l"(X + (th2_6 * 4 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 2560), "l"(X + ((th2_6 * 4 + 1) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 3840), "l"(X + ((th2_6 * 4 + 2) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 5120), "l"(X + ((th2_6 * 4 + 3) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_503 = __shfl_xor_sync(0xFFFFFFFF, hist[22], 4);
        float s_54;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_54) : "f"(hist[22]), "f"(_shfl_xor_sync_503));
        float _shfl_xor_sync_505 = __shfl_xor_sync(0xFFFFFFFF, s_54, 24);
        float _add_504;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_504) : "f"(s_54), "f"(_shfl_xor_sync_505));
        float v_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_15) : "f"(_add_504), "f"(hc_eps));
        float m_15;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_15) : "f"(g_14), "f"(hist[23]));
        float _shfl_xor_sync_506 = __shfl_xor_sync(0xFFFFFFFF, m_15, 4);
        float ip_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_15) : "f"(m_15), "f"(_shfl_xor_sync_506));
        float _shfl_xor_sync_509 = __shfl_xor_sync(0xFFFFFFFF, ip_15, 24);
        float _add_508;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_508) : "f"(ip_15), "f"(_shfl_xor_sync_509));
        float _sub_507;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_507) : "f"(g_14), "f"(_add_508));
        float g_15 = __fdiv_rn(_sub_507, v_15);
        int upper_4 = lane_1 & 16;
        float send_5 = ((upper_4 != 0) ? dots_p_3[0] : dots_p_3[2]);
        float keep_5 = ((upper_4 != 0) ? dots_p_3[2] : dots_p_3[0]);
        float _shfl_xor_sync_510 = __shfl_xor_sync(0xFFFFFFFF, send_5, 16);
        float add_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_12) : "f"(keep_5), "f"(_shfl_xor_sync_510));
        float send_1_3 = ((upper_4 != 0) ? dots_p_3[1] : dots_p_3[3]);
        float keep_1_3 = ((upper_4 != 0) ? dots_p_3[3] : dots_p_3[1]);
        float _shfl_xor_sync_511 = __shfl_xor_sync(0xFFFFFFFF, send_1_3, 16);
        float add_1_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_7) : "f"(keep_1_3), "f"(_shfl_xor_sync_511));
        int upper_1_3 = lane_1 & 8;
        float send_2_3 = ((upper_1_3 != 0) ? add_12 : add_1_7);
        float keep_2_3 = ((upper_1_3 != 0) ? add_1_7 : add_12);
        float _shfl_xor_sync_512 = __shfl_xor_sync(0xFFFFFFFF, send_2_3, 8);
        float add_2_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_3) : "f"(keep_2_3), "f"(_shfl_xor_sync_512));
        float _shfl_xor_sync_513 = __shfl_xor_sync(0xFFFFFFFF, hist[21], 1);
        float s_55;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_55) : "f"(hist[21]), "f"(_shfl_xor_sync_513));
        float _shfl_xor_sync_515 = __shfl_xor_sync(0xFFFFFFFF, s_55, 2);
        float _add_514;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_514) : "f"(s_55), "f"(_shfl_xor_sync_515));
        float v_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_16) : "f"(_add_514), "f"(hc_eps));
        float m_16;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_16) : "f"(g_15), "f"(hist[22]));
        float _shfl_xor_sync_516 = __shfl_xor_sync(0xFFFFFFFF, m_16, 1);
        float ip_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_16) : "f"(m_16), "f"(_shfl_xor_sync_516));
        float _shfl_xor_sync_519 = __shfl_xor_sync(0xFFFFFFFF, ip_16, 2);
        float _add_518;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_518) : "f"(ip_16), "f"(_shfl_xor_sync_519));
        float _sub_517;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_517) : "f"(g_15), "f"(_add_518));
        float g_16 = __fdiv_rn(_sub_517, v_16);
        float _shfl_xor_sync_520 = __shfl_xor_sync(0xFFFFFFFF, add_2_3, 4);
        float add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_13) : "f"(add_2_3), "f"(_shfl_xor_sync_520));
        float _shfl_xor_sync_521 = __shfl_xor_sync(0xFFFFFFFF, add_13, 2);
        float add_1_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_8) : "f"(add_13), "f"(_shfl_xor_sync_521));
        float _shfl_xor_sync_523 = __shfl_xor_sync(0xFFFFFFFF, add_1_8, 1);
        float _add_522;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_522) : "f"(add_1_8), "f"(_shfl_xor_sync_523));
        dpre_part[own_base + 128] = _add_522;
        float _shfl_xor_sync_524 = __shfl_xor_sync(0xFFFFFFFF, hist[20], 4);
        float s_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_56) : "f"(hist[20]), "f"(_shfl_xor_sync_524));
        float _shfl_xor_sync_526 = __shfl_xor_sync(0xFFFFFFFF, s_56, 24);
        float _add_525;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_525) : "f"(s_56), "f"(_shfl_xor_sync_526));
        float v_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_17) : "f"(_add_525), "f"(hc_eps));
        float m_17;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_17) : "f"(g_16), "f"(hist[21]));
        float _shfl_xor_sync_527 = __shfl_xor_sync(0xFFFFFFFF, m_17, 4);
        float ip_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_17) : "f"(m_17), "f"(_shfl_xor_sync_527));
        float _shfl_xor_sync_530 = __shfl_xor_sync(0xFFFFFFFF, ip_17, 24);
        float _add_529;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_529) : "f"(ip_17), "f"(_shfl_xor_sync_530));
        float _sub_528;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_528) : "f"(g_16), "f"(_add_529));
        float g_17 = __fdiv_rn(_sub_528, v_17);
        mbarrier_wait(ring_bar_addr + (st__1_1) * 8, 0);
        unsigned int ring_pk_24[10];
        unsigned int lane_row_24 = sa__1_1 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[(0) + 1]))
            : "r"(lane_row_24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[(2) + 1]))
            : "r"(lane_row_24 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[(4) + 1]))
            : "r"(lane_row_24 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[(6) + 1]))
            : "r"(lane_row_24 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_24[(8) + 1]))
            : "r"(lane_row_24 + 32));
        unsigned int ring_pk_25[10];
        unsigned int lane_row_25 = sa__1_1 + 1280 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[(0) + 1]))
            : "r"(lane_row_25));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[(2) + 1]))
            : "r"(lane_row_25 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[(4) + 1]))
            : "r"(lane_row_25 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[(6) + 1]))
            : "r"(lane_row_25 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_25[(8) + 1]))
            : "r"(lane_row_25 + 32));
        float dots_p_4[4];
        float _shfl_xor_sync_531 = __shfl_xor_sync(0xFFFFFFFF, hist[19], 1);
        float s_57;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_57) : "f"(hist[19]), "f"(_shfl_xor_sync_531));
        float _shfl_xor_sync_533 = __shfl_xor_sync(0xFFFFFFFF, s_57, 2);
        float _add_532;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_532) : "f"(s_57), "f"(_shfl_xor_sync_533));
        float v_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_18) : "f"(_add_532), "f"(hc_eps));
        float m_18;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_18) : "f"(g_17), "f"(hist[20]));
        float _shfl_xor_sync_534 = __shfl_xor_sync(0xFFFFFFFF, m_18, 1);
        float ip_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_18) : "f"(m_18), "f"(_shfl_xor_sync_534));
        float _shfl_xor_sync_537 = __shfl_xor_sync(0xFFFFFFFF, ip_18, 2);
        float _add_536;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_536) : "f"(ip_18), "f"(_shfl_xor_sync_537));
        float _sub_535;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_535) : "f"(g_17), "f"(_add_536));
        float g_18 = __fdiv_rn(_sub_535, v_18);
        float _dot_538;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_538) : "r"(ring_pk_25[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[0] = _dot_538;
        float _dot_539;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_539) : "r"(ring_pk_25[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_539;
        float _dot_540;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_540) : "r"(ring_pk_25[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_540;
        float _dot_541;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_541) : "r"(ring_pk_25[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_541;
        float _dot_542;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_542) : "r"(ring_pk_25[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_542;
        float _dot_543;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_543) : "r"(ring_pk_25[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_543;
        float _dot_544;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_544) : "r"(ring_pk_25[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_544;
        float _dot_545;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_545) : "r"(ring_pk_25[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_545;
        float _dot_546;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_546) : "r"(ring_pk_25[8]), "r"(ring_pk_24[8]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_546;
        float _dot_547;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_547) : "r"(ring_pk_25[9]), "r"(ring_pk_24[9]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_547;
        unsigned int ring_pk_26[10];
        unsigned int lane_row_26 = sa__1_1 + 2560 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(0) + 1]))
            : "r"(lane_row_26));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(2) + 1]))
            : "r"(lane_row_26 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(4) + 1]))
            : "r"(lane_row_26 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(6) + 1]))
            : "r"(lane_row_26 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_26[(8) + 1]))
            : "r"(lane_row_26 + 32));
        float _shfl_xor_sync_548 = __shfl_xor_sync(0xFFFFFFFF, hist[18], 4);
        float s_58;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_58) : "f"(hist[18]), "f"(_shfl_xor_sync_548));
        float _shfl_xor_sync_550 = __shfl_xor_sync(0xFFFFFFFF, s_58, 24);
        float _add_549;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_549) : "f"(s_58), "f"(_shfl_xor_sync_550));
        float v_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_19) : "f"(_add_549), "f"(hc_eps));
        float m_19;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_19) : "f"(g_18), "f"(hist[19]));
        float _shfl_xor_sync_551 = __shfl_xor_sync(0xFFFFFFFF, m_19, 4);
        float ip_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_19) : "f"(m_19), "f"(_shfl_xor_sync_551));
        float _shfl_xor_sync_554 = __shfl_xor_sync(0xFFFFFFFF, ip_19, 24);
        float _add_553;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_553) : "f"(ip_19), "f"(_shfl_xor_sync_554));
        float _sub_552;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_552) : "f"(g_18), "f"(_add_553));
        float g_19 = __fdiv_rn(_sub_552, v_19);
        float _dot_555;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_555) : "r"(ring_pk_26[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[1] = _dot_555;
        float _dot_556;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_556) : "r"(ring_pk_26[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_556;
        float _dot_557;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_557) : "r"(ring_pk_26[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_557;
        float _dot_558;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_558) : "r"(ring_pk_26[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_558;
        float _dot_559;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_559) : "r"(ring_pk_26[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_559;
        float _dot_560;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_560) : "r"(ring_pk_26[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_560;
        float _dot_561;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_561) : "r"(ring_pk_26[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_561;
        float _dot_562;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_562) : "r"(ring_pk_26[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_562;
        float _dot_563;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_563) : "r"(ring_pk_26[8]), "r"(ring_pk_24[8]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_563;
        float _dot_564;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_564) : "r"(ring_pk_26[9]), "r"(ring_pk_24[9]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_564;
        unsigned int ring_pk_27[10];
        unsigned int lane_row_27 = sa__1_1 + 3840 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(0) + 1]))
            : "r"(lane_row_27));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(2) + 1]))
            : "r"(lane_row_27 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(4) + 1]))
            : "r"(lane_row_27 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(6) + 1]))
            : "r"(lane_row_27 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_27[(8) + 1]))
            : "r"(lane_row_27 + 32));
        float _shfl_xor_sync_565 = __shfl_xor_sync(0xFFFFFFFF, hist[17], 1);
        float s_59;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_59) : "f"(hist[17]), "f"(_shfl_xor_sync_565));
        float _shfl_xor_sync_567 = __shfl_xor_sync(0xFFFFFFFF, s_59, 2);
        float _add_566;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_566) : "f"(s_59), "f"(_shfl_xor_sync_567));
        float v_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_20) : "f"(_add_566), "f"(hc_eps));
        float m_20;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_20) : "f"(g_19), "f"(hist[18]));
        float _shfl_xor_sync_568 = __shfl_xor_sync(0xFFFFFFFF, m_20, 1);
        float ip_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_20) : "f"(m_20), "f"(_shfl_xor_sync_568));
        float _shfl_xor_sync_571 = __shfl_xor_sync(0xFFFFFFFF, ip_20, 2);
        float _add_570;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_570) : "f"(ip_20), "f"(_shfl_xor_sync_571));
        float _sub_569;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_569) : "f"(g_19), "f"(_add_570));
        float g_20 = __fdiv_rn(_sub_569, v_20);
        float _dot_572;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_572) : "r"(ring_pk_27[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[2] = _dot_572;
        float _dot_573;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_573) : "r"(ring_pk_27[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_573;
        float _dot_574;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_574) : "r"(ring_pk_27[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_574;
        float _dot_575;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_575) : "r"(ring_pk_27[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_575;
        float _dot_576;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_576) : "r"(ring_pk_27[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_576;
        float _dot_577;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_577) : "r"(ring_pk_27[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_577;
        float _dot_578;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_578) : "r"(ring_pk_27[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_578;
        float _dot_579;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_579) : "r"(ring_pk_27[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_579;
        float _dot_580;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_580) : "r"(ring_pk_27[8]), "r"(ring_pk_24[8]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_580;
        float _dot_581;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_581) : "r"(ring_pk_27[9]), "r"(ring_pk_24[9]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_581;
        unsigned int ring_pk_28[10];
        unsigned int lane_row_28 = sa__1_1 + 5120 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(0) + 1]))
            : "r"(lane_row_28));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(2) + 1]))
            : "r"(lane_row_28 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(4) + 1]))
            : "r"(lane_row_28 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(6) + 1]))
            : "r"(lane_row_28 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_28[(8) + 1]))
            : "r"(lane_row_28 + 32));
        float _shfl_xor_sync_582 = __shfl_xor_sync(0xFFFFFFFF, hist[16], 4);
        float s_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_60) : "f"(hist[16]), "f"(_shfl_xor_sync_582));
        float _shfl_xor_sync_584 = __shfl_xor_sync(0xFFFFFFFF, s_60, 24);
        float _add_583;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_583) : "f"(s_60), "f"(_shfl_xor_sync_584));
        float v_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_21) : "f"(_add_583), "f"(hc_eps));
        float m_21;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_21) : "f"(g_20), "f"(hist[17]));
        float _shfl_xor_sync_585 = __shfl_xor_sync(0xFFFFFFFF, m_21, 4);
        float ip_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_21) : "f"(m_21), "f"(_shfl_xor_sync_585));
        float _shfl_xor_sync_588 = __shfl_xor_sync(0xFFFFFFFF, ip_21, 24);
        float _add_587;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_587) : "f"(ip_21), "f"(_shfl_xor_sync_588));
        float _sub_586;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_586) : "f"(g_20), "f"(_add_587));
        float g_21 = __fdiv_rn(_sub_586, v_21);
        float _dot_589;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_589) : "r"(ring_pk_28[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[3] = _dot_589;
        float _dot_590;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_590) : "r"(ring_pk_28[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_590;
        float _dot_591;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_591) : "r"(ring_pk_28[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_591;
        float _dot_592;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_592) : "r"(ring_pk_28[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_592;
        float _dot_593;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_593) : "r"(ring_pk_28[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_593;
        float _dot_594;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_594) : "r"(ring_pk_28[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_594;
        float _dot_595;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_595) : "r"(ring_pk_28[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_595;
        float _dot_596;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_596) : "r"(ring_pk_28[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_596;
        float _dot_597;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_597) : "r"(ring_pk_28[8]), "r"(ring_pk_24[8]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_597;
        float _dot_598;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_598) : "r"(ring_pk_28[9]), "r"(ring_pk_24[9]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_598;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_1) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1), "l"(dU + (th2_7 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 1280), "l"(X + (th2_7 * 4 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 2560), "l"(X + ((th2_7 * 4 + 1) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 3840), "l"(X + ((th2_7 * 4 + 2) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 5120), "l"(X + ((th2_7 * 4 + 3) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_599 = __shfl_xor_sync(0xFFFFFFFF, hist[15], 1);
        float s_61;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_61) : "f"(hist[15]), "f"(_shfl_xor_sync_599));
        float _shfl_xor_sync_601 = __shfl_xor_sync(0xFFFFFFFF, s_61, 2);
        float _add_600;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_600) : "f"(s_61), "f"(_shfl_xor_sync_601));
        float v_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_22) : "f"(_add_600), "f"(hc_eps));
        float m_22;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_22) : "f"(g_21), "f"(hist[16]));
        float _shfl_xor_sync_602 = __shfl_xor_sync(0xFFFFFFFF, m_22, 1);
        float ip_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_22) : "f"(m_22), "f"(_shfl_xor_sync_602));
        float _shfl_xor_sync_605 = __shfl_xor_sync(0xFFFFFFFF, ip_22, 2);
        float _add_604;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_604) : "f"(ip_22), "f"(_shfl_xor_sync_605));
        float _sub_603;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_603) : "f"(g_21), "f"(_add_604));
        float g_22 = __fdiv_rn(_sub_603, v_22);
        int upper_5 = lane_1 & 16;
        float send_6 = ((upper_5 != 0) ? dots_p_4[0] : dots_p_4[2]);
        float keep_6 = ((upper_5 != 0) ? dots_p_4[2] : dots_p_4[0]);
        float _shfl_xor_sync_606 = __shfl_xor_sync(0xFFFFFFFF, send_6, 16);
        float add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_14) : "f"(keep_6), "f"(_shfl_xor_sync_606));
        float send_1_4 = ((upper_5 != 0) ? dots_p_4[1] : dots_p_4[3]);
        float keep_1_4 = ((upper_5 != 0) ? dots_p_4[3] : dots_p_4[1]);
        float _shfl_xor_sync_607 = __shfl_xor_sync(0xFFFFFFFF, send_1_4, 16);
        float add_1_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_9) : "f"(keep_1_4), "f"(_shfl_xor_sync_607));
        int upper_1_4 = lane_1 & 8;
        float send_2_4 = ((upper_1_4 != 0) ? add_14 : add_1_9);
        float keep_2_4 = ((upper_1_4 != 0) ? add_1_9 : add_14);
        float _shfl_xor_sync_608 = __shfl_xor_sync(0xFFFFFFFF, send_2_4, 8);
        float add_2_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_4) : "f"(keep_2_4), "f"(_shfl_xor_sync_608));
        float _shfl_xor_sync_609 = __shfl_xor_sync(0xFFFFFFFF, hist[14], 4);
        float s_62;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_62) : "f"(hist[14]), "f"(_shfl_xor_sync_609));
        float _shfl_xor_sync_611 = __shfl_xor_sync(0xFFFFFFFF, s_62, 24);
        float _add_610;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_610) : "f"(s_62), "f"(_shfl_xor_sync_611));
        float v_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_23) : "f"(_add_610), "f"(hc_eps));
        float m_23;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_23) : "f"(g_22), "f"(hist[15]));
        float _shfl_xor_sync_612 = __shfl_xor_sync(0xFFFFFFFF, m_23, 4);
        float ip_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_23) : "f"(m_23), "f"(_shfl_xor_sync_612));
        float _shfl_xor_sync_615 = __shfl_xor_sync(0xFFFFFFFF, ip_23, 24);
        float _add_614;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_614) : "f"(ip_23), "f"(_shfl_xor_sync_615));
        float _sub_613;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_613) : "f"(g_22), "f"(_add_614));
        float g_23 = __fdiv_rn(_sub_613, v_23);
        float _shfl_xor_sync_616 = __shfl_xor_sync(0xFFFFFFFF, add_2_4, 4);
        float add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_15) : "f"(add_2_4), "f"(_shfl_xor_sync_616));
        float _shfl_xor_sync_617 = __shfl_xor_sync(0xFFFFFFFF, add_15, 2);
        float add_1_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_10) : "f"(add_15), "f"(_shfl_xor_sync_617));
        float _shfl_xor_sync_619 = __shfl_xor_sync(0xFFFFFFFF, add_1_10, 1);
        float _add_618;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_618) : "f"(add_1_10), "f"(_shfl_xor_sync_619));
        dpre_part[own_base + 160] = _add_618;
        float _shfl_xor_sync_620 = __shfl_xor_sync(0xFFFFFFFF, hist[13], 1);
        float s_63;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_63) : "f"(hist[13]), "f"(_shfl_xor_sync_620));
        float _shfl_xor_sync_622 = __shfl_xor_sync(0xFFFFFFFF, s_63, 2);
        float _add_621;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_621) : "f"(s_63), "f"(_shfl_xor_sync_622));
        float v_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_24) : "f"(_add_621), "f"(hc_eps));
        float m_24;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_24) : "f"(g_23), "f"(hist[14]));
        float _shfl_xor_sync_623 = __shfl_xor_sync(0xFFFFFFFF, m_24, 1);
        float ip_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_24) : "f"(m_24), "f"(_shfl_xor_sync_623));
        float _shfl_xor_sync_626 = __shfl_xor_sync(0xFFFFFFFF, ip_24, 2);
        float _add_625;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_625) : "f"(ip_24), "f"(_shfl_xor_sync_626));
        float _sub_624;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_624) : "f"(g_23), "f"(_add_625));
        float g_24 = __fdiv_rn(_sub_624, v_24);
        mbarrier_wait(ring_bar_addr + (st__1_0) * 8, 1);
        unsigned int ring_pk_29[10];
        unsigned int lane_row_29 = sa__1_0 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[(0) + 1]))
            : "r"(lane_row_29));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[(2) + 1]))
            : "r"(lane_row_29 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[(4) + 1]))
            : "r"(lane_row_29 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[(6) + 1]))
            : "r"(lane_row_29 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_29[(8) + 1]))
            : "r"(lane_row_29 + 32));
        unsigned int ring_pk_30[10];
        unsigned int lane_row_30 = sa__1_0 + 1280 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[(0) + 1]))
            : "r"(lane_row_30));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[(2) + 1]))
            : "r"(lane_row_30 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[(4) + 1]))
            : "r"(lane_row_30 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[(6) + 1]))
            : "r"(lane_row_30 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_30[(8) + 1]))
            : "r"(lane_row_30 + 32));
        float dots_p_5[4];
        float _shfl_xor_sync_627 = __shfl_xor_sync(0xFFFFFFFF, hist[12], 4);
        float s_64;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_64) : "f"(hist[12]), "f"(_shfl_xor_sync_627));
        float _shfl_xor_sync_629 = __shfl_xor_sync(0xFFFFFFFF, s_64, 24);
        float _add_628;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_628) : "f"(s_64), "f"(_shfl_xor_sync_629));
        float v_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_25) : "f"(_add_628), "f"(hc_eps));
        float m_25;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_25) : "f"(g_24), "f"(hist[13]));
        float _shfl_xor_sync_630 = __shfl_xor_sync(0xFFFFFFFF, m_25, 4);
        float ip_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_25) : "f"(m_25), "f"(_shfl_xor_sync_630));
        float _shfl_xor_sync_633 = __shfl_xor_sync(0xFFFFFFFF, ip_25, 24);
        float _add_632;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_632) : "f"(ip_25), "f"(_shfl_xor_sync_633));
        float _sub_631;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_631) : "f"(g_24), "f"(_add_632));
        float g_25 = __fdiv_rn(_sub_631, v_25);
        float _dot_634;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_634) : "r"(ring_pk_30[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[0] = _dot_634;
        float _dot_635;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_635) : "r"(ring_pk_30[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_635;
        float _dot_636;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_636) : "r"(ring_pk_30[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_636;
        float _dot_637;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_637) : "r"(ring_pk_30[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_637;
        float _dot_638;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_638) : "r"(ring_pk_30[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_638;
        float _dot_639;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_639) : "r"(ring_pk_30[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_639;
        float _dot_640;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_640) : "r"(ring_pk_30[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_640;
        float _dot_641;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_641) : "r"(ring_pk_30[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_641;
        float _dot_642;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_642) : "r"(ring_pk_30[8]), "r"(ring_pk_29[8]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_642;
        float _dot_643;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_643) : "r"(ring_pk_30[9]), "r"(ring_pk_29[9]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_643;
        unsigned int ring_pk_31[10];
        unsigned int lane_row_31 = sa__1_0 + 2560 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(0) + 1]))
            : "r"(lane_row_31));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(2) + 1]))
            : "r"(lane_row_31 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(4) + 1]))
            : "r"(lane_row_31 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(6) + 1]))
            : "r"(lane_row_31 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_31[(8) + 1]))
            : "r"(lane_row_31 + 32));
        float _shfl_xor_sync_644 = __shfl_xor_sync(0xFFFFFFFF, hist[11], 1);
        float s_65;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_65) : "f"(hist[11]), "f"(_shfl_xor_sync_644));
        float _shfl_xor_sync_646 = __shfl_xor_sync(0xFFFFFFFF, s_65, 2);
        float _add_645;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_645) : "f"(s_65), "f"(_shfl_xor_sync_646));
        float v_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_26) : "f"(_add_645), "f"(hc_eps));
        float m_26;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_26) : "f"(g_25), "f"(hist[12]));
        float _shfl_xor_sync_647 = __shfl_xor_sync(0xFFFFFFFF, m_26, 1);
        float ip_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_26) : "f"(m_26), "f"(_shfl_xor_sync_647));
        float _shfl_xor_sync_650 = __shfl_xor_sync(0xFFFFFFFF, ip_26, 2);
        float _add_649;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_649) : "f"(ip_26), "f"(_shfl_xor_sync_650));
        float _sub_648;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_648) : "f"(g_25), "f"(_add_649));
        float g_26 = __fdiv_rn(_sub_648, v_26);
        float _dot_651;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_651) : "r"(ring_pk_31[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[1] = _dot_651;
        float _dot_652;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_652) : "r"(ring_pk_31[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_652;
        float _dot_653;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_653) : "r"(ring_pk_31[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_653;
        float _dot_654;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_654) : "r"(ring_pk_31[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_654;
        float _dot_655;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_655) : "r"(ring_pk_31[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_655;
        float _dot_656;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_656) : "r"(ring_pk_31[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_656;
        float _dot_657;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_657) : "r"(ring_pk_31[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_657;
        float _dot_658;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_658) : "r"(ring_pk_31[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_658;
        float _dot_659;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_659) : "r"(ring_pk_31[8]), "r"(ring_pk_29[8]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_659;
        float _dot_660;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_660) : "r"(ring_pk_31[9]), "r"(ring_pk_29[9]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_660;
        unsigned int ring_pk_32[10];
        unsigned int lane_row_32 = sa__1_0 + 3840 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(0) + 1]))
            : "r"(lane_row_32));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(2) + 1]))
            : "r"(lane_row_32 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(4) + 1]))
            : "r"(lane_row_32 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(6) + 1]))
            : "r"(lane_row_32 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_32[(8) + 1]))
            : "r"(lane_row_32 + 32));
        float _shfl_xor_sync_661 = __shfl_xor_sync(0xFFFFFFFF, hist[10], 4);
        float s_66;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_66) : "f"(hist[10]), "f"(_shfl_xor_sync_661));
        float _shfl_xor_sync_663 = __shfl_xor_sync(0xFFFFFFFF, s_66, 24);
        float _add_662;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_662) : "f"(s_66), "f"(_shfl_xor_sync_663));
        float v_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_27) : "f"(_add_662), "f"(hc_eps));
        float m_27;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_27) : "f"(g_26), "f"(hist[11]));
        float _shfl_xor_sync_664 = __shfl_xor_sync(0xFFFFFFFF, m_27, 4);
        float ip_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_27) : "f"(m_27), "f"(_shfl_xor_sync_664));
        float _shfl_xor_sync_667 = __shfl_xor_sync(0xFFFFFFFF, ip_27, 24);
        float _add_666;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_666) : "f"(ip_27), "f"(_shfl_xor_sync_667));
        float _sub_665;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_665) : "f"(g_26), "f"(_add_666));
        float g_27 = __fdiv_rn(_sub_665, v_27);
        float _dot_668;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_668) : "r"(ring_pk_32[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[2] = _dot_668;
        float _dot_669;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_669) : "r"(ring_pk_32[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_669;
        float _dot_670;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_670) : "r"(ring_pk_32[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_670;
        float _dot_671;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_671) : "r"(ring_pk_32[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_671;
        float _dot_672;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_672) : "r"(ring_pk_32[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_672;
        float _dot_673;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_673) : "r"(ring_pk_32[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_673;
        float _dot_674;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_674) : "r"(ring_pk_32[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_674;
        float _dot_675;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_675) : "r"(ring_pk_32[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_675;
        float _dot_676;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_676) : "r"(ring_pk_32[8]), "r"(ring_pk_29[8]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_676;
        float _dot_677;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_677) : "r"(ring_pk_32[9]), "r"(ring_pk_29[9]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_677;
        unsigned int ring_pk_33[10];
        unsigned int lane_row_33 = sa__1_0 + 5120 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(0) + 1]))
            : "r"(lane_row_33));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(2) + 1]))
            : "r"(lane_row_33 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(4) + 1]))
            : "r"(lane_row_33 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(6) + 1]))
            : "r"(lane_row_33 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_33[(8) + 1]))
            : "r"(lane_row_33 + 32));
        float _shfl_xor_sync_678 = __shfl_xor_sync(0xFFFFFFFF, hist[9], 1);
        float s_67;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_67) : "f"(hist[9]), "f"(_shfl_xor_sync_678));
        float _shfl_xor_sync_680 = __shfl_xor_sync(0xFFFFFFFF, s_67, 2);
        float _add_679;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_679) : "f"(s_67), "f"(_shfl_xor_sync_680));
        float v_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_28) : "f"(_add_679), "f"(hc_eps));
        float m_28;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_28) : "f"(g_27), "f"(hist[10]));
        float _shfl_xor_sync_681 = __shfl_xor_sync(0xFFFFFFFF, m_28, 1);
        float ip_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_28) : "f"(m_28), "f"(_shfl_xor_sync_681));
        float _shfl_xor_sync_684 = __shfl_xor_sync(0xFFFFFFFF, ip_28, 2);
        float _add_683;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_683) : "f"(ip_28), "f"(_shfl_xor_sync_684));
        float _sub_682;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_682) : "f"(g_27), "f"(_add_683));
        float g_28 = __fdiv_rn(_sub_682, v_28);
        float _dot_685;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_685) : "r"(ring_pk_33[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[3] = _dot_685;
        float _dot_686;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_686) : "r"(ring_pk_33[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_686;
        float _dot_687;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_687) : "r"(ring_pk_33[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_687;
        float _dot_688;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_688) : "r"(ring_pk_33[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_688;
        float _dot_689;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_689) : "r"(ring_pk_33[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_689;
        float _dot_690;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_690) : "r"(ring_pk_33[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_690;
        float _dot_691;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_691) : "r"(ring_pk_33[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_691;
        float _dot_692;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_692) : "r"(ring_pk_33[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_692;
        float _dot_693;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_693) : "r"(ring_pk_33[8]), "r"(ring_pk_29[8]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_693;
        float _dot_694;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_694) : "r"(ring_pk_33[9]), "r"(ring_pk_29[9]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_694;
        __syncwarp();
        bool elected_0 = elect_sync();
        bool defer_el_0 = elected_0 && tail_pred;
        if (defer_el_0) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_0) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0), "l"(dU + (th2_8 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 1280), "l"(X + (th2_8 * 4 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 2560), "l"(X + ((th2_8 * 4 + 1) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 3840), "l"(X + ((th2_8 * 4 + 2) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_0 + 5120), "l"(X + ((th2_8 * 4 + 3) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_0) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_695 = __shfl_xor_sync(0xFFFFFFFF, hist[8], 4);
        float s_68;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_68) : "f"(hist[8]), "f"(_shfl_xor_sync_695));
        float _shfl_xor_sync_697 = __shfl_xor_sync(0xFFFFFFFF, s_68, 24);
        float _add_696;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_696) : "f"(s_68), "f"(_shfl_xor_sync_697));
        float v_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_29) : "f"(_add_696), "f"(hc_eps));
        float m_29;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_29) : "f"(g_28), "f"(hist[9]));
        float _shfl_xor_sync_698 = __shfl_xor_sync(0xFFFFFFFF, m_29, 4);
        float ip_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_29) : "f"(m_29), "f"(_shfl_xor_sync_698));
        float _shfl_xor_sync_701 = __shfl_xor_sync(0xFFFFFFFF, ip_29, 24);
        float _add_700;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_700) : "f"(ip_29), "f"(_shfl_xor_sync_701));
        float _sub_699;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_699) : "f"(g_28), "f"(_add_700));
        float g_29 = __fdiv_rn(_sub_699, v_29);
        int upper_6 = lane_1 & 16;
        float send_7 = ((upper_6 != 0) ? dots_p_5[0] : dots_p_5[2]);
        float keep_7 = ((upper_6 != 0) ? dots_p_5[2] : dots_p_5[0]);
        float _shfl_xor_sync_702 = __shfl_xor_sync(0xFFFFFFFF, send_7, 16);
        float add_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_16) : "f"(keep_7), "f"(_shfl_xor_sync_702));
        float send_1_5 = ((upper_6 != 0) ? dots_p_5[1] : dots_p_5[3]);
        float keep_1_5 = ((upper_6 != 0) ? dots_p_5[3] : dots_p_5[1]);
        float _shfl_xor_sync_703 = __shfl_xor_sync(0xFFFFFFFF, send_1_5, 16);
        float add_1_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_11) : "f"(keep_1_5), "f"(_shfl_xor_sync_703));
        int upper_1_5 = lane_1 & 8;
        float send_2_5 = ((upper_1_5 != 0) ? add_16 : add_1_11);
        float keep_2_5 = ((upper_1_5 != 0) ? add_1_11 : add_16);
        float _shfl_xor_sync_704 = __shfl_xor_sync(0xFFFFFFFF, send_2_5, 8);
        float add_2_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_5) : "f"(keep_2_5), "f"(_shfl_xor_sync_704));
        float _shfl_xor_sync_705 = __shfl_xor_sync(0xFFFFFFFF, hist[7], 1);
        float s_69;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_69) : "f"(hist[7]), "f"(_shfl_xor_sync_705));
        float _shfl_xor_sync_707 = __shfl_xor_sync(0xFFFFFFFF, s_69, 2);
        float _add_706;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_706) : "f"(s_69), "f"(_shfl_xor_sync_707));
        float v_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_30) : "f"(_add_706), "f"(hc_eps));
        float m_30;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_30) : "f"(g_29), "f"(hist[8]));
        float _shfl_xor_sync_708 = __shfl_xor_sync(0xFFFFFFFF, m_30, 1);
        float ip_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_30) : "f"(m_30), "f"(_shfl_xor_sync_708));
        float _shfl_xor_sync_711 = __shfl_xor_sync(0xFFFFFFFF, ip_30, 2);
        float _add_710;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_710) : "f"(ip_30), "f"(_shfl_xor_sync_711));
        float _sub_709;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_709) : "f"(g_29), "f"(_add_710));
        float g_30 = __fdiv_rn(_sub_709, v_30);
        float _shfl_xor_sync_712 = __shfl_xor_sync(0xFFFFFFFF, add_2_5, 4);
        float add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_17) : "f"(add_2_5), "f"(_shfl_xor_sync_712));
        float _shfl_xor_sync_713 = __shfl_xor_sync(0xFFFFFFFF, add_17, 2);
        float add_1_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_12) : "f"(add_17), "f"(_shfl_xor_sync_713));
        float _shfl_xor_sync_715 = __shfl_xor_sync(0xFFFFFFFF, add_1_12, 1);
        float _add_714;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_714) : "f"(add_1_12), "f"(_shfl_xor_sync_715));
        dpre_part[own_base + 192] = _add_714;
        float _shfl_xor_sync_716 = __shfl_xor_sync(0xFFFFFFFF, hist[6], 4);
        float s_70;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_70) : "f"(hist[6]), "f"(_shfl_xor_sync_716));
        float _shfl_xor_sync_718 = __shfl_xor_sync(0xFFFFFFFF, s_70, 24);
        float _add_717;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_717) : "f"(s_70), "f"(_shfl_xor_sync_718));
        float v_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_31) : "f"(_add_717), "f"(hc_eps));
        float m_31;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_31) : "f"(g_30), "f"(hist[7]));
        float _shfl_xor_sync_719 = __shfl_xor_sync(0xFFFFFFFF, m_31, 4);
        float ip_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_31) : "f"(m_31), "f"(_shfl_xor_sync_719));
        float _shfl_xor_sync_722 = __shfl_xor_sync(0xFFFFFFFF, ip_31, 24);
        float _add_721;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_721) : "f"(ip_31), "f"(_shfl_xor_sync_722));
        float _sub_720;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_720) : "f"(g_30), "f"(_add_721));
        float g_31 = __fdiv_rn(_sub_720, v_31);
        mbarrier_wait(ring_bar_addr + (st__1_1) * 8, 1);
        unsigned int ring_pk_34[10];
        unsigned int lane_row_34 = sa__1_1 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[(0) + 1]))
            : "r"(lane_row_34));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[(2) + 1]))
            : "r"(lane_row_34 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[(4) + 1]))
            : "r"(lane_row_34 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[(6) + 1]))
            : "r"(lane_row_34 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_34[(8) + 1]))
            : "r"(lane_row_34 + 32));
        unsigned int ring_pk_35[10];
        unsigned int lane_row_35 = sa__1_1 + 1280 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[(0) + 1]))
            : "r"(lane_row_35));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[(2) + 1]))
            : "r"(lane_row_35 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[(4) + 1]))
            : "r"(lane_row_35 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[(6) + 1]))
            : "r"(lane_row_35 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_35[(8) + 1]))
            : "r"(lane_row_35 + 32));
        float dots_p_6[4];
        float _shfl_xor_sync_723 = __shfl_xor_sync(0xFFFFFFFF, hist[5], 1);
        float s_71;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_71) : "f"(hist[5]), "f"(_shfl_xor_sync_723));
        float _shfl_xor_sync_725 = __shfl_xor_sync(0xFFFFFFFF, s_71, 2);
        float _add_724;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_724) : "f"(s_71), "f"(_shfl_xor_sync_725));
        float v_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_32) : "f"(_add_724), "f"(hc_eps));
        float m_32;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_32) : "f"(g_31), "f"(hist[6]));
        float _shfl_xor_sync_726 = __shfl_xor_sync(0xFFFFFFFF, m_32, 1);
        float ip_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_32) : "f"(m_32), "f"(_shfl_xor_sync_726));
        float _shfl_xor_sync_729 = __shfl_xor_sync(0xFFFFFFFF, ip_32, 2);
        float _add_728;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_728) : "f"(ip_32), "f"(_shfl_xor_sync_729));
        float _sub_727;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_727) : "f"(g_31), "f"(_add_728));
        float g_32 = __fdiv_rn(_sub_727, v_32);
        float _dot_730;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_730) : "r"(ring_pk_35[0]), "r"(ring_pk_34[0]), "f"(-0.0f));
        dots_p_6[0] = _dot_730;
        float _dot_731;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_731) : "r"(ring_pk_35[1]), "r"(ring_pk_34[1]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_731;
        float _dot_732;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_732) : "r"(ring_pk_35[2]), "r"(ring_pk_34[2]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_732;
        float _dot_733;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_733) : "r"(ring_pk_35[3]), "r"(ring_pk_34[3]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_733;
        float _dot_734;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_734) : "r"(ring_pk_35[4]), "r"(ring_pk_34[4]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_734;
        float _dot_735;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_735) : "r"(ring_pk_35[5]), "r"(ring_pk_34[5]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_735;
        float _dot_736;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_736) : "r"(ring_pk_35[6]), "r"(ring_pk_34[6]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_736;
        float _dot_737;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_737) : "r"(ring_pk_35[7]), "r"(ring_pk_34[7]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_737;
        float _dot_738;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_738) : "r"(ring_pk_35[8]), "r"(ring_pk_34[8]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_738;
        float _dot_739;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_739) : "r"(ring_pk_35[9]), "r"(ring_pk_34[9]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_739;
        unsigned int ring_pk_36[10];
        unsigned int lane_row_36 = sa__1_1 + 2560 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[(0) + 1]))
            : "r"(lane_row_36));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[(2) + 1]))
            : "r"(lane_row_36 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[(4) + 1]))
            : "r"(lane_row_36 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[(6) + 1]))
            : "r"(lane_row_36 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_36[(8) + 1]))
            : "r"(lane_row_36 + 32));
        float _shfl_xor_sync_740 = __shfl_xor_sync(0xFFFFFFFF, hist[4], 4);
        float s_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_72) : "f"(hist[4]), "f"(_shfl_xor_sync_740));
        float _shfl_xor_sync_742 = __shfl_xor_sync(0xFFFFFFFF, s_72, 24);
        float _add_741;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_741) : "f"(s_72), "f"(_shfl_xor_sync_742));
        float v_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_33) : "f"(_add_741), "f"(hc_eps));
        float m_33;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_33) : "f"(g_32), "f"(hist[5]));
        float _shfl_xor_sync_743 = __shfl_xor_sync(0xFFFFFFFF, m_33, 4);
        float ip_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_33) : "f"(m_33), "f"(_shfl_xor_sync_743));
        float _shfl_xor_sync_746 = __shfl_xor_sync(0xFFFFFFFF, ip_33, 24);
        float _add_745;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_745) : "f"(ip_33), "f"(_shfl_xor_sync_746));
        float _sub_744;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_744) : "f"(g_32), "f"(_add_745));
        float g_33 = __fdiv_rn(_sub_744, v_33);
        float _dot_747;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_747) : "r"(ring_pk_36[0]), "r"(ring_pk_34[0]), "f"(-0.0f));
        dots_p_6[1] = _dot_747;
        float _dot_748;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_748) : "r"(ring_pk_36[1]), "r"(ring_pk_34[1]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_748;
        float _dot_749;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_749) : "r"(ring_pk_36[2]), "r"(ring_pk_34[2]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_749;
        float _dot_750;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_750) : "r"(ring_pk_36[3]), "r"(ring_pk_34[3]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_750;
        float _dot_751;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_751) : "r"(ring_pk_36[4]), "r"(ring_pk_34[4]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_751;
        float _dot_752;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_752) : "r"(ring_pk_36[5]), "r"(ring_pk_34[5]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_752;
        float _dot_753;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_753) : "r"(ring_pk_36[6]), "r"(ring_pk_34[6]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_753;
        float _dot_754;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_754) : "r"(ring_pk_36[7]), "r"(ring_pk_34[7]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_754;
        float _dot_755;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_755) : "r"(ring_pk_36[8]), "r"(ring_pk_34[8]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_755;
        float _dot_756;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_756) : "r"(ring_pk_36[9]), "r"(ring_pk_34[9]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_756;
        unsigned int ring_pk_37[10];
        unsigned int lane_row_37 = sa__1_1 + 3840 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[(0) + 1]))
            : "r"(lane_row_37));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[(2) + 1]))
            : "r"(lane_row_37 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[(4) + 1]))
            : "r"(lane_row_37 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[(6) + 1]))
            : "r"(lane_row_37 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_37[(8) + 1]))
            : "r"(lane_row_37 + 32));
        float _shfl_xor_sync_757 = __shfl_xor_sync(0xFFFFFFFF, hist[3], 1);
        float s_73;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_73) : "f"(hist[3]), "f"(_shfl_xor_sync_757));
        float _shfl_xor_sync_759 = __shfl_xor_sync(0xFFFFFFFF, s_73, 2);
        float _add_758;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_758) : "f"(s_73), "f"(_shfl_xor_sync_759));
        float v_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_34) : "f"(_add_758), "f"(hc_eps));
        float m_34;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_34) : "f"(g_33), "f"(hist[4]));
        float _shfl_xor_sync_760 = __shfl_xor_sync(0xFFFFFFFF, m_34, 1);
        float ip_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_34) : "f"(m_34), "f"(_shfl_xor_sync_760));
        float _shfl_xor_sync_763 = __shfl_xor_sync(0xFFFFFFFF, ip_34, 2);
        float _add_762;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_762) : "f"(ip_34), "f"(_shfl_xor_sync_763));
        float _sub_761;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_761) : "f"(g_33), "f"(_add_762));
        float g_34 = __fdiv_rn(_sub_761, v_34);
        float _dot_764;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_764) : "r"(ring_pk_37[0]), "r"(ring_pk_34[0]), "f"(-0.0f));
        dots_p_6[2] = _dot_764;
        float _dot_765;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_765) : "r"(ring_pk_37[1]), "r"(ring_pk_34[1]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_765;
        float _dot_766;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_766) : "r"(ring_pk_37[2]), "r"(ring_pk_34[2]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_766;
        float _dot_767;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_767) : "r"(ring_pk_37[3]), "r"(ring_pk_34[3]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_767;
        float _dot_768;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_768) : "r"(ring_pk_37[4]), "r"(ring_pk_34[4]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_768;
        float _dot_769;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_769) : "r"(ring_pk_37[5]), "r"(ring_pk_34[5]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_769;
        float _dot_770;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_770) : "r"(ring_pk_37[6]), "r"(ring_pk_34[6]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_770;
        float _dot_771;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_771) : "r"(ring_pk_37[7]), "r"(ring_pk_34[7]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_771;
        float _dot_772;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_772) : "r"(ring_pk_37[8]), "r"(ring_pk_34[8]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_772;
        float _dot_773;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_773) : "r"(ring_pk_37[9]), "r"(ring_pk_34[9]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_773;
        unsigned int ring_pk_38[10];
        unsigned int lane_row_38 = sa__1_1 + 5120 + lane32;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[0])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[(0) + 1]))
            : "r"(lane_row_38));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[2])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[(2) + 1]))
            : "r"(lane_row_38 + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[4])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[(4) + 1]))
            : "r"(lane_row_38 + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[6])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[(6) + 1]))
            : "r"(lane_row_38 + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[8])), "=r"(*reinterpret_cast<uint32_t*>(&ring_pk_38[(8) + 1]))
            : "r"(lane_row_38 + 32));
        float _shfl_xor_sync_774 = __shfl_xor_sync(0xFFFFFFFF, hist[2], 4);
        float s_74;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_74) : "f"(hist[2]), "f"(_shfl_xor_sync_774));
        float _shfl_xor_sync_776 = __shfl_xor_sync(0xFFFFFFFF, s_74, 24);
        float _add_775;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_775) : "f"(s_74), "f"(_shfl_xor_sync_776));
        float v_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_35) : "f"(_add_775), "f"(hc_eps));
        float m_35;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_35) : "f"(g_34), "f"(hist[3]));
        float _shfl_xor_sync_777 = __shfl_xor_sync(0xFFFFFFFF, m_35, 4);
        float ip_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_35) : "f"(m_35), "f"(_shfl_xor_sync_777));
        float _shfl_xor_sync_780 = __shfl_xor_sync(0xFFFFFFFF, ip_35, 24);
        float _add_779;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_779) : "f"(ip_35), "f"(_shfl_xor_sync_780));
        float _sub_778;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_778) : "f"(g_34), "f"(_add_779));
        float g_35 = __fdiv_rn(_sub_778, v_35);
        float _dot_781;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_781) : "r"(ring_pk_38[0]), "r"(ring_pk_34[0]), "f"(-0.0f));
        dots_p_6[3] = _dot_781;
        float _dot_782;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_782) : "r"(ring_pk_38[1]), "r"(ring_pk_34[1]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_782;
        float _dot_783;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_783) : "r"(ring_pk_38[2]), "r"(ring_pk_34[2]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_783;
        float _dot_784;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_784) : "r"(ring_pk_38[3]), "r"(ring_pk_34[3]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_784;
        float _dot_785;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_785) : "r"(ring_pk_38[4]), "r"(ring_pk_34[4]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_785;
        float _dot_786;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_786) : "r"(ring_pk_38[5]), "r"(ring_pk_34[5]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_786;
        float _dot_787;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_787) : "r"(ring_pk_38[6]), "r"(ring_pk_34[6]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_787;
        float _dot_788;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_788) : "r"(ring_pk_38[7]), "r"(ring_pk_34[7]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_788;
        float _dot_789;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_789) : "r"(ring_pk_38[8]), "r"(ring_pk_34[8]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_789;
        float _dot_790;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_790) : "r"(ring_pk_38[9]), "r"(ring_pk_34[9]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_790;
        __syncwarp();
        bool elected_1 = elect_sync();
        bool defer_el_1 = elected_1 && tail_pred;
        if (defer_el_1) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st__1_1) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1), "l"(dU + (th2_9 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 1280), "l"(X + (th2_9 * 4 * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 2560), "l"(X + ((th2_9 * 4 + 1) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 3840), "l"(X + ((th2_9 * 4 + 2) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa__1_1 + 5120), "l"(X + ((th2_9 * 4 + 3) * 5120 + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st__1_1) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_791 = __shfl_xor_sync(0xFFFFFFFF, hist[1], 1);
        float s_75;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_75) : "f"(hist[1]), "f"(_shfl_xor_sync_791));
        float _shfl_xor_sync_793 = __shfl_xor_sync(0xFFFFFFFF, s_75, 2);
        float _add_792;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_792) : "f"(s_75), "f"(_shfl_xor_sync_793));
        float v_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_36) : "f"(_add_792), "f"(hc_eps));
        float m_36;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_36) : "f"(g_35), "f"(hist[2]));
        float _shfl_xor_sync_794 = __shfl_xor_sync(0xFFFFFFFF, m_36, 1);
        float ip_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_36) : "f"(m_36), "f"(_shfl_xor_sync_794));
        float _shfl_xor_sync_797 = __shfl_xor_sync(0xFFFFFFFF, ip_36, 2);
        float _add_796;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_796) : "f"(ip_36), "f"(_shfl_xor_sync_797));
        float _sub_795;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_795) : "f"(g_35), "f"(_add_796));
        float g_36 = __fdiv_rn(_sub_795, v_36);
        int upper_7 = lane_1 & 16;
        float send_8 = ((upper_7 != 0) ? dots_p_6[0] : dots_p_6[2]);
        float keep_8 = ((upper_7 != 0) ? dots_p_6[2] : dots_p_6[0]);
        float _shfl_xor_sync_798 = __shfl_xor_sync(0xFFFFFFFF, send_8, 16);
        float add_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_18) : "f"(keep_8), "f"(_shfl_xor_sync_798));
        float send_1_6 = ((upper_7 != 0) ? dots_p_6[1] : dots_p_6[3]);
        float keep_1_6 = ((upper_7 != 0) ? dots_p_6[3] : dots_p_6[1]);
        float _shfl_xor_sync_799 = __shfl_xor_sync(0xFFFFFFFF, send_1_6, 16);
        float add_1_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_13) : "f"(keep_1_6), "f"(_shfl_xor_sync_799));
        int upper_1_6 = lane_1 & 8;
        float send_2_6 = ((upper_1_6 != 0) ? add_18 : add_1_13);
        float keep_2_6 = ((upper_1_6 != 0) ? add_1_13 : add_18);
        float _shfl_xor_sync_800 = __shfl_xor_sync(0xFFFFFFFF, send_2_6, 8);
        float add_2_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_6) : "f"(keep_2_6), "f"(_shfl_xor_sync_800));
        float _add_195;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(hist[0]), "f"(hc_eps));
        float _shfl_xor_sync_801 = __shfl_xor_sync(0xFFFFFFFF, _add_195, 4);
        float add_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_19) : "f"(_add_195), "f"(_shfl_xor_sync_801));
        float _shfl_xor_sync_803 = __shfl_xor_sync(0xFFFFFFFF, add_19, 24);
        float _add_802;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_802) : "f"(add_19), "f"(_shfl_xor_sync_803));
        float _add_197;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(_add_802), "f"(hc_eps));
        float _mul_39;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(g_36), "f"(hist[1]));
        float _shfl_xor_sync_804 = __shfl_xor_sync(0xFFFFFFFF, _mul_39, 4);
        float add_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_20) : "f"(_mul_39), "f"(_shfl_xor_sync_804));
        float _shfl_xor_sync_807 = __shfl_xor_sync(0xFFFFFFFF, add_20, 24);
        float _add_806;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_806) : "f"(add_20), "f"(_shfl_xor_sync_807));
        float _sub_805;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_805) : "f"(g_36), "f"(_add_806));
        float _fdiv_rn_80 = __fdiv_rn(_sub_805, _add_197);
        float _shfl_xor_sync_808 = __shfl_xor_sync(0xFFFFFFFF, add_2_6, 4);
        float add_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_21) : "f"(add_2_6), "f"(_shfl_xor_sync_808));
        float _shfl_xor_sync_809 = __shfl_xor_sync(0xFFFFFFFF, add_21, 2);
        float add_1_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_14) : "f"(add_21), "f"(_shfl_xor_sync_809));
        float _shfl_xor_sync_811 = __shfl_xor_sync(0xFFFFFFFF, add_1_14, 1);
        float _add_810;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_810) : "f"(add_1_14), "f"(_shfl_xor_sync_811));
        dpre_part[own_base + 224] = _add_810;
        float acc_162 = 0.0f;
        float _shfl_xor_sync_812 = __shfl_xor_sync(0xFFFFFFFF, hist[0], 2);
        float _shfl_xor_sync_814 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_80, 2);
        float _sub_813;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_813) : "f"(_fdiv_rn_80), "f"(_shfl_xor_sync_814));
        float _shfl_xor_sync_816 = __shfl_xor_sync(0xFFFFFFFF, hist[0], 1);
        float _shfl_xor_sync_818 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_80, 1);
        float _sub_817;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_817) : "f"(_fdiv_rn_80), "f"(_shfl_xor_sync_818));
        float _fma_815;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_815) : "f"(_shfl_xor_sync_816), "f"(_sub_817), "f"(acc_162));
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(_shfl_xor_sync_812), "f"(_sub_813), "f"(_fma_815));
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, hist[0], 3);
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_80, 3);
        float _sub_820;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_820) : "f"(_fdiv_rn_80), "f"(_shfl_xor_1));
        float _fma_819;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_819) : "f"(_shfl_xor_0), "f"(_sub_820), "f"(_fma_0));
        float da = ((lane_1 >= 8 && lane_1 < 24) ? mul_f32_rn_noftz(hist[0], _fma_819) : mul_f32_rn_noftz(1.0f, dsig));
        __syncthreads();
        dp[0] = dpre_part[cur_base + lc4];
        float _add_821;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_821) : "f"(dp[0]), "f"(dpre_part[cur_base + 4 + lc4]));
        dp[0] = _add_821;
        float _add_822;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_822) : "f"(dp[0]), "f"(dpre_part[cur_base + 8 + lc4]));
        dp[0] = _add_822;
        float _add_823;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_823) : "f"(dp[0]), "f"(dpre_part[cur_base + 12 + lc4]));
        dp[0] = _add_823;
        float _add_824;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_824) : "f"(dp[0]), "f"(dpre_part[cur_base + 16 + lc4]));
        dp[0] = _add_824;
        float _add_825;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_825) : "f"(dp[0]), "f"(dpre_part[cur_base + 20 + lc4]));
        dp[0] = _add_825;
        float _add_826;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_826) : "f"(dp[0]), "f"(dpre_part[cur_base + 24 + lc4]));
        dp[0] = _add_826;
        float _add_827;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_827) : "f"(dp[0]), "f"(dpre_part[cur_base + 28 + lc4]));
        dp[0] = _add_827;
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
        float c_t = (float)((double)rr * (double)_warp_reduce_0 * 4.8828125e-05);
        float dzr;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dzr) : "f"(dz), "f"(r));
        float cvt_f32 = (float)(__nv_bfloat16)dzr;
        float sub;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub) : "f"(dzr), "f"(cvt_f32));
        float cvt_f32_1 = (float)(__nv_bfloat16)sub;
        float _sub_828;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_828) : "f"(sub), "f"(cvt_f32_1));
        float cvt_f32_2 = (float)(__nv_bfloat16)_sub_828;
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
        float _sub_830;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_830) : "f"(_add_2), "f"(acc[0]));
        float _sub_829;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_829) : "f"(_sub_830), "f"(_sub_0_1));
        comp[0] = ((tok < t1) ? _sub_829 : comp[0]);
        acc[0] = ((tok < t1) ? _add_2 : acc[0]);
        float _sub_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(daz), "f"(comp[1]));
        float _add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(acc[1]), "f"(_sub_1));
        float _sub_832;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_832) : "f"(_add_4), "f"(acc[1]));
        float _sub_831;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_831) : "f"(_sub_832), "f"(_sub_1));
        comp[1] = ((tok < t1) ? _sub_831 : comp[1]);
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
            float add_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(o[0]), "f"(sub_1));
            float _sub_834;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_834) : "f"(add_3), "f"(o[0]));
            float _sub_833;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_833) : "f"(_sub_834), "f"(sub_1));
            oc[0] = _sub_833;
            o[0] = add_3;
            float _sub_835;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_835) : "f"(0.0f), "f"(fold[src + 1]));
            float sub_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_0) : "f"(_sub_835), "f"(oc[0]));
            float add_22;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_22) : "f"(o[0]), "f"(sub_0));
            float _sub_837;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_837) : "f"(add_22), "f"(o[0]));
            float _sub_836;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_836) : "f"(_sub_837), "f"(sub_0));
            oc[0] = _sub_836;
            o[0] = add_22;
            float sub_1_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1_1) : "f"(fold[src + 2]), "f"(oc[1]));
            float add_23;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_23) : "f"(o[1]), "f"(sub_1_1));
            float _sub_839;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_839) : "f"(add_23), "f"(o[1]));
            float _sub_838;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_838) : "f"(_sub_839), "f"(sub_1_1));
            oc[1] = _sub_838;
            o[1] = add_23;
            float _sub_840;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_840) : "f"(0.0f), "f"(fold[src + 3]));
            float sub_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_2) : "f"(_sub_840), "f"(oc[1]));
            float add_24;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_24) : "f"(o[1]), "f"(sub_2));
            float _sub_842;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_842) : "f"(add_24), "f"(o[1]));
            float _sub_841;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_841) : "f"(_sub_842), "f"(sub_2));
            oc[1] = _sub_841;
            o[1] = add_24;
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
                float add_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(ga[0]), "f"(sub_3));
                float _sub_844;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_844) : "f"(add_4), "f"(ga[0]));
                float _sub_843;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_843) : "f"(_sub_844), "f"(sub_3));
                gc[0] = _sub_843;
                ga[0] = add_4;
            }
            float _sub_845;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_845) : "f"(ga[0]), "f"(gc[0]));
            *(reinterpret_cast<float*>(param_partial + (prow + 24 + lane64)) + (0)) = _sub_845;
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
