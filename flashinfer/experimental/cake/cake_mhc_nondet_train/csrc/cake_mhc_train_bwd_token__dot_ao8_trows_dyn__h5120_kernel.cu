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


extern "C" __device__ float __nv_expf(float);




__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_mhc_train_bwd_token__dot_ao8_trows_dyn__h5120(__nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ dU, float* __restrict__ stats, float* __restrict__ gstats, float* __restrict__ scale, float* __restrict__ base, __nv_bfloat16* __restrict__ dzp, float* __restrict__ tokp, float* __restrict__ param_partial, int T, int t_pad, int rows_per_cta, float hc_eps, unsigned int* __restrict__ ticket)
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
    #define tk_full_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* dpre_part = reinterpret_cast<float*>(smem_raw + 1024);
    const int dpre_part_addr = smem + 1024;
    unsigned int* _tkv = reinterpret_cast<unsigned int*>(smem_raw + 3072);
    const int tkv_addr = smem + 3072;
    float* gsumv = reinterpret_cast<float*>(smem_raw + 3584);
    const int gsumv_addr = smem + 3584;
    float* gfold = reinterpret_cast<float*>(smem_raw + 5120);
    const int gfold_addr = smem + 5120;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 11264);
    const int ring_addr = smem + 11264;

    // Mbarrier init (18 barriers at smem_raw[0..144))

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
            // --- pipeline 'tk_pipe' ---
            // tk_full: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
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
    int n = ((lane_1 < 24) ? lane_1 : 23);
    long long n64 = (long long)n;
    long long lane64 = (long long)lane_1;
    float sc0 = scale[0];
    float sc1 = scale[1];
    float sc2 = scale[2];
    float sc = ((lane_1 < 4) ? sc0 : ((lane_1 < 8) ? sc1 : sc2));
    float bs = base[n64];
    int lane_hi = lane_1 >> 3;
    int lc4 = ((lane_1 < 4) ? lane_1 : 3);
    int gcol = ((lane_1 >= 4) ? lane_1 - 4 : 0);
    long long wcol = (long long)warp_0 * 640ll;
    unsigned int lane32 = (unsigned int)(lane_1 * 40);
    int st_a = warp_0 * 2;
    int st_b = warp_0 * 2 + 1;
    unsigned int sa_a = (unsigned int)(ring_addr + (unsigned int)(st_a * 6400));
    unsigned int sa_b = (unsigned int)(ring_addr + (unsigned int)(st_b * 6400));
    int n_groups = T + 7 >> 3;
    int grid = (int)num_bids;
    bool dyn_on = n_groups > grid;
    unsigned int tk_limit = (unsigned int)(n_groups - 1);
    int g_end = n_groups;
    unsigned int tc = 0;
    int lo = bid * 8;
    int hi = n_groups * 8;
    int gf_end = ((lo + 8 < T) ? lo + 8 : T);
    int gf1 = ((lo + 1 < gf_end) ? lo + 1 : gf_end - 1);
    long long tf0 = (long long)lo;
    long long td64 = (long long)gf1;
    if (lo < hi) {
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_a) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a), "l"(dU + (tf0 * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 1280), "l"(X + (tf0 * 4ll * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 2560), "l"(X + ((tf0 * 4ll + 1ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 3840), "l"(X + ((tf0 * 4ll + 2ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 5120), "l"(X + ((tf0 * 4ll + 3ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_b) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b), "l"(dU + (td64 * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 1280), "l"(X + (td64 * 4ll * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 2560), "l"(X + ((td64 * 4ll + 1ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 3840), "l"(X + ((td64 * 4ll + 2ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 5120), "l"(X + ((td64 * 4ll + 3ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    int gi[1];
    gi[0] = 0;
    int pg[1];
    pg[0] = 0;
    int gc[1];
    gc[0] = bid;
    int own_last[1];
    own_last[0] = 0;
    int ring_grp = 0;
    unsigned int _phase_tk_full = 0;
    #pragma unroll 1
    for (int itv = lo; itv < hi; itv += 8) {
        int gcur = gc[0];
        int g0 = gcur * 8;
        own_last[0] = ((gcur == n_groups - 1) ? 1 : own_last[0]);
        int t1 = ((g0 + 8 < T) ? g0 + 8 : T);
        int t_last = t1 - 1;
        int th_1_0 = g0 + 2;
        long long th_t_0 = (long long)((th_1_0 < t1) ? th_1_0 : t_last);
        int th_1_1 = g0 + 3;
        long long th_t_1 = (long long)((th_1_1 < t1) ? th_1_1 : t_last);
        int th_1_2 = g0 + 4;
        long long th_t_2 = (long long)((th_1_2 < t1) ? th_1_2 : t_last);
        int th_1_3 = g0 + 5;
        long long th_t_3 = (long long)((th_1_3 < t1) ? th_1_3 : t_last);
        int th_1_4 = g0 + 6;
        long long th_t_4 = (long long)((th_1_4 < t1) ? th_1_4 : t_last);
        int th_1_5 = g0 + 7;
        long long th_t_5 = (long long)((th_1_5 < t1) ? th_1_5 : t_last);
        int tok = g0 + warp_0;
        int cur_base = (gi[0] & 1) * 256 + warp_0 * 32;
        int own_base = (gi[0] & 1) * 256 + warp_0 * 4 + lane_hi;
        bool first_grp = gi[0] == 0;
        unsigned int drawn = (unsigned int)n_groups;
        if (dyn_on) {
            if (warp_0 == 0) {
                if (lane == 0) {
                    uint32_t _atomic_fetch_inc_0;
                    asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(_atomic_fetch_inc_0) : "l"(&ticket[32]), "r"(static_cast<uint32_t>(tk_limit)) : "memory");
                    drawn = _atomic_fetch_inc_0;
                }
            }
        }
        int gnext_in = g_end;
        unsigned int gvalid = ((dyn_on) ? (unsigned int)(n_groups - grid) : 0);
        unsigned int gpub = ((drawn < gvalid) ? drawn + (unsigned int)grid : (unsigned int)n_groups);
        int tokc = ((tok < t1) ? tok : t_last);
        long long tok64 = (long long)tokc;
        float dp[1];
        long long srow = tok64 * 64ll;
        float r = stats[srow];
        float z_load = stats[srow + 1ll + n64];
        float gval = gstats[tok64 * 32ll + (long long)gcol];
        float z = ((lane_1 < 24) ? z_load : 0.0f);
        float a_map;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a_map) : "f"(z), "f"(sc), "f"(bs));
        float a = ((lane_1 < 24) ? a_map : 0.0f);
        float gtwo;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(gtwo) : "f"(gval), "f"(2.0f));
        float hist[40];
        float _sub_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(0.0f), "f"(fabsf(a)));
        float _exp_0 = __nv_expf(_sub_1);
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
        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
        float _max_0 = fmaxf(logit, _shfl_xor_sync_2);
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, _max_0, 2);
        float _sub_3;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(logit), "f"(fmaxf(_max_0, _shfl_xor_sync_4)));
        float _exp_1 = __nv_expf(_sub_3);
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
        float add;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(_exp_1), "f"(_shfl_xor_sync_5));
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, add, 2);
        float _add_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(add), "f"(_shfl_xor_sync_7));
        float _fdiv_rn_2 = __fdiv_rn(_exp_1, _add_6);
        hist[0] = _fdiv_rn_2;
        float _add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(_fdiv_rn_2), "f"(hc_eps));
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, _add_3, 4);
        float s;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s) : "f"(_add_3), "f"(_shfl_xor_sync_8));
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, s, 24);
        float _add_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(s), "f"(_shfl_xor_sync_11));
        float _add_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(_add_10), "f"(hc_eps));
        float c = __fdiv_rn(_add_3, _add_9);
        hist[1] = c;
        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, c, 1);
        float s_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_0) : "f"(c), "f"(_shfl_xor_sync_12));
        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, s_0, 2);
        float _add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(s_0), "f"(_shfl_xor_sync_15));
        float _add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(_add_14), "f"(hc_eps));
        float c_0 = __fdiv_rn(c, _add_13);
        hist[2] = c_0;
        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, c_0, 4);
        float s_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_1) : "f"(c_0), "f"(_shfl_xor_sync_16));
        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, s_1, 24);
        float _add_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(s_1), "f"(_shfl_xor_sync_19));
        float _add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(_add_18), "f"(hc_eps));
        float c_1 = __fdiv_rn(c_0, _add_17);
        hist[3] = c_1;
        float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, c_1, 1);
        float s_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_2) : "f"(c_1), "f"(_shfl_xor_sync_20));
        float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, s_2, 2);
        float _add_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(s_2), "f"(_shfl_xor_sync_23));
        float _add_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(_add_22), "f"(hc_eps));
        float c_2 = __fdiv_rn(c_1, _add_21);
        hist[4] = c_2;
        float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, c_2, 4);
        float s_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_3) : "f"(c_2), "f"(_shfl_xor_sync_24));
        float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, s_3, 24);
        float _add_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(s_3), "f"(_shfl_xor_sync_27));
        float _add_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(_add_26), "f"(hc_eps));
        float c_3 = __fdiv_rn(c_2, _add_25);
        hist[5] = c_3;
        float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, c_3, 1);
        float s_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_4) : "f"(c_3), "f"(_shfl_xor_sync_28));
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, s_4, 2);
        float _add_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(s_4), "f"(_shfl_xor_sync_31));
        float _add_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(_add_30), "f"(hc_eps));
        float c_4 = __fdiv_rn(c_3, _add_29);
        hist[6] = c_4;
        float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, c_4, 4);
        float s_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_5) : "f"(c_4), "f"(_shfl_xor_sync_32));
        float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, s_5, 24);
        float _add_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(s_5), "f"(_shfl_xor_sync_35));
        float _add_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(_add_34), "f"(hc_eps));
        float c_5 = __fdiv_rn(c_4, _add_33);
        hist[7] = c_5;
        float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, c_5, 1);
        float s_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_6) : "f"(c_5), "f"(_shfl_xor_sync_36));
        float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, s_6, 2);
        float _add_38;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(s_6), "f"(_shfl_xor_sync_39));
        float _add_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(_add_38), "f"(hc_eps));
        float c_6 = __fdiv_rn(c_5, _add_37);
        hist[8] = c_6;
        float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, c_6, 4);
        float s_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_7) : "f"(c_6), "f"(_shfl_xor_sync_40));
        float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, s_7, 24);
        float _add_42;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(s_7), "f"(_shfl_xor_sync_43));
        float _add_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(_add_42), "f"(hc_eps));
        float c_7 = __fdiv_rn(c_6, _add_41);
        hist[9] = c_7;
        mbarrier_wait(ring_bar_addr + (st_a) * 8, 0);
        unsigned int ring_pk[10];
        unsigned int lane_row = sa_a + lane32;
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
        unsigned int lane_row_0 = sa_a + 1280 + lane32;
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
        float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, c_7, 1);
        float s_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_8) : "f"(c_7), "f"(_shfl_xor_sync_44));
        float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, s_8, 2);
        float _add_46;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(s_8), "f"(_shfl_xor_sync_47));
        float _add_45;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(_add_46), "f"(hc_eps));
        float c_8 = __fdiv_rn(c_7, _add_45);
        hist[10] = c_8;
        float _dot_48;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_48) : "r"(ring_pk_0[0]), "r"(ring_pk[0]), "f"(-0.0f));
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
            : "=f"(_dot_49) : "r"(ring_pk_0[1]), "r"(ring_pk[1]), "f"(dots_p[0]));
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
            : "=f"(_dot_50) : "r"(ring_pk_0[2]), "r"(ring_pk[2]), "f"(dots_p[0]));
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
            : "=f"(_dot_51) : "r"(ring_pk_0[3]), "r"(ring_pk[3]), "f"(dots_p[0]));
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
            : "=f"(_dot_52) : "r"(ring_pk_0[4]), "r"(ring_pk[4]), "f"(dots_p[0]));
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
            : "=f"(_dot_53) : "r"(ring_pk_0[5]), "r"(ring_pk[5]), "f"(dots_p[0]));
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
            : "=f"(_dot_54) : "r"(ring_pk_0[6]), "r"(ring_pk[6]), "f"(dots_p[0]));
        dots_p[0] = _dot_54;
        float _dot_55;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_55) : "r"(ring_pk_0[7]), "r"(ring_pk[7]), "f"(dots_p[0]));
        dots_p[0] = _dot_55;
        float _dot_56;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_56) : "r"(ring_pk_0[8]), "r"(ring_pk[8]), "f"(dots_p[0]));
        dots_p[0] = _dot_56;
        float _dot_57;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_57) : "r"(ring_pk_0[9]), "r"(ring_pk[9]), "f"(dots_p[0]));
        dots_p[0] = _dot_57;
        unsigned int ring_pk_1[10];
        unsigned int lane_row_1 = sa_a + 2560 + lane32;
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
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, c_8, 4);
        float s_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_9) : "f"(c_8), "f"(_shfl_xor_sync_58));
        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, s_9, 24);
        float _add_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(s_9), "f"(_shfl_xor_sync_61));
        float _add_59;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(_add_60), "f"(hc_eps));
        float c_9 = __fdiv_rn(c_8, _add_59);
        hist[11] = c_9;
        float _dot_62;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_62) : "r"(ring_pk_1[0]), "r"(ring_pk[0]), "f"(-0.0f));
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
            : "=f"(_dot_63) : "r"(ring_pk_1[1]), "r"(ring_pk[1]), "f"(dots_p[1]));
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
            : "=f"(_dot_64) : "r"(ring_pk_1[2]), "r"(ring_pk[2]), "f"(dots_p[1]));
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
            : "=f"(_dot_65) : "r"(ring_pk_1[3]), "r"(ring_pk[3]), "f"(dots_p[1]));
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
            : "=f"(_dot_66) : "r"(ring_pk_1[4]), "r"(ring_pk[4]), "f"(dots_p[1]));
        dots_p[1] = _dot_66;
        float _dot_67;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_67) : "r"(ring_pk_1[5]), "r"(ring_pk[5]), "f"(dots_p[1]));
        dots_p[1] = _dot_67;
        float _dot_68;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_68) : "r"(ring_pk_1[6]), "r"(ring_pk[6]), "f"(dots_p[1]));
        dots_p[1] = _dot_68;
        float _dot_69;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_69) : "r"(ring_pk_1[7]), "r"(ring_pk[7]), "f"(dots_p[1]));
        dots_p[1] = _dot_69;
        float _dot_70;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_70) : "r"(ring_pk_1[8]), "r"(ring_pk[8]), "f"(dots_p[1]));
        dots_p[1] = _dot_70;
        float _dot_71;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_71) : "r"(ring_pk_1[9]), "r"(ring_pk[9]), "f"(dots_p[1]));
        dots_p[1] = _dot_71;
        unsigned int ring_pk_2[10];
        unsigned int lane_row_2 = sa_a + 3840 + lane32;
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
        float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, c_9, 1);
        float s_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_10) : "f"(c_9), "f"(_shfl_xor_sync_72));
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, s_10, 2);
        float _add_74;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(s_10), "f"(_shfl_xor_sync_75));
        float _add_73;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(_add_74), "f"(hc_eps));
        float c_10 = __fdiv_rn(c_9, _add_73);
        hist[12] = c_10;
        float _dot_76;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_76) : "r"(ring_pk_2[0]), "r"(ring_pk[0]), "f"(-0.0f));
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
            : "=f"(_dot_77) : "r"(ring_pk_2[1]), "r"(ring_pk[1]), "f"(dots_p[2]));
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
            : "=f"(_dot_78) : "r"(ring_pk_2[2]), "r"(ring_pk[2]), "f"(dots_p[2]));
        dots_p[2] = _dot_78;
        float _dot_79;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_79) : "r"(ring_pk_2[3]), "r"(ring_pk[3]), "f"(dots_p[2]));
        dots_p[2] = _dot_79;
        float _dot_80;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_80) : "r"(ring_pk_2[4]), "r"(ring_pk[4]), "f"(dots_p[2]));
        dots_p[2] = _dot_80;
        float _dot_81;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_81) : "r"(ring_pk_2[5]), "r"(ring_pk[5]), "f"(dots_p[2]));
        dots_p[2] = _dot_81;
        float _dot_82;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_82) : "r"(ring_pk_2[6]), "r"(ring_pk[6]), "f"(dots_p[2]));
        dots_p[2] = _dot_82;
        float _dot_83;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_83) : "r"(ring_pk_2[7]), "r"(ring_pk[7]), "f"(dots_p[2]));
        dots_p[2] = _dot_83;
        float _dot_84;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_84) : "r"(ring_pk_2[8]), "r"(ring_pk[8]), "f"(dots_p[2]));
        dots_p[2] = _dot_84;
        float _dot_85;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_85) : "r"(ring_pk_2[9]), "r"(ring_pk[9]), "f"(dots_p[2]));
        dots_p[2] = _dot_85;
        unsigned int ring_pk_3[10];
        unsigned int lane_row_3 = sa_a + 5120 + lane32;
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
        float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, c_10, 4);
        float s_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_11) : "f"(c_10), "f"(_shfl_xor_sync_86));
        float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, s_11, 24);
        float _add_88;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(s_11), "f"(_shfl_xor_sync_89));
        float _add_87;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(_add_88), "f"(hc_eps));
        float c_11 = __fdiv_rn(c_10, _add_87);
        hist[13] = c_11;
        float _dot_90;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_90) : "r"(ring_pk_3[0]), "r"(ring_pk[0]), "f"(-0.0f));
        dots_p[3] = _dot_90;
        float _dot_91;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_91) : "r"(ring_pk_3[1]), "r"(ring_pk[1]), "f"(dots_p[3]));
        dots_p[3] = _dot_91;
        float _dot_92;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_92) : "r"(ring_pk_3[2]), "r"(ring_pk[2]), "f"(dots_p[3]));
        dots_p[3] = _dot_92;
        float _dot_93;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_93) : "r"(ring_pk_3[3]), "r"(ring_pk[3]), "f"(dots_p[3]));
        dots_p[3] = _dot_93;
        float _dot_94;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_94) : "r"(ring_pk_3[4]), "r"(ring_pk[4]), "f"(dots_p[3]));
        dots_p[3] = _dot_94;
        float _dot_95;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_95) : "r"(ring_pk_3[5]), "r"(ring_pk[5]), "f"(dots_p[3]));
        dots_p[3] = _dot_95;
        float _dot_96;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_96) : "r"(ring_pk_3[6]), "r"(ring_pk[6]), "f"(dots_p[3]));
        dots_p[3] = _dot_96;
        float _dot_97;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_97) : "r"(ring_pk_3[7]), "r"(ring_pk[7]), "f"(dots_p[3]));
        dots_p[3] = _dot_97;
        float _dot_98;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_98) : "r"(ring_pk_3[8]), "r"(ring_pk[8]), "f"(dots_p[3]));
        dots_p[3] = _dot_98;
        float _dot_99;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_99) : "r"(ring_pk_3[9]), "r"(ring_pk[9]), "f"(dots_p[3]));
        dots_p[3] = _dot_99;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_a) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a), "l"(dU + (th_t_0 * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 1280), "l"(X + (th_t_0 * 4ll * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 2560), "l"(X + ((th_t_0 * 4ll + 1ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 3840), "l"(X + ((th_t_0 * 4ll + 2ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 5120), "l"(X + ((th_t_0 * 4ll + 3ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, c_11, 1);
        float s_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_12) : "f"(c_11), "f"(_shfl_xor_sync_100));
        float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, s_12, 2);
        float _add_102;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(s_12), "f"(_shfl_xor_sync_103));
        float _add_101;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(_add_102), "f"(hc_eps));
        float c_12 = __fdiv_rn(c_11, _add_101);
        hist[14] = c_12;
        int upper = lane_1 & 16;
        float send = ((upper != 0) ? dots_p[0] : dots_p[2]);
        float keep = ((upper != 0) ? dots_p[2] : dots_p[0]);
        float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
        float add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_0) : "f"(keep), "f"(_shfl_xor_sync_104));
        float send_1 = ((upper != 0) ? dots_p[1] : dots_p[3]);
        float keep_1 = ((upper != 0) ? dots_p[3] : dots_p[1]);
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, send_1, 16);
        float add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(keep_1), "f"(_shfl_xor_sync_105));
        int upper_1 = lane_1 & 8;
        float send_2 = ((upper_1 != 0) ? add_0 : add_1);
        float keep_2 = ((upper_1 != 0) ? add_1 : add_0);
        float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, send_2, 8);
        float add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(keep_2), "f"(_shfl_xor_sync_106));
        float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, c_12, 4);
        float s_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_13) : "f"(c_12), "f"(_shfl_xor_sync_107));
        float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, s_13, 24);
        float _add_109;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(s_13), "f"(_shfl_xor_sync_110));
        float _add_108;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(_add_109), "f"(hc_eps));
        float c_13 = __fdiv_rn(c_12, _add_108);
        hist[15] = c_13;
        float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, add_2, 4);
        float add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(add_2), "f"(_shfl_xor_sync_111));
        float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, add_3, 2);
        float add_1_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_0) : "f"(add_3), "f"(_shfl_xor_sync_112));
        float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, add_1_0, 1);
        float _add_113;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(add_1_0), "f"(_shfl_xor_sync_114));
        dpre_part[own_base] = _add_113;
        float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, c_13, 1);
        float s_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_14) : "f"(c_13), "f"(_shfl_xor_sync_115));
        float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, s_14, 2);
        float _add_117;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(s_14), "f"(_shfl_xor_sync_118));
        float _add_116;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(_add_117), "f"(hc_eps));
        float c_14 = __fdiv_rn(c_13, _add_116);
        hist[16] = c_14;
        float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, c_14, 4);
        float s_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_15) : "f"(c_14), "f"(_shfl_xor_sync_119));
        float _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, s_15, 24);
        float _add_121;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(s_15), "f"(_shfl_xor_sync_122));
        float _add_120;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(_add_121), "f"(hc_eps));
        float c_15 = __fdiv_rn(c_14, _add_120);
        hist[17] = c_15;
        float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, c_15, 1);
        float s_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_16) : "f"(c_15), "f"(_shfl_xor_sync_123));
        float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, s_16, 2);
        float _add_125;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(s_16), "f"(_shfl_xor_sync_126));
        float _add_124;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(_add_125), "f"(hc_eps));
        float c_16 = __fdiv_rn(c_15, _add_124);
        hist[18] = c_16;
        mbarrier_wait(ring_bar_addr + (st_b) * 8, 0);
        unsigned int ring_pk_4[10];
        unsigned int lane_row_4 = sa_b + lane32;
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
        unsigned int lane_row_5 = sa_b + 1280 + lane32;
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
        float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, c_16, 4);
        float s_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_17) : "f"(c_16), "f"(_shfl_xor_sync_127));
        float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, s_17, 24);
        float _add_129;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(s_17), "f"(_shfl_xor_sync_130));
        float _add_128;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(_add_129), "f"(hc_eps));
        float c_17 = __fdiv_rn(c_16, _add_128);
        hist[19] = c_17;
        float _dot_131;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_131) : "r"(ring_pk_5[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[0] = _dot_131;
        float _dot_132;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_132) : "r"(ring_pk_5[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_132;
        float _dot_133;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_133) : "r"(ring_pk_5[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_133;
        float _dot_134;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_134) : "r"(ring_pk_5[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_134;
        float _dot_135;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_135) : "r"(ring_pk_5[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_135;
        float _dot_136;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_136) : "r"(ring_pk_5[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_136;
        float _dot_137;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_137) : "r"(ring_pk_5[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_137;
        float _dot_138;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_138) : "r"(ring_pk_5[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_138;
        float _dot_139;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_139) : "r"(ring_pk_5[8]), "r"(ring_pk_4[8]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_139;
        float _dot_140;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_140) : "r"(ring_pk_5[9]), "r"(ring_pk_4[9]), "f"(dots_p_0[0]));
        dots_p_0[0] = _dot_140;
        unsigned int ring_pk_6[10];
        unsigned int lane_row_6 = sa_b + 2560 + lane32;
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
        float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, c_17, 1);
        float s_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_18) : "f"(c_17), "f"(_shfl_xor_sync_141));
        float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, s_18, 2);
        float _add_143;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(s_18), "f"(_shfl_xor_sync_144));
        float _add_142;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(_add_143), "f"(hc_eps));
        float c_18 = __fdiv_rn(c_17, _add_142);
        hist[20] = c_18;
        float _dot_145;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_145) : "r"(ring_pk_6[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[1] = _dot_145;
        float _dot_146;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_146) : "r"(ring_pk_6[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_146;
        float _dot_147;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_147) : "r"(ring_pk_6[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_147;
        float _dot_148;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_148) : "r"(ring_pk_6[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_148;
        float _dot_149;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_149) : "r"(ring_pk_6[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_149;
        float _dot_150;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_150) : "r"(ring_pk_6[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_150;
        float _dot_151;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_151) : "r"(ring_pk_6[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_151;
        float _dot_152;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_152) : "r"(ring_pk_6[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_152;
        float _dot_153;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_153) : "r"(ring_pk_6[8]), "r"(ring_pk_4[8]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_153;
        float _dot_154;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_154) : "r"(ring_pk_6[9]), "r"(ring_pk_4[9]), "f"(dots_p_0[1]));
        dots_p_0[1] = _dot_154;
        unsigned int ring_pk_7[10];
        unsigned int lane_row_7 = sa_b + 3840 + lane32;
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
        float _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, c_18, 4);
        float s_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_19) : "f"(c_18), "f"(_shfl_xor_sync_155));
        float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, s_19, 24);
        float _add_157;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(s_19), "f"(_shfl_xor_sync_158));
        float _add_156;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(_add_157), "f"(hc_eps));
        float c_19 = __fdiv_rn(c_18, _add_156);
        hist[21] = c_19;
        float _dot_159;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_159) : "r"(ring_pk_7[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[2] = _dot_159;
        float _dot_160;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_160) : "r"(ring_pk_7[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_160;
        float _dot_161;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_161) : "r"(ring_pk_7[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_161;
        float _dot_162;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_162) : "r"(ring_pk_7[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_162;
        float _dot_163;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_163) : "r"(ring_pk_7[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_163;
        float _dot_164;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_164) : "r"(ring_pk_7[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_164;
        float _dot_165;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_165) : "r"(ring_pk_7[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_165;
        float _dot_166;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_166) : "r"(ring_pk_7[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_166;
        float _dot_167;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_167) : "r"(ring_pk_7[8]), "r"(ring_pk_4[8]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_167;
        float _dot_168;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_168) : "r"(ring_pk_7[9]), "r"(ring_pk_4[9]), "f"(dots_p_0[2]));
        dots_p_0[2] = _dot_168;
        unsigned int ring_pk_8[10];
        unsigned int lane_row_8 = sa_b + 5120 + lane32;
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
        float _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, c_19, 1);
        float s_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_20) : "f"(c_19), "f"(_shfl_xor_sync_169));
        float _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, s_20, 2);
        float _add_171;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171) : "f"(s_20), "f"(_shfl_xor_sync_172));
        float _add_170;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_170) : "f"(_add_171), "f"(hc_eps));
        float c_20 = __fdiv_rn(c_19, _add_170);
        hist[22] = c_20;
        float _dot_173;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_173) : "r"(ring_pk_8[0]), "r"(ring_pk_4[0]), "f"(-0.0f));
        dots_p_0[3] = _dot_173;
        float _dot_174;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_174) : "r"(ring_pk_8[1]), "r"(ring_pk_4[1]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_174;
        float _dot_175;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_175) : "r"(ring_pk_8[2]), "r"(ring_pk_4[2]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_175;
        float _dot_176;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_176) : "r"(ring_pk_8[3]), "r"(ring_pk_4[3]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_176;
        float _dot_177;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_177) : "r"(ring_pk_8[4]), "r"(ring_pk_4[4]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_177;
        float _dot_178;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_178) : "r"(ring_pk_8[5]), "r"(ring_pk_4[5]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_178;
        float _dot_179;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_179) : "r"(ring_pk_8[6]), "r"(ring_pk_4[6]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_179;
        float _dot_180;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_180) : "r"(ring_pk_8[7]), "r"(ring_pk_4[7]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_180;
        float _dot_181;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_181) : "r"(ring_pk_8[8]), "r"(ring_pk_4[8]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_181;
        float _dot_182;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_182) : "r"(ring_pk_8[9]), "r"(ring_pk_4[9]), "f"(dots_p_0[3]));
        dots_p_0[3] = _dot_182;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_b) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b), "l"(dU + (th_t_1 * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 1280), "l"(X + (th_t_1 * 4ll * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 2560), "l"(X + ((th_t_1 * 4ll + 1ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 3840), "l"(X + ((th_t_1 * 4ll + 2ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 5120), "l"(X + ((th_t_1 * 4ll + 3ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_183 = __shfl_xor_sync(0xFFFFFFFF, c_20, 4);
        float s_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_21) : "f"(c_20), "f"(_shfl_xor_sync_183));
        float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, s_21, 24);
        float _add_185;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_185) : "f"(s_21), "f"(_shfl_xor_sync_186));
        float _add_184;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_184) : "f"(_add_185), "f"(hc_eps));
        float c_21 = __fdiv_rn(c_20, _add_184);
        hist[23] = c_21;
        int upper_0 = lane_1 & 16;
        float send_0 = ((upper_0 != 0) ? dots_p_0[0] : dots_p_0[2]);
        float keep_0 = ((upper_0 != 0) ? dots_p_0[2] : dots_p_0[0]);
        float _shfl_xor_sync_187 = __shfl_xor_sync(0xFFFFFFFF, send_0, 16);
        float add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(keep_0), "f"(_shfl_xor_sync_187));
        float send_1_0 = ((upper_0 != 0) ? dots_p_0[1] : dots_p_0[3]);
        float keep_1_0 = ((upper_0 != 0) ? dots_p_0[3] : dots_p_0[1]);
        float _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, send_1_0, 16);
        float add_1_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_1) : "f"(keep_1_0), "f"(_shfl_xor_sync_188));
        int upper_1_0 = lane_1 & 8;
        float send_2_0 = ((upper_1_0 != 0) ? add_4 : add_1_1);
        float keep_2_0 = ((upper_1_0 != 0) ? add_1_1 : add_4);
        float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, send_2_0, 8);
        float add_2_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_0) : "f"(keep_2_0), "f"(_shfl_xor_sync_189));
        float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, c_21, 1);
        float s_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_22) : "f"(c_21), "f"(_shfl_xor_sync_190));
        float _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, s_22, 2);
        float _add_192;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_192) : "f"(s_22), "f"(_shfl_xor_sync_193));
        float _add_191;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191) : "f"(_add_192), "f"(hc_eps));
        float c_22 = __fdiv_rn(c_21, _add_191);
        hist[24] = c_22;
        float _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, add_2_0, 4);
        float add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(add_2_0), "f"(_shfl_xor_sync_194));
        float _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, add_5, 2);
        float add_1_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_2) : "f"(add_5), "f"(_shfl_xor_sync_195));
        float _shfl_xor_sync_197 = __shfl_xor_sync(0xFFFFFFFF, add_1_2, 1);
        float _add_196;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_196) : "f"(add_1_2), "f"(_shfl_xor_sync_197));
        dpre_part[own_base + 32] = _add_196;
        float _shfl_xor_sync_198 = __shfl_xor_sync(0xFFFFFFFF, c_22, 4);
        float s_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_23) : "f"(c_22), "f"(_shfl_xor_sync_198));
        float _shfl_xor_sync_201 = __shfl_xor_sync(0xFFFFFFFF, s_23, 24);
        float _add_200;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_200) : "f"(s_23), "f"(_shfl_xor_sync_201));
        float _add_199;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_199) : "f"(_add_200), "f"(hc_eps));
        float c_23 = __fdiv_rn(c_22, _add_199);
        hist[25] = c_23;
        float _shfl_xor_sync_202 = __shfl_xor_sync(0xFFFFFFFF, c_23, 1);
        float s_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_24) : "f"(c_23), "f"(_shfl_xor_sync_202));
        float _shfl_xor_sync_205 = __shfl_xor_sync(0xFFFFFFFF, s_24, 2);
        float _add_204;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_204) : "f"(s_24), "f"(_shfl_xor_sync_205));
        float _add_203;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_203) : "f"(_add_204), "f"(hc_eps));
        float c_24 = __fdiv_rn(c_23, _add_203);
        hist[26] = c_24;
        float _shfl_xor_sync_206 = __shfl_xor_sync(0xFFFFFFFF, c_24, 4);
        float s_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_25) : "f"(c_24), "f"(_shfl_xor_sync_206));
        float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, s_25, 24);
        float _add_208;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(s_25), "f"(_shfl_xor_sync_209));
        float _add_207;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_207) : "f"(_add_208), "f"(hc_eps));
        float c_25 = __fdiv_rn(c_24, _add_207);
        hist[27] = c_25;
        mbarrier_wait(ring_bar_addr + (st_a) * 8, 1);
        unsigned int ring_pk_9[10];
        unsigned int lane_row_9 = sa_a + lane32;
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
        unsigned int lane_row_10 = sa_a + 1280 + lane32;
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
        float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, c_25, 1);
        float s_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_26) : "f"(c_25), "f"(_shfl_xor_sync_210));
        float _shfl_xor_sync_213 = __shfl_xor_sync(0xFFFFFFFF, s_26, 2);
        float _add_212;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_212) : "f"(s_26), "f"(_shfl_xor_sync_213));
        float _add_211;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_211) : "f"(_add_212), "f"(hc_eps));
        float c_26 = __fdiv_rn(c_25, _add_211);
        hist[28] = c_26;
        float _dot_214;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_214) : "r"(ring_pk_10[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[0] = _dot_214;
        float _dot_215;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_215) : "r"(ring_pk_10[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_215;
        float _dot_216;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_216) : "r"(ring_pk_10[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_216;
        float _dot_217;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_217) : "r"(ring_pk_10[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_217;
        float _dot_218;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_218) : "r"(ring_pk_10[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_218;
        float _dot_219;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_219) : "r"(ring_pk_10[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_219;
        float _dot_220;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_220) : "r"(ring_pk_10[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_220;
        float _dot_221;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_221) : "r"(ring_pk_10[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_221;
        float _dot_222;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_222) : "r"(ring_pk_10[8]), "r"(ring_pk_9[8]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_222;
        float _dot_223;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_223) : "r"(ring_pk_10[9]), "r"(ring_pk_9[9]), "f"(dots_p_1[0]));
        dots_p_1[0] = _dot_223;
        unsigned int ring_pk_11[10];
        unsigned int lane_row_11 = sa_a + 2560 + lane32;
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
        float _shfl_xor_sync_224 = __shfl_xor_sync(0xFFFFFFFF, c_26, 4);
        float s_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_27) : "f"(c_26), "f"(_shfl_xor_sync_224));
        float _shfl_xor_sync_227 = __shfl_xor_sync(0xFFFFFFFF, s_27, 24);
        float _add_226;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_226) : "f"(s_27), "f"(_shfl_xor_sync_227));
        float _add_225;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_225) : "f"(_add_226), "f"(hc_eps));
        float c_27 = __fdiv_rn(c_26, _add_225);
        hist[29] = c_27;
        float _dot_228;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_228) : "r"(ring_pk_11[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[1] = _dot_228;
        float _dot_229;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_229) : "r"(ring_pk_11[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_229;
        float _dot_230;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_230) : "r"(ring_pk_11[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_230;
        float _dot_231;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_231) : "r"(ring_pk_11[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_231;
        float _dot_232;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_232) : "r"(ring_pk_11[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_232;
        float _dot_233;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_233) : "r"(ring_pk_11[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_233;
        float _dot_234;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_234) : "r"(ring_pk_11[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_234;
        float _dot_235;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_235) : "r"(ring_pk_11[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_235;
        float _dot_236;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_236) : "r"(ring_pk_11[8]), "r"(ring_pk_9[8]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_236;
        float _dot_237;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_237) : "r"(ring_pk_11[9]), "r"(ring_pk_9[9]), "f"(dots_p_1[1]));
        dots_p_1[1] = _dot_237;
        unsigned int ring_pk_12[10];
        unsigned int lane_row_12 = sa_a + 3840 + lane32;
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
        float _shfl_xor_sync_238 = __shfl_xor_sync(0xFFFFFFFF, c_27, 1);
        float s_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_28) : "f"(c_27), "f"(_shfl_xor_sync_238));
        float _shfl_xor_sync_241 = __shfl_xor_sync(0xFFFFFFFF, s_28, 2);
        float _add_240;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_240) : "f"(s_28), "f"(_shfl_xor_sync_241));
        float _add_239;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_239) : "f"(_add_240), "f"(hc_eps));
        float c_28 = __fdiv_rn(c_27, _add_239);
        hist[30] = c_28;
        float _dot_242;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_242) : "r"(ring_pk_12[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[2] = _dot_242;
        float _dot_243;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_243) : "r"(ring_pk_12[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_243;
        float _dot_244;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_244) : "r"(ring_pk_12[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_244;
        float _dot_245;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_245) : "r"(ring_pk_12[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_245;
        float _dot_246;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_246) : "r"(ring_pk_12[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_246;
        float _dot_247;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_247) : "r"(ring_pk_12[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_247;
        float _dot_248;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_248) : "r"(ring_pk_12[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_248;
        float _dot_249;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_249) : "r"(ring_pk_12[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_249;
        float _dot_250;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_250) : "r"(ring_pk_12[8]), "r"(ring_pk_9[8]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_250;
        float _dot_251;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_251) : "r"(ring_pk_12[9]), "r"(ring_pk_9[9]), "f"(dots_p_1[2]));
        dots_p_1[2] = _dot_251;
        unsigned int ring_pk_13[10];
        unsigned int lane_row_13 = sa_a + 5120 + lane32;
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
        float _shfl_xor_sync_252 = __shfl_xor_sync(0xFFFFFFFF, c_28, 4);
        float s_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_29) : "f"(c_28), "f"(_shfl_xor_sync_252));
        float _shfl_xor_sync_255 = __shfl_xor_sync(0xFFFFFFFF, s_29, 24);
        float _add_254;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_254) : "f"(s_29), "f"(_shfl_xor_sync_255));
        float _add_253;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_253) : "f"(_add_254), "f"(hc_eps));
        float c_29 = __fdiv_rn(c_28, _add_253);
        hist[31] = c_29;
        float _dot_256;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_256) : "r"(ring_pk_13[0]), "r"(ring_pk_9[0]), "f"(-0.0f));
        dots_p_1[3] = _dot_256;
        float _dot_257;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_257) : "r"(ring_pk_13[1]), "r"(ring_pk_9[1]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_257;
        float _dot_258;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_258) : "r"(ring_pk_13[2]), "r"(ring_pk_9[2]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_258;
        float _dot_259;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_259) : "r"(ring_pk_13[3]), "r"(ring_pk_9[3]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_259;
        float _dot_260;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_260) : "r"(ring_pk_13[4]), "r"(ring_pk_9[4]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_260;
        float _dot_261;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_261) : "r"(ring_pk_13[5]), "r"(ring_pk_9[5]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_261;
        float _dot_262;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_262) : "r"(ring_pk_13[6]), "r"(ring_pk_9[6]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_262;
        float _dot_263;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_263) : "r"(ring_pk_13[7]), "r"(ring_pk_9[7]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_263;
        float _dot_264;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_264) : "r"(ring_pk_13[8]), "r"(ring_pk_9[8]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_264;
        float _dot_265;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_265) : "r"(ring_pk_13[9]), "r"(ring_pk_9[9]), "f"(dots_p_1[3]));
        dots_p_1[3] = _dot_265;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_a) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a), "l"(dU + (th_t_2 * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 1280), "l"(X + (th_t_2 * 4ll * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 2560), "l"(X + ((th_t_2 * 4ll + 1ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 3840), "l"(X + ((th_t_2 * 4ll + 2ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 5120), "l"(X + ((th_t_2 * 4ll + 3ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_266 = __shfl_xor_sync(0xFFFFFFFF, c_29, 1);
        float s_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_30) : "f"(c_29), "f"(_shfl_xor_sync_266));
        float _shfl_xor_sync_269 = __shfl_xor_sync(0xFFFFFFFF, s_30, 2);
        float _add_268;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_268) : "f"(s_30), "f"(_shfl_xor_sync_269));
        float _add_267;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_267) : "f"(_add_268), "f"(hc_eps));
        float c_30 = __fdiv_rn(c_29, _add_267);
        hist[32] = c_30;
        int upper_2 = lane_1 & 16;
        float send_3 = ((upper_2 != 0) ? dots_p_1[0] : dots_p_1[2]);
        float keep_3 = ((upper_2 != 0) ? dots_p_1[2] : dots_p_1[0]);
        float _shfl_xor_sync_270 = __shfl_xor_sync(0xFFFFFFFF, send_3, 16);
        float add_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_6) : "f"(keep_3), "f"(_shfl_xor_sync_270));
        float send_1_1 = ((upper_2 != 0) ? dots_p_1[1] : dots_p_1[3]);
        float keep_1_1 = ((upper_2 != 0) ? dots_p_1[3] : dots_p_1[1]);
        float _shfl_xor_sync_271 = __shfl_xor_sync(0xFFFFFFFF, send_1_1, 16);
        float add_1_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_3) : "f"(keep_1_1), "f"(_shfl_xor_sync_271));
        int upper_1_1 = lane_1 & 8;
        float send_2_1 = ((upper_1_1 != 0) ? add_6 : add_1_3);
        float keep_2_1 = ((upper_1_1 != 0) ? add_1_3 : add_6);
        float _shfl_xor_sync_272 = __shfl_xor_sync(0xFFFFFFFF, send_2_1, 8);
        float add_2_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_1) : "f"(keep_2_1), "f"(_shfl_xor_sync_272));
        float _shfl_xor_sync_273 = __shfl_xor_sync(0xFFFFFFFF, c_30, 4);
        float s_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_31) : "f"(c_30), "f"(_shfl_xor_sync_273));
        float _shfl_xor_sync_276 = __shfl_xor_sync(0xFFFFFFFF, s_31, 24);
        float _add_275;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_275) : "f"(s_31), "f"(_shfl_xor_sync_276));
        float _add_274;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_274) : "f"(_add_275), "f"(hc_eps));
        float c_31 = __fdiv_rn(c_30, _add_274);
        hist[33] = c_31;
        float _shfl_xor_sync_277 = __shfl_xor_sync(0xFFFFFFFF, add_2_1, 4);
        float add_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7) : "f"(add_2_1), "f"(_shfl_xor_sync_277));
        float _shfl_xor_sync_278 = __shfl_xor_sync(0xFFFFFFFF, add_7, 2);
        float add_1_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_4) : "f"(add_7), "f"(_shfl_xor_sync_278));
        float _shfl_xor_sync_280 = __shfl_xor_sync(0xFFFFFFFF, add_1_4, 1);
        float _add_279;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_279) : "f"(add_1_4), "f"(_shfl_xor_sync_280));
        dpre_part[own_base + 64] = _add_279;
        float _shfl_xor_sync_281 = __shfl_xor_sync(0xFFFFFFFF, c_31, 1);
        float s_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_32) : "f"(c_31), "f"(_shfl_xor_sync_281));
        float _shfl_xor_sync_284 = __shfl_xor_sync(0xFFFFFFFF, s_32, 2);
        float _add_283;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_283) : "f"(s_32), "f"(_shfl_xor_sync_284));
        float _add_282;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_282) : "f"(_add_283), "f"(hc_eps));
        float c_32 = __fdiv_rn(c_31, _add_282);
        hist[34] = c_32;
        float _shfl_xor_sync_285 = __shfl_xor_sync(0xFFFFFFFF, c_32, 4);
        float s_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_33) : "f"(c_32), "f"(_shfl_xor_sync_285));
        float _shfl_xor_sync_288 = __shfl_xor_sync(0xFFFFFFFF, s_33, 24);
        float _add_287;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_287) : "f"(s_33), "f"(_shfl_xor_sync_288));
        float _add_286;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_286) : "f"(_add_287), "f"(hc_eps));
        float c_33 = __fdiv_rn(c_32, _add_286);
        hist[35] = c_33;
        float _shfl_xor_sync_289 = __shfl_xor_sync(0xFFFFFFFF, c_33, 1);
        float s_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_34) : "f"(c_33), "f"(_shfl_xor_sync_289));
        float _shfl_xor_sync_292 = __shfl_xor_sync(0xFFFFFFFF, s_34, 2);
        float _add_291;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_291) : "f"(s_34), "f"(_shfl_xor_sync_292));
        float _add_290;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_290) : "f"(_add_291), "f"(hc_eps));
        float c_34 = __fdiv_rn(c_33, _add_290);
        hist[36] = c_34;
        mbarrier_wait(ring_bar_addr + (st_b) * 8, 1);
        unsigned int ring_pk_14[10];
        unsigned int lane_row_14 = sa_b + lane32;
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
        unsigned int lane_row_15 = sa_b + 1280 + lane32;
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
        float _shfl_xor_sync_293 = __shfl_xor_sync(0xFFFFFFFF, c_34, 4);
        float s_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_35) : "f"(c_34), "f"(_shfl_xor_sync_293));
        float _shfl_xor_sync_296 = __shfl_xor_sync(0xFFFFFFFF, s_35, 24);
        float _add_295;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_295) : "f"(s_35), "f"(_shfl_xor_sync_296));
        float _add_294;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_294) : "f"(_add_295), "f"(hc_eps));
        float c_35 = __fdiv_rn(c_34, _add_294);
        hist[37] = c_35;
        float _dot_297;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_297) : "r"(ring_pk_15[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[0] = _dot_297;
        float _dot_298;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_298) : "r"(ring_pk_15[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_298;
        float _dot_299;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_299) : "r"(ring_pk_15[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_299;
        float _dot_300;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_300) : "r"(ring_pk_15[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_300;
        float _dot_301;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_301) : "r"(ring_pk_15[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_301;
        float _dot_302;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_302) : "r"(ring_pk_15[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_302;
        float _dot_303;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_303) : "r"(ring_pk_15[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_303;
        float _dot_304;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_304) : "r"(ring_pk_15[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_304;
        float _dot_305;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_305) : "r"(ring_pk_15[8]), "r"(ring_pk_14[8]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_305;
        float _dot_306;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_306) : "r"(ring_pk_15[9]), "r"(ring_pk_14[9]), "f"(dots_p_2[0]));
        dots_p_2[0] = _dot_306;
        unsigned int ring_pk_16[10];
        unsigned int lane_row_16 = sa_b + 2560 + lane32;
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
        float _shfl_xor_sync_307 = __shfl_xor_sync(0xFFFFFFFF, c_35, 1);
        float s_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_36) : "f"(c_35), "f"(_shfl_xor_sync_307));
        float _shfl_xor_sync_310 = __shfl_xor_sync(0xFFFFFFFF, s_36, 2);
        float _add_309;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_309) : "f"(s_36), "f"(_shfl_xor_sync_310));
        float _add_308;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_308) : "f"(_add_309), "f"(hc_eps));
        float c_36 = __fdiv_rn(c_35, _add_308);
        hist[38] = c_36;
        float _dot_311;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_311) : "r"(ring_pk_16[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[1] = _dot_311;
        float _dot_312;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_312) : "r"(ring_pk_16[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_312;
        float _dot_313;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_313) : "r"(ring_pk_16[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_313;
        float _dot_314;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_314) : "r"(ring_pk_16[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_314;
        float _dot_315;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_315) : "r"(ring_pk_16[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_315;
        float _dot_316;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_316) : "r"(ring_pk_16[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_316;
        float _dot_317;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_317) : "r"(ring_pk_16[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_317;
        float _dot_318;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_318) : "r"(ring_pk_16[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_318;
        float _dot_319;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_319) : "r"(ring_pk_16[8]), "r"(ring_pk_14[8]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_319;
        float _dot_320;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_320) : "r"(ring_pk_16[9]), "r"(ring_pk_14[9]), "f"(dots_p_2[1]));
        dots_p_2[1] = _dot_320;
        unsigned int ring_pk_17[10];
        unsigned int lane_row_17 = sa_b + 3840 + lane32;
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
        float _shfl_xor_sync_321 = __shfl_xor_sync(0xFFFFFFFF, c_36, 4);
        float s_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_37) : "f"(c_36), "f"(_shfl_xor_sync_321));
        float _shfl_xor_sync_324 = __shfl_xor_sync(0xFFFFFFFF, s_37, 24);
        float _add_323;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_323) : "f"(s_37), "f"(_shfl_xor_sync_324));
        float _add_322;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_322) : "f"(_add_323), "f"(hc_eps));
        float c_37 = __fdiv_rn(c_36, _add_322);
        hist[39] = c_37;
        float _dot_325;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_325) : "r"(ring_pk_17[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[2] = _dot_325;
        float _dot_326;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_326) : "r"(ring_pk_17[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_326;
        float _dot_327;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_327) : "r"(ring_pk_17[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_327;
        float _dot_328;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_328) : "r"(ring_pk_17[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_328;
        float _dot_329;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_329) : "r"(ring_pk_17[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_329;
        float _dot_330;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_330) : "r"(ring_pk_17[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_330;
        float _dot_331;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_331) : "r"(ring_pk_17[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_331;
        float _dot_332;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_332) : "r"(ring_pk_17[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_332;
        float _dot_333;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_333) : "r"(ring_pk_17[8]), "r"(ring_pk_14[8]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_333;
        float _dot_334;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_334) : "r"(ring_pk_17[9]), "r"(ring_pk_14[9]), "f"(dots_p_2[2]));
        dots_p_2[2] = _dot_334;
        unsigned int ring_pk_18[10];
        unsigned int lane_row_18 = sa_b + 5120 + lane32;
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
        float dsig;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dsig) : "f"(sig), "f"(sneg));
        float G = ((lane_1 >= 8 && lane_1 < 24) ? gval : 0.0f);
        float _shfl_xor_sync_335 = __shfl_xor_sync(0xFFFFFFFF, hist[38], 4);
        float s_38;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_38) : "f"(hist[38]), "f"(_shfl_xor_sync_335));
        float _shfl_xor_sync_337 = __shfl_xor_sync(0xFFFFFFFF, s_38, 24);
        float _add_336;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_336) : "f"(s_38), "f"(_shfl_xor_sync_337));
        float v;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(_add_336), "f"(hc_eps));
        float m;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m) : "f"(G), "f"(hist[39]));
        float _shfl_xor_sync_338 = __shfl_xor_sync(0xFFFFFFFF, m, 4);
        float ip;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip) : "f"(m), "f"(_shfl_xor_sync_338));
        float _shfl_xor_sync_341 = __shfl_xor_sync(0xFFFFFFFF, ip, 24);
        float _add_340;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_340) : "f"(ip), "f"(_shfl_xor_sync_341));
        float _sub_339;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_339) : "f"(G), "f"(_add_340));
        float g = __fdiv_rn(_sub_339, v);
        float _dot_342;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_342) : "r"(ring_pk_18[0]), "r"(ring_pk_14[0]), "f"(-0.0f));
        dots_p_2[3] = _dot_342;
        float _dot_343;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_343) : "r"(ring_pk_18[1]), "r"(ring_pk_14[1]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_343;
        float _dot_344;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_344) : "r"(ring_pk_18[2]), "r"(ring_pk_14[2]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_344;
        float _dot_345;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_345) : "r"(ring_pk_18[3]), "r"(ring_pk_14[3]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_345;
        float _dot_346;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_346) : "r"(ring_pk_18[4]), "r"(ring_pk_14[4]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_346;
        float _dot_347;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_347) : "r"(ring_pk_18[5]), "r"(ring_pk_14[5]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_347;
        float _dot_348;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_348) : "r"(ring_pk_18[6]), "r"(ring_pk_14[6]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_348;
        float _dot_349;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_349) : "r"(ring_pk_18[7]), "r"(ring_pk_14[7]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_349;
        float _dot_350;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_350) : "r"(ring_pk_18[8]), "r"(ring_pk_14[8]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_350;
        float _dot_351;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_351) : "r"(ring_pk_18[9]), "r"(ring_pk_14[9]), "f"(dots_p_2[3]));
        dots_p_2[3] = _dot_351;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_b) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b), "l"(dU + (th_t_3 * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 1280), "l"(X + (th_t_3 * 4ll * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 2560), "l"(X + ((th_t_3 * 4ll + 1ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 3840), "l"(X + ((th_t_3 * 4ll + 2ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 5120), "l"(X + ((th_t_3 * 4ll + 3ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_352 = __shfl_xor_sync(0xFFFFFFFF, hist[37], 1);
        float s_39;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_39) : "f"(hist[37]), "f"(_shfl_xor_sync_352));
        float _shfl_xor_sync_354 = __shfl_xor_sync(0xFFFFFFFF, s_39, 2);
        float _add_353;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_353) : "f"(s_39), "f"(_shfl_xor_sync_354));
        float v_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_0) : "f"(_add_353), "f"(hc_eps));
        float m_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_0) : "f"(g), "f"(hist[38]));
        float _shfl_xor_sync_355 = __shfl_xor_sync(0xFFFFFFFF, m_0, 1);
        float ip_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_0) : "f"(m_0), "f"(_shfl_xor_sync_355));
        float _shfl_xor_sync_358 = __shfl_xor_sync(0xFFFFFFFF, ip_0, 2);
        float _add_357;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_357) : "f"(ip_0), "f"(_shfl_xor_sync_358));
        float _sub_356;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_356) : "f"(g), "f"(_add_357));
        float g_0 = __fdiv_rn(_sub_356, v_0);
        int upper_3 = lane_1 & 16;
        float send_4 = ((upper_3 != 0) ? dots_p_2[0] : dots_p_2[2]);
        float keep_4 = ((upper_3 != 0) ? dots_p_2[2] : dots_p_2[0]);
        float _shfl_xor_sync_359 = __shfl_xor_sync(0xFFFFFFFF, send_4, 16);
        float add_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8) : "f"(keep_4), "f"(_shfl_xor_sync_359));
        float send_1_2 = ((upper_3 != 0) ? dots_p_2[1] : dots_p_2[3]);
        float keep_1_2 = ((upper_3 != 0) ? dots_p_2[3] : dots_p_2[1]);
        float _shfl_xor_sync_360 = __shfl_xor_sync(0xFFFFFFFF, send_1_2, 16);
        float add_1_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_5) : "f"(keep_1_2), "f"(_shfl_xor_sync_360));
        int upper_1_2 = lane_1 & 8;
        float send_2_2 = ((upper_1_2 != 0) ? add_8 : add_1_5);
        float keep_2_2 = ((upper_1_2 != 0) ? add_1_5 : add_8);
        float _shfl_xor_sync_361 = __shfl_xor_sync(0xFFFFFFFF, send_2_2, 8);
        float add_2_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_2) : "f"(keep_2_2), "f"(_shfl_xor_sync_361));
        float _shfl_xor_sync_362 = __shfl_xor_sync(0xFFFFFFFF, hist[36], 4);
        float s_40;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_40) : "f"(hist[36]), "f"(_shfl_xor_sync_362));
        float _shfl_xor_sync_364 = __shfl_xor_sync(0xFFFFFFFF, s_40, 24);
        float _add_363;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_363) : "f"(s_40), "f"(_shfl_xor_sync_364));
        float v_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_1) : "f"(_add_363), "f"(hc_eps));
        float m_1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_1) : "f"(g_0), "f"(hist[37]));
        float _shfl_xor_sync_365 = __shfl_xor_sync(0xFFFFFFFF, m_1, 4);
        float ip_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_1) : "f"(m_1), "f"(_shfl_xor_sync_365));
        float _shfl_xor_sync_368 = __shfl_xor_sync(0xFFFFFFFF, ip_1, 24);
        float _add_367;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_367) : "f"(ip_1), "f"(_shfl_xor_sync_368));
        float _sub_366;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_366) : "f"(g_0), "f"(_add_367));
        float g_1 = __fdiv_rn(_sub_366, v_1);
        float _shfl_xor_sync_369 = __shfl_xor_sync(0xFFFFFFFF, add_2_2, 4);
        float add_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_9) : "f"(add_2_2), "f"(_shfl_xor_sync_369));
        float _shfl_xor_sync_370 = __shfl_xor_sync(0xFFFFFFFF, add_9, 2);
        float add_1_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_6) : "f"(add_9), "f"(_shfl_xor_sync_370));
        float _shfl_xor_sync_372 = __shfl_xor_sync(0xFFFFFFFF, add_1_6, 1);
        float _add_371;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_371) : "f"(add_1_6), "f"(_shfl_xor_sync_372));
        dpre_part[own_base + 96] = _add_371;
        float _shfl_xor_sync_373 = __shfl_xor_sync(0xFFFFFFFF, hist[35], 1);
        float s_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_41) : "f"(hist[35]), "f"(_shfl_xor_sync_373));
        float _shfl_xor_sync_375 = __shfl_xor_sync(0xFFFFFFFF, s_41, 2);
        float _add_374;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_374) : "f"(s_41), "f"(_shfl_xor_sync_375));
        float v_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_2) : "f"(_add_374), "f"(hc_eps));
        float m_2;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_2) : "f"(g_1), "f"(hist[36]));
        float _shfl_xor_sync_376 = __shfl_xor_sync(0xFFFFFFFF, m_2, 1);
        float ip_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_2) : "f"(m_2), "f"(_shfl_xor_sync_376));
        float _shfl_xor_sync_379 = __shfl_xor_sync(0xFFFFFFFF, ip_2, 2);
        float _add_378;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_378) : "f"(ip_2), "f"(_shfl_xor_sync_379));
        float _sub_377;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_377) : "f"(g_1), "f"(_add_378));
        float g_2 = __fdiv_rn(_sub_377, v_2);
        float _shfl_xor_sync_380 = __shfl_xor_sync(0xFFFFFFFF, hist[34], 4);
        float s_42;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_42) : "f"(hist[34]), "f"(_shfl_xor_sync_380));
        float _shfl_xor_sync_382 = __shfl_xor_sync(0xFFFFFFFF, s_42, 24);
        float _add_381;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_381) : "f"(s_42), "f"(_shfl_xor_sync_382));
        float v_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_3) : "f"(_add_381), "f"(hc_eps));
        float m_3;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_3) : "f"(g_2), "f"(hist[35]));
        float _shfl_xor_sync_383 = __shfl_xor_sync(0xFFFFFFFF, m_3, 4);
        float ip_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_3) : "f"(m_3), "f"(_shfl_xor_sync_383));
        float _shfl_xor_sync_386 = __shfl_xor_sync(0xFFFFFFFF, ip_3, 24);
        float _add_385;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_385) : "f"(ip_3), "f"(_shfl_xor_sync_386));
        float _sub_384;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_384) : "f"(g_2), "f"(_add_385));
        float g_3 = __fdiv_rn(_sub_384, v_3);
        float _shfl_xor_sync_387 = __shfl_xor_sync(0xFFFFFFFF, hist[33], 1);
        float s_43;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_43) : "f"(hist[33]), "f"(_shfl_xor_sync_387));
        float _shfl_xor_sync_389 = __shfl_xor_sync(0xFFFFFFFF, s_43, 2);
        float _add_388;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_388) : "f"(s_43), "f"(_shfl_xor_sync_389));
        float v_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_4) : "f"(_add_388), "f"(hc_eps));
        float m_4;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_4) : "f"(g_3), "f"(hist[34]));
        float _shfl_xor_sync_390 = __shfl_xor_sync(0xFFFFFFFF, m_4, 1);
        float ip_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_4) : "f"(m_4), "f"(_shfl_xor_sync_390));
        float _shfl_xor_sync_393 = __shfl_xor_sync(0xFFFFFFFF, ip_4, 2);
        float _add_392;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_392) : "f"(ip_4), "f"(_shfl_xor_sync_393));
        float _sub_391;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_391) : "f"(g_3), "f"(_add_392));
        float g_4 = __fdiv_rn(_sub_391, v_4);
        if (warp_0 == 0) {
            if (lane == 0) {
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(tkv_addr + tc * 4), "r"(gpub) : "memory");
                mbarrier_arrive(tk_full_addr + (tc) * 8);
            }
        }
        mbarrier_wait(ring_bar_addr + (st_a) * 8, 0);
        unsigned int ring_pk_19[10];
        unsigned int lane_row_19 = sa_a + lane32;
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
        unsigned int lane_row_20 = sa_a + 1280 + lane32;
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
        float _shfl_xor_sync_394 = __shfl_xor_sync(0xFFFFFFFF, hist[32], 4);
        float s_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_44) : "f"(hist[32]), "f"(_shfl_xor_sync_394));
        float _shfl_xor_sync_396 = __shfl_xor_sync(0xFFFFFFFF, s_44, 24);
        float _add_395;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_395) : "f"(s_44), "f"(_shfl_xor_sync_396));
        float v_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_5) : "f"(_add_395), "f"(hc_eps));
        float m_5;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_5) : "f"(g_4), "f"(hist[33]));
        float _shfl_xor_sync_397 = __shfl_xor_sync(0xFFFFFFFF, m_5, 4);
        float ip_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_5) : "f"(m_5), "f"(_shfl_xor_sync_397));
        float _shfl_xor_sync_400 = __shfl_xor_sync(0xFFFFFFFF, ip_5, 24);
        float _add_399;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_399) : "f"(ip_5), "f"(_shfl_xor_sync_400));
        float _sub_398;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_398) : "f"(g_4), "f"(_add_399));
        float g_5 = __fdiv_rn(_sub_398, v_5);
        float _dot_401;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_401) : "r"(ring_pk_20[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[0] = _dot_401;
        float _dot_402;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_402) : "r"(ring_pk_20[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_402;
        float _dot_403;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_403) : "r"(ring_pk_20[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_403;
        float _dot_404;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_404) : "r"(ring_pk_20[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_404;
        float _dot_405;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_405) : "r"(ring_pk_20[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_405;
        float _dot_406;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_406) : "r"(ring_pk_20[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_406;
        float _dot_407;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_407) : "r"(ring_pk_20[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_407;
        float _dot_408;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_408) : "r"(ring_pk_20[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_408;
        float _dot_409;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_409) : "r"(ring_pk_20[8]), "r"(ring_pk_19[8]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_409;
        float _dot_410;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_410) : "r"(ring_pk_20[9]), "r"(ring_pk_19[9]), "f"(dots_p_3[0]));
        dots_p_3[0] = _dot_410;
        unsigned int ring_pk_21[10];
        unsigned int lane_row_21 = sa_a + 2560 + lane32;
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
        float _shfl_xor_sync_411 = __shfl_xor_sync(0xFFFFFFFF, hist[31], 1);
        float s_45;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_45) : "f"(hist[31]), "f"(_shfl_xor_sync_411));
        float _shfl_xor_sync_413 = __shfl_xor_sync(0xFFFFFFFF, s_45, 2);
        float _add_412;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_412) : "f"(s_45), "f"(_shfl_xor_sync_413));
        float v_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_6) : "f"(_add_412), "f"(hc_eps));
        float m_6;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_6) : "f"(g_5), "f"(hist[32]));
        float _shfl_xor_sync_414 = __shfl_xor_sync(0xFFFFFFFF, m_6, 1);
        float ip_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_6) : "f"(m_6), "f"(_shfl_xor_sync_414));
        float _shfl_xor_sync_417 = __shfl_xor_sync(0xFFFFFFFF, ip_6, 2);
        float _add_416;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_416) : "f"(ip_6), "f"(_shfl_xor_sync_417));
        float _sub_415;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_415) : "f"(g_5), "f"(_add_416));
        float g_6 = __fdiv_rn(_sub_415, v_6);
        float _dot_418;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_418) : "r"(ring_pk_21[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[1] = _dot_418;
        float _dot_419;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_419) : "r"(ring_pk_21[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_419;
        float _dot_420;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_420) : "r"(ring_pk_21[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_420;
        float _dot_421;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_421) : "r"(ring_pk_21[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_421;
        float _dot_422;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_422) : "r"(ring_pk_21[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_422;
        float _dot_423;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_423) : "r"(ring_pk_21[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_423;
        float _dot_424;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_424) : "r"(ring_pk_21[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_424;
        float _dot_425;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_425) : "r"(ring_pk_21[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[1]));
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
            : "=f"(_dot_426) : "r"(ring_pk_21[8]), "r"(ring_pk_19[8]), "f"(dots_p_3[1]));
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
            : "=f"(_dot_427) : "r"(ring_pk_21[9]), "r"(ring_pk_19[9]), "f"(dots_p_3[1]));
        dots_p_3[1] = _dot_427;
        unsigned int ring_pk_22[10];
        unsigned int lane_row_22 = sa_a + 3840 + lane32;
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
        float _shfl_xor_sync_428 = __shfl_xor_sync(0xFFFFFFFF, hist[30], 4);
        float s_46;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_46) : "f"(hist[30]), "f"(_shfl_xor_sync_428));
        float _shfl_xor_sync_430 = __shfl_xor_sync(0xFFFFFFFF, s_46, 24);
        float _add_429;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_429) : "f"(s_46), "f"(_shfl_xor_sync_430));
        float v_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_7) : "f"(_add_429), "f"(hc_eps));
        float m_7;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_7) : "f"(g_6), "f"(hist[31]));
        float _shfl_xor_sync_431 = __shfl_xor_sync(0xFFFFFFFF, m_7, 4);
        float ip_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_7) : "f"(m_7), "f"(_shfl_xor_sync_431));
        float _shfl_xor_sync_434 = __shfl_xor_sync(0xFFFFFFFF, ip_7, 24);
        float _add_433;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_433) : "f"(ip_7), "f"(_shfl_xor_sync_434));
        float _sub_432;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_432) : "f"(g_6), "f"(_add_433));
        float g_7 = __fdiv_rn(_sub_432, v_7);
        float _dot_435;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_435) : "r"(ring_pk_22[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[2] = _dot_435;
        float _dot_436;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_436) : "r"(ring_pk_22[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_436;
        float _dot_437;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_437) : "r"(ring_pk_22[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_437;
        float _dot_438;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_438) : "r"(ring_pk_22[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_438;
        float _dot_439;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_439) : "r"(ring_pk_22[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_439;
        float _dot_440;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_440) : "r"(ring_pk_22[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[2]));
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
            : "=f"(_dot_441) : "r"(ring_pk_22[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[2]));
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
            : "=f"(_dot_442) : "r"(ring_pk_22[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[2]));
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
            : "=f"(_dot_443) : "r"(ring_pk_22[8]), "r"(ring_pk_19[8]), "f"(dots_p_3[2]));
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
            : "=f"(_dot_444) : "r"(ring_pk_22[9]), "r"(ring_pk_19[9]), "f"(dots_p_3[2]));
        dots_p_3[2] = _dot_444;
        unsigned int ring_pk_23[10];
        unsigned int lane_row_23 = sa_a + 5120 + lane32;
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
        float _shfl_xor_sync_445 = __shfl_xor_sync(0xFFFFFFFF, hist[29], 1);
        float s_47;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_47) : "f"(hist[29]), "f"(_shfl_xor_sync_445));
        float _shfl_xor_sync_447 = __shfl_xor_sync(0xFFFFFFFF, s_47, 2);
        float _add_446;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_446) : "f"(s_47), "f"(_shfl_xor_sync_447));
        float v_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_8) : "f"(_add_446), "f"(hc_eps));
        float m_8;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_8) : "f"(g_7), "f"(hist[30]));
        float _shfl_xor_sync_448 = __shfl_xor_sync(0xFFFFFFFF, m_8, 1);
        float ip_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_8) : "f"(m_8), "f"(_shfl_xor_sync_448));
        float _shfl_xor_sync_451 = __shfl_xor_sync(0xFFFFFFFF, ip_8, 2);
        float _add_450;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_450) : "f"(ip_8), "f"(_shfl_xor_sync_451));
        float _sub_449;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_449) : "f"(g_7), "f"(_add_450));
        float g_8 = __fdiv_rn(_sub_449, v_8);
        float _dot_452;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_452) : "r"(ring_pk_23[0]), "r"(ring_pk_19[0]), "f"(-0.0f));
        dots_p_3[3] = _dot_452;
        float _dot_453;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_453) : "r"(ring_pk_23[1]), "r"(ring_pk_19[1]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_453;
        float _dot_454;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_454) : "r"(ring_pk_23[2]), "r"(ring_pk_19[2]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_454;
        float _dot_455;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_455) : "r"(ring_pk_23[3]), "r"(ring_pk_19[3]), "f"(dots_p_3[3]));
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
            : "=f"(_dot_456) : "r"(ring_pk_23[4]), "r"(ring_pk_19[4]), "f"(dots_p_3[3]));
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
            : "=f"(_dot_457) : "r"(ring_pk_23[5]), "r"(ring_pk_19[5]), "f"(dots_p_3[3]));
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
            : "=f"(_dot_458) : "r"(ring_pk_23[6]), "r"(ring_pk_19[6]), "f"(dots_p_3[3]));
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
            : "=f"(_dot_459) : "r"(ring_pk_23[7]), "r"(ring_pk_19[7]), "f"(dots_p_3[3]));
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
            : "=f"(_dot_460) : "r"(ring_pk_23[8]), "r"(ring_pk_19[8]), "f"(dots_p_3[3]));
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
            : "=f"(_dot_461) : "r"(ring_pk_23[9]), "r"(ring_pk_19[9]), "f"(dots_p_3[3]));
        dots_p_3[3] = _dot_461;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_a) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a), "l"(dU + (th_t_4 * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 1280), "l"(X + (th_t_4 * 4ll * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 2560), "l"(X + ((th_t_4 * 4ll + 1ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 3840), "l"(X + ((th_t_4 * 4ll + 2ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 5120), "l"(X + ((th_t_4 * 4ll + 3ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_462 = __shfl_xor_sync(0xFFFFFFFF, hist[28], 4);
        float s_48;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_48) : "f"(hist[28]), "f"(_shfl_xor_sync_462));
        float _shfl_xor_sync_464 = __shfl_xor_sync(0xFFFFFFFF, s_48, 24);
        float _add_463;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_463) : "f"(s_48), "f"(_shfl_xor_sync_464));
        float v_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_9) : "f"(_add_463), "f"(hc_eps));
        float m_9;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_9) : "f"(g_8), "f"(hist[29]));
        float _shfl_xor_sync_465 = __shfl_xor_sync(0xFFFFFFFF, m_9, 4);
        float ip_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_9) : "f"(m_9), "f"(_shfl_xor_sync_465));
        float _shfl_xor_sync_468 = __shfl_xor_sync(0xFFFFFFFF, ip_9, 24);
        float _add_467;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_467) : "f"(ip_9), "f"(_shfl_xor_sync_468));
        float _sub_466;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_466) : "f"(g_8), "f"(_add_467));
        float g_9 = __fdiv_rn(_sub_466, v_9);
        int upper_4 = lane_1 & 16;
        float send_5 = ((upper_4 != 0) ? dots_p_3[0] : dots_p_3[2]);
        float keep_5 = ((upper_4 != 0) ? dots_p_3[2] : dots_p_3[0]);
        float _shfl_xor_sync_469 = __shfl_xor_sync(0xFFFFFFFF, send_5, 16);
        float add_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_10) : "f"(keep_5), "f"(_shfl_xor_sync_469));
        float send_1_3 = ((upper_4 != 0) ? dots_p_3[1] : dots_p_3[3]);
        float keep_1_3 = ((upper_4 != 0) ? dots_p_3[3] : dots_p_3[1]);
        float _shfl_xor_sync_470 = __shfl_xor_sync(0xFFFFFFFF, send_1_3, 16);
        float add_1_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_7) : "f"(keep_1_3), "f"(_shfl_xor_sync_470));
        int upper_1_3 = lane_1 & 8;
        float send_2_3 = ((upper_1_3 != 0) ? add_10 : add_1_7);
        float keep_2_3 = ((upper_1_3 != 0) ? add_1_7 : add_10);
        float _shfl_xor_sync_471 = __shfl_xor_sync(0xFFFFFFFF, send_2_3, 8);
        float add_2_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_3) : "f"(keep_2_3), "f"(_shfl_xor_sync_471));
        float _shfl_xor_sync_472 = __shfl_xor_sync(0xFFFFFFFF, hist[27], 1);
        float s_49;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_49) : "f"(hist[27]), "f"(_shfl_xor_sync_472));
        float _shfl_xor_sync_474 = __shfl_xor_sync(0xFFFFFFFF, s_49, 2);
        float _add_473;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_473) : "f"(s_49), "f"(_shfl_xor_sync_474));
        float v_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_10) : "f"(_add_473), "f"(hc_eps));
        float m_10;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_10) : "f"(g_9), "f"(hist[28]));
        float _shfl_xor_sync_475 = __shfl_xor_sync(0xFFFFFFFF, m_10, 1);
        float ip_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_10) : "f"(m_10), "f"(_shfl_xor_sync_475));
        float _shfl_xor_sync_478 = __shfl_xor_sync(0xFFFFFFFF, ip_10, 2);
        float _add_477;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_477) : "f"(ip_10), "f"(_shfl_xor_sync_478));
        float _sub_476;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_476) : "f"(g_9), "f"(_add_477));
        float g_10 = __fdiv_rn(_sub_476, v_10);
        float _shfl_xor_sync_479 = __shfl_xor_sync(0xFFFFFFFF, add_2_3, 4);
        float add_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_11) : "f"(add_2_3), "f"(_shfl_xor_sync_479));
        float _shfl_xor_sync_480 = __shfl_xor_sync(0xFFFFFFFF, add_11, 2);
        float add_1_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_8) : "f"(add_11), "f"(_shfl_xor_sync_480));
        float _shfl_xor_sync_482 = __shfl_xor_sync(0xFFFFFFFF, add_1_8, 1);
        float _add_481;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_481) : "f"(add_1_8), "f"(_shfl_xor_sync_482));
        dpre_part[own_base + 128] = _add_481;
        float _shfl_xor_sync_483 = __shfl_xor_sync(0xFFFFFFFF, hist[26], 4);
        float s_50;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_50) : "f"(hist[26]), "f"(_shfl_xor_sync_483));
        float _shfl_xor_sync_485 = __shfl_xor_sync(0xFFFFFFFF, s_50, 24);
        float _add_484;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_484) : "f"(s_50), "f"(_shfl_xor_sync_485));
        float v_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_11) : "f"(_add_484), "f"(hc_eps));
        float m_11;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_11) : "f"(g_10), "f"(hist[27]));
        float _shfl_xor_sync_486 = __shfl_xor_sync(0xFFFFFFFF, m_11, 4);
        float ip_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_11) : "f"(m_11), "f"(_shfl_xor_sync_486));
        float _shfl_xor_sync_489 = __shfl_xor_sync(0xFFFFFFFF, ip_11, 24);
        float _add_488;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_488) : "f"(ip_11), "f"(_shfl_xor_sync_489));
        float _sub_487;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_487) : "f"(g_10), "f"(_add_488));
        float g_11 = __fdiv_rn(_sub_487, v_11);
        float _shfl_xor_sync_490 = __shfl_xor_sync(0xFFFFFFFF, hist[25], 1);
        float s_51;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_51) : "f"(hist[25]), "f"(_shfl_xor_sync_490));
        float _shfl_xor_sync_492 = __shfl_xor_sync(0xFFFFFFFF, s_51, 2);
        float _add_491;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_491) : "f"(s_51), "f"(_shfl_xor_sync_492));
        float v_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_12) : "f"(_add_491), "f"(hc_eps));
        float m_12;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_12) : "f"(g_11), "f"(hist[26]));
        float _shfl_xor_sync_493 = __shfl_xor_sync(0xFFFFFFFF, m_12, 1);
        float ip_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_12) : "f"(m_12), "f"(_shfl_xor_sync_493));
        float _shfl_xor_sync_496 = __shfl_xor_sync(0xFFFFFFFF, ip_12, 2);
        float _add_495;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_495) : "f"(ip_12), "f"(_shfl_xor_sync_496));
        float _sub_494;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_494) : "f"(g_11), "f"(_add_495));
        float g_12 = __fdiv_rn(_sub_494, v_12);
        float _shfl_xor_sync_497 = __shfl_xor_sync(0xFFFFFFFF, hist[24], 4);
        float s_52;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_52) : "f"(hist[24]), "f"(_shfl_xor_sync_497));
        float _shfl_xor_sync_499 = __shfl_xor_sync(0xFFFFFFFF, s_52, 24);
        float _add_498;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_498) : "f"(s_52), "f"(_shfl_xor_sync_499));
        float v_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_13) : "f"(_add_498), "f"(hc_eps));
        float m_13;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_13) : "f"(g_12), "f"(hist[25]));
        float _shfl_xor_sync_500 = __shfl_xor_sync(0xFFFFFFFF, m_13, 4);
        float ip_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_13) : "f"(m_13), "f"(_shfl_xor_sync_500));
        float _shfl_xor_sync_503 = __shfl_xor_sync(0xFFFFFFFF, ip_13, 24);
        float _add_502;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_502) : "f"(ip_13), "f"(_shfl_xor_sync_503));
        float _sub_501;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_501) : "f"(g_12), "f"(_add_502));
        float g_13 = __fdiv_rn(_sub_501, v_13);
        mbarrier_wait(ring_bar_addr + (st_b) * 8, 0);
        unsigned int ring_pk_24[10];
        unsigned int lane_row_24 = sa_b + lane32;
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
        unsigned int lane_row_25 = sa_b + 1280 + lane32;
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
        float _shfl_xor_sync_504 = __shfl_xor_sync(0xFFFFFFFF, hist[23], 1);
        float s_53;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_53) : "f"(hist[23]), "f"(_shfl_xor_sync_504));
        float _shfl_xor_sync_506 = __shfl_xor_sync(0xFFFFFFFF, s_53, 2);
        float _add_505;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_505) : "f"(s_53), "f"(_shfl_xor_sync_506));
        float v_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_14) : "f"(_add_505), "f"(hc_eps));
        float m_14;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_14) : "f"(g_13), "f"(hist[24]));
        float _shfl_xor_sync_507 = __shfl_xor_sync(0xFFFFFFFF, m_14, 1);
        float ip_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_14) : "f"(m_14), "f"(_shfl_xor_sync_507));
        float _shfl_xor_sync_510 = __shfl_xor_sync(0xFFFFFFFF, ip_14, 2);
        float _add_509;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_509) : "f"(ip_14), "f"(_shfl_xor_sync_510));
        float _sub_508;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_508) : "f"(g_13), "f"(_add_509));
        float g_14 = __fdiv_rn(_sub_508, v_14);
        float _dot_511;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_511) : "r"(ring_pk_25[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[0] = _dot_511;
        float _dot_512;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_512) : "r"(ring_pk_25[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_512;
        float _dot_513;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_513) : "r"(ring_pk_25[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_513;
        float _dot_514;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_514) : "r"(ring_pk_25[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_514;
        float _dot_515;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_515) : "r"(ring_pk_25[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_515;
        float _dot_516;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_516) : "r"(ring_pk_25[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_516;
        float _dot_517;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_517) : "r"(ring_pk_25[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_517;
        float _dot_518;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_518) : "r"(ring_pk_25[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_518;
        float _dot_519;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_519) : "r"(ring_pk_25[8]), "r"(ring_pk_24[8]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_519;
        float _dot_520;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_520) : "r"(ring_pk_25[9]), "r"(ring_pk_24[9]), "f"(dots_p_4[0]));
        dots_p_4[0] = _dot_520;
        unsigned int ring_pk_26[10];
        unsigned int lane_row_26 = sa_b + 2560 + lane32;
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
        float _shfl_xor_sync_521 = __shfl_xor_sync(0xFFFFFFFF, hist[22], 4);
        float s_54;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_54) : "f"(hist[22]), "f"(_shfl_xor_sync_521));
        float _shfl_xor_sync_523 = __shfl_xor_sync(0xFFFFFFFF, s_54, 24);
        float _add_522;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_522) : "f"(s_54), "f"(_shfl_xor_sync_523));
        float v_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_15) : "f"(_add_522), "f"(hc_eps));
        float m_15;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_15) : "f"(g_14), "f"(hist[23]));
        float _shfl_xor_sync_524 = __shfl_xor_sync(0xFFFFFFFF, m_15, 4);
        float ip_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_15) : "f"(m_15), "f"(_shfl_xor_sync_524));
        float _shfl_xor_sync_527 = __shfl_xor_sync(0xFFFFFFFF, ip_15, 24);
        float _add_526;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_526) : "f"(ip_15), "f"(_shfl_xor_sync_527));
        float _sub_525;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_525) : "f"(g_14), "f"(_add_526));
        float g_15 = __fdiv_rn(_sub_525, v_15);
        float _dot_528;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_528) : "r"(ring_pk_26[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[1] = _dot_528;
        float _dot_529;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_529) : "r"(ring_pk_26[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_529;
        float _dot_530;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_530) : "r"(ring_pk_26[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_530;
        float _dot_531;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_531) : "r"(ring_pk_26[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_531;
        float _dot_532;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_532) : "r"(ring_pk_26[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_532;
        float _dot_533;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_533) : "r"(ring_pk_26[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_533;
        float _dot_534;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_534) : "r"(ring_pk_26[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_534;
        float _dot_535;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_535) : "r"(ring_pk_26[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_535;
        float _dot_536;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_536) : "r"(ring_pk_26[8]), "r"(ring_pk_24[8]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_536;
        float _dot_537;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_537) : "r"(ring_pk_26[9]), "r"(ring_pk_24[9]), "f"(dots_p_4[1]));
        dots_p_4[1] = _dot_537;
        unsigned int ring_pk_27[10];
        unsigned int lane_row_27 = sa_b + 3840 + lane32;
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
        float _shfl_xor_sync_538 = __shfl_xor_sync(0xFFFFFFFF, hist[21], 1);
        float s_55;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_55) : "f"(hist[21]), "f"(_shfl_xor_sync_538));
        float _shfl_xor_sync_540 = __shfl_xor_sync(0xFFFFFFFF, s_55, 2);
        float _add_539;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_539) : "f"(s_55), "f"(_shfl_xor_sync_540));
        float v_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_16) : "f"(_add_539), "f"(hc_eps));
        float m_16;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_16) : "f"(g_15), "f"(hist[22]));
        float _shfl_xor_sync_541 = __shfl_xor_sync(0xFFFFFFFF, m_16, 1);
        float ip_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_16) : "f"(m_16), "f"(_shfl_xor_sync_541));
        float _shfl_xor_sync_544 = __shfl_xor_sync(0xFFFFFFFF, ip_16, 2);
        float _add_543;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_543) : "f"(ip_16), "f"(_shfl_xor_sync_544));
        float _sub_542;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_542) : "f"(g_15), "f"(_add_543));
        float g_16 = __fdiv_rn(_sub_542, v_16);
        float _dot_545;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_545) : "r"(ring_pk_27[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[2] = _dot_545;
        float _dot_546;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_546) : "r"(ring_pk_27[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_546;
        float _dot_547;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_547) : "r"(ring_pk_27[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_547;
        float _dot_548;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_548) : "r"(ring_pk_27[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_548;
        float _dot_549;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_549) : "r"(ring_pk_27[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_549;
        float _dot_550;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_550) : "r"(ring_pk_27[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_550;
        float _dot_551;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_551) : "r"(ring_pk_27[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_551;
        float _dot_552;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_552) : "r"(ring_pk_27[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_552;
        float _dot_553;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_553) : "r"(ring_pk_27[8]), "r"(ring_pk_24[8]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_553;
        float _dot_554;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_554) : "r"(ring_pk_27[9]), "r"(ring_pk_24[9]), "f"(dots_p_4[2]));
        dots_p_4[2] = _dot_554;
        unsigned int ring_pk_28[10];
        unsigned int lane_row_28 = sa_b + 5120 + lane32;
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
        float _shfl_xor_sync_555 = __shfl_xor_sync(0xFFFFFFFF, hist[20], 4);
        float s_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_56) : "f"(hist[20]), "f"(_shfl_xor_sync_555));
        float _shfl_xor_sync_557 = __shfl_xor_sync(0xFFFFFFFF, s_56, 24);
        float _add_556;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_556) : "f"(s_56), "f"(_shfl_xor_sync_557));
        float v_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_17) : "f"(_add_556), "f"(hc_eps));
        float m_17;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_17) : "f"(g_16), "f"(hist[21]));
        float _shfl_xor_sync_558 = __shfl_xor_sync(0xFFFFFFFF, m_17, 4);
        float ip_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_17) : "f"(m_17), "f"(_shfl_xor_sync_558));
        float _shfl_xor_sync_561 = __shfl_xor_sync(0xFFFFFFFF, ip_17, 24);
        float _add_560;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_560) : "f"(ip_17), "f"(_shfl_xor_sync_561));
        float _sub_559;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_559) : "f"(g_16), "f"(_add_560));
        float g_17 = __fdiv_rn(_sub_559, v_17);
        float _dot_562;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_562) : "r"(ring_pk_28[0]), "r"(ring_pk_24[0]), "f"(-0.0f));
        dots_p_4[3] = _dot_562;
        float _dot_563;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_563) : "r"(ring_pk_28[1]), "r"(ring_pk_24[1]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_563;
        float _dot_564;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_564) : "r"(ring_pk_28[2]), "r"(ring_pk_24[2]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_564;
        float _dot_565;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_565) : "r"(ring_pk_28[3]), "r"(ring_pk_24[3]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_565;
        float _dot_566;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_566) : "r"(ring_pk_28[4]), "r"(ring_pk_24[4]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_566;
        float _dot_567;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_567) : "r"(ring_pk_28[5]), "r"(ring_pk_24[5]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_567;
        float _dot_568;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_568) : "r"(ring_pk_28[6]), "r"(ring_pk_24[6]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_568;
        float _dot_569;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_569) : "r"(ring_pk_28[7]), "r"(ring_pk_24[7]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_569;
        float _dot_570;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_570) : "r"(ring_pk_28[8]), "r"(ring_pk_24[8]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_570;
        float _dot_571;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_571) : "r"(ring_pk_28[9]), "r"(ring_pk_24[9]), "f"(dots_p_4[3]));
        dots_p_4[3] = _dot_571;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_b) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b), "l"(dU + (th_t_5 * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 1280), "l"(X + (th_t_5 * 4ll * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 2560), "l"(X + ((th_t_5 * 4ll + 1ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 3840), "l"(X + ((th_t_5 * 4ll + 2ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 5120), "l"(X + ((th_t_5 * 4ll + 3ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_572 = __shfl_xor_sync(0xFFFFFFFF, hist[19], 1);
        float s_57;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_57) : "f"(hist[19]), "f"(_shfl_xor_sync_572));
        float _shfl_xor_sync_574 = __shfl_xor_sync(0xFFFFFFFF, s_57, 2);
        float _add_573;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_573) : "f"(s_57), "f"(_shfl_xor_sync_574));
        float v_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_18) : "f"(_add_573), "f"(hc_eps));
        float m_18;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_18) : "f"(g_17), "f"(hist[20]));
        float _shfl_xor_sync_575 = __shfl_xor_sync(0xFFFFFFFF, m_18, 1);
        float ip_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_18) : "f"(m_18), "f"(_shfl_xor_sync_575));
        float _shfl_xor_sync_578 = __shfl_xor_sync(0xFFFFFFFF, ip_18, 2);
        float _add_577;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_577) : "f"(ip_18), "f"(_shfl_xor_sync_578));
        float _sub_576;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_576) : "f"(g_17), "f"(_add_577));
        float g_18 = __fdiv_rn(_sub_576, v_18);
        int upper_5 = lane_1 & 16;
        float send_6 = ((upper_5 != 0) ? dots_p_4[0] : dots_p_4[2]);
        float keep_6 = ((upper_5 != 0) ? dots_p_4[2] : dots_p_4[0]);
        float _shfl_xor_sync_579 = __shfl_xor_sync(0xFFFFFFFF, send_6, 16);
        float add_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_12) : "f"(keep_6), "f"(_shfl_xor_sync_579));
        float send_1_4 = ((upper_5 != 0) ? dots_p_4[1] : dots_p_4[3]);
        float keep_1_4 = ((upper_5 != 0) ? dots_p_4[3] : dots_p_4[1]);
        float _shfl_xor_sync_580 = __shfl_xor_sync(0xFFFFFFFF, send_1_4, 16);
        float add_1_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_9) : "f"(keep_1_4), "f"(_shfl_xor_sync_580));
        int upper_1_4 = lane_1 & 8;
        float send_2_4 = ((upper_1_4 != 0) ? add_12 : add_1_9);
        float keep_2_4 = ((upper_1_4 != 0) ? add_1_9 : add_12);
        float _shfl_xor_sync_581 = __shfl_xor_sync(0xFFFFFFFF, send_2_4, 8);
        float add_2_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_4) : "f"(keep_2_4), "f"(_shfl_xor_sync_581));
        float _shfl_xor_sync_582 = __shfl_xor_sync(0xFFFFFFFF, hist[18], 4);
        float s_58;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_58) : "f"(hist[18]), "f"(_shfl_xor_sync_582));
        float _shfl_xor_sync_584 = __shfl_xor_sync(0xFFFFFFFF, s_58, 24);
        float _add_583;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_583) : "f"(s_58), "f"(_shfl_xor_sync_584));
        float v_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_19) : "f"(_add_583), "f"(hc_eps));
        float m_19;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_19) : "f"(g_18), "f"(hist[19]));
        float _shfl_xor_sync_585 = __shfl_xor_sync(0xFFFFFFFF, m_19, 4);
        float ip_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_19) : "f"(m_19), "f"(_shfl_xor_sync_585));
        float _shfl_xor_sync_588 = __shfl_xor_sync(0xFFFFFFFF, ip_19, 24);
        float _add_587;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_587) : "f"(ip_19), "f"(_shfl_xor_sync_588));
        float _sub_586;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_586) : "f"(g_18), "f"(_add_587));
        float g_19 = __fdiv_rn(_sub_586, v_19);
        float _shfl_xor_sync_589 = __shfl_xor_sync(0xFFFFFFFF, add_2_4, 4);
        float add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_13) : "f"(add_2_4), "f"(_shfl_xor_sync_589));
        float _shfl_xor_sync_590 = __shfl_xor_sync(0xFFFFFFFF, add_13, 2);
        float add_1_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_10) : "f"(add_13), "f"(_shfl_xor_sync_590));
        float _shfl_xor_sync_592 = __shfl_xor_sync(0xFFFFFFFF, add_1_10, 1);
        float _add_591;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_591) : "f"(add_1_10), "f"(_shfl_xor_sync_592));
        dpre_part[own_base + 160] = _add_591;
        float _shfl_xor_sync_593 = __shfl_xor_sync(0xFFFFFFFF, hist[17], 1);
        float s_59;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_59) : "f"(hist[17]), "f"(_shfl_xor_sync_593));
        float _shfl_xor_sync_595 = __shfl_xor_sync(0xFFFFFFFF, s_59, 2);
        float _add_594;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_594) : "f"(s_59), "f"(_shfl_xor_sync_595));
        float v_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_20) : "f"(_add_594), "f"(hc_eps));
        float m_20;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_20) : "f"(g_19), "f"(hist[18]));
        float _shfl_xor_sync_596 = __shfl_xor_sync(0xFFFFFFFF, m_20, 1);
        float ip_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_20) : "f"(m_20), "f"(_shfl_xor_sync_596));
        float _shfl_xor_sync_599 = __shfl_xor_sync(0xFFFFFFFF, ip_20, 2);
        float _add_598;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_598) : "f"(ip_20), "f"(_shfl_xor_sync_599));
        float _sub_597;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_597) : "f"(g_19), "f"(_add_598));
        float g_20 = __fdiv_rn(_sub_597, v_20);
        float _shfl_xor_sync_600 = __shfl_xor_sync(0xFFFFFFFF, hist[16], 4);
        float s_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_60) : "f"(hist[16]), "f"(_shfl_xor_sync_600));
        float _shfl_xor_sync_602 = __shfl_xor_sync(0xFFFFFFFF, s_60, 24);
        float _add_601;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_601) : "f"(s_60), "f"(_shfl_xor_sync_602));
        float v_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_21) : "f"(_add_601), "f"(hc_eps));
        float m_21;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_21) : "f"(g_20), "f"(hist[17]));
        float _shfl_xor_sync_603 = __shfl_xor_sync(0xFFFFFFFF, m_21, 4);
        float ip_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_21) : "f"(m_21), "f"(_shfl_xor_sync_603));
        float _shfl_xor_sync_606 = __shfl_xor_sync(0xFFFFFFFF, ip_21, 24);
        float _add_605;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_605) : "f"(ip_21), "f"(_shfl_xor_sync_606));
        float _sub_604;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_604) : "f"(g_20), "f"(_add_605));
        float g_21 = __fdiv_rn(_sub_604, v_21);
        float _shfl_xor_sync_607 = __shfl_xor_sync(0xFFFFFFFF, hist[15], 1);
        float s_61;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_61) : "f"(hist[15]), "f"(_shfl_xor_sync_607));
        float _shfl_xor_sync_609 = __shfl_xor_sync(0xFFFFFFFF, s_61, 2);
        float _add_608;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_608) : "f"(s_61), "f"(_shfl_xor_sync_609));
        float v_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_22) : "f"(_add_608), "f"(hc_eps));
        float m_22;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_22) : "f"(g_21), "f"(hist[16]));
        float _shfl_xor_sync_610 = __shfl_xor_sync(0xFFFFFFFF, m_22, 1);
        float ip_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_22) : "f"(m_22), "f"(_shfl_xor_sync_610));
        float _shfl_xor_sync_613 = __shfl_xor_sync(0xFFFFFFFF, ip_22, 2);
        float _add_612;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_612) : "f"(ip_22), "f"(_shfl_xor_sync_613));
        float _sub_611;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_611) : "f"(g_21), "f"(_add_612));
        float g_22 = __fdiv_rn(_sub_611, v_22);
        mbarrier_wait(tk_full_addr + (tc) * 8, _phase_tk_full);
        unsigned int tku[1];
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tku[0])) : "r"(tkv_addr + tc * 4) : "memory");
        int gtk;
        asm volatile("redux.sync.max.s32 %0, %1, 0xFFFFFFFF;" : "=r"(gtk) : "r"((int)tku[0]));
        bool nv = gtk < g_end;
        int gn8 = gtk * 8;
        int gn_end = ((gn8 + 8 < T) ? gn8 + 8 : T);
        int gn1 = ((gn8 + 1 < gn_end) ? gn8 + 1 : gn_end - 1);
        long long t6 = (long long)((nv) ? gn8 : t_last);
        long long t7 = (long long)((nv) ? gn1 : t_last);
        mbarrier_wait(ring_bar_addr + (st_a) * 8, 1);
        unsigned int ring_pk_29[10];
        unsigned int lane_row_29 = sa_a + lane32;
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
        unsigned int lane_row_30 = sa_a + 1280 + lane32;
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
        float _shfl_xor_sync_614 = __shfl_xor_sync(0xFFFFFFFF, hist[14], 4);
        float s_62;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_62) : "f"(hist[14]), "f"(_shfl_xor_sync_614));
        float _shfl_xor_sync_616 = __shfl_xor_sync(0xFFFFFFFF, s_62, 24);
        float _add_615;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_615) : "f"(s_62), "f"(_shfl_xor_sync_616));
        float v_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_23) : "f"(_add_615), "f"(hc_eps));
        float m_23;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_23) : "f"(g_22), "f"(hist[15]));
        float _shfl_xor_sync_617 = __shfl_xor_sync(0xFFFFFFFF, m_23, 4);
        float ip_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_23) : "f"(m_23), "f"(_shfl_xor_sync_617));
        float _shfl_xor_sync_620 = __shfl_xor_sync(0xFFFFFFFF, ip_23, 24);
        float _add_619;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_619) : "f"(ip_23), "f"(_shfl_xor_sync_620));
        float _sub_618;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_618) : "f"(g_22), "f"(_add_619));
        float g_23 = __fdiv_rn(_sub_618, v_23);
        float _dot_621;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_621) : "r"(ring_pk_30[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[0] = _dot_621;
        float _dot_622;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_622) : "r"(ring_pk_30[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_622;
        float _dot_623;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_623) : "r"(ring_pk_30[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_623;
        float _dot_624;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_624) : "r"(ring_pk_30[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_624;
        float _dot_625;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_625) : "r"(ring_pk_30[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_625;
        float _dot_626;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_626) : "r"(ring_pk_30[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_626;
        float _dot_627;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_627) : "r"(ring_pk_30[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_627;
        float _dot_628;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_628) : "r"(ring_pk_30[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_628;
        float _dot_629;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_629) : "r"(ring_pk_30[8]), "r"(ring_pk_29[8]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_629;
        float _dot_630;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_630) : "r"(ring_pk_30[9]), "r"(ring_pk_29[9]), "f"(dots_p_5[0]));
        dots_p_5[0] = _dot_630;
        unsigned int ring_pk_31[10];
        unsigned int lane_row_31 = sa_a + 2560 + lane32;
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
        float _shfl_xor_sync_631 = __shfl_xor_sync(0xFFFFFFFF, hist[13], 1);
        float s_63;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_63) : "f"(hist[13]), "f"(_shfl_xor_sync_631));
        float _shfl_xor_sync_633 = __shfl_xor_sync(0xFFFFFFFF, s_63, 2);
        float _add_632;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_632) : "f"(s_63), "f"(_shfl_xor_sync_633));
        float v_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_24) : "f"(_add_632), "f"(hc_eps));
        float m_24;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_24) : "f"(g_23), "f"(hist[14]));
        float _shfl_xor_sync_634 = __shfl_xor_sync(0xFFFFFFFF, m_24, 1);
        float ip_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_24) : "f"(m_24), "f"(_shfl_xor_sync_634));
        float _shfl_xor_sync_637 = __shfl_xor_sync(0xFFFFFFFF, ip_24, 2);
        float _add_636;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_636) : "f"(ip_24), "f"(_shfl_xor_sync_637));
        float _sub_635;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_635) : "f"(g_23), "f"(_add_636));
        float g_24 = __fdiv_rn(_sub_635, v_24);
        float _dot_638;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_638) : "r"(ring_pk_31[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[1] = _dot_638;
        float _dot_639;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_639) : "r"(ring_pk_31[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_639;
        float _dot_640;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_640) : "r"(ring_pk_31[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_640;
        float _dot_641;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_641) : "r"(ring_pk_31[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_641;
        float _dot_642;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_642) : "r"(ring_pk_31[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_642;
        float _dot_643;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_643) : "r"(ring_pk_31[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_643;
        float _dot_644;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_644) : "r"(ring_pk_31[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_644;
        float _dot_645;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_645) : "r"(ring_pk_31[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_645;
        float _dot_646;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_646) : "r"(ring_pk_31[8]), "r"(ring_pk_29[8]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_646;
        float _dot_647;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_647) : "r"(ring_pk_31[9]), "r"(ring_pk_29[9]), "f"(dots_p_5[1]));
        dots_p_5[1] = _dot_647;
        unsigned int ring_pk_32[10];
        unsigned int lane_row_32 = sa_a + 3840 + lane32;
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
        float _shfl_xor_sync_648 = __shfl_xor_sync(0xFFFFFFFF, hist[12], 4);
        float s_64;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_64) : "f"(hist[12]), "f"(_shfl_xor_sync_648));
        float _shfl_xor_sync_650 = __shfl_xor_sync(0xFFFFFFFF, s_64, 24);
        float _add_649;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_649) : "f"(s_64), "f"(_shfl_xor_sync_650));
        float v_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_25) : "f"(_add_649), "f"(hc_eps));
        float m_25;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_25) : "f"(g_24), "f"(hist[13]));
        float _shfl_xor_sync_651 = __shfl_xor_sync(0xFFFFFFFF, m_25, 4);
        float ip_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_25) : "f"(m_25), "f"(_shfl_xor_sync_651));
        float _shfl_xor_sync_654 = __shfl_xor_sync(0xFFFFFFFF, ip_25, 24);
        float _add_653;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_653) : "f"(ip_25), "f"(_shfl_xor_sync_654));
        float _sub_652;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_652) : "f"(g_24), "f"(_add_653));
        float g_25 = __fdiv_rn(_sub_652, v_25);
        float _dot_655;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_655) : "r"(ring_pk_32[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[2] = _dot_655;
        float _dot_656;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_656) : "r"(ring_pk_32[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_656;
        float _dot_657;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_657) : "r"(ring_pk_32[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_657;
        float _dot_658;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_658) : "r"(ring_pk_32[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_658;
        float _dot_659;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_659) : "r"(ring_pk_32[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_659;
        float _dot_660;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_660) : "r"(ring_pk_32[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_660;
        float _dot_661;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_661) : "r"(ring_pk_32[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_661;
        float _dot_662;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_662) : "r"(ring_pk_32[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_662;
        float _dot_663;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_663) : "r"(ring_pk_32[8]), "r"(ring_pk_29[8]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_663;
        float _dot_664;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_664) : "r"(ring_pk_32[9]), "r"(ring_pk_29[9]), "f"(dots_p_5[2]));
        dots_p_5[2] = _dot_664;
        unsigned int ring_pk_33[10];
        unsigned int lane_row_33 = sa_a + 5120 + lane32;
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
        float _shfl_xor_sync_665 = __shfl_xor_sync(0xFFFFFFFF, hist[11], 1);
        float s_65;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_65) : "f"(hist[11]), "f"(_shfl_xor_sync_665));
        float _shfl_xor_sync_667 = __shfl_xor_sync(0xFFFFFFFF, s_65, 2);
        float _add_666;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_666) : "f"(s_65), "f"(_shfl_xor_sync_667));
        float v_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_26) : "f"(_add_666), "f"(hc_eps));
        float m_26;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_26) : "f"(g_25), "f"(hist[12]));
        float _shfl_xor_sync_668 = __shfl_xor_sync(0xFFFFFFFF, m_26, 1);
        float ip_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_26) : "f"(m_26), "f"(_shfl_xor_sync_668));
        float _shfl_xor_sync_671 = __shfl_xor_sync(0xFFFFFFFF, ip_26, 2);
        float _add_670;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_670) : "f"(ip_26), "f"(_shfl_xor_sync_671));
        float _sub_669;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_669) : "f"(g_25), "f"(_add_670));
        float g_26 = __fdiv_rn(_sub_669, v_26);
        float _dot_672;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_672) : "r"(ring_pk_33[0]), "r"(ring_pk_29[0]), "f"(-0.0f));
        dots_p_5[3] = _dot_672;
        float _dot_673;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_673) : "r"(ring_pk_33[1]), "r"(ring_pk_29[1]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_673;
        float _dot_674;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_674) : "r"(ring_pk_33[2]), "r"(ring_pk_29[2]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_674;
        float _dot_675;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_675) : "r"(ring_pk_33[3]), "r"(ring_pk_29[3]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_675;
        float _dot_676;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_676) : "r"(ring_pk_33[4]), "r"(ring_pk_29[4]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_676;
        float _dot_677;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_677) : "r"(ring_pk_33[5]), "r"(ring_pk_29[5]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_677;
        float _dot_678;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_678) : "r"(ring_pk_33[6]), "r"(ring_pk_29[6]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_678;
        float _dot_679;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_679) : "r"(ring_pk_33[7]), "r"(ring_pk_29[7]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_679;
        float _dot_680;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_680) : "r"(ring_pk_33[8]), "r"(ring_pk_29[8]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_680;
        float _dot_681;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_681) : "r"(ring_pk_33[9]), "r"(ring_pk_29[9]), "f"(dots_p_5[3]));
        dots_p_5[3] = _dot_681;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_a) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a), "l"(dU + (t6 * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 1280), "l"(X + (t6 * 4ll * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 2560), "l"(X + ((t6 * 4ll + 1ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 3840), "l"(X + ((t6 * 4ll + 2ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_a + 5120), "l"(X + ((t6 * 4ll + 3ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_a) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_682 = __shfl_xor_sync(0xFFFFFFFF, hist[10], 4);
        float s_66;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_66) : "f"(hist[10]), "f"(_shfl_xor_sync_682));
        float _shfl_xor_sync_684 = __shfl_xor_sync(0xFFFFFFFF, s_66, 24);
        float _add_683;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_683) : "f"(s_66), "f"(_shfl_xor_sync_684));
        float v_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_27) : "f"(_add_683), "f"(hc_eps));
        float m_27;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_27) : "f"(g_26), "f"(hist[11]));
        float _shfl_xor_sync_685 = __shfl_xor_sync(0xFFFFFFFF, m_27, 4);
        float ip_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_27) : "f"(m_27), "f"(_shfl_xor_sync_685));
        float _shfl_xor_sync_688 = __shfl_xor_sync(0xFFFFFFFF, ip_27, 24);
        float _add_687;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_687) : "f"(ip_27), "f"(_shfl_xor_sync_688));
        float _sub_686;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_686) : "f"(g_26), "f"(_add_687));
        float g_27 = __fdiv_rn(_sub_686, v_27);
        int upper_6 = lane_1 & 16;
        float send_7 = ((upper_6 != 0) ? dots_p_5[0] : dots_p_5[2]);
        float keep_7 = ((upper_6 != 0) ? dots_p_5[2] : dots_p_5[0]);
        float _shfl_xor_sync_689 = __shfl_xor_sync(0xFFFFFFFF, send_7, 16);
        float add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_14) : "f"(keep_7), "f"(_shfl_xor_sync_689));
        float send_1_5 = ((upper_6 != 0) ? dots_p_5[1] : dots_p_5[3]);
        float keep_1_5 = ((upper_6 != 0) ? dots_p_5[3] : dots_p_5[1]);
        float _shfl_xor_sync_690 = __shfl_xor_sync(0xFFFFFFFF, send_1_5, 16);
        float add_1_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_11) : "f"(keep_1_5), "f"(_shfl_xor_sync_690));
        int upper_1_5 = lane_1 & 8;
        float send_2_5 = ((upper_1_5 != 0) ? add_14 : add_1_11);
        float keep_2_5 = ((upper_1_5 != 0) ? add_1_11 : add_14);
        float _shfl_xor_sync_691 = __shfl_xor_sync(0xFFFFFFFF, send_2_5, 8);
        float add_2_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_5) : "f"(keep_2_5), "f"(_shfl_xor_sync_691));
        float _shfl_xor_sync_692 = __shfl_xor_sync(0xFFFFFFFF, hist[9], 1);
        float s_67;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_67) : "f"(hist[9]), "f"(_shfl_xor_sync_692));
        float _shfl_xor_sync_694 = __shfl_xor_sync(0xFFFFFFFF, s_67, 2);
        float _add_693;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_693) : "f"(s_67), "f"(_shfl_xor_sync_694));
        float v_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_28) : "f"(_add_693), "f"(hc_eps));
        float m_28;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_28) : "f"(g_27), "f"(hist[10]));
        float _shfl_xor_sync_695 = __shfl_xor_sync(0xFFFFFFFF, m_28, 1);
        float ip_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_28) : "f"(m_28), "f"(_shfl_xor_sync_695));
        float _shfl_xor_sync_698 = __shfl_xor_sync(0xFFFFFFFF, ip_28, 2);
        float _add_697;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_697) : "f"(ip_28), "f"(_shfl_xor_sync_698));
        float _sub_696;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_696) : "f"(g_27), "f"(_add_697));
        float g_28 = __fdiv_rn(_sub_696, v_28);
        float _shfl_xor_sync_699 = __shfl_xor_sync(0xFFFFFFFF, add_2_5, 4);
        float add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_15) : "f"(add_2_5), "f"(_shfl_xor_sync_699));
        float _shfl_xor_sync_700 = __shfl_xor_sync(0xFFFFFFFF, add_15, 2);
        float add_1_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_12) : "f"(add_15), "f"(_shfl_xor_sync_700));
        float _shfl_xor_sync_702 = __shfl_xor_sync(0xFFFFFFFF, add_1_12, 1);
        float _add_701;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_701) : "f"(add_1_12), "f"(_shfl_xor_sync_702));
        dpre_part[own_base + 192] = _add_701;
        float _shfl_xor_sync_703 = __shfl_xor_sync(0xFFFFFFFF, hist[8], 4);
        float s_68;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_68) : "f"(hist[8]), "f"(_shfl_xor_sync_703));
        float _shfl_xor_sync_705 = __shfl_xor_sync(0xFFFFFFFF, s_68, 24);
        float _add_704;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_704) : "f"(s_68), "f"(_shfl_xor_sync_705));
        float v_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_29) : "f"(_add_704), "f"(hc_eps));
        float m_29;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_29) : "f"(g_28), "f"(hist[9]));
        float _shfl_xor_sync_706 = __shfl_xor_sync(0xFFFFFFFF, m_29, 4);
        float ip_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_29) : "f"(m_29), "f"(_shfl_xor_sync_706));
        float _shfl_xor_sync_709 = __shfl_xor_sync(0xFFFFFFFF, ip_29, 24);
        float _add_708;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_708) : "f"(ip_29), "f"(_shfl_xor_sync_709));
        float _sub_707;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_707) : "f"(g_28), "f"(_add_708));
        float g_29 = __fdiv_rn(_sub_707, v_29);
        float _shfl_xor_sync_710 = __shfl_xor_sync(0xFFFFFFFF, hist[7], 1);
        float s_69;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_69) : "f"(hist[7]), "f"(_shfl_xor_sync_710));
        float _shfl_xor_sync_712 = __shfl_xor_sync(0xFFFFFFFF, s_69, 2);
        float _add_711;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_711) : "f"(s_69), "f"(_shfl_xor_sync_712));
        float v_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_30) : "f"(_add_711), "f"(hc_eps));
        float m_30;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_30) : "f"(g_29), "f"(hist[8]));
        float _shfl_xor_sync_713 = __shfl_xor_sync(0xFFFFFFFF, m_30, 1);
        float ip_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_30) : "f"(m_30), "f"(_shfl_xor_sync_713));
        float _shfl_xor_sync_716 = __shfl_xor_sync(0xFFFFFFFF, ip_30, 2);
        float _add_715;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_715) : "f"(ip_30), "f"(_shfl_xor_sync_716));
        float _sub_714;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_714) : "f"(g_29), "f"(_add_715));
        float g_30 = __fdiv_rn(_sub_714, v_30);
        float _shfl_xor_sync_717 = __shfl_xor_sync(0xFFFFFFFF, hist[6], 4);
        float s_70;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_70) : "f"(hist[6]), "f"(_shfl_xor_sync_717));
        float _shfl_xor_sync_719 = __shfl_xor_sync(0xFFFFFFFF, s_70, 24);
        float _add_718;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_718) : "f"(s_70), "f"(_shfl_xor_sync_719));
        float v_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_31) : "f"(_add_718), "f"(hc_eps));
        float m_31;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_31) : "f"(g_30), "f"(hist[7]));
        float _shfl_xor_sync_720 = __shfl_xor_sync(0xFFFFFFFF, m_31, 4);
        float ip_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_31) : "f"(m_31), "f"(_shfl_xor_sync_720));
        float _shfl_xor_sync_723 = __shfl_xor_sync(0xFFFFFFFF, ip_31, 24);
        float _add_722;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_722) : "f"(ip_31), "f"(_shfl_xor_sync_723));
        float _sub_721;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_721) : "f"(g_30), "f"(_add_722));
        float g_31 = __fdiv_rn(_sub_721, v_31);
        mbarrier_wait(ring_bar_addr + (st_b) * 8, 1);
        unsigned int ring_pk_34[10];
        unsigned int lane_row_34 = sa_b + lane32;
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
        unsigned int lane_row_35 = sa_b + 1280 + lane32;
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
        float _shfl_xor_sync_724 = __shfl_xor_sync(0xFFFFFFFF, hist[5], 1);
        float s_71;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_71) : "f"(hist[5]), "f"(_shfl_xor_sync_724));
        float _shfl_xor_sync_726 = __shfl_xor_sync(0xFFFFFFFF, s_71, 2);
        float _add_725;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_725) : "f"(s_71), "f"(_shfl_xor_sync_726));
        float v_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_32) : "f"(_add_725), "f"(hc_eps));
        float m_32;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_32) : "f"(g_31), "f"(hist[6]));
        float _shfl_xor_sync_727 = __shfl_xor_sync(0xFFFFFFFF, m_32, 1);
        float ip_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_32) : "f"(m_32), "f"(_shfl_xor_sync_727));
        float _shfl_xor_sync_730 = __shfl_xor_sync(0xFFFFFFFF, ip_32, 2);
        float _add_729;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_729) : "f"(ip_32), "f"(_shfl_xor_sync_730));
        float _sub_728;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_728) : "f"(g_31), "f"(_add_729));
        float g_32 = __fdiv_rn(_sub_728, v_32);
        float _dot_731;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_731) : "r"(ring_pk_35[0]), "r"(ring_pk_34[0]), "f"(-0.0f));
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
            : "=f"(_dot_732) : "r"(ring_pk_35[1]), "r"(ring_pk_34[1]), "f"(dots_p_6[0]));
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
            : "=f"(_dot_733) : "r"(ring_pk_35[2]), "r"(ring_pk_34[2]), "f"(dots_p_6[0]));
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
            : "=f"(_dot_734) : "r"(ring_pk_35[3]), "r"(ring_pk_34[3]), "f"(dots_p_6[0]));
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
            : "=f"(_dot_735) : "r"(ring_pk_35[4]), "r"(ring_pk_34[4]), "f"(dots_p_6[0]));
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
            : "=f"(_dot_736) : "r"(ring_pk_35[5]), "r"(ring_pk_34[5]), "f"(dots_p_6[0]));
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
            : "=f"(_dot_737) : "r"(ring_pk_35[6]), "r"(ring_pk_34[6]), "f"(dots_p_6[0]));
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
            : "=f"(_dot_738) : "r"(ring_pk_35[7]), "r"(ring_pk_34[7]), "f"(dots_p_6[0]));
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
            : "=f"(_dot_739) : "r"(ring_pk_35[8]), "r"(ring_pk_34[8]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_739;
        float _dot_740;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_740) : "r"(ring_pk_35[9]), "r"(ring_pk_34[9]), "f"(dots_p_6[0]));
        dots_p_6[0] = _dot_740;
        unsigned int ring_pk_36[10];
        unsigned int lane_row_36 = sa_b + 2560 + lane32;
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
        float _shfl_xor_sync_741 = __shfl_xor_sync(0xFFFFFFFF, hist[4], 4);
        float s_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_72) : "f"(hist[4]), "f"(_shfl_xor_sync_741));
        float _shfl_xor_sync_743 = __shfl_xor_sync(0xFFFFFFFF, s_72, 24);
        float _add_742;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_742) : "f"(s_72), "f"(_shfl_xor_sync_743));
        float v_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_33) : "f"(_add_742), "f"(hc_eps));
        float m_33;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_33) : "f"(g_32), "f"(hist[5]));
        float _shfl_xor_sync_744 = __shfl_xor_sync(0xFFFFFFFF, m_33, 4);
        float ip_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_33) : "f"(m_33), "f"(_shfl_xor_sync_744));
        float _shfl_xor_sync_747 = __shfl_xor_sync(0xFFFFFFFF, ip_33, 24);
        float _add_746;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_746) : "f"(ip_33), "f"(_shfl_xor_sync_747));
        float _sub_745;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_745) : "f"(g_32), "f"(_add_746));
        float g_33 = __fdiv_rn(_sub_745, v_33);
        float _dot_748;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_748) : "r"(ring_pk_36[0]), "r"(ring_pk_34[0]), "f"(-0.0f));
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
            : "=f"(_dot_749) : "r"(ring_pk_36[1]), "r"(ring_pk_34[1]), "f"(dots_p_6[1]));
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
            : "=f"(_dot_750) : "r"(ring_pk_36[2]), "r"(ring_pk_34[2]), "f"(dots_p_6[1]));
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
            : "=f"(_dot_751) : "r"(ring_pk_36[3]), "r"(ring_pk_34[3]), "f"(dots_p_6[1]));
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
            : "=f"(_dot_752) : "r"(ring_pk_36[4]), "r"(ring_pk_34[4]), "f"(dots_p_6[1]));
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
            : "=f"(_dot_753) : "r"(ring_pk_36[5]), "r"(ring_pk_34[5]), "f"(dots_p_6[1]));
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
            : "=f"(_dot_754) : "r"(ring_pk_36[6]), "r"(ring_pk_34[6]), "f"(dots_p_6[1]));
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
            : "=f"(_dot_755) : "r"(ring_pk_36[7]), "r"(ring_pk_34[7]), "f"(dots_p_6[1]));
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
            : "=f"(_dot_756) : "r"(ring_pk_36[8]), "r"(ring_pk_34[8]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_756;
        float _dot_757;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_757) : "r"(ring_pk_36[9]), "r"(ring_pk_34[9]), "f"(dots_p_6[1]));
        dots_p_6[1] = _dot_757;
        unsigned int ring_pk_37[10];
        unsigned int lane_row_37 = sa_b + 3840 + lane32;
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
        float _shfl_xor_sync_758 = __shfl_xor_sync(0xFFFFFFFF, hist[3], 1);
        float s_73;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_73) : "f"(hist[3]), "f"(_shfl_xor_sync_758));
        float _shfl_xor_sync_760 = __shfl_xor_sync(0xFFFFFFFF, s_73, 2);
        float _add_759;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_759) : "f"(s_73), "f"(_shfl_xor_sync_760));
        float v_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_34) : "f"(_add_759), "f"(hc_eps));
        float m_34;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_34) : "f"(g_33), "f"(hist[4]));
        float _shfl_xor_sync_761 = __shfl_xor_sync(0xFFFFFFFF, m_34, 1);
        float ip_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_34) : "f"(m_34), "f"(_shfl_xor_sync_761));
        float _shfl_xor_sync_764 = __shfl_xor_sync(0xFFFFFFFF, ip_34, 2);
        float _add_763;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_763) : "f"(ip_34), "f"(_shfl_xor_sync_764));
        float _sub_762;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_762) : "f"(g_33), "f"(_add_763));
        float g_34 = __fdiv_rn(_sub_762, v_34);
        float _dot_765;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_765) : "r"(ring_pk_37[0]), "r"(ring_pk_34[0]), "f"(-0.0f));
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
            : "=f"(_dot_766) : "r"(ring_pk_37[1]), "r"(ring_pk_34[1]), "f"(dots_p_6[2]));
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
            : "=f"(_dot_767) : "r"(ring_pk_37[2]), "r"(ring_pk_34[2]), "f"(dots_p_6[2]));
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
            : "=f"(_dot_768) : "r"(ring_pk_37[3]), "r"(ring_pk_34[3]), "f"(dots_p_6[2]));
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
            : "=f"(_dot_769) : "r"(ring_pk_37[4]), "r"(ring_pk_34[4]), "f"(dots_p_6[2]));
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
            : "=f"(_dot_770) : "r"(ring_pk_37[5]), "r"(ring_pk_34[5]), "f"(dots_p_6[2]));
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
            : "=f"(_dot_771) : "r"(ring_pk_37[6]), "r"(ring_pk_34[6]), "f"(dots_p_6[2]));
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
            : "=f"(_dot_772) : "r"(ring_pk_37[7]), "r"(ring_pk_34[7]), "f"(dots_p_6[2]));
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
            : "=f"(_dot_773) : "r"(ring_pk_37[8]), "r"(ring_pk_34[8]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_773;
        float _dot_774;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_774) : "r"(ring_pk_37[9]), "r"(ring_pk_34[9]), "f"(dots_p_6[2]));
        dots_p_6[2] = _dot_774;
        unsigned int ring_pk_38[10];
        unsigned int lane_row_38 = sa_b + 5120 + lane32;
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
        float _shfl_xor_sync_775 = __shfl_xor_sync(0xFFFFFFFF, hist[2], 4);
        float s_74;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_74) : "f"(hist[2]), "f"(_shfl_xor_sync_775));
        float _shfl_xor_sync_777 = __shfl_xor_sync(0xFFFFFFFF, s_74, 24);
        float _add_776;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_776) : "f"(s_74), "f"(_shfl_xor_sync_777));
        float v_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_35) : "f"(_add_776), "f"(hc_eps));
        float m_35;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_35) : "f"(g_34), "f"(hist[3]));
        float _shfl_xor_sync_778 = __shfl_xor_sync(0xFFFFFFFF, m_35, 4);
        float ip_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_35) : "f"(m_35), "f"(_shfl_xor_sync_778));
        float _shfl_xor_sync_781 = __shfl_xor_sync(0xFFFFFFFF, ip_35, 24);
        float _add_780;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_780) : "f"(ip_35), "f"(_shfl_xor_sync_781));
        float _sub_779;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_779) : "f"(g_34), "f"(_add_780));
        float g_35 = __fdiv_rn(_sub_779, v_35);
        float _dot_782;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_782) : "r"(ring_pk_38[0]), "r"(ring_pk_34[0]), "f"(-0.0f));
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
            : "=f"(_dot_783) : "r"(ring_pk_38[1]), "r"(ring_pk_34[1]), "f"(dots_p_6[3]));
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
            : "=f"(_dot_784) : "r"(ring_pk_38[2]), "r"(ring_pk_34[2]), "f"(dots_p_6[3]));
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
            : "=f"(_dot_785) : "r"(ring_pk_38[3]), "r"(ring_pk_34[3]), "f"(dots_p_6[3]));
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
            : "=f"(_dot_786) : "r"(ring_pk_38[4]), "r"(ring_pk_34[4]), "f"(dots_p_6[3]));
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
            : "=f"(_dot_787) : "r"(ring_pk_38[5]), "r"(ring_pk_34[5]), "f"(dots_p_6[3]));
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
            : "=f"(_dot_788) : "r"(ring_pk_38[6]), "r"(ring_pk_34[6]), "f"(dots_p_6[3]));
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
            : "=f"(_dot_789) : "r"(ring_pk_38[7]), "r"(ring_pk_34[7]), "f"(dots_p_6[3]));
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
            : "=f"(_dot_790) : "r"(ring_pk_38[8]), "r"(ring_pk_34[8]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_790;
        float _dot_791;
        asm(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_791) : "r"(ring_pk_38[9]), "r"(ring_pk_34[9]), "f"(dots_p_6[3]));
        dots_p_6[3] = _dot_791;
        __syncwarp();
        if (elect_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive_expect_tx(ring_bar_addr + (st_b) * 8, 6400);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b), "l"(dU + (t7 * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 1280), "l"(X + (t7 * 4ll * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 2560), "l"(X + ((t7 * 4ll + 1ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 3840), "l"(X + ((t7 * 4ll + 2ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sa_b + 5120), "l"(X + ((t7 * 4ll + 3ll) * 5120ll + wcol)), "r"((unsigned)(1280)), "r"(ring_bar_addr + (st_b) * 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        float _shfl_xor_sync_792 = __shfl_xor_sync(0xFFFFFFFF, hist[1], 1);
        float s_75;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_75) : "f"(hist[1]), "f"(_shfl_xor_sync_792));
        float _shfl_xor_sync_794 = __shfl_xor_sync(0xFFFFFFFF, s_75, 2);
        float _add_793;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_793) : "f"(s_75), "f"(_shfl_xor_sync_794));
        float v_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(v_36) : "f"(_add_793), "f"(hc_eps));
        float m_36;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(m_36) : "f"(g_35), "f"(hist[2]));
        float _shfl_xor_sync_795 = __shfl_xor_sync(0xFFFFFFFF, m_36, 1);
        float ip_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ip_36) : "f"(m_36), "f"(_shfl_xor_sync_795));
        float _shfl_xor_sync_798 = __shfl_xor_sync(0xFFFFFFFF, ip_36, 2);
        float _add_797;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_797) : "f"(ip_36), "f"(_shfl_xor_sync_798));
        float _sub_796;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_796) : "f"(g_35), "f"(_add_797));
        float g_36 = __fdiv_rn(_sub_796, v_36);
        int upper_7 = lane_1 & 16;
        float send_8 = ((upper_7 != 0) ? dots_p_6[0] : dots_p_6[2]);
        float keep_8 = ((upper_7 != 0) ? dots_p_6[2] : dots_p_6[0]);
        float _shfl_xor_sync_799 = __shfl_xor_sync(0xFFFFFFFF, send_8, 16);
        float add_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_16) : "f"(keep_8), "f"(_shfl_xor_sync_799));
        float send_1_6 = ((upper_7 != 0) ? dots_p_6[1] : dots_p_6[3]);
        float keep_1_6 = ((upper_7 != 0) ? dots_p_6[3] : dots_p_6[1]);
        float _shfl_xor_sync_800 = __shfl_xor_sync(0xFFFFFFFF, send_1_6, 16);
        float add_1_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_13) : "f"(keep_1_6), "f"(_shfl_xor_sync_800));
        int upper_1_6 = lane_1 & 8;
        float send_2_6 = ((upper_1_6 != 0) ? add_16 : add_1_13);
        float keep_2_6 = ((upper_1_6 != 0) ? add_1_13 : add_16);
        float _shfl_xor_sync_801 = __shfl_xor_sync(0xFFFFFFFF, send_2_6, 8);
        float add_2_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_6) : "f"(keep_2_6), "f"(_shfl_xor_sync_801));
        float _add_195;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(hist[0]), "f"(hc_eps));
        float _shfl_xor_sync_802 = __shfl_xor_sync(0xFFFFFFFF, _add_195, 4);
        float add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_17) : "f"(_add_195), "f"(_shfl_xor_sync_802));
        float _shfl_xor_sync_804 = __shfl_xor_sync(0xFFFFFFFF, add_17, 24);
        float _add_803;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_803) : "f"(add_17), "f"(_shfl_xor_sync_804));
        float _add_197;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(_add_803), "f"(hc_eps));
        float _mul_39;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(g_36), "f"(hist[1]));
        float _shfl_xor_sync_805 = __shfl_xor_sync(0xFFFFFFFF, _mul_39, 4);
        float add_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_18) : "f"(_mul_39), "f"(_shfl_xor_sync_805));
        float _shfl_xor_sync_808 = __shfl_xor_sync(0xFFFFFFFF, add_18, 24);
        float _add_807;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_807) : "f"(add_18), "f"(_shfl_xor_sync_808));
        float _sub_806;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_806) : "f"(g_36), "f"(_add_807));
        float _fdiv_rn_80 = __fdiv_rn(_sub_806, _add_197);
        float _shfl_xor_sync_809 = __shfl_xor_sync(0xFFFFFFFF, add_2_6, 4);
        float add_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_19) : "f"(add_2_6), "f"(_shfl_xor_sync_809));
        float _shfl_xor_sync_810 = __shfl_xor_sync(0xFFFFFFFF, add_19, 2);
        float add_1_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_14) : "f"(add_19), "f"(_shfl_xor_sync_810));
        float _shfl_xor_sync_812 = __shfl_xor_sync(0xFFFFFFFF, add_1_14, 1);
        float _add_811;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_811) : "f"(add_1_14), "f"(_shfl_xor_sync_812));
        dpre_part[own_base + 224] = _add_811;
        float acc_162 = 0.0f;
        float _shfl_xor_sync_813 = __shfl_xor_sync(0xFFFFFFFF, hist[0], 2);
        float _shfl_xor_sync_815 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_80, 2);
        float _sub_814;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_814) : "f"(_fdiv_rn_80), "f"(_shfl_xor_sync_815));
        float _shfl_xor_sync_817 = __shfl_xor_sync(0xFFFFFFFF, hist[0], 1);
        float _shfl_xor_sync_819 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_80, 1);
        float _sub_818;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_818) : "f"(_fdiv_rn_80), "f"(_shfl_xor_sync_819));
        float _fma_816;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_816) : "f"(_shfl_xor_sync_817), "f"(_sub_818), "f"(acc_162));
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(_shfl_xor_sync_813), "f"(_sub_814), "f"(_fma_816));
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, hist[0], 3);
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_80, 3);
        float _sub_821;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_821) : "f"(_fdiv_rn_80), "f"(_shfl_xor_1));
        float _fma_820;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_820) : "f"(_shfl_xor_0), "f"(_sub_821), "f"(_fma_0));
        float da = ((lane_1 >= 8 && lane_1 < 24) ? mul_f32_rn_noftz(hist[0], _fma_820) : mul_f32_rn_noftz(1.0f, dsig));
        __syncthreads();
        dp[0] = dpre_part[cur_base + lc4];
        float _add_822;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_822) : "f"(dp[0]), "f"(dpre_part[cur_base + 4 + lc4]));
        dp[0] = _add_822;
        float _add_823;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_823) : "f"(dp[0]), "f"(dpre_part[cur_base + 8 + lc4]));
        dp[0] = _add_823;
        float _add_824;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_824) : "f"(dp[0]), "f"(dpre_part[cur_base + 12 + lc4]));
        dp[0] = _add_824;
        float _add_825;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_825) : "f"(dp[0]), "f"(dpre_part[cur_base + 16 + lc4]));
        dp[0] = _add_825;
        float _add_826;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_826) : "f"(dp[0]), "f"(dpre_part[cur_base + 20 + lc4]));
        dp[0] = _add_826;
        float _add_827;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_827) : "f"(dp[0]), "f"(dpre_part[cur_base + 24 + lc4]));
        dp[0] = _add_827;
        float _add_828;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_828) : "f"(dp[0]), "f"(dpre_part[cur_base + 28 + lc4]));
        dp[0] = _add_828;
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
        float _sub_829;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_829) : "f"(sub), "f"(cvt_f32_1));
        float cvt_f32_2 = (float)(__nv_bfloat16)_sub_829;
        float p1 = ((lane_1 < 24) ? cvt_f32 : 0.0f);
        float p2 = ((lane_1 < 24) ? cvt_f32_1 : 0.0f);
        float p3 = ((lane_1 < 24) ? cvt_f32_2 : 0.0f);
        long long drow = tok64 * 96ll + lane64;
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow])), "h"(__bfloat16_as_ushort((__nv_bfloat16)p1)), "r"((unsigned int)(tok < t1)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow + 32ll])), "h"(__bfloat16_as_ushort((__nv_bfloat16)p2)), "r"((unsigned int)(tok < t1)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow + 64ll])), "h"(__bfloat16_as_ushort((__nv_bfloat16)p3)), "r"((unsigned int)(tok < t1)) : "memory");
        long long trow = tok64 * 24ll;
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow])), "f"(r), "r"((unsigned int)(tok < t1 && lane_1 == 0)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + 1ll])), "f"(c_t), "r"((unsigned int)(tok < t1 && lane_1 == 0)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + 2ll + lane64])), "f"(_add_1), "r"((unsigned int)(tok < t1 && lane_1 < 4)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + lane64 - 2ll])), "f"(c_37), "r"((unsigned int)(tok < t1 && (lane_1 >= 8 && lane_1 < 24))) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + lane64 - 2ll])), "f"(0.0f), "r"((unsigned int)(tok < t1 && (lane_1 >= 24 && lane_1 < 26))) : "memory");
        float _sub_830;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_830) : "f"(0.0f), "f"(daz));
        float dazl;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dazl) : "f"(da_0), "f"(z), "f"(_sub_830));
        gfold[(gi[0] & 1) * 768 + warp_0 * 96 + lane_1] = ((tok < t1) ? da_0 : 0.0f);
        gfold[(gi[0] & 1) * 768 + warp_0 * 96 + 32 + lane_1] = ((tok < t1) ? daz : 0.0f);
        gfold[(gi[0] & 1) * 768 + warp_0 * 96 + 64 + lane_1] = ((tok < t1) ? dazl : 0.0f);
        if (gi[0] > 0) {
            if (warp_0 == (gi[0] & 7)) {
                float o[2];
                float oc[2];
                float ol[1];
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    o[e] = 0.0f;
                    oc[e] = 0.0f;
                }
                ol[0] = 0.0f;
                #pragma unroll
                for (int w = 0; w < 8; w++) {
                    int src = (gi[0] - 1 & 1) * 768 + w * 96 + lane_1;
                    float sub_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_0) : "f"(gfold[src]), "f"(oc[0]));
                    float add_20;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_20) : "f"(o[0]), "f"(sub_0));
                    float _sub_832;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_832) : "f"(add_20), "f"(o[0]));
                    float _sub_831;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_831) : "f"(_sub_832), "f"(sub_0));
                    oc[0] = _sub_831;
                    o[0] = add_20;
                    float sub_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1) : "f"(gfold[src + 32]), "f"(oc[1]));
                    float add_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_21) : "f"(o[1]), "f"(sub_1));
                    float _sub_834;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_834) : "f"(add_21), "f"(o[1]));
                    float _sub_833;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_833) : "f"(_sub_834), "f"(sub_1));
                    oc[1] = _sub_833;
                    o[1] = add_21;
                    float _add_835;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_835) : "f"(ol[0]), "f"(gfold[src + 64]));
                    ol[0] = _add_835;
                }
                float dbase_n;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(dbase_n) : "f"(o[0]), "f"(oc[0]));
                float _sub_836;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_836) : "f"(o[0]), "f"(dbase_n));
                float dbase_l;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(dbase_l) : "f"(_sub_836), "f"(oc[0]));
                float zc;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(zc) : "f"(oc[1]), "f"(ol[0]));
                float zh;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(zh) : "f"(o[1]), "f"(zc));
                gsumv[lane_1] = zh;
                float _sub_838;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_838) : "f"(o[1]), "f"(zh));
                float _sub_837;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_837) : "f"(_sub_838), "f"(zc));
                gsumv[32 + lane_1] = _sub_837;
                __syncwarp();
                int gj = ((lane_1 < 27) ? lane_1 - 24 : lane_1 - 27);
                int glo = ((gj < 2) ? gj * 4 : 8);
                int ghi = ((gj < 2) ? gj * 4 + 4 : 24);
                float ga[1];
                float gc_0[1];
                float gl[1];
                ga[0] = 0.0f;
                gc_0[0] = 0.0f;
                gl[0] = 0.0f;
                if (lane_1 >= 24 && lane_1 < 30) {
                    #pragma unroll
                    for (int m_37 = 0; m_37 < 16; m_37++) {
                        if (glo + m_37 < ghi) {
                            float sub_2;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_2) : "f"(gsumv[glo + m_37]), "f"(gc_0[0]));
                            float add_22;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_22) : "f"(ga[0]), "f"(sub_2));
                            float _sub_840;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_840) : "f"(add_22), "f"(ga[0]));
                            float _sub_839;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_839) : "f"(_sub_840), "f"(sub_2));
                            gc_0[0] = _sub_839;
                            ga[0] = add_22;
                            float _add_841;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_841) : "f"(gl[0]), "f"(gsumv[32 + glo + m_37]));
                            gl[0] = _add_841;
                        }
                    }
                }
                float gcl;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(gcl) : "f"(gc_0[0]), "f"(gl[0]));
                float gval_0;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(gval_0) : "f"(ga[0]), "f"(gcl));
                float _sub_842;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_842) : "f"(ga[0]), "f"(gval_0));
                float glow;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(glow) : "f"(_sub_842), "f"(gcl));
                float v_37 = ((lane_1 < 24) ? dbase_n : ((lane_1 < 27) ? gval_0 : ((lane_1 < 30) ? glow : 0.0f)));
                *(reinterpret_cast<float*>(param_partial + (lane64 * (long long)n_groups + (long long)pg[0])) + (0)) = v_37;
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&param_partial[(lane64 + 32ll) * (long long)n_groups + (long long)pg[0]])), "f"(dbase_l), "r"((unsigned int)(lane_1 < 24)) : "memory");
            }
        }
        pg[0] = gcur;
        gi[0] = gi[0] + 1;
        ring_grp += 1;
        gc[0] = gtk;
        tc += 1;
        if (tc == 2) { tc = 0; _phase_tk_full ^= 1; }
        if (gtk >= g_end) {
            break;
        }
    }
    if (lo < hi) {
        mbarrier_wait(ring_bar_addr + (st_a) * 8, 0);
        mbarrier_wait(ring_bar_addr + (st_b) * 8, 0);
    }
    if (lo < hi) {
        __syncthreads();
        if (warp_0 == (gi[0] & 7)) {
            float o_1[2];
            float oc_1[2];
            float ol_1[1];
            #pragma unroll
            for (int e_1 = 0; e_1 < 2; e_1++) {
                o_1[e_1] = 0.0f;
                oc_1[e_1] = 0.0f;
            }
            ol_1[0] = 0.0f;
            #pragma unroll
            for (int w_1 = 0; w_1 < 8; w_1++) {
                int src_1 = (gi[0] - 1 & 1) * 768 + w_1 * 96 + lane_1;
                float sub_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_3) : "f"(gfold[src_1]), "f"(oc_1[0]));
                float add_23;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_23) : "f"(o_1[0]), "f"(sub_3));
                float _sub_844;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_844) : "f"(add_23), "f"(o_1[0]));
                float _sub_843;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_843) : "f"(_sub_844), "f"(sub_3));
                oc_1[0] = _sub_843;
                o_1[0] = add_23;
                float sub_3_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_3_1) : "f"(gfold[src_1 + 32]), "f"(oc_1[1]));
                float add_23_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_23_1) : "f"(o_1[1]), "f"(sub_3_1));
                float _sub_846;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_846) : "f"(add_23_1), "f"(o_1[1]));
                float _sub_845;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_845) : "f"(_sub_846), "f"(sub_3_1));
                oc_1[1] = _sub_845;
                o_1[1] = add_23_1;
                float _add_847;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_847) : "f"(ol_1[0]), "f"(gfold[src_1 + 64]));
                ol_1[0] = _add_847;
            }
            float dbase_n_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(dbase_n_1) : "f"(o_1[0]), "f"(oc_1[0]));
            float _sub_848;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_848) : "f"(o_1[0]), "f"(dbase_n_1));
            float dbase_l_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(dbase_l_1) : "f"(_sub_848), "f"(oc_1[0]));
            float zc_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(zc_1) : "f"(oc_1[1]), "f"(ol_1[0]));
            float zh_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(zh_1) : "f"(o_1[1]), "f"(zc_1));
            gsumv[lane_1] = zh_1;
            float _sub_850;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_850) : "f"(o_1[1]), "f"(zh_1));
            float _sub_849;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_849) : "f"(_sub_850), "f"(zc_1));
            gsumv[32 + lane_1] = _sub_849;
            __syncwarp();
            int gj_1 = ((lane_1 < 27) ? lane_1 - 24 : lane_1 - 27);
            int glo_1 = ((gj_1 < 2) ? gj_1 * 4 : 8);
            int ghi_1 = ((gj_1 < 2) ? gj_1 * 4 + 4 : 24);
            float ga_1[1];
            float gc_1[1];
            float gl_1[1];
            ga_1[0] = 0.0f;
            gc_1[0] = 0.0f;
            gl_1[0] = 0.0f;
            if (lane_1 >= 24 && lane_1 < 30) {
                #pragma unroll
                for (int m_38 = 0; m_38 < 16; m_38++) {
                    if (glo_1 + m_38 < ghi_1) {
                        float sub_4;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_4) : "f"(gsumv[glo_1 + m_38]), "f"(gc_1[0]));
                        float add_24;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_24) : "f"(ga_1[0]), "f"(sub_4));
                        float _sub_852;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_852) : "f"(add_24), "f"(ga_1[0]));
                        float _sub_851;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_851) : "f"(_sub_852), "f"(sub_4));
                        gc_1[0] = _sub_851;
                        ga_1[0] = add_24;
                        float _add_853;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_853) : "f"(gl_1[0]), "f"(gsumv[32 + glo_1 + m_38]));
                        gl_1[0] = _add_853;
                    }
                }
            }
            float gcl_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gcl_1) : "f"(gc_1[0]), "f"(gl_1[0]));
            float gval_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gval_1) : "f"(ga_1[0]), "f"(gcl_1));
            float _sub_854;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_854) : "f"(ga_1[0]), "f"(gval_1));
            float glow_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(glow_1) : "f"(_sub_854), "f"(gcl_1));
            float v_38 = ((lane_1 < 24) ? dbase_n_1 : ((lane_1 < 27) ? gval_1 : ((lane_1 < 30) ? glow_1 : 0.0f)));
            *(reinterpret_cast<float*>(param_partial + (lane64 * (long long)n_groups + (long long)pg[0])) + (0)) = v_38;
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&param_partial[(lane64 + 32ll) * (long long)n_groups + (long long)pg[0]])), "f"(dbase_l_1), "r"((unsigned int)(lane_1 < 24)) : "memory");
        }
    }
    if (own_last[0] != 0) {
        float zero = 0.0f;
        int npad = t_pad - T;
        long long t64 = (long long)T;
        int nz = npad * 96;
        int trips_z = (nz + 255) / 256;
        #pragma unroll 1
        for (int k = 0; k < trips_z; k++) {
            int ez = k * 256 + tid_2;
            if (ez < nz) {
                *(reinterpret_cast<__nv_bfloat16*>(dzp + (t64 * 96ll + (long long)ez)) + (0)) = (__nv_bfloat16)zero;
            }
        }
        int nt = npad * 24;
        int trips_t = (nt + 255) / 256;
        #pragma unroll 1
        for (int k_1 = 0; k_1 < trips_t; k_1++) {
            int et = k_1 * 256 + tid_2;
            if (et < nt) {
                *(reinterpret_cast<float*>(tokp + (t64 * 24ll + (long long)et)) + (0)) = zero;
            }
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
