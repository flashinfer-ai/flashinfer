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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

template <typename T, int N>
struct CakeParamArray {
    T v[N];
    __device__ __forceinline__ const T& operator[](int i) const { return v[i]; }
};

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_K_SMEM_OFF 17408
#define SMEM_K_SMEM_STAGE_BYTES 16384
#define SMEM_K_SMEM_STRIDE 16384
#define SMEM_VT_SMEM_OFF 115712
#define SMEM_VT_SMEM_STAGE_BYTES 16384
#define SMEM_VT_SMEM_STRIDE 16384
#define SMEM_TOTAL 214016
#define THREADS 128

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

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
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


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_cake_vsa_sm90_d8fb281dbbcd7358026e(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, __nv_bfloat16* __restrict__ O, const CakeParamArray<int16_t, 1750> plan, int seqlen_q, int seqlen_k, float scale_log2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define k_full0_addr (mbar_base + 8)
    #define k_full1_addr (mbar_base + 16)
    #define k_full2_addr (mbar_base + 24)
    #define k_full3_addr (mbar_base + 32)
    #define k_full4_addr (mbar_base + 40)
    #define k_full5_addr (mbar_base + 48)
    #define v_full0_addr (mbar_base + 56)
    #define v_full1_addr (mbar_base + 64)
    #define v_full2_addr (mbar_base + 72)
    #define v_full3_addr (mbar_base + 80)
    #define v_full4_addr (mbar_base + 88)
    #define v_full5_addr (mbar_base + 96)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int k_smem_addr = smem + 17408;
    __nv_bfloat16* vt_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int vt_smem_addr = smem + 115712;

    // Mbarrier init (13 pipeline groups, 0 ordered-sequence groups, 13 barriers)
    // Mbarriers at smem_raw[0..104)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // k_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // k_full1: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // k_full2: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // k_full3: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // k_full4: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // k_full5: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // v_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // v_full1: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // v_full2: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // v_full3: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // v_full4: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // v_full5: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    int tile = bid;
    int mb = seqlen_q / 64;
    int head = tile / mb;
    int qb = tile - head * mb;
    int plan_base = tile * 7;
    int cnt = plan[plan_base];
    int q_row = head * seqlen_q + qb * 64;
    int kv_base = head * seqlen_k;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(q_full_addr, 16384);
            tma_3d_gmem2smem(q_smem_addr, (&Q), 0, q_row, 0, q_full_addr);
            if (cnt > 0) {
                int blk = plan[plan_base + 1];
                int blk_row = kv_base + blk * 64;
                mbarrier_arrive_expect_tx(k_full0_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr, (&K), 0, blk_row, 0, k_full0_addr);
                mbarrier_arrive_expect_tx(v_full0_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr, (&Vt), 0, 0, blk_row / 8, 0, v_full0_addr);
            }
            if (cnt > 1) {
                int blk_1 = plan[plan_base + 1 + 1];
                int blk_row_1 = kv_base + blk_1 * 64;
                mbarrier_arrive_expect_tx(k_full1_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr + 16384, (&K), 0, blk_row_1, 0, k_full1_addr);
                mbarrier_arrive_expect_tx(v_full1_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr + 16384, (&Vt), 0, 0, blk_row_1 / 8, 0, v_full1_addr);
            }
            if (cnt > 2) {
                int blk_2 = plan[plan_base + 1 + 2];
                int blk_row_2 = kv_base + blk_2 * 64;
                mbarrier_arrive_expect_tx(k_full2_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr + 32768, (&K), 0, blk_row_2, 0, k_full2_addr);
                mbarrier_arrive_expect_tx(v_full2_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr + 32768, (&Vt), 0, 0, blk_row_2 / 8, 0, v_full2_addr);
            }
            if (cnt > 3) {
                int blk_3 = plan[plan_base + 1 + 3];
                int blk_row_3 = kv_base + blk_3 * 64;
                mbarrier_arrive_expect_tx(k_full3_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr + 49152, (&K), 0, blk_row_3, 0, k_full3_addr);
                mbarrier_arrive_expect_tx(v_full3_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr + 49152, (&Vt), 0, 0, blk_row_3 / 8, 0, v_full3_addr);
            }
            if (cnt > 4) {
                int blk_4 = plan[plan_base + 1 + 4];
                int blk_row_4 = kv_base + blk_4 * 64;
                mbarrier_arrive_expect_tx(k_full4_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr + 65536, (&K), 0, blk_row_4, 0, k_full4_addr);
                mbarrier_arrive_expect_tx(v_full4_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr + 65536, (&Vt), 0, 0, blk_row_4 / 8, 0, v_full4_addr);
            }
            if (cnt > 5) {
                int blk_5 = plan[plan_base + 1 + 5];
                int blk_row_5 = kv_base + blk_5 * 64;
                mbarrier_arrive_expect_tx(k_full5_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr + 81920, (&K), 0, blk_row_5, 0, k_full5_addr);
                mbarrier_arrive_expect_tx(v_full5_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr + 81920, (&Vt), 0, 0, blk_row_5 / 8, 0, v_full5_addr);
            }
        }
    }
    int m0_local = warp * 16 + lane / 4;
    int m1_local = m0_local + 8;
    float d_o[64];
    float d_qk[32];
    unsigned int p_bf16[16];
    float row_max0 = -CAKE_INF;
    float row_max1 = -CAKE_INF;
    float row_sum0 = 0.0f;
    float row_sum1 = 0.0f;
    unsigned int _phase_q_full_0 = 0;
    mbarrier_wait(q_full_addr, _phase_q_full_0);
    _phase_q_full_0 ^= 1;
    unsigned int _phase_k_full0_0 = 0;
    uint64_t _wgmma_desc_0 = (((uint64_t)(((q_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
    uint64_t _wgmma_desc_1 = (((uint64_t)(((k_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
    uint64_t _wgmma_desc_2 = (((uint64_t)(((q_smem_addr + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
    uint64_t _wgmma_desc_3 = (((uint64_t)(((k_smem_addr + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
    unsigned int _phase_v_full0_0 = 0;
    uint64_t _wgmma_desc_4 = (((uint64_t)(((vt_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
    if (cnt > 0) {
        mbarrier_wait(k_full0_addr, _phase_k_full0_0);
        _phase_k_full0_0 ^= 1;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_1 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_1 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_1 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_3)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 2), "l"(_wgmma_b_0_3 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 4), "l"(_wgmma_b_0_3 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 6), "l"(_wgmma_b_0_3 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        d_qk[0] = d_qk[0] * scale_log2;
        d_qk[1] = d_qk[1] * scale_log2;
        d_qk[2] = d_qk[2] * scale_log2;
        d_qk[3] = d_qk[3] * scale_log2;
        d_qk[4] = d_qk[4] * scale_log2;
        d_qk[5] = d_qk[5] * scale_log2;
        d_qk[6] = d_qk[6] * scale_log2;
        d_qk[7] = d_qk[7] * scale_log2;
        d_qk[8] = d_qk[8] * scale_log2;
        d_qk[9] = d_qk[9] * scale_log2;
        d_qk[10] = d_qk[10] * scale_log2;
        d_qk[11] = d_qk[11] * scale_log2;
        d_qk[12] = d_qk[12] * scale_log2;
        d_qk[13] = d_qk[13] * scale_log2;
        d_qk[14] = d_qk[14] * scale_log2;
        d_qk[15] = d_qk[15] * scale_log2;
        d_qk[16] = d_qk[16] * scale_log2;
        d_qk[17] = d_qk[17] * scale_log2;
        d_qk[18] = d_qk[18] * scale_log2;
        d_qk[19] = d_qk[19] * scale_log2;
        d_qk[20] = d_qk[20] * scale_log2;
        d_qk[21] = d_qk[21] * scale_log2;
        d_qk[22] = d_qk[22] * scale_log2;
        d_qk[23] = d_qk[23] * scale_log2;
        d_qk[24] = d_qk[24] * scale_log2;
        d_qk[25] = d_qk[25] * scale_log2;
        d_qk[26] = d_qk[26] * scale_log2;
        d_qk[27] = d_qk[27] * scale_log2;
        d_qk[28] = d_qk[28] * scale_log2;
        d_qk[29] = d_qk[29] * scale_log2;
        d_qk[30] = d_qk[30] * scale_log2;
        d_qk[31] = d_qk[31] * scale_log2;
        float new_max0 = -CAKE_INF;
        float new_max1 = -CAKE_INF;
        float _max_0 = max_noftz(new_max0, d_qk[0]);
        new_max0 = _max_0;
        float _max_1 = max_noftz(new_max0, d_qk[1]);
        new_max0 = _max_1;
        float _max_2 = max_noftz(new_max0, d_qk[4]);
        new_max0 = _max_2;
        float _max_3 = max_noftz(new_max0, d_qk[5]);
        new_max0 = _max_3;
        float _max_4 = max_noftz(new_max0, d_qk[8]);
        new_max0 = _max_4;
        float _max_5 = max_noftz(new_max0, d_qk[9]);
        new_max0 = _max_5;
        float _max_6 = max_noftz(new_max0, d_qk[12]);
        new_max0 = _max_6;
        float _max_7 = max_noftz(new_max0, d_qk[13]);
        new_max0 = _max_7;
        float _max_8 = max_noftz(new_max0, d_qk[16]);
        new_max0 = _max_8;
        float _max_9 = max_noftz(new_max0, d_qk[17]);
        new_max0 = _max_9;
        float _max_10 = max_noftz(new_max0, d_qk[20]);
        new_max0 = _max_10;
        float _max_11 = max_noftz(new_max0, d_qk[21]);
        new_max0 = _max_11;
        float _max_12 = max_noftz(new_max0, d_qk[24]);
        new_max0 = _max_12;
        float _max_13 = max_noftz(new_max0, d_qk[25]);
        new_max0 = _max_13;
        float _max_14 = max_noftz(new_max0, d_qk[28]);
        new_max0 = _max_14;
        float _max_15 = max_noftz(new_max0, d_qk[29]);
        new_max0 = _max_15;
        float _max_16 = max_noftz(new_max1, d_qk[2]);
        new_max1 = _max_16;
        float _max_17 = max_noftz(new_max1, d_qk[3]);
        new_max1 = _max_17;
        float _max_18 = max_noftz(new_max1, d_qk[6]);
        new_max1 = _max_18;
        float _max_19 = max_noftz(new_max1, d_qk[7]);
        new_max1 = _max_19;
        float _max_20 = max_noftz(new_max1, d_qk[10]);
        new_max1 = _max_20;
        float _max_21 = max_noftz(new_max1, d_qk[11]);
        new_max1 = _max_21;
        float _max_22 = max_noftz(new_max1, d_qk[14]);
        new_max1 = _max_22;
        float _max_23 = max_noftz(new_max1, d_qk[15]);
        new_max1 = _max_23;
        float _max_24 = max_noftz(new_max1, d_qk[18]);
        new_max1 = _max_24;
        float _max_25 = max_noftz(new_max1, d_qk[19]);
        new_max1 = _max_25;
        float _max_26 = max_noftz(new_max1, d_qk[22]);
        new_max1 = _max_26;
        float _max_27 = max_noftz(new_max1, d_qk[23]);
        new_max1 = _max_27;
        float _max_28 = max_noftz(new_max1, d_qk[26]);
        new_max1 = _max_28;
        float _max_29 = max_noftz(new_max1, d_qk[27]);
        new_max1 = _max_29;
        float _max_30 = max_noftz(new_max1, d_qk[30]);
        new_max1 = _max_30;
        float _max_31 = max_noftz(new_max1, d_qk[31]);
        new_max1 = _max_31;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, new_max0, 2);
        float _max_32 = max_noftz(new_max0, _shfl_xor_0);
        new_max0 = _max_32;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, new_max0, 1);
        float _max_33 = max_noftz(new_max0, _shfl_xor_1);
        new_max0 = _max_33;
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, new_max1, 2);
        float _max_34 = max_noftz(new_max1, _shfl_xor_2);
        new_max1 = _max_34;
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, new_max1, 1);
        float _max_35 = max_noftz(new_max1, _shfl_xor_3);
        new_max1 = _max_35;
        {
            row_max0 = new_max0;
            row_max1 = new_max1;
        }
        float _exp2_2 = approx_exp2(d_qk[0] - row_max0);
        float _exp2_3 = approx_exp2(d_qk[1] - row_max0);
        d_qk[0] = _exp2_2;
        d_qk[1] = _exp2_3;
        row_sum0 += _exp2_2 + _exp2_3;
        float _exp2_4 = approx_exp2(d_qk[2] - row_max1);
        float _exp2_5 = approx_exp2(d_qk[3] - row_max1);
        d_qk[2] = _exp2_4;
        d_qk[3] = _exp2_5;
        row_sum1 += _exp2_4 + _exp2_5;
        float _exp2_6 = approx_exp2(d_qk[4] - row_max0);
        float _exp2_7 = approx_exp2(d_qk[5] - row_max0);
        d_qk[4] = _exp2_6;
        d_qk[5] = _exp2_7;
        row_sum0 += _exp2_6 + _exp2_7;
        float _exp2_8 = approx_exp2(d_qk[6] - row_max1);
        float _exp2_9 = approx_exp2(d_qk[7] - row_max1);
        d_qk[6] = _exp2_8;
        d_qk[7] = _exp2_9;
        row_sum1 += _exp2_8 + _exp2_9;
        float _exp2_10 = approx_exp2(d_qk[8] - row_max0);
        float _exp2_11 = approx_exp2(d_qk[9] - row_max0);
        d_qk[8] = _exp2_10;
        d_qk[9] = _exp2_11;
        row_sum0 += _exp2_10 + _exp2_11;
        float _exp2_12 = approx_exp2(d_qk[10] - row_max1);
        float _exp2_13 = approx_exp2(d_qk[11] - row_max1);
        d_qk[10] = _exp2_12;
        d_qk[11] = _exp2_13;
        row_sum1 += _exp2_12 + _exp2_13;
        float _exp2_14 = approx_exp2(d_qk[12] - row_max0);
        float _exp2_15 = approx_exp2(d_qk[13] - row_max0);
        d_qk[12] = _exp2_14;
        d_qk[13] = _exp2_15;
        row_sum0 += _exp2_14 + _exp2_15;
        float _exp2_16 = approx_exp2(d_qk[14] - row_max1);
        float _exp2_17 = approx_exp2(d_qk[15] - row_max1);
        d_qk[14] = _exp2_16;
        d_qk[15] = _exp2_17;
        row_sum1 += _exp2_16 + _exp2_17;
        float _exp2_18 = approx_exp2(d_qk[16] - row_max0);
        float _exp2_19 = approx_exp2(d_qk[17] - row_max0);
        d_qk[16] = _exp2_18;
        d_qk[17] = _exp2_19;
        row_sum0 += _exp2_18 + _exp2_19;
        float _exp2_20 = approx_exp2(d_qk[18] - row_max1);
        float _exp2_21 = approx_exp2(d_qk[19] - row_max1);
        d_qk[18] = _exp2_20;
        d_qk[19] = _exp2_21;
        row_sum1 += _exp2_20 + _exp2_21;
        float _exp2_22 = approx_exp2(d_qk[20] - row_max0);
        float _exp2_23 = approx_exp2(d_qk[21] - row_max0);
        d_qk[20] = _exp2_22;
        d_qk[21] = _exp2_23;
        row_sum0 += _exp2_22 + _exp2_23;
        float _exp2_24 = approx_exp2(d_qk[22] - row_max1);
        float _exp2_25 = approx_exp2(d_qk[23] - row_max1);
        d_qk[22] = _exp2_24;
        d_qk[23] = _exp2_25;
        row_sum1 += _exp2_24 + _exp2_25;
        float _exp2_26 = approx_exp2(d_qk[24] - row_max0);
        float _exp2_27 = approx_exp2(d_qk[25] - row_max0);
        d_qk[24] = _exp2_26;
        d_qk[25] = _exp2_27;
        row_sum0 += _exp2_26 + _exp2_27;
        float _exp2_28 = approx_exp2(d_qk[26] - row_max1);
        float _exp2_29 = approx_exp2(d_qk[27] - row_max1);
        d_qk[26] = _exp2_28;
        d_qk[27] = _exp2_29;
        row_sum1 += _exp2_28 + _exp2_29;
        float _exp2_30 = approx_exp2(d_qk[28] - row_max0);
        float _exp2_31 = approx_exp2(d_qk[29] - row_max0);
        d_qk[28] = _exp2_30;
        d_qk[29] = _exp2_31;
        row_sum0 += _exp2_30 + _exp2_31;
        float _exp2_32 = approx_exp2(d_qk[30] - row_max1);
        float _exp2_33 = approx_exp2(d_qk[31] - row_max1);
        d_qk[30] = _exp2_32;
        d_qk[31] = _exp2_33;
        row_sum1 += _exp2_32 + _exp2_33;
        __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bf16x2_0)[0];
        __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bf16x2_1)[0];
        __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bf16x2_2)[0];
        __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bf16x2_3)[0];
        __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bf16x2_4)[0];
        __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bf16x2_5)[0];
        __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bf16x2_6)[0];
        __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bf16x2_7)[0];
        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bf16x2_8)[0];
        __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bf16x2_9)[0];
        __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bf16x2_10)[0];
        __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bf16x2_11)[0];
        __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bf16x2_12)[0];
        __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bf16x2_13)[0];
        __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bf16x2_14)[0];
        __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bf16x2_15)[0];
        mbarrier_wait(v_full0_addr, _phase_v_full0_0);
        _phase_v_full0_0 ^= 1;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_b_0_4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_b_0_4 + 128)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_b_0_4 + 256)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_b_0_4 + 384)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    unsigned int _phase_k_full1_0 = 0;
    uint64_t _wgmma_desc_5 = (((uint64_t)(((k_smem_addr + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_5 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_5 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_5);
    uint64_t _wgmma_desc_6 = (((uint64_t)(((k_smem_addr + 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_6 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_6 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_6);
    unsigned int _phase_v_full1_0 = 0;
    uint64_t _wgmma_desc_7 = (((uint64_t)(((vt_smem_addr + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_7 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_7 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_7);
    if (cnt > 1) {
        mbarrier_wait(k_full1_addr, _phase_k_full1_0);
        _phase_k_full1_0 ^= 1;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_5)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_5 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_5 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_5 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 2), "l"(_wgmma_b_0_6 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 4), "l"(_wgmma_b_0_6 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 6), "l"(_wgmma_b_0_6 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        d_qk[0] = d_qk[0] * scale_log2;
        d_qk[1] = d_qk[1] * scale_log2;
        d_qk[2] = d_qk[2] * scale_log2;
        d_qk[3] = d_qk[3] * scale_log2;
        d_qk[4] = d_qk[4] * scale_log2;
        d_qk[5] = d_qk[5] * scale_log2;
        d_qk[6] = d_qk[6] * scale_log2;
        d_qk[7] = d_qk[7] * scale_log2;
        d_qk[8] = d_qk[8] * scale_log2;
        d_qk[9] = d_qk[9] * scale_log2;
        d_qk[10] = d_qk[10] * scale_log2;
        d_qk[11] = d_qk[11] * scale_log2;
        d_qk[12] = d_qk[12] * scale_log2;
        d_qk[13] = d_qk[13] * scale_log2;
        d_qk[14] = d_qk[14] * scale_log2;
        d_qk[15] = d_qk[15] * scale_log2;
        d_qk[16] = d_qk[16] * scale_log2;
        d_qk[17] = d_qk[17] * scale_log2;
        d_qk[18] = d_qk[18] * scale_log2;
        d_qk[19] = d_qk[19] * scale_log2;
        d_qk[20] = d_qk[20] * scale_log2;
        d_qk[21] = d_qk[21] * scale_log2;
        d_qk[22] = d_qk[22] * scale_log2;
        d_qk[23] = d_qk[23] * scale_log2;
        d_qk[24] = d_qk[24] * scale_log2;
        d_qk[25] = d_qk[25] * scale_log2;
        d_qk[26] = d_qk[26] * scale_log2;
        d_qk[27] = d_qk[27] * scale_log2;
        d_qk[28] = d_qk[28] * scale_log2;
        d_qk[29] = d_qk[29] * scale_log2;
        d_qk[30] = d_qk[30] * scale_log2;
        d_qk[31] = d_qk[31] * scale_log2;
        float new_max0_1 = -CAKE_INF;
        float new_max1_1 = -CAKE_INF;
        float _max_38 = max_noftz(new_max0_1, d_qk[0]);
        new_max0_1 = _max_38;
        float _max_39 = max_noftz(new_max0_1, d_qk[1]);
        new_max0_1 = _max_39;
        float _max_40 = max_noftz(new_max0_1, d_qk[4]);
        new_max0_1 = _max_40;
        float _max_41 = max_noftz(new_max0_1, d_qk[5]);
        new_max0_1 = _max_41;
        float _max_42 = max_noftz(new_max0_1, d_qk[8]);
        new_max0_1 = _max_42;
        float _max_43 = max_noftz(new_max0_1, d_qk[9]);
        new_max0_1 = _max_43;
        float _max_44 = max_noftz(new_max0_1, d_qk[12]);
        new_max0_1 = _max_44;
        float _max_45 = max_noftz(new_max0_1, d_qk[13]);
        new_max0_1 = _max_45;
        float _max_46 = max_noftz(new_max0_1, d_qk[16]);
        new_max0_1 = _max_46;
        float _max_47 = max_noftz(new_max0_1, d_qk[17]);
        new_max0_1 = _max_47;
        float _max_48 = max_noftz(new_max0_1, d_qk[20]);
        new_max0_1 = _max_48;
        float _max_49 = max_noftz(new_max0_1, d_qk[21]);
        new_max0_1 = _max_49;
        float _max_50 = max_noftz(new_max0_1, d_qk[24]);
        new_max0_1 = _max_50;
        float _max_51 = max_noftz(new_max0_1, d_qk[25]);
        new_max0_1 = _max_51;
        float _max_52 = max_noftz(new_max0_1, d_qk[28]);
        new_max0_1 = _max_52;
        float _max_53 = max_noftz(new_max0_1, d_qk[29]);
        new_max0_1 = _max_53;
        float _max_54 = max_noftz(new_max1_1, d_qk[2]);
        new_max1_1 = _max_54;
        float _max_55 = max_noftz(new_max1_1, d_qk[3]);
        new_max1_1 = _max_55;
        float _max_56 = max_noftz(new_max1_1, d_qk[6]);
        new_max1_1 = _max_56;
        float _max_57 = max_noftz(new_max1_1, d_qk[7]);
        new_max1_1 = _max_57;
        float _max_58 = max_noftz(new_max1_1, d_qk[10]);
        new_max1_1 = _max_58;
        float _max_59 = max_noftz(new_max1_1, d_qk[11]);
        new_max1_1 = _max_59;
        float _max_60 = max_noftz(new_max1_1, d_qk[14]);
        new_max1_1 = _max_60;
        float _max_61 = max_noftz(new_max1_1, d_qk[15]);
        new_max1_1 = _max_61;
        float _max_62 = max_noftz(new_max1_1, d_qk[18]);
        new_max1_1 = _max_62;
        float _max_63 = max_noftz(new_max1_1, d_qk[19]);
        new_max1_1 = _max_63;
        float _max_64 = max_noftz(new_max1_1, d_qk[22]);
        new_max1_1 = _max_64;
        float _max_65 = max_noftz(new_max1_1, d_qk[23]);
        new_max1_1 = _max_65;
        float _max_66 = max_noftz(new_max1_1, d_qk[26]);
        new_max1_1 = _max_66;
        float _max_67 = max_noftz(new_max1_1, d_qk[27]);
        new_max1_1 = _max_67;
        float _max_68 = max_noftz(new_max1_1, d_qk[30]);
        new_max1_1 = _max_68;
        float _max_69 = max_noftz(new_max1_1, d_qk[31]);
        new_max1_1 = _max_69;
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, new_max0_1, 2);
        float _max_70 = max_noftz(new_max0_1, _shfl_xor_4);
        new_max0_1 = _max_70;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, new_max0_1, 1);
        float _max_71 = max_noftz(new_max0_1, _shfl_xor_5);
        new_max0_1 = _max_71;
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, new_max1_1, 2);
        float _max_72 = max_noftz(new_max1_1, _shfl_xor_6);
        new_max1_1 = _max_72;
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, new_max1_1, 1);
        float _max_73 = max_noftz(new_max1_1, _shfl_xor_7);
        new_max1_1 = _max_73;
        {
            float _max_74 = max_noftz(row_max0, new_max0_1);
            float merged_max0 = _max_74;
            float _max_75 = max_noftz(row_max1, new_max1_1);
            float merged_max1 = _max_75;
            float _exp2_34 = approx_exp2(row_max0 - merged_max0);
            float _exp2_35 = approx_exp2(row_max1 - merged_max1);
            d_o[0] = d_o[0] * _exp2_34;
            d_o[1] = d_o[1] * _exp2_34;
            d_o[4] = d_o[4] * _exp2_34;
            d_o[5] = d_o[5] * _exp2_34;
            d_o[8] = d_o[8] * _exp2_34;
            d_o[9] = d_o[9] * _exp2_34;
            d_o[12] = d_o[12] * _exp2_34;
            d_o[13] = d_o[13] * _exp2_34;
            d_o[16] = d_o[16] * _exp2_34;
            d_o[17] = d_o[17] * _exp2_34;
            d_o[20] = d_o[20] * _exp2_34;
            d_o[21] = d_o[21] * _exp2_34;
            d_o[24] = d_o[24] * _exp2_34;
            d_o[25] = d_o[25] * _exp2_34;
            d_o[28] = d_o[28] * _exp2_34;
            d_o[29] = d_o[29] * _exp2_34;
            d_o[32] = d_o[32] * _exp2_34;
            d_o[33] = d_o[33] * _exp2_34;
            d_o[36] = d_o[36] * _exp2_34;
            d_o[37] = d_o[37] * _exp2_34;
            d_o[40] = d_o[40] * _exp2_34;
            d_o[41] = d_o[41] * _exp2_34;
            d_o[44] = d_o[44] * _exp2_34;
            d_o[45] = d_o[45] * _exp2_34;
            d_o[48] = d_o[48] * _exp2_34;
            d_o[49] = d_o[49] * _exp2_34;
            d_o[52] = d_o[52] * _exp2_34;
            d_o[53] = d_o[53] * _exp2_34;
            d_o[56] = d_o[56] * _exp2_34;
            d_o[57] = d_o[57] * _exp2_34;
            d_o[60] = d_o[60] * _exp2_34;
            d_o[61] = d_o[61] * _exp2_34;
            d_o[2] = d_o[2] * _exp2_35;
            d_o[3] = d_o[3] * _exp2_35;
            d_o[6] = d_o[6] * _exp2_35;
            d_o[7] = d_o[7] * _exp2_35;
            d_o[10] = d_o[10] * _exp2_35;
            d_o[11] = d_o[11] * _exp2_35;
            d_o[14] = d_o[14] * _exp2_35;
            d_o[15] = d_o[15] * _exp2_35;
            d_o[18] = d_o[18] * _exp2_35;
            d_o[19] = d_o[19] * _exp2_35;
            d_o[22] = d_o[22] * _exp2_35;
            d_o[23] = d_o[23] * _exp2_35;
            d_o[26] = d_o[26] * _exp2_35;
            d_o[27] = d_o[27] * _exp2_35;
            d_o[30] = d_o[30] * _exp2_35;
            d_o[31] = d_o[31] * _exp2_35;
            d_o[34] = d_o[34] * _exp2_35;
            d_o[35] = d_o[35] * _exp2_35;
            d_o[38] = d_o[38] * _exp2_35;
            d_o[39] = d_o[39] * _exp2_35;
            d_o[42] = d_o[42] * _exp2_35;
            d_o[43] = d_o[43] * _exp2_35;
            d_o[46] = d_o[46] * _exp2_35;
            d_o[47] = d_o[47] * _exp2_35;
            d_o[50] = d_o[50] * _exp2_35;
            d_o[51] = d_o[51] * _exp2_35;
            d_o[54] = d_o[54] * _exp2_35;
            d_o[55] = d_o[55] * _exp2_35;
            d_o[58] = d_o[58] * _exp2_35;
            d_o[59] = d_o[59] * _exp2_35;
            d_o[62] = d_o[62] * _exp2_35;
            d_o[63] = d_o[63] * _exp2_35;
            row_sum0 = row_sum0 * _exp2_34;
            row_sum1 = row_sum1 * _exp2_35;
            row_max0 = merged_max0;
            row_max1 = merged_max1;
        }
        float _exp2_36 = approx_exp2(d_qk[0] - row_max0);
        float _exp2_37 = approx_exp2(d_qk[1] - row_max0);
        d_qk[0] = _exp2_36;
        d_qk[1] = _exp2_37;
        row_sum0 += _exp2_36 + _exp2_37;
        float _exp2_38 = approx_exp2(d_qk[2] - row_max1);
        float _exp2_39 = approx_exp2(d_qk[3] - row_max1);
        d_qk[2] = _exp2_38;
        d_qk[3] = _exp2_39;
        row_sum1 += _exp2_38 + _exp2_39;
        float _exp2_40 = approx_exp2(d_qk[4] - row_max0);
        float _exp2_41 = approx_exp2(d_qk[5] - row_max0);
        d_qk[4] = _exp2_40;
        d_qk[5] = _exp2_41;
        row_sum0 += _exp2_40 + _exp2_41;
        float _exp2_42 = approx_exp2(d_qk[6] - row_max1);
        float _exp2_43 = approx_exp2(d_qk[7] - row_max1);
        d_qk[6] = _exp2_42;
        d_qk[7] = _exp2_43;
        row_sum1 += _exp2_42 + _exp2_43;
        float _exp2_44 = approx_exp2(d_qk[8] - row_max0);
        float _exp2_45 = approx_exp2(d_qk[9] - row_max0);
        d_qk[8] = _exp2_44;
        d_qk[9] = _exp2_45;
        row_sum0 += _exp2_44 + _exp2_45;
        float _exp2_46 = approx_exp2(d_qk[10] - row_max1);
        float _exp2_47 = approx_exp2(d_qk[11] - row_max1);
        d_qk[10] = _exp2_46;
        d_qk[11] = _exp2_47;
        row_sum1 += _exp2_46 + _exp2_47;
        float _exp2_48 = approx_exp2(d_qk[12] - row_max0);
        float _exp2_49 = approx_exp2(d_qk[13] - row_max0);
        d_qk[12] = _exp2_48;
        d_qk[13] = _exp2_49;
        row_sum0 += _exp2_48 + _exp2_49;
        float _exp2_50 = approx_exp2(d_qk[14] - row_max1);
        float _exp2_51 = approx_exp2(d_qk[15] - row_max1);
        d_qk[14] = _exp2_50;
        d_qk[15] = _exp2_51;
        row_sum1 += _exp2_50 + _exp2_51;
        float _exp2_52 = approx_exp2(d_qk[16] - row_max0);
        float _exp2_53 = approx_exp2(d_qk[17] - row_max0);
        d_qk[16] = _exp2_52;
        d_qk[17] = _exp2_53;
        row_sum0 += _exp2_52 + _exp2_53;
        float _exp2_54 = approx_exp2(d_qk[18] - row_max1);
        float _exp2_55 = approx_exp2(d_qk[19] - row_max1);
        d_qk[18] = _exp2_54;
        d_qk[19] = _exp2_55;
        row_sum1 += _exp2_54 + _exp2_55;
        float _exp2_56 = approx_exp2(d_qk[20] - row_max0);
        float _exp2_57 = approx_exp2(d_qk[21] - row_max0);
        d_qk[20] = _exp2_56;
        d_qk[21] = _exp2_57;
        row_sum0 += _exp2_56 + _exp2_57;
        float _exp2_58 = approx_exp2(d_qk[22] - row_max1);
        float _exp2_59 = approx_exp2(d_qk[23] - row_max1);
        d_qk[22] = _exp2_58;
        d_qk[23] = _exp2_59;
        row_sum1 += _exp2_58 + _exp2_59;
        float _exp2_60 = approx_exp2(d_qk[24] - row_max0);
        float _exp2_61 = approx_exp2(d_qk[25] - row_max0);
        d_qk[24] = _exp2_60;
        d_qk[25] = _exp2_61;
        row_sum0 += _exp2_60 + _exp2_61;
        float _exp2_62 = approx_exp2(d_qk[26] - row_max1);
        float _exp2_63 = approx_exp2(d_qk[27] - row_max1);
        d_qk[26] = _exp2_62;
        d_qk[27] = _exp2_63;
        row_sum1 += _exp2_62 + _exp2_63;
        float _exp2_64 = approx_exp2(d_qk[28] - row_max0);
        float _exp2_65 = approx_exp2(d_qk[29] - row_max0);
        d_qk[28] = _exp2_64;
        d_qk[29] = _exp2_65;
        row_sum0 += _exp2_64 + _exp2_65;
        float _exp2_66 = approx_exp2(d_qk[30] - row_max1);
        float _exp2_67 = approx_exp2(d_qk[31] - row_max1);
        d_qk[30] = _exp2_66;
        d_qk[31] = _exp2_67;
        row_sum1 += _exp2_66 + _exp2_67;
        __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bf16x2_16)[0];
        __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bf16x2_17)[0];
        __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bf16x2_18)[0];
        __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bf16x2_19)[0];
        __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bf16x2_20)[0];
        __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bf16x2_21)[0];
        __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bf16x2_22)[0];
        __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bf16x2_23)[0];
        __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bf16x2_24)[0];
        __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bf16x2_25)[0];
        __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bf16x2_26)[0];
        __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bf16x2_27)[0];
        __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bf16x2_28)[0];
        __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bf16x2_29)[0];
        __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bf16x2_30)[0];
        __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bf16x2_31)[0];
        mbarrier_wait(v_full1_addr, _phase_v_full1_0);
        _phase_v_full1_0 ^= 1;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_b_0_7)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_b_0_7 + 128)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_b_0_7 + 256)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_b_0_7 + 384)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    unsigned int _phase_k_full2_0 = 0;
    uint64_t _wgmma_desc_8 = (((uint64_t)(((k_smem_addr + 32768)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_8 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_8 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_8);
    uint64_t _wgmma_desc_9 = (((uint64_t)(((k_smem_addr + 32768 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_9 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_9 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_9);
    unsigned int _phase_v_full2_0 = 0;
    uint64_t _wgmma_desc_10 = (((uint64_t)(((vt_smem_addr + 32768)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_10 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_10 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_10);
    if (cnt > 2) {
        mbarrier_wait(k_full2_addr, _phase_k_full2_0);
        _phase_k_full2_0 ^= 1;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_8)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_8 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_8 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_8 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_9)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 2), "l"(_wgmma_b_0_9 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 4), "l"(_wgmma_b_0_9 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 6), "l"(_wgmma_b_0_9 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        d_qk[0] = d_qk[0] * scale_log2;
        d_qk[1] = d_qk[1] * scale_log2;
        d_qk[2] = d_qk[2] * scale_log2;
        d_qk[3] = d_qk[3] * scale_log2;
        d_qk[4] = d_qk[4] * scale_log2;
        d_qk[5] = d_qk[5] * scale_log2;
        d_qk[6] = d_qk[6] * scale_log2;
        d_qk[7] = d_qk[7] * scale_log2;
        d_qk[8] = d_qk[8] * scale_log2;
        d_qk[9] = d_qk[9] * scale_log2;
        d_qk[10] = d_qk[10] * scale_log2;
        d_qk[11] = d_qk[11] * scale_log2;
        d_qk[12] = d_qk[12] * scale_log2;
        d_qk[13] = d_qk[13] * scale_log2;
        d_qk[14] = d_qk[14] * scale_log2;
        d_qk[15] = d_qk[15] * scale_log2;
        d_qk[16] = d_qk[16] * scale_log2;
        d_qk[17] = d_qk[17] * scale_log2;
        d_qk[18] = d_qk[18] * scale_log2;
        d_qk[19] = d_qk[19] * scale_log2;
        d_qk[20] = d_qk[20] * scale_log2;
        d_qk[21] = d_qk[21] * scale_log2;
        d_qk[22] = d_qk[22] * scale_log2;
        d_qk[23] = d_qk[23] * scale_log2;
        d_qk[24] = d_qk[24] * scale_log2;
        d_qk[25] = d_qk[25] * scale_log2;
        d_qk[26] = d_qk[26] * scale_log2;
        d_qk[27] = d_qk[27] * scale_log2;
        d_qk[28] = d_qk[28] * scale_log2;
        d_qk[29] = d_qk[29] * scale_log2;
        d_qk[30] = d_qk[30] * scale_log2;
        d_qk[31] = d_qk[31] * scale_log2;
        float new_max0_2 = -CAKE_INF;
        float new_max1_2 = -CAKE_INF;
        float _max_76 = max_noftz(new_max0_2, d_qk[0]);
        new_max0_2 = _max_76;
        float _max_77 = max_noftz(new_max0_2, d_qk[1]);
        new_max0_2 = _max_77;
        float _max_78 = max_noftz(new_max0_2, d_qk[4]);
        new_max0_2 = _max_78;
        float _max_79 = max_noftz(new_max0_2, d_qk[5]);
        new_max0_2 = _max_79;
        float _max_80 = max_noftz(new_max0_2, d_qk[8]);
        new_max0_2 = _max_80;
        float _max_81 = max_noftz(new_max0_2, d_qk[9]);
        new_max0_2 = _max_81;
        float _max_82 = max_noftz(new_max0_2, d_qk[12]);
        new_max0_2 = _max_82;
        float _max_83 = max_noftz(new_max0_2, d_qk[13]);
        new_max0_2 = _max_83;
        float _max_84 = max_noftz(new_max0_2, d_qk[16]);
        new_max0_2 = _max_84;
        float _max_85 = max_noftz(new_max0_2, d_qk[17]);
        new_max0_2 = _max_85;
        float _max_86 = max_noftz(new_max0_2, d_qk[20]);
        new_max0_2 = _max_86;
        float _max_87 = max_noftz(new_max0_2, d_qk[21]);
        new_max0_2 = _max_87;
        float _max_88 = max_noftz(new_max0_2, d_qk[24]);
        new_max0_2 = _max_88;
        float _max_89 = max_noftz(new_max0_2, d_qk[25]);
        new_max0_2 = _max_89;
        float _max_90 = max_noftz(new_max0_2, d_qk[28]);
        new_max0_2 = _max_90;
        float _max_91 = max_noftz(new_max0_2, d_qk[29]);
        new_max0_2 = _max_91;
        float _max_92 = max_noftz(new_max1_2, d_qk[2]);
        new_max1_2 = _max_92;
        float _max_93 = max_noftz(new_max1_2, d_qk[3]);
        new_max1_2 = _max_93;
        float _max_94 = max_noftz(new_max1_2, d_qk[6]);
        new_max1_2 = _max_94;
        float _max_95 = max_noftz(new_max1_2, d_qk[7]);
        new_max1_2 = _max_95;
        float _max_96 = max_noftz(new_max1_2, d_qk[10]);
        new_max1_2 = _max_96;
        float _max_97 = max_noftz(new_max1_2, d_qk[11]);
        new_max1_2 = _max_97;
        float _max_98 = max_noftz(new_max1_2, d_qk[14]);
        new_max1_2 = _max_98;
        float _max_99 = max_noftz(new_max1_2, d_qk[15]);
        new_max1_2 = _max_99;
        float _max_100 = max_noftz(new_max1_2, d_qk[18]);
        new_max1_2 = _max_100;
        float _max_101 = max_noftz(new_max1_2, d_qk[19]);
        new_max1_2 = _max_101;
        float _max_102 = max_noftz(new_max1_2, d_qk[22]);
        new_max1_2 = _max_102;
        float _max_103 = max_noftz(new_max1_2, d_qk[23]);
        new_max1_2 = _max_103;
        float _max_104 = max_noftz(new_max1_2, d_qk[26]);
        new_max1_2 = _max_104;
        float _max_105 = max_noftz(new_max1_2, d_qk[27]);
        new_max1_2 = _max_105;
        float _max_106 = max_noftz(new_max1_2, d_qk[30]);
        new_max1_2 = _max_106;
        float _max_107 = max_noftz(new_max1_2, d_qk[31]);
        new_max1_2 = _max_107;
        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, new_max0_2, 2);
        float _max_108 = max_noftz(new_max0_2, _shfl_xor_8);
        new_max0_2 = _max_108;
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, new_max0_2, 1);
        float _max_109 = max_noftz(new_max0_2, _shfl_xor_9);
        new_max0_2 = _max_109;
        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, new_max1_2, 2);
        float _max_110 = max_noftz(new_max1_2, _shfl_xor_10);
        new_max1_2 = _max_110;
        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, new_max1_2, 1);
        float _max_111 = max_noftz(new_max1_2, _shfl_xor_11);
        new_max1_2 = _max_111;
        {
            float _max_112 = max_noftz(row_max0, new_max0_2);
            float merged_max0_1 = _max_112;
            float _max_113 = max_noftz(row_max1, new_max1_2);
            float merged_max1_1 = _max_113;
            float _exp2_68 = approx_exp2(row_max0 - merged_max0_1);
            float _exp2_69 = approx_exp2(row_max1 - merged_max1_1);
            d_o[0] = d_o[0] * _exp2_68;
            d_o[1] = d_o[1] * _exp2_68;
            d_o[4] = d_o[4] * _exp2_68;
            d_o[5] = d_o[5] * _exp2_68;
            d_o[8] = d_o[8] * _exp2_68;
            d_o[9] = d_o[9] * _exp2_68;
            d_o[12] = d_o[12] * _exp2_68;
            d_o[13] = d_o[13] * _exp2_68;
            d_o[16] = d_o[16] * _exp2_68;
            d_o[17] = d_o[17] * _exp2_68;
            d_o[20] = d_o[20] * _exp2_68;
            d_o[21] = d_o[21] * _exp2_68;
            d_o[24] = d_o[24] * _exp2_68;
            d_o[25] = d_o[25] * _exp2_68;
            d_o[28] = d_o[28] * _exp2_68;
            d_o[29] = d_o[29] * _exp2_68;
            d_o[32] = d_o[32] * _exp2_68;
            d_o[33] = d_o[33] * _exp2_68;
            d_o[36] = d_o[36] * _exp2_68;
            d_o[37] = d_o[37] * _exp2_68;
            d_o[40] = d_o[40] * _exp2_68;
            d_o[41] = d_o[41] * _exp2_68;
            d_o[44] = d_o[44] * _exp2_68;
            d_o[45] = d_o[45] * _exp2_68;
            d_o[48] = d_o[48] * _exp2_68;
            d_o[49] = d_o[49] * _exp2_68;
            d_o[52] = d_o[52] * _exp2_68;
            d_o[53] = d_o[53] * _exp2_68;
            d_o[56] = d_o[56] * _exp2_68;
            d_o[57] = d_o[57] * _exp2_68;
            d_o[60] = d_o[60] * _exp2_68;
            d_o[61] = d_o[61] * _exp2_68;
            d_o[2] = d_o[2] * _exp2_69;
            d_o[3] = d_o[3] * _exp2_69;
            d_o[6] = d_o[6] * _exp2_69;
            d_o[7] = d_o[7] * _exp2_69;
            d_o[10] = d_o[10] * _exp2_69;
            d_o[11] = d_o[11] * _exp2_69;
            d_o[14] = d_o[14] * _exp2_69;
            d_o[15] = d_o[15] * _exp2_69;
            d_o[18] = d_o[18] * _exp2_69;
            d_o[19] = d_o[19] * _exp2_69;
            d_o[22] = d_o[22] * _exp2_69;
            d_o[23] = d_o[23] * _exp2_69;
            d_o[26] = d_o[26] * _exp2_69;
            d_o[27] = d_o[27] * _exp2_69;
            d_o[30] = d_o[30] * _exp2_69;
            d_o[31] = d_o[31] * _exp2_69;
            d_o[34] = d_o[34] * _exp2_69;
            d_o[35] = d_o[35] * _exp2_69;
            d_o[38] = d_o[38] * _exp2_69;
            d_o[39] = d_o[39] * _exp2_69;
            d_o[42] = d_o[42] * _exp2_69;
            d_o[43] = d_o[43] * _exp2_69;
            d_o[46] = d_o[46] * _exp2_69;
            d_o[47] = d_o[47] * _exp2_69;
            d_o[50] = d_o[50] * _exp2_69;
            d_o[51] = d_o[51] * _exp2_69;
            d_o[54] = d_o[54] * _exp2_69;
            d_o[55] = d_o[55] * _exp2_69;
            d_o[58] = d_o[58] * _exp2_69;
            d_o[59] = d_o[59] * _exp2_69;
            d_o[62] = d_o[62] * _exp2_69;
            d_o[63] = d_o[63] * _exp2_69;
            row_sum0 = row_sum0 * _exp2_68;
            row_sum1 = row_sum1 * _exp2_69;
            row_max0 = merged_max0_1;
            row_max1 = merged_max1_1;
        }
        float _exp2_70 = approx_exp2(d_qk[0] - row_max0);
        float _exp2_71 = approx_exp2(d_qk[1] - row_max0);
        d_qk[0] = _exp2_70;
        d_qk[1] = _exp2_71;
        row_sum0 += _exp2_70 + _exp2_71;
        float _exp2_72 = approx_exp2(d_qk[2] - row_max1);
        float _exp2_73 = approx_exp2(d_qk[3] - row_max1);
        d_qk[2] = _exp2_72;
        d_qk[3] = _exp2_73;
        row_sum1 += _exp2_72 + _exp2_73;
        float _exp2_74 = approx_exp2(d_qk[4] - row_max0);
        float _exp2_75 = approx_exp2(d_qk[5] - row_max0);
        d_qk[4] = _exp2_74;
        d_qk[5] = _exp2_75;
        row_sum0 += _exp2_74 + _exp2_75;
        float _exp2_76 = approx_exp2(d_qk[6] - row_max1);
        float _exp2_77 = approx_exp2(d_qk[7] - row_max1);
        d_qk[6] = _exp2_76;
        d_qk[7] = _exp2_77;
        row_sum1 += _exp2_76 + _exp2_77;
        float _exp2_78 = approx_exp2(d_qk[8] - row_max0);
        float _exp2_79 = approx_exp2(d_qk[9] - row_max0);
        d_qk[8] = _exp2_78;
        d_qk[9] = _exp2_79;
        row_sum0 += _exp2_78 + _exp2_79;
        float _exp2_80 = approx_exp2(d_qk[10] - row_max1);
        float _exp2_81 = approx_exp2(d_qk[11] - row_max1);
        d_qk[10] = _exp2_80;
        d_qk[11] = _exp2_81;
        row_sum1 += _exp2_80 + _exp2_81;
        float _exp2_82 = approx_exp2(d_qk[12] - row_max0);
        float _exp2_83 = approx_exp2(d_qk[13] - row_max0);
        d_qk[12] = _exp2_82;
        d_qk[13] = _exp2_83;
        row_sum0 += _exp2_82 + _exp2_83;
        float _exp2_84 = approx_exp2(d_qk[14] - row_max1);
        float _exp2_85 = approx_exp2(d_qk[15] - row_max1);
        d_qk[14] = _exp2_84;
        d_qk[15] = _exp2_85;
        row_sum1 += _exp2_84 + _exp2_85;
        float _exp2_86 = approx_exp2(d_qk[16] - row_max0);
        float _exp2_87 = approx_exp2(d_qk[17] - row_max0);
        d_qk[16] = _exp2_86;
        d_qk[17] = _exp2_87;
        row_sum0 += _exp2_86 + _exp2_87;
        float _exp2_88 = approx_exp2(d_qk[18] - row_max1);
        float _exp2_89 = approx_exp2(d_qk[19] - row_max1);
        d_qk[18] = _exp2_88;
        d_qk[19] = _exp2_89;
        row_sum1 += _exp2_88 + _exp2_89;
        float _exp2_90 = approx_exp2(d_qk[20] - row_max0);
        float _exp2_91 = approx_exp2(d_qk[21] - row_max0);
        d_qk[20] = _exp2_90;
        d_qk[21] = _exp2_91;
        row_sum0 += _exp2_90 + _exp2_91;
        float _exp2_92 = approx_exp2(d_qk[22] - row_max1);
        float _exp2_93 = approx_exp2(d_qk[23] - row_max1);
        d_qk[22] = _exp2_92;
        d_qk[23] = _exp2_93;
        row_sum1 += _exp2_92 + _exp2_93;
        float _exp2_94 = approx_exp2(d_qk[24] - row_max0);
        float _exp2_95 = approx_exp2(d_qk[25] - row_max0);
        d_qk[24] = _exp2_94;
        d_qk[25] = _exp2_95;
        row_sum0 += _exp2_94 + _exp2_95;
        float _exp2_96 = approx_exp2(d_qk[26] - row_max1);
        float _exp2_97 = approx_exp2(d_qk[27] - row_max1);
        d_qk[26] = _exp2_96;
        d_qk[27] = _exp2_97;
        row_sum1 += _exp2_96 + _exp2_97;
        float _exp2_98 = approx_exp2(d_qk[28] - row_max0);
        float _exp2_99 = approx_exp2(d_qk[29] - row_max0);
        d_qk[28] = _exp2_98;
        d_qk[29] = _exp2_99;
        row_sum0 += _exp2_98 + _exp2_99;
        float _exp2_100 = approx_exp2(d_qk[30] - row_max1);
        float _exp2_101 = approx_exp2(d_qk[31] - row_max1);
        d_qk[30] = _exp2_100;
        d_qk[31] = _exp2_101;
        row_sum1 += _exp2_100 + _exp2_101;
        __nv_bfloat162 _bf16x2_32 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bf16x2_32)[0];
        __nv_bfloat162 _bf16x2_33 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bf16x2_33)[0];
        __nv_bfloat162 _bf16x2_34 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bf16x2_34)[0];
        __nv_bfloat162 _bf16x2_35 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bf16x2_35)[0];
        __nv_bfloat162 _bf16x2_36 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bf16x2_36)[0];
        __nv_bfloat162 _bf16x2_37 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bf16x2_37)[0];
        __nv_bfloat162 _bf16x2_38 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bf16x2_38)[0];
        __nv_bfloat162 _bf16x2_39 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bf16x2_39)[0];
        __nv_bfloat162 _bf16x2_40 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bf16x2_40)[0];
        __nv_bfloat162 _bf16x2_41 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bf16x2_41)[0];
        __nv_bfloat162 _bf16x2_42 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bf16x2_42)[0];
        __nv_bfloat162 _bf16x2_43 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bf16x2_43)[0];
        __nv_bfloat162 _bf16x2_44 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bf16x2_44)[0];
        __nv_bfloat162 _bf16x2_45 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bf16x2_45)[0];
        __nv_bfloat162 _bf16x2_46 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bf16x2_46)[0];
        __nv_bfloat162 _bf16x2_47 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bf16x2_47)[0];
        mbarrier_wait(v_full2_addr, _phase_v_full2_0);
        _phase_v_full2_0 ^= 1;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_b_0_10)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_b_0_10 + 128)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_b_0_10 + 256)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_b_0_10 + 384)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    unsigned int _phase_k_full3_0 = 0;
    uint64_t _wgmma_desc_11 = (((uint64_t)(((k_smem_addr + 49152)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_11 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_11 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_11);
    uint64_t _wgmma_desc_12 = (((uint64_t)(((k_smem_addr + 49152 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_12 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_12 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_12);
    unsigned int _phase_v_full3_0 = 0;
    uint64_t _wgmma_desc_13 = (((uint64_t)(((vt_smem_addr + 49152)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_13 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_13 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_13);
    if (cnt > 3) {
        mbarrier_wait(k_full3_addr, _phase_k_full3_0);
        _phase_k_full3_0 ^= 1;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_11)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_11 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_11 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_11 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_12)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 2), "l"(_wgmma_b_0_12 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 4), "l"(_wgmma_b_0_12 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 6), "l"(_wgmma_b_0_12 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        d_qk[0] = d_qk[0] * scale_log2;
        d_qk[1] = d_qk[1] * scale_log2;
        d_qk[2] = d_qk[2] * scale_log2;
        d_qk[3] = d_qk[3] * scale_log2;
        d_qk[4] = d_qk[4] * scale_log2;
        d_qk[5] = d_qk[5] * scale_log2;
        d_qk[6] = d_qk[6] * scale_log2;
        d_qk[7] = d_qk[7] * scale_log2;
        d_qk[8] = d_qk[8] * scale_log2;
        d_qk[9] = d_qk[9] * scale_log2;
        d_qk[10] = d_qk[10] * scale_log2;
        d_qk[11] = d_qk[11] * scale_log2;
        d_qk[12] = d_qk[12] * scale_log2;
        d_qk[13] = d_qk[13] * scale_log2;
        d_qk[14] = d_qk[14] * scale_log2;
        d_qk[15] = d_qk[15] * scale_log2;
        d_qk[16] = d_qk[16] * scale_log2;
        d_qk[17] = d_qk[17] * scale_log2;
        d_qk[18] = d_qk[18] * scale_log2;
        d_qk[19] = d_qk[19] * scale_log2;
        d_qk[20] = d_qk[20] * scale_log2;
        d_qk[21] = d_qk[21] * scale_log2;
        d_qk[22] = d_qk[22] * scale_log2;
        d_qk[23] = d_qk[23] * scale_log2;
        d_qk[24] = d_qk[24] * scale_log2;
        d_qk[25] = d_qk[25] * scale_log2;
        d_qk[26] = d_qk[26] * scale_log2;
        d_qk[27] = d_qk[27] * scale_log2;
        d_qk[28] = d_qk[28] * scale_log2;
        d_qk[29] = d_qk[29] * scale_log2;
        d_qk[30] = d_qk[30] * scale_log2;
        d_qk[31] = d_qk[31] * scale_log2;
        float new_max0_3 = -CAKE_INF;
        float new_max1_3 = -CAKE_INF;
        float _max_114 = max_noftz(new_max0_3, d_qk[0]);
        new_max0_3 = _max_114;
        float _max_115 = max_noftz(new_max0_3, d_qk[1]);
        new_max0_3 = _max_115;
        float _max_116 = max_noftz(new_max0_3, d_qk[4]);
        new_max0_3 = _max_116;
        float _max_117 = max_noftz(new_max0_3, d_qk[5]);
        new_max0_3 = _max_117;
        float _max_118 = max_noftz(new_max0_3, d_qk[8]);
        new_max0_3 = _max_118;
        float _max_119 = max_noftz(new_max0_3, d_qk[9]);
        new_max0_3 = _max_119;
        float _max_120 = max_noftz(new_max0_3, d_qk[12]);
        new_max0_3 = _max_120;
        float _max_121 = max_noftz(new_max0_3, d_qk[13]);
        new_max0_3 = _max_121;
        float _max_122 = max_noftz(new_max0_3, d_qk[16]);
        new_max0_3 = _max_122;
        float _max_123 = max_noftz(new_max0_3, d_qk[17]);
        new_max0_3 = _max_123;
        float _max_124 = max_noftz(new_max0_3, d_qk[20]);
        new_max0_3 = _max_124;
        float _max_125 = max_noftz(new_max0_3, d_qk[21]);
        new_max0_3 = _max_125;
        float _max_126 = max_noftz(new_max0_3, d_qk[24]);
        new_max0_3 = _max_126;
        float _max_127 = max_noftz(new_max0_3, d_qk[25]);
        new_max0_3 = _max_127;
        float _max_128 = max_noftz(new_max0_3, d_qk[28]);
        new_max0_3 = _max_128;
        float _max_129 = max_noftz(new_max0_3, d_qk[29]);
        new_max0_3 = _max_129;
        float _max_130 = max_noftz(new_max1_3, d_qk[2]);
        new_max1_3 = _max_130;
        float _max_131 = max_noftz(new_max1_3, d_qk[3]);
        new_max1_3 = _max_131;
        float _max_132 = max_noftz(new_max1_3, d_qk[6]);
        new_max1_3 = _max_132;
        float _max_133 = max_noftz(new_max1_3, d_qk[7]);
        new_max1_3 = _max_133;
        float _max_134 = max_noftz(new_max1_3, d_qk[10]);
        new_max1_3 = _max_134;
        float _max_135 = max_noftz(new_max1_3, d_qk[11]);
        new_max1_3 = _max_135;
        float _max_136 = max_noftz(new_max1_3, d_qk[14]);
        new_max1_3 = _max_136;
        float _max_137 = max_noftz(new_max1_3, d_qk[15]);
        new_max1_3 = _max_137;
        float _max_138 = max_noftz(new_max1_3, d_qk[18]);
        new_max1_3 = _max_138;
        float _max_139 = max_noftz(new_max1_3, d_qk[19]);
        new_max1_3 = _max_139;
        float _max_140 = max_noftz(new_max1_3, d_qk[22]);
        new_max1_3 = _max_140;
        float _max_141 = max_noftz(new_max1_3, d_qk[23]);
        new_max1_3 = _max_141;
        float _max_142 = max_noftz(new_max1_3, d_qk[26]);
        new_max1_3 = _max_142;
        float _max_143 = max_noftz(new_max1_3, d_qk[27]);
        new_max1_3 = _max_143;
        float _max_144 = max_noftz(new_max1_3, d_qk[30]);
        new_max1_3 = _max_144;
        float _max_145 = max_noftz(new_max1_3, d_qk[31]);
        new_max1_3 = _max_145;
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, new_max0_3, 2);
        float _max_146 = max_noftz(new_max0_3, _shfl_xor_12);
        new_max0_3 = _max_146;
        float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, new_max0_3, 1);
        float _max_147 = max_noftz(new_max0_3, _shfl_xor_13);
        new_max0_3 = _max_147;
        float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, new_max1_3, 2);
        float _max_148 = max_noftz(new_max1_3, _shfl_xor_14);
        new_max1_3 = _max_148;
        float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, new_max1_3, 1);
        float _max_149 = max_noftz(new_max1_3, _shfl_xor_15);
        new_max1_3 = _max_149;
        {
            float _max_150 = max_noftz(row_max0, new_max0_3);
            float merged_max0_2 = _max_150;
            float _max_151 = max_noftz(row_max1, new_max1_3);
            float merged_max1_2 = _max_151;
            float _exp2_102 = approx_exp2(row_max0 - merged_max0_2);
            float _exp2_103 = approx_exp2(row_max1 - merged_max1_2);
            d_o[0] = d_o[0] * _exp2_102;
            d_o[1] = d_o[1] * _exp2_102;
            d_o[4] = d_o[4] * _exp2_102;
            d_o[5] = d_o[5] * _exp2_102;
            d_o[8] = d_o[8] * _exp2_102;
            d_o[9] = d_o[9] * _exp2_102;
            d_o[12] = d_o[12] * _exp2_102;
            d_o[13] = d_o[13] * _exp2_102;
            d_o[16] = d_o[16] * _exp2_102;
            d_o[17] = d_o[17] * _exp2_102;
            d_o[20] = d_o[20] * _exp2_102;
            d_o[21] = d_o[21] * _exp2_102;
            d_o[24] = d_o[24] * _exp2_102;
            d_o[25] = d_o[25] * _exp2_102;
            d_o[28] = d_o[28] * _exp2_102;
            d_o[29] = d_o[29] * _exp2_102;
            d_o[32] = d_o[32] * _exp2_102;
            d_o[33] = d_o[33] * _exp2_102;
            d_o[36] = d_o[36] * _exp2_102;
            d_o[37] = d_o[37] * _exp2_102;
            d_o[40] = d_o[40] * _exp2_102;
            d_o[41] = d_o[41] * _exp2_102;
            d_o[44] = d_o[44] * _exp2_102;
            d_o[45] = d_o[45] * _exp2_102;
            d_o[48] = d_o[48] * _exp2_102;
            d_o[49] = d_o[49] * _exp2_102;
            d_o[52] = d_o[52] * _exp2_102;
            d_o[53] = d_o[53] * _exp2_102;
            d_o[56] = d_o[56] * _exp2_102;
            d_o[57] = d_o[57] * _exp2_102;
            d_o[60] = d_o[60] * _exp2_102;
            d_o[61] = d_o[61] * _exp2_102;
            d_o[2] = d_o[2] * _exp2_103;
            d_o[3] = d_o[3] * _exp2_103;
            d_o[6] = d_o[6] * _exp2_103;
            d_o[7] = d_o[7] * _exp2_103;
            d_o[10] = d_o[10] * _exp2_103;
            d_o[11] = d_o[11] * _exp2_103;
            d_o[14] = d_o[14] * _exp2_103;
            d_o[15] = d_o[15] * _exp2_103;
            d_o[18] = d_o[18] * _exp2_103;
            d_o[19] = d_o[19] * _exp2_103;
            d_o[22] = d_o[22] * _exp2_103;
            d_o[23] = d_o[23] * _exp2_103;
            d_o[26] = d_o[26] * _exp2_103;
            d_o[27] = d_o[27] * _exp2_103;
            d_o[30] = d_o[30] * _exp2_103;
            d_o[31] = d_o[31] * _exp2_103;
            d_o[34] = d_o[34] * _exp2_103;
            d_o[35] = d_o[35] * _exp2_103;
            d_o[38] = d_o[38] * _exp2_103;
            d_o[39] = d_o[39] * _exp2_103;
            d_o[42] = d_o[42] * _exp2_103;
            d_o[43] = d_o[43] * _exp2_103;
            d_o[46] = d_o[46] * _exp2_103;
            d_o[47] = d_o[47] * _exp2_103;
            d_o[50] = d_o[50] * _exp2_103;
            d_o[51] = d_o[51] * _exp2_103;
            d_o[54] = d_o[54] * _exp2_103;
            d_o[55] = d_o[55] * _exp2_103;
            d_o[58] = d_o[58] * _exp2_103;
            d_o[59] = d_o[59] * _exp2_103;
            d_o[62] = d_o[62] * _exp2_103;
            d_o[63] = d_o[63] * _exp2_103;
            row_sum0 = row_sum0 * _exp2_102;
            row_sum1 = row_sum1 * _exp2_103;
            row_max0 = merged_max0_2;
            row_max1 = merged_max1_2;
        }
        float _exp2_104 = approx_exp2(d_qk[0] - row_max0);
        float _exp2_105 = approx_exp2(d_qk[1] - row_max0);
        d_qk[0] = _exp2_104;
        d_qk[1] = _exp2_105;
        row_sum0 += _exp2_104 + _exp2_105;
        float _exp2_106 = approx_exp2(d_qk[2] - row_max1);
        float _exp2_107 = approx_exp2(d_qk[3] - row_max1);
        d_qk[2] = _exp2_106;
        d_qk[3] = _exp2_107;
        row_sum1 += _exp2_106 + _exp2_107;
        float _exp2_108 = approx_exp2(d_qk[4] - row_max0);
        float _exp2_109 = approx_exp2(d_qk[5] - row_max0);
        d_qk[4] = _exp2_108;
        d_qk[5] = _exp2_109;
        row_sum0 += _exp2_108 + _exp2_109;
        float _exp2_110 = approx_exp2(d_qk[6] - row_max1);
        float _exp2_111 = approx_exp2(d_qk[7] - row_max1);
        d_qk[6] = _exp2_110;
        d_qk[7] = _exp2_111;
        row_sum1 += _exp2_110 + _exp2_111;
        float _exp2_112 = approx_exp2(d_qk[8] - row_max0);
        float _exp2_113 = approx_exp2(d_qk[9] - row_max0);
        d_qk[8] = _exp2_112;
        d_qk[9] = _exp2_113;
        row_sum0 += _exp2_112 + _exp2_113;
        float _exp2_114 = approx_exp2(d_qk[10] - row_max1);
        float _exp2_115 = approx_exp2(d_qk[11] - row_max1);
        d_qk[10] = _exp2_114;
        d_qk[11] = _exp2_115;
        row_sum1 += _exp2_114 + _exp2_115;
        float _exp2_116 = approx_exp2(d_qk[12] - row_max0);
        float _exp2_117 = approx_exp2(d_qk[13] - row_max0);
        d_qk[12] = _exp2_116;
        d_qk[13] = _exp2_117;
        row_sum0 += _exp2_116 + _exp2_117;
        float _exp2_118 = approx_exp2(d_qk[14] - row_max1);
        float _exp2_119 = approx_exp2(d_qk[15] - row_max1);
        d_qk[14] = _exp2_118;
        d_qk[15] = _exp2_119;
        row_sum1 += _exp2_118 + _exp2_119;
        float _exp2_120 = approx_exp2(d_qk[16] - row_max0);
        float _exp2_121 = approx_exp2(d_qk[17] - row_max0);
        d_qk[16] = _exp2_120;
        d_qk[17] = _exp2_121;
        row_sum0 += _exp2_120 + _exp2_121;
        float _exp2_122 = approx_exp2(d_qk[18] - row_max1);
        float _exp2_123 = approx_exp2(d_qk[19] - row_max1);
        d_qk[18] = _exp2_122;
        d_qk[19] = _exp2_123;
        row_sum1 += _exp2_122 + _exp2_123;
        float _exp2_124 = approx_exp2(d_qk[20] - row_max0);
        float _exp2_125 = approx_exp2(d_qk[21] - row_max0);
        d_qk[20] = _exp2_124;
        d_qk[21] = _exp2_125;
        row_sum0 += _exp2_124 + _exp2_125;
        float _exp2_126 = approx_exp2(d_qk[22] - row_max1);
        float _exp2_127 = approx_exp2(d_qk[23] - row_max1);
        d_qk[22] = _exp2_126;
        d_qk[23] = _exp2_127;
        row_sum1 += _exp2_126 + _exp2_127;
        float _exp2_128 = approx_exp2(d_qk[24] - row_max0);
        float _exp2_129 = approx_exp2(d_qk[25] - row_max0);
        d_qk[24] = _exp2_128;
        d_qk[25] = _exp2_129;
        row_sum0 += _exp2_128 + _exp2_129;
        float _exp2_130 = approx_exp2(d_qk[26] - row_max1);
        float _exp2_131 = approx_exp2(d_qk[27] - row_max1);
        d_qk[26] = _exp2_130;
        d_qk[27] = _exp2_131;
        row_sum1 += _exp2_130 + _exp2_131;
        float _exp2_132 = approx_exp2(d_qk[28] - row_max0);
        float _exp2_133 = approx_exp2(d_qk[29] - row_max0);
        d_qk[28] = _exp2_132;
        d_qk[29] = _exp2_133;
        row_sum0 += _exp2_132 + _exp2_133;
        float _exp2_134 = approx_exp2(d_qk[30] - row_max1);
        float _exp2_135 = approx_exp2(d_qk[31] - row_max1);
        d_qk[30] = _exp2_134;
        d_qk[31] = _exp2_135;
        row_sum1 += _exp2_134 + _exp2_135;
        __nv_bfloat162 _bf16x2_48 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bf16x2_48)[0];
        __nv_bfloat162 _bf16x2_49 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bf16x2_49)[0];
        __nv_bfloat162 _bf16x2_50 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bf16x2_50)[0];
        __nv_bfloat162 _bf16x2_51 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bf16x2_51)[0];
        __nv_bfloat162 _bf16x2_52 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bf16x2_52)[0];
        __nv_bfloat162 _bf16x2_53 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bf16x2_53)[0];
        __nv_bfloat162 _bf16x2_54 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bf16x2_54)[0];
        __nv_bfloat162 _bf16x2_55 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bf16x2_55)[0];
        __nv_bfloat162 _bf16x2_56 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bf16x2_56)[0];
        __nv_bfloat162 _bf16x2_57 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bf16x2_57)[0];
        __nv_bfloat162 _bf16x2_58 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bf16x2_58)[0];
        __nv_bfloat162 _bf16x2_59 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bf16x2_59)[0];
        __nv_bfloat162 _bf16x2_60 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bf16x2_60)[0];
        __nv_bfloat162 _bf16x2_61 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bf16x2_61)[0];
        __nv_bfloat162 _bf16x2_62 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bf16x2_62)[0];
        __nv_bfloat162 _bf16x2_63 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bf16x2_63)[0];
        mbarrier_wait(v_full3_addr, _phase_v_full3_0);
        _phase_v_full3_0 ^= 1;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_b_0_13)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_b_0_13 + 128)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_b_0_13 + 256)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_b_0_13 + 384)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    unsigned int _phase_k_full4_0 = 0;
    uint64_t _wgmma_desc_14 = (((uint64_t)(((k_smem_addr + 65536)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_14 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_14 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_14);
    uint64_t _wgmma_desc_15 = (((uint64_t)(((k_smem_addr + 65536 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_15 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_15 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_15);
    unsigned int _phase_v_full4_0 = 0;
    uint64_t _wgmma_desc_16 = (((uint64_t)(((vt_smem_addr + 65536)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_16 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_16 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_16);
    if (cnt > 4) {
        mbarrier_wait(k_full4_addr, _phase_k_full4_0);
        _phase_k_full4_0 ^= 1;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_14)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_14 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_14 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_14 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_15)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 2), "l"(_wgmma_b_0_15 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 4), "l"(_wgmma_b_0_15 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 6), "l"(_wgmma_b_0_15 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        d_qk[0] = d_qk[0] * scale_log2;
        d_qk[1] = d_qk[1] * scale_log2;
        d_qk[2] = d_qk[2] * scale_log2;
        d_qk[3] = d_qk[3] * scale_log2;
        d_qk[4] = d_qk[4] * scale_log2;
        d_qk[5] = d_qk[5] * scale_log2;
        d_qk[6] = d_qk[6] * scale_log2;
        d_qk[7] = d_qk[7] * scale_log2;
        d_qk[8] = d_qk[8] * scale_log2;
        d_qk[9] = d_qk[9] * scale_log2;
        d_qk[10] = d_qk[10] * scale_log2;
        d_qk[11] = d_qk[11] * scale_log2;
        d_qk[12] = d_qk[12] * scale_log2;
        d_qk[13] = d_qk[13] * scale_log2;
        d_qk[14] = d_qk[14] * scale_log2;
        d_qk[15] = d_qk[15] * scale_log2;
        d_qk[16] = d_qk[16] * scale_log2;
        d_qk[17] = d_qk[17] * scale_log2;
        d_qk[18] = d_qk[18] * scale_log2;
        d_qk[19] = d_qk[19] * scale_log2;
        d_qk[20] = d_qk[20] * scale_log2;
        d_qk[21] = d_qk[21] * scale_log2;
        d_qk[22] = d_qk[22] * scale_log2;
        d_qk[23] = d_qk[23] * scale_log2;
        d_qk[24] = d_qk[24] * scale_log2;
        d_qk[25] = d_qk[25] * scale_log2;
        d_qk[26] = d_qk[26] * scale_log2;
        d_qk[27] = d_qk[27] * scale_log2;
        d_qk[28] = d_qk[28] * scale_log2;
        d_qk[29] = d_qk[29] * scale_log2;
        d_qk[30] = d_qk[30] * scale_log2;
        d_qk[31] = d_qk[31] * scale_log2;
        float new_max0_4 = -CAKE_INF;
        float new_max1_4 = -CAKE_INF;
        float _max_152 = max_noftz(new_max0_4, d_qk[0]);
        new_max0_4 = _max_152;
        float _max_153 = max_noftz(new_max0_4, d_qk[1]);
        new_max0_4 = _max_153;
        float _max_154 = max_noftz(new_max0_4, d_qk[4]);
        new_max0_4 = _max_154;
        float _max_155 = max_noftz(new_max0_4, d_qk[5]);
        new_max0_4 = _max_155;
        float _max_156 = max_noftz(new_max0_4, d_qk[8]);
        new_max0_4 = _max_156;
        float _max_157 = max_noftz(new_max0_4, d_qk[9]);
        new_max0_4 = _max_157;
        float _max_158 = max_noftz(new_max0_4, d_qk[12]);
        new_max0_4 = _max_158;
        float _max_159 = max_noftz(new_max0_4, d_qk[13]);
        new_max0_4 = _max_159;
        float _max_160 = max_noftz(new_max0_4, d_qk[16]);
        new_max0_4 = _max_160;
        float _max_161 = max_noftz(new_max0_4, d_qk[17]);
        new_max0_4 = _max_161;
        float _max_162 = max_noftz(new_max0_4, d_qk[20]);
        new_max0_4 = _max_162;
        float _max_163 = max_noftz(new_max0_4, d_qk[21]);
        new_max0_4 = _max_163;
        float _max_164 = max_noftz(new_max0_4, d_qk[24]);
        new_max0_4 = _max_164;
        float _max_165 = max_noftz(new_max0_4, d_qk[25]);
        new_max0_4 = _max_165;
        float _max_166 = max_noftz(new_max0_4, d_qk[28]);
        new_max0_4 = _max_166;
        float _max_167 = max_noftz(new_max0_4, d_qk[29]);
        new_max0_4 = _max_167;
        float _max_168 = max_noftz(new_max1_4, d_qk[2]);
        new_max1_4 = _max_168;
        float _max_169 = max_noftz(new_max1_4, d_qk[3]);
        new_max1_4 = _max_169;
        float _max_170 = max_noftz(new_max1_4, d_qk[6]);
        new_max1_4 = _max_170;
        float _max_171 = max_noftz(new_max1_4, d_qk[7]);
        new_max1_4 = _max_171;
        float _max_172 = max_noftz(new_max1_4, d_qk[10]);
        new_max1_4 = _max_172;
        float _max_173 = max_noftz(new_max1_4, d_qk[11]);
        new_max1_4 = _max_173;
        float _max_174 = max_noftz(new_max1_4, d_qk[14]);
        new_max1_4 = _max_174;
        float _max_175 = max_noftz(new_max1_4, d_qk[15]);
        new_max1_4 = _max_175;
        float _max_176 = max_noftz(new_max1_4, d_qk[18]);
        new_max1_4 = _max_176;
        float _max_177 = max_noftz(new_max1_4, d_qk[19]);
        new_max1_4 = _max_177;
        float _max_178 = max_noftz(new_max1_4, d_qk[22]);
        new_max1_4 = _max_178;
        float _max_179 = max_noftz(new_max1_4, d_qk[23]);
        new_max1_4 = _max_179;
        float _max_180 = max_noftz(new_max1_4, d_qk[26]);
        new_max1_4 = _max_180;
        float _max_181 = max_noftz(new_max1_4, d_qk[27]);
        new_max1_4 = _max_181;
        float _max_182 = max_noftz(new_max1_4, d_qk[30]);
        new_max1_4 = _max_182;
        float _max_183 = max_noftz(new_max1_4, d_qk[31]);
        new_max1_4 = _max_183;
        float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, new_max0_4, 2);
        float _max_184 = max_noftz(new_max0_4, _shfl_xor_16);
        new_max0_4 = _max_184;
        float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, new_max0_4, 1);
        float _max_185 = max_noftz(new_max0_4, _shfl_xor_17);
        new_max0_4 = _max_185;
        float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, new_max1_4, 2);
        float _max_186 = max_noftz(new_max1_4, _shfl_xor_18);
        new_max1_4 = _max_186;
        float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, new_max1_4, 1);
        float _max_187 = max_noftz(new_max1_4, _shfl_xor_19);
        new_max1_4 = _max_187;
        {
            float _max_188 = max_noftz(row_max0, new_max0_4);
            float merged_max0_3 = _max_188;
            float _max_189 = max_noftz(row_max1, new_max1_4);
            float merged_max1_3 = _max_189;
            float _exp2_136 = approx_exp2(row_max0 - merged_max0_3);
            float _exp2_137 = approx_exp2(row_max1 - merged_max1_3);
            d_o[0] = d_o[0] * _exp2_136;
            d_o[1] = d_o[1] * _exp2_136;
            d_o[4] = d_o[4] * _exp2_136;
            d_o[5] = d_o[5] * _exp2_136;
            d_o[8] = d_o[8] * _exp2_136;
            d_o[9] = d_o[9] * _exp2_136;
            d_o[12] = d_o[12] * _exp2_136;
            d_o[13] = d_o[13] * _exp2_136;
            d_o[16] = d_o[16] * _exp2_136;
            d_o[17] = d_o[17] * _exp2_136;
            d_o[20] = d_o[20] * _exp2_136;
            d_o[21] = d_o[21] * _exp2_136;
            d_o[24] = d_o[24] * _exp2_136;
            d_o[25] = d_o[25] * _exp2_136;
            d_o[28] = d_o[28] * _exp2_136;
            d_o[29] = d_o[29] * _exp2_136;
            d_o[32] = d_o[32] * _exp2_136;
            d_o[33] = d_o[33] * _exp2_136;
            d_o[36] = d_o[36] * _exp2_136;
            d_o[37] = d_o[37] * _exp2_136;
            d_o[40] = d_o[40] * _exp2_136;
            d_o[41] = d_o[41] * _exp2_136;
            d_o[44] = d_o[44] * _exp2_136;
            d_o[45] = d_o[45] * _exp2_136;
            d_o[48] = d_o[48] * _exp2_136;
            d_o[49] = d_o[49] * _exp2_136;
            d_o[52] = d_o[52] * _exp2_136;
            d_o[53] = d_o[53] * _exp2_136;
            d_o[56] = d_o[56] * _exp2_136;
            d_o[57] = d_o[57] * _exp2_136;
            d_o[60] = d_o[60] * _exp2_136;
            d_o[61] = d_o[61] * _exp2_136;
            d_o[2] = d_o[2] * _exp2_137;
            d_o[3] = d_o[3] * _exp2_137;
            d_o[6] = d_o[6] * _exp2_137;
            d_o[7] = d_o[7] * _exp2_137;
            d_o[10] = d_o[10] * _exp2_137;
            d_o[11] = d_o[11] * _exp2_137;
            d_o[14] = d_o[14] * _exp2_137;
            d_o[15] = d_o[15] * _exp2_137;
            d_o[18] = d_o[18] * _exp2_137;
            d_o[19] = d_o[19] * _exp2_137;
            d_o[22] = d_o[22] * _exp2_137;
            d_o[23] = d_o[23] * _exp2_137;
            d_o[26] = d_o[26] * _exp2_137;
            d_o[27] = d_o[27] * _exp2_137;
            d_o[30] = d_o[30] * _exp2_137;
            d_o[31] = d_o[31] * _exp2_137;
            d_o[34] = d_o[34] * _exp2_137;
            d_o[35] = d_o[35] * _exp2_137;
            d_o[38] = d_o[38] * _exp2_137;
            d_o[39] = d_o[39] * _exp2_137;
            d_o[42] = d_o[42] * _exp2_137;
            d_o[43] = d_o[43] * _exp2_137;
            d_o[46] = d_o[46] * _exp2_137;
            d_o[47] = d_o[47] * _exp2_137;
            d_o[50] = d_o[50] * _exp2_137;
            d_o[51] = d_o[51] * _exp2_137;
            d_o[54] = d_o[54] * _exp2_137;
            d_o[55] = d_o[55] * _exp2_137;
            d_o[58] = d_o[58] * _exp2_137;
            d_o[59] = d_o[59] * _exp2_137;
            d_o[62] = d_o[62] * _exp2_137;
            d_o[63] = d_o[63] * _exp2_137;
            row_sum0 = row_sum0 * _exp2_136;
            row_sum1 = row_sum1 * _exp2_137;
            row_max0 = merged_max0_3;
            row_max1 = merged_max1_3;
        }
        float _exp2_138 = approx_exp2(d_qk[0] - row_max0);
        float _exp2_139 = approx_exp2(d_qk[1] - row_max0);
        d_qk[0] = _exp2_138;
        d_qk[1] = _exp2_139;
        row_sum0 += _exp2_138 + _exp2_139;
        float _exp2_140 = approx_exp2(d_qk[2] - row_max1);
        float _exp2_141 = approx_exp2(d_qk[3] - row_max1);
        d_qk[2] = _exp2_140;
        d_qk[3] = _exp2_141;
        row_sum1 += _exp2_140 + _exp2_141;
        float _exp2_142 = approx_exp2(d_qk[4] - row_max0);
        float _exp2_143 = approx_exp2(d_qk[5] - row_max0);
        d_qk[4] = _exp2_142;
        d_qk[5] = _exp2_143;
        row_sum0 += _exp2_142 + _exp2_143;
        float _exp2_144 = approx_exp2(d_qk[6] - row_max1);
        float _exp2_145 = approx_exp2(d_qk[7] - row_max1);
        d_qk[6] = _exp2_144;
        d_qk[7] = _exp2_145;
        row_sum1 += _exp2_144 + _exp2_145;
        float _exp2_146 = approx_exp2(d_qk[8] - row_max0);
        float _exp2_147 = approx_exp2(d_qk[9] - row_max0);
        d_qk[8] = _exp2_146;
        d_qk[9] = _exp2_147;
        row_sum0 += _exp2_146 + _exp2_147;
        float _exp2_148 = approx_exp2(d_qk[10] - row_max1);
        float _exp2_149 = approx_exp2(d_qk[11] - row_max1);
        d_qk[10] = _exp2_148;
        d_qk[11] = _exp2_149;
        row_sum1 += _exp2_148 + _exp2_149;
        float _exp2_150 = approx_exp2(d_qk[12] - row_max0);
        float _exp2_151 = approx_exp2(d_qk[13] - row_max0);
        d_qk[12] = _exp2_150;
        d_qk[13] = _exp2_151;
        row_sum0 += _exp2_150 + _exp2_151;
        float _exp2_152 = approx_exp2(d_qk[14] - row_max1);
        float _exp2_153 = approx_exp2(d_qk[15] - row_max1);
        d_qk[14] = _exp2_152;
        d_qk[15] = _exp2_153;
        row_sum1 += _exp2_152 + _exp2_153;
        float _exp2_154 = approx_exp2(d_qk[16] - row_max0);
        float _exp2_155 = approx_exp2(d_qk[17] - row_max0);
        d_qk[16] = _exp2_154;
        d_qk[17] = _exp2_155;
        row_sum0 += _exp2_154 + _exp2_155;
        float _exp2_156 = approx_exp2(d_qk[18] - row_max1);
        float _exp2_157 = approx_exp2(d_qk[19] - row_max1);
        d_qk[18] = _exp2_156;
        d_qk[19] = _exp2_157;
        row_sum1 += _exp2_156 + _exp2_157;
        float _exp2_158 = approx_exp2(d_qk[20] - row_max0);
        float _exp2_159 = approx_exp2(d_qk[21] - row_max0);
        d_qk[20] = _exp2_158;
        d_qk[21] = _exp2_159;
        row_sum0 += _exp2_158 + _exp2_159;
        float _exp2_160 = approx_exp2(d_qk[22] - row_max1);
        float _exp2_161 = approx_exp2(d_qk[23] - row_max1);
        d_qk[22] = _exp2_160;
        d_qk[23] = _exp2_161;
        row_sum1 += _exp2_160 + _exp2_161;
        float _exp2_162 = approx_exp2(d_qk[24] - row_max0);
        float _exp2_163 = approx_exp2(d_qk[25] - row_max0);
        d_qk[24] = _exp2_162;
        d_qk[25] = _exp2_163;
        row_sum0 += _exp2_162 + _exp2_163;
        float _exp2_164 = approx_exp2(d_qk[26] - row_max1);
        float _exp2_165 = approx_exp2(d_qk[27] - row_max1);
        d_qk[26] = _exp2_164;
        d_qk[27] = _exp2_165;
        row_sum1 += _exp2_164 + _exp2_165;
        float _exp2_166 = approx_exp2(d_qk[28] - row_max0);
        float _exp2_167 = approx_exp2(d_qk[29] - row_max0);
        d_qk[28] = _exp2_166;
        d_qk[29] = _exp2_167;
        row_sum0 += _exp2_166 + _exp2_167;
        float _exp2_168 = approx_exp2(d_qk[30] - row_max1);
        float _exp2_169 = approx_exp2(d_qk[31] - row_max1);
        d_qk[30] = _exp2_168;
        d_qk[31] = _exp2_169;
        row_sum1 += _exp2_168 + _exp2_169;
        __nv_bfloat162 _bf16x2_64 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bf16x2_64)[0];
        __nv_bfloat162 _bf16x2_65 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bf16x2_65)[0];
        __nv_bfloat162 _bf16x2_66 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bf16x2_66)[0];
        __nv_bfloat162 _bf16x2_67 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bf16x2_67)[0];
        __nv_bfloat162 _bf16x2_68 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bf16x2_68)[0];
        __nv_bfloat162 _bf16x2_69 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bf16x2_69)[0];
        __nv_bfloat162 _bf16x2_70 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bf16x2_70)[0];
        __nv_bfloat162 _bf16x2_71 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bf16x2_71)[0];
        __nv_bfloat162 _bf16x2_72 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bf16x2_72)[0];
        __nv_bfloat162 _bf16x2_73 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bf16x2_73)[0];
        __nv_bfloat162 _bf16x2_74 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bf16x2_74)[0];
        __nv_bfloat162 _bf16x2_75 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bf16x2_75)[0];
        __nv_bfloat162 _bf16x2_76 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bf16x2_76)[0];
        __nv_bfloat162 _bf16x2_77 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bf16x2_77)[0];
        __nv_bfloat162 _bf16x2_78 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bf16x2_78)[0];
        __nv_bfloat162 _bf16x2_79 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bf16x2_79)[0];
        mbarrier_wait(v_full4_addr, _phase_v_full4_0);
        _phase_v_full4_0 ^= 1;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_b_0_16)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_b_0_16 + 128)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_b_0_16 + 256)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_b_0_16 + 384)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    unsigned int _phase_k_full5_0 = 0;
    uint64_t _wgmma_desc_17 = (((uint64_t)(((k_smem_addr + 81920)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_17 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_17 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_17);
    uint64_t _wgmma_desc_18 = (((uint64_t)(((k_smem_addr + 81920 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_18 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_18 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_18);
    unsigned int _phase_v_full5_0 = 0;
    uint64_t _wgmma_desc_19 = (((uint64_t)(((vt_smem_addr + 81920)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_19 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_19 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_19);
    if (cnt > 5) {
        mbarrier_wait(k_full5_addr, _phase_k_full5_0);
        _phase_k_full5_0 ^= 1;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_17)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_17 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_17 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_17 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_18)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 2), "l"(_wgmma_b_0_18 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 4), "l"(_wgmma_b_0_18 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0_2 + 6), "l"(_wgmma_b_0_18 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        d_qk[0] = d_qk[0] * scale_log2;
        d_qk[1] = d_qk[1] * scale_log2;
        d_qk[2] = d_qk[2] * scale_log2;
        d_qk[3] = d_qk[3] * scale_log2;
        d_qk[4] = d_qk[4] * scale_log2;
        d_qk[5] = d_qk[5] * scale_log2;
        d_qk[6] = d_qk[6] * scale_log2;
        d_qk[7] = d_qk[7] * scale_log2;
        d_qk[8] = d_qk[8] * scale_log2;
        d_qk[9] = d_qk[9] * scale_log2;
        d_qk[10] = d_qk[10] * scale_log2;
        d_qk[11] = d_qk[11] * scale_log2;
        d_qk[12] = d_qk[12] * scale_log2;
        d_qk[13] = d_qk[13] * scale_log2;
        d_qk[14] = d_qk[14] * scale_log2;
        d_qk[15] = d_qk[15] * scale_log2;
        d_qk[16] = d_qk[16] * scale_log2;
        d_qk[17] = d_qk[17] * scale_log2;
        d_qk[18] = d_qk[18] * scale_log2;
        d_qk[19] = d_qk[19] * scale_log2;
        d_qk[20] = d_qk[20] * scale_log2;
        d_qk[21] = d_qk[21] * scale_log2;
        d_qk[22] = d_qk[22] * scale_log2;
        d_qk[23] = d_qk[23] * scale_log2;
        d_qk[24] = d_qk[24] * scale_log2;
        d_qk[25] = d_qk[25] * scale_log2;
        d_qk[26] = d_qk[26] * scale_log2;
        d_qk[27] = d_qk[27] * scale_log2;
        d_qk[28] = d_qk[28] * scale_log2;
        d_qk[29] = d_qk[29] * scale_log2;
        d_qk[30] = d_qk[30] * scale_log2;
        d_qk[31] = d_qk[31] * scale_log2;
        float new_max0_5 = -CAKE_INF;
        float new_max1_5 = -CAKE_INF;
        float _max_190 = max_noftz(new_max0_5, d_qk[0]);
        new_max0_5 = _max_190;
        float _max_191 = max_noftz(new_max0_5, d_qk[1]);
        new_max0_5 = _max_191;
        float _max_192 = max_noftz(new_max0_5, d_qk[4]);
        new_max0_5 = _max_192;
        float _max_193 = max_noftz(new_max0_5, d_qk[5]);
        new_max0_5 = _max_193;
        float _max_194 = max_noftz(new_max0_5, d_qk[8]);
        new_max0_5 = _max_194;
        float _max_195 = max_noftz(new_max0_5, d_qk[9]);
        new_max0_5 = _max_195;
        float _max_196 = max_noftz(new_max0_5, d_qk[12]);
        new_max0_5 = _max_196;
        float _max_197 = max_noftz(new_max0_5, d_qk[13]);
        new_max0_5 = _max_197;
        float _max_198 = max_noftz(new_max0_5, d_qk[16]);
        new_max0_5 = _max_198;
        float _max_199 = max_noftz(new_max0_5, d_qk[17]);
        new_max0_5 = _max_199;
        float _max_200 = max_noftz(new_max0_5, d_qk[20]);
        new_max0_5 = _max_200;
        float _max_201 = max_noftz(new_max0_5, d_qk[21]);
        new_max0_5 = _max_201;
        float _max_202 = max_noftz(new_max0_5, d_qk[24]);
        new_max0_5 = _max_202;
        float _max_203 = max_noftz(new_max0_5, d_qk[25]);
        new_max0_5 = _max_203;
        float _max_204 = max_noftz(new_max0_5, d_qk[28]);
        new_max0_5 = _max_204;
        float _max_205 = max_noftz(new_max0_5, d_qk[29]);
        new_max0_5 = _max_205;
        float _max_206 = max_noftz(new_max1_5, d_qk[2]);
        new_max1_5 = _max_206;
        float _max_207 = max_noftz(new_max1_5, d_qk[3]);
        new_max1_5 = _max_207;
        float _max_208 = max_noftz(new_max1_5, d_qk[6]);
        new_max1_5 = _max_208;
        float _max_209 = max_noftz(new_max1_5, d_qk[7]);
        new_max1_5 = _max_209;
        float _max_210 = max_noftz(new_max1_5, d_qk[10]);
        new_max1_5 = _max_210;
        float _max_211 = max_noftz(new_max1_5, d_qk[11]);
        new_max1_5 = _max_211;
        float _max_212 = max_noftz(new_max1_5, d_qk[14]);
        new_max1_5 = _max_212;
        float _max_213 = max_noftz(new_max1_5, d_qk[15]);
        new_max1_5 = _max_213;
        float _max_214 = max_noftz(new_max1_5, d_qk[18]);
        new_max1_5 = _max_214;
        float _max_215 = max_noftz(new_max1_5, d_qk[19]);
        new_max1_5 = _max_215;
        float _max_216 = max_noftz(new_max1_5, d_qk[22]);
        new_max1_5 = _max_216;
        float _max_217 = max_noftz(new_max1_5, d_qk[23]);
        new_max1_5 = _max_217;
        float _max_218 = max_noftz(new_max1_5, d_qk[26]);
        new_max1_5 = _max_218;
        float _max_219 = max_noftz(new_max1_5, d_qk[27]);
        new_max1_5 = _max_219;
        float _max_220 = max_noftz(new_max1_5, d_qk[30]);
        new_max1_5 = _max_220;
        float _max_221 = max_noftz(new_max1_5, d_qk[31]);
        new_max1_5 = _max_221;
        float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, new_max0_5, 2);
        float _max_222 = max_noftz(new_max0_5, _shfl_xor_20);
        new_max0_5 = _max_222;
        float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, new_max0_5, 1);
        float _max_223 = max_noftz(new_max0_5, _shfl_xor_21);
        new_max0_5 = _max_223;
        float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, new_max1_5, 2);
        float _max_224 = max_noftz(new_max1_5, _shfl_xor_22);
        new_max1_5 = _max_224;
        float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, new_max1_5, 1);
        float _max_225 = max_noftz(new_max1_5, _shfl_xor_23);
        new_max1_5 = _max_225;
        {
            float _max_226 = max_noftz(row_max0, new_max0_5);
            float merged_max0_4 = _max_226;
            float _max_227 = max_noftz(row_max1, new_max1_5);
            float merged_max1_4 = _max_227;
            float _exp2_170 = approx_exp2(row_max0 - merged_max0_4);
            float _exp2_171 = approx_exp2(row_max1 - merged_max1_4);
            d_o[0] = d_o[0] * _exp2_170;
            d_o[1] = d_o[1] * _exp2_170;
            d_o[4] = d_o[4] * _exp2_170;
            d_o[5] = d_o[5] * _exp2_170;
            d_o[8] = d_o[8] * _exp2_170;
            d_o[9] = d_o[9] * _exp2_170;
            d_o[12] = d_o[12] * _exp2_170;
            d_o[13] = d_o[13] * _exp2_170;
            d_o[16] = d_o[16] * _exp2_170;
            d_o[17] = d_o[17] * _exp2_170;
            d_o[20] = d_o[20] * _exp2_170;
            d_o[21] = d_o[21] * _exp2_170;
            d_o[24] = d_o[24] * _exp2_170;
            d_o[25] = d_o[25] * _exp2_170;
            d_o[28] = d_o[28] * _exp2_170;
            d_o[29] = d_o[29] * _exp2_170;
            d_o[32] = d_o[32] * _exp2_170;
            d_o[33] = d_o[33] * _exp2_170;
            d_o[36] = d_o[36] * _exp2_170;
            d_o[37] = d_o[37] * _exp2_170;
            d_o[40] = d_o[40] * _exp2_170;
            d_o[41] = d_o[41] * _exp2_170;
            d_o[44] = d_o[44] * _exp2_170;
            d_o[45] = d_o[45] * _exp2_170;
            d_o[48] = d_o[48] * _exp2_170;
            d_o[49] = d_o[49] * _exp2_170;
            d_o[52] = d_o[52] * _exp2_170;
            d_o[53] = d_o[53] * _exp2_170;
            d_o[56] = d_o[56] * _exp2_170;
            d_o[57] = d_o[57] * _exp2_170;
            d_o[60] = d_o[60] * _exp2_170;
            d_o[61] = d_o[61] * _exp2_170;
            d_o[2] = d_o[2] * _exp2_171;
            d_o[3] = d_o[3] * _exp2_171;
            d_o[6] = d_o[6] * _exp2_171;
            d_o[7] = d_o[7] * _exp2_171;
            d_o[10] = d_o[10] * _exp2_171;
            d_o[11] = d_o[11] * _exp2_171;
            d_o[14] = d_o[14] * _exp2_171;
            d_o[15] = d_o[15] * _exp2_171;
            d_o[18] = d_o[18] * _exp2_171;
            d_o[19] = d_o[19] * _exp2_171;
            d_o[22] = d_o[22] * _exp2_171;
            d_o[23] = d_o[23] * _exp2_171;
            d_o[26] = d_o[26] * _exp2_171;
            d_o[27] = d_o[27] * _exp2_171;
            d_o[30] = d_o[30] * _exp2_171;
            d_o[31] = d_o[31] * _exp2_171;
            d_o[34] = d_o[34] * _exp2_171;
            d_o[35] = d_o[35] * _exp2_171;
            d_o[38] = d_o[38] * _exp2_171;
            d_o[39] = d_o[39] * _exp2_171;
            d_o[42] = d_o[42] * _exp2_171;
            d_o[43] = d_o[43] * _exp2_171;
            d_o[46] = d_o[46] * _exp2_171;
            d_o[47] = d_o[47] * _exp2_171;
            d_o[50] = d_o[50] * _exp2_171;
            d_o[51] = d_o[51] * _exp2_171;
            d_o[54] = d_o[54] * _exp2_171;
            d_o[55] = d_o[55] * _exp2_171;
            d_o[58] = d_o[58] * _exp2_171;
            d_o[59] = d_o[59] * _exp2_171;
            d_o[62] = d_o[62] * _exp2_171;
            d_o[63] = d_o[63] * _exp2_171;
            row_sum0 = row_sum0 * _exp2_170;
            row_sum1 = row_sum1 * _exp2_171;
            row_max0 = merged_max0_4;
            row_max1 = merged_max1_4;
        }
        float _exp2_172 = approx_exp2(d_qk[0] - row_max0);
        float _exp2_173 = approx_exp2(d_qk[1] - row_max0);
        d_qk[0] = _exp2_172;
        d_qk[1] = _exp2_173;
        row_sum0 += _exp2_172 + _exp2_173;
        float _exp2_174 = approx_exp2(d_qk[2] - row_max1);
        float _exp2_175 = approx_exp2(d_qk[3] - row_max1);
        d_qk[2] = _exp2_174;
        d_qk[3] = _exp2_175;
        row_sum1 += _exp2_174 + _exp2_175;
        float _exp2_176 = approx_exp2(d_qk[4] - row_max0);
        float _exp2_177 = approx_exp2(d_qk[5] - row_max0);
        d_qk[4] = _exp2_176;
        d_qk[5] = _exp2_177;
        row_sum0 += _exp2_176 + _exp2_177;
        float _exp2_178 = approx_exp2(d_qk[6] - row_max1);
        float _exp2_179 = approx_exp2(d_qk[7] - row_max1);
        d_qk[6] = _exp2_178;
        d_qk[7] = _exp2_179;
        row_sum1 += _exp2_178 + _exp2_179;
        float _exp2_180 = approx_exp2(d_qk[8] - row_max0);
        float _exp2_181 = approx_exp2(d_qk[9] - row_max0);
        d_qk[8] = _exp2_180;
        d_qk[9] = _exp2_181;
        row_sum0 += _exp2_180 + _exp2_181;
        float _exp2_182 = approx_exp2(d_qk[10] - row_max1);
        float _exp2_183 = approx_exp2(d_qk[11] - row_max1);
        d_qk[10] = _exp2_182;
        d_qk[11] = _exp2_183;
        row_sum1 += _exp2_182 + _exp2_183;
        float _exp2_184 = approx_exp2(d_qk[12] - row_max0);
        float _exp2_185 = approx_exp2(d_qk[13] - row_max0);
        d_qk[12] = _exp2_184;
        d_qk[13] = _exp2_185;
        row_sum0 += _exp2_184 + _exp2_185;
        float _exp2_186 = approx_exp2(d_qk[14] - row_max1);
        float _exp2_187 = approx_exp2(d_qk[15] - row_max1);
        d_qk[14] = _exp2_186;
        d_qk[15] = _exp2_187;
        row_sum1 += _exp2_186 + _exp2_187;
        float _exp2_188 = approx_exp2(d_qk[16] - row_max0);
        float _exp2_189 = approx_exp2(d_qk[17] - row_max0);
        d_qk[16] = _exp2_188;
        d_qk[17] = _exp2_189;
        row_sum0 += _exp2_188 + _exp2_189;
        float _exp2_190 = approx_exp2(d_qk[18] - row_max1);
        float _exp2_191 = approx_exp2(d_qk[19] - row_max1);
        d_qk[18] = _exp2_190;
        d_qk[19] = _exp2_191;
        row_sum1 += _exp2_190 + _exp2_191;
        float _exp2_192 = approx_exp2(d_qk[20] - row_max0);
        float _exp2_193 = approx_exp2(d_qk[21] - row_max0);
        d_qk[20] = _exp2_192;
        d_qk[21] = _exp2_193;
        row_sum0 += _exp2_192 + _exp2_193;
        float _exp2_194 = approx_exp2(d_qk[22] - row_max1);
        float _exp2_195 = approx_exp2(d_qk[23] - row_max1);
        d_qk[22] = _exp2_194;
        d_qk[23] = _exp2_195;
        row_sum1 += _exp2_194 + _exp2_195;
        float _exp2_196 = approx_exp2(d_qk[24] - row_max0);
        float _exp2_197 = approx_exp2(d_qk[25] - row_max0);
        d_qk[24] = _exp2_196;
        d_qk[25] = _exp2_197;
        row_sum0 += _exp2_196 + _exp2_197;
        float _exp2_198 = approx_exp2(d_qk[26] - row_max1);
        float _exp2_199 = approx_exp2(d_qk[27] - row_max1);
        d_qk[26] = _exp2_198;
        d_qk[27] = _exp2_199;
        row_sum1 += _exp2_198 + _exp2_199;
        float _exp2_200 = approx_exp2(d_qk[28] - row_max0);
        float _exp2_201 = approx_exp2(d_qk[29] - row_max0);
        d_qk[28] = _exp2_200;
        d_qk[29] = _exp2_201;
        row_sum0 += _exp2_200 + _exp2_201;
        float _exp2_202 = approx_exp2(d_qk[30] - row_max1);
        float _exp2_203 = approx_exp2(d_qk[31] - row_max1);
        d_qk[30] = _exp2_202;
        d_qk[31] = _exp2_203;
        row_sum1 += _exp2_202 + _exp2_203;
        __nv_bfloat162 _bf16x2_80 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bf16x2_80)[0];
        __nv_bfloat162 _bf16x2_81 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bf16x2_81)[0];
        __nv_bfloat162 _bf16x2_82 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bf16x2_82)[0];
        __nv_bfloat162 _bf16x2_83 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bf16x2_83)[0];
        __nv_bfloat162 _bf16x2_84 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bf16x2_84)[0];
        __nv_bfloat162 _bf16x2_85 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bf16x2_85)[0];
        __nv_bfloat162 _bf16x2_86 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bf16x2_86)[0];
        __nv_bfloat162 _bf16x2_87 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bf16x2_87)[0];
        __nv_bfloat162 _bf16x2_88 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bf16x2_88)[0];
        __nv_bfloat162 _bf16x2_89 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bf16x2_89)[0];
        __nv_bfloat162 _bf16x2_90 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bf16x2_90)[0];
        __nv_bfloat162 _bf16x2_91 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bf16x2_91)[0];
        __nv_bfloat162 _bf16x2_92 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bf16x2_92)[0];
        __nv_bfloat162 _bf16x2_93 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bf16x2_93)[0];
        __nv_bfloat162 _bf16x2_94 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bf16x2_94)[0];
        __nv_bfloat162 _bf16x2_95 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bf16x2_95)[0];
        mbarrier_wait(v_full5_addr, _phase_v_full5_0);
        _phase_v_full5_0 ^= 1;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_b_0_19)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_b_0_19 + 128)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_b_0_19 + 256)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_b_0_19 + 384)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
    row_sum0 += _shfl_xor_24;
    float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
    row_sum0 += _shfl_xor_25;
    float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
    row_sum1 += _shfl_xor_26;
    float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
    row_sum1 += _shfl_xor_27;
    float _rcp_0 = approx_rcp(row_sum0);
    float _rcp_1 = approx_rcp(row_sum1);
    d_o[0] = d_o[0] * _rcp_0;
    d_o[1] = d_o[1] * _rcp_0;
    d_o[4] = d_o[4] * _rcp_0;
    d_o[5] = d_o[5] * _rcp_0;
    d_o[8] = d_o[8] * _rcp_0;
    d_o[9] = d_o[9] * _rcp_0;
    d_o[12] = d_o[12] * _rcp_0;
    d_o[13] = d_o[13] * _rcp_0;
    d_o[16] = d_o[16] * _rcp_0;
    d_o[17] = d_o[17] * _rcp_0;
    d_o[20] = d_o[20] * _rcp_0;
    d_o[21] = d_o[21] * _rcp_0;
    d_o[24] = d_o[24] * _rcp_0;
    d_o[25] = d_o[25] * _rcp_0;
    d_o[28] = d_o[28] * _rcp_0;
    d_o[29] = d_o[29] * _rcp_0;
    d_o[32] = d_o[32] * _rcp_0;
    d_o[33] = d_o[33] * _rcp_0;
    d_o[36] = d_o[36] * _rcp_0;
    d_o[37] = d_o[37] * _rcp_0;
    d_o[40] = d_o[40] * _rcp_0;
    d_o[41] = d_o[41] * _rcp_0;
    d_o[44] = d_o[44] * _rcp_0;
    d_o[45] = d_o[45] * _rcp_0;
    d_o[48] = d_o[48] * _rcp_0;
    d_o[49] = d_o[49] * _rcp_0;
    d_o[52] = d_o[52] * _rcp_0;
    d_o[53] = d_o[53] * _rcp_0;
    d_o[56] = d_o[56] * _rcp_0;
    d_o[57] = d_o[57] * _rcp_0;
    d_o[60] = d_o[60] * _rcp_0;
    d_o[61] = d_o[61] * _rcp_0;
    d_o[2] = d_o[2] * _rcp_1;
    d_o[3] = d_o[3] * _rcp_1;
    d_o[6] = d_o[6] * _rcp_1;
    d_o[7] = d_o[7] * _rcp_1;
    d_o[10] = d_o[10] * _rcp_1;
    d_o[11] = d_o[11] * _rcp_1;
    d_o[14] = d_o[14] * _rcp_1;
    d_o[15] = d_o[15] * _rcp_1;
    d_o[18] = d_o[18] * _rcp_1;
    d_o[19] = d_o[19] * _rcp_1;
    d_o[22] = d_o[22] * _rcp_1;
    d_o[23] = d_o[23] * _rcp_1;
    d_o[26] = d_o[26] * _rcp_1;
    d_o[27] = d_o[27] * _rcp_1;
    d_o[30] = d_o[30] * _rcp_1;
    d_o[31] = d_o[31] * _rcp_1;
    d_o[34] = d_o[34] * _rcp_1;
    d_o[35] = d_o[35] * _rcp_1;
    d_o[38] = d_o[38] * _rcp_1;
    d_o[39] = d_o[39] * _rcp_1;
    d_o[42] = d_o[42] * _rcp_1;
    d_o[43] = d_o[43] * _rcp_1;
    d_o[46] = d_o[46] * _rcp_1;
    d_o[47] = d_o[47] * _rcp_1;
    d_o[50] = d_o[50] * _rcp_1;
    d_o[51] = d_o[51] * _rcp_1;
    d_o[54] = d_o[54] * _rcp_1;
    d_o[55] = d_o[55] * _rcp_1;
    d_o[58] = d_o[58] * _rcp_1;
    d_o[59] = d_o[59] * _rcp_1;
    d_o[62] = d_o[62] * _rcp_1;
    d_o[63] = d_o[63] * _rcp_1;
    int qj = lane & 3;
    int qj1 = qj & 1;
    int qj2 = qj & 2;
    unsigned int o_vec[4];
    unsigned int o_tmp[4];
    int o_row_base = q_row * 128;
    int m_local_r = ((1) ? m0_local : m1_local);
    __nv_bfloat162 _bf16x2_96 = __float22bfloat162_rn(make_float2(d_o[0], d_o[1]));
    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_96)[0];
    __nv_bfloat162 _bf16x2_97 = __float22bfloat162_rn(make_float2(d_o[4], d_o[5]));
    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_97)[0];
    __nv_bfloat162 _bf16x2_98 = __float22bfloat162_rn(make_float2(d_o[8], d_o[9]));
    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_98)[0];
    __nv_bfloat162 _bf16x2_99 = __float22bfloat162_rn(make_float2(d_o[12], d_o[13]));
    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_99)[0];
    unsigned int _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
    o_tmp[0] = _shfl_xor_28;
    unsigned int _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
    o_tmp[1] = _shfl_xor_29;
    unsigned int _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
    o_tmp[2] = _shfl_xor_30;
    unsigned int _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
    o_tmp[3] = _shfl_xor_31;
    {
        o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
    }
    unsigned int _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
    o_tmp[0] = _shfl_xor_32;
    unsigned int _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
    o_tmp[1] = _shfl_xor_33;
    unsigned int _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
    o_tmp[2] = _shfl_xor_34;
    unsigned int _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
    o_tmp[3] = _shfl_xor_35;
    {
        o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
    }
    int o_off = o_row_base + m_local_r * 128 + qj * 8;
    reinterpret_cast<int4*>(O + o_off)[0] = reinterpret_cast<int4*>(o_vec)[0];
    __nv_bfloat162 _bf16x2_100 = __float22bfloat162_rn(make_float2(d_o[16], d_o[17]));
    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_100)[0];
    __nv_bfloat162 _bf16x2_101 = __float22bfloat162_rn(make_float2(d_o[20], d_o[21]));
    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_101)[0];
    __nv_bfloat162 _bf16x2_102 = __float22bfloat162_rn(make_float2(d_o[24], d_o[25]));
    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_102)[0];
    __nv_bfloat162 _bf16x2_103 = __float22bfloat162_rn(make_float2(d_o[28], d_o[29]));
    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_103)[0];
    unsigned int _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
    o_tmp[0] = _shfl_xor_36;
    unsigned int _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
    o_tmp[1] = _shfl_xor_37;
    unsigned int _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
    o_tmp[2] = _shfl_xor_38;
    unsigned int _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
    o_tmp[3] = _shfl_xor_39;
    {
        o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
    }
    unsigned int _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
    o_tmp[0] = _shfl_xor_40;
    unsigned int _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
    o_tmp[1] = _shfl_xor_41;
    unsigned int _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
    o_tmp[2] = _shfl_xor_42;
    unsigned int _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
    o_tmp[3] = _shfl_xor_43;
    {
        o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
    }
    int o_off_0 = o_row_base + m_local_r * 128 + (4 + qj) * 8;
    reinterpret_cast<int4*>(O + o_off_0)[0] = reinterpret_cast<int4*>(o_vec)[0];
    __nv_bfloat162 _bf16x2_104 = __float22bfloat162_rn(make_float2(d_o[32], d_o[33]));
    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_104)[0];
    __nv_bfloat162 _bf16x2_105 = __float22bfloat162_rn(make_float2(d_o[36], d_o[37]));
    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_105)[0];
    __nv_bfloat162 _bf16x2_106 = __float22bfloat162_rn(make_float2(d_o[40], d_o[41]));
    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_106)[0];
    __nv_bfloat162 _bf16x2_107 = __float22bfloat162_rn(make_float2(d_o[44], d_o[45]));
    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_107)[0];
    unsigned int _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
    o_tmp[0] = _shfl_xor_44;
    unsigned int _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
    o_tmp[1] = _shfl_xor_45;
    unsigned int _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
    o_tmp[2] = _shfl_xor_46;
    unsigned int _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
    o_tmp[3] = _shfl_xor_47;
    {
        o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
    }
    unsigned int _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
    o_tmp[0] = _shfl_xor_48;
    unsigned int _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
    o_tmp[1] = _shfl_xor_49;
    unsigned int _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
    o_tmp[2] = _shfl_xor_50;
    unsigned int _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
    o_tmp[3] = _shfl_xor_51;
    {
        o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
    }
    int o_off_1 = o_row_base + m_local_r * 128 + (8 + qj) * 8;
    reinterpret_cast<int4*>(O + o_off_1)[0] = reinterpret_cast<int4*>(o_vec)[0];
    __nv_bfloat162 _bf16x2_108 = __float22bfloat162_rn(make_float2(d_o[48], d_o[49]));
    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_108)[0];
    __nv_bfloat162 _bf16x2_109 = __float22bfloat162_rn(make_float2(d_o[52], d_o[53]));
    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_109)[0];
    __nv_bfloat162 _bf16x2_110 = __float22bfloat162_rn(make_float2(d_o[56], d_o[57]));
    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_110)[0];
    __nv_bfloat162 _bf16x2_111 = __float22bfloat162_rn(make_float2(d_o[60], d_o[61]));
    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_111)[0];
    unsigned int _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
    o_tmp[0] = _shfl_xor_52;
    unsigned int _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
    o_tmp[1] = _shfl_xor_53;
    unsigned int _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
    o_tmp[2] = _shfl_xor_54;
    unsigned int _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
    o_tmp[3] = _shfl_xor_55;
    {
        o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
    }
    unsigned int _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
    o_tmp[0] = _shfl_xor_56;
    unsigned int _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
    o_tmp[1] = _shfl_xor_57;
    unsigned int _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
    o_tmp[2] = _shfl_xor_58;
    unsigned int _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
    o_tmp[3] = _shfl_xor_59;
    {
        o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
    }
    int o_off_2 = o_row_base + m_local_r * 128 + (12 + qj) * 8;
    reinterpret_cast<int4*>(O + o_off_2)[0] = reinterpret_cast<int4*>(o_vec)[0];
    int m_local_r_3 = ((0) ? m0_local : m1_local);
    __nv_bfloat162 _bf16x2_112 = __float22bfloat162_rn(make_float2(d_o[2], d_o[3]));
    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_112)[0];
    __nv_bfloat162 _bf16x2_113 = __float22bfloat162_rn(make_float2(d_o[6], d_o[7]));
    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_113)[0];
    __nv_bfloat162 _bf16x2_114 = __float22bfloat162_rn(make_float2(d_o[10], d_o[11]));
    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_114)[0];
    __nv_bfloat162 _bf16x2_115 = __float22bfloat162_rn(make_float2(d_o[14], d_o[15]));
    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_115)[0];
    unsigned int _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
    o_tmp[0] = _shfl_xor_60;
    unsigned int _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
    o_tmp[1] = _shfl_xor_61;
    unsigned int _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
    o_tmp[2] = _shfl_xor_62;
    unsigned int _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
    o_tmp[3] = _shfl_xor_63;
    {
        o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
    }
    unsigned int _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
    o_tmp[0] = _shfl_xor_64;
    unsigned int _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
    o_tmp[1] = _shfl_xor_65;
    unsigned int _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
    o_tmp[2] = _shfl_xor_66;
    unsigned int _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
    o_tmp[3] = _shfl_xor_67;
    {
        o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
    }
    int o_off_4 = o_row_base + m_local_r_3 * 128 + qj * 8;
    reinterpret_cast<int4*>(O + o_off_4)[0] = reinterpret_cast<int4*>(o_vec)[0];
    __nv_bfloat162 _bf16x2_116 = __float22bfloat162_rn(make_float2(d_o[18], d_o[19]));
    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_116)[0];
    __nv_bfloat162 _bf16x2_117 = __float22bfloat162_rn(make_float2(d_o[22], d_o[23]));
    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_117)[0];
    __nv_bfloat162 _bf16x2_118 = __float22bfloat162_rn(make_float2(d_o[26], d_o[27]));
    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_118)[0];
    __nv_bfloat162 _bf16x2_119 = __float22bfloat162_rn(make_float2(d_o[30], d_o[31]));
    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_119)[0];
    unsigned int _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
    o_tmp[0] = _shfl_xor_68;
    unsigned int _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
    o_tmp[1] = _shfl_xor_69;
    unsigned int _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
    o_tmp[2] = _shfl_xor_70;
    unsigned int _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
    o_tmp[3] = _shfl_xor_71;
    {
        o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
    }
    unsigned int _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
    o_tmp[0] = _shfl_xor_72;
    unsigned int _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
    o_tmp[1] = _shfl_xor_73;
    unsigned int _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
    o_tmp[2] = _shfl_xor_74;
    unsigned int _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
    o_tmp[3] = _shfl_xor_75;
    {
        o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
    }
    int o_off_5 = o_row_base + m_local_r_3 * 128 + (4 + qj) * 8;
    reinterpret_cast<int4*>(O + o_off_5)[0] = reinterpret_cast<int4*>(o_vec)[0];
    __nv_bfloat162 _bf16x2_120 = __float22bfloat162_rn(make_float2(d_o[34], d_o[35]));
    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_120)[0];
    __nv_bfloat162 _bf16x2_121 = __float22bfloat162_rn(make_float2(d_o[38], d_o[39]));
    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_121)[0];
    __nv_bfloat162 _bf16x2_122 = __float22bfloat162_rn(make_float2(d_o[42], d_o[43]));
    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_122)[0];
    __nv_bfloat162 _bf16x2_123 = __float22bfloat162_rn(make_float2(d_o[46], d_o[47]));
    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_123)[0];
    unsigned int _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
    o_tmp[0] = _shfl_xor_76;
    unsigned int _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
    o_tmp[1] = _shfl_xor_77;
    unsigned int _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
    o_tmp[2] = _shfl_xor_78;
    unsigned int _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
    o_tmp[3] = _shfl_xor_79;
    {
        o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
    }
    unsigned int _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
    o_tmp[0] = _shfl_xor_80;
    unsigned int _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
    o_tmp[1] = _shfl_xor_81;
    unsigned int _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
    o_tmp[2] = _shfl_xor_82;
    unsigned int _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
    o_tmp[3] = _shfl_xor_83;
    {
        o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
    }
    int o_off_6 = o_row_base + m_local_r_3 * 128 + (8 + qj) * 8;
    reinterpret_cast<int4*>(O + o_off_6)[0] = reinterpret_cast<int4*>(o_vec)[0];
    __nv_bfloat162 _bf16x2_124 = __float22bfloat162_rn(make_float2(d_o[50], d_o[51]));
    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_124)[0];
    __nv_bfloat162 _bf16x2_125 = __float22bfloat162_rn(make_float2(d_o[54], d_o[55]));
    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_125)[0];
    __nv_bfloat162 _bf16x2_126 = __float22bfloat162_rn(make_float2(d_o[58], d_o[59]));
    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_126)[0];
    __nv_bfloat162 _bf16x2_127 = __float22bfloat162_rn(make_float2(d_o[62], d_o[63]));
    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_127)[0];
    unsigned int _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
    o_tmp[0] = _shfl_xor_84;
    unsigned int _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
    o_tmp[1] = _shfl_xor_85;
    unsigned int _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
    o_tmp[2] = _shfl_xor_86;
    unsigned int _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
    o_tmp[3] = _shfl_xor_87;
    {
        o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
    }
    unsigned int _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
    o_tmp[0] = _shfl_xor_88;
    unsigned int _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
    o_tmp[1] = _shfl_xor_89;
    unsigned int _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
    o_tmp[2] = _shfl_xor_90;
    unsigned int _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
    o_tmp[3] = _shfl_xor_91;
    {
        o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
    }
    {
        o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
    }
    {
        o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
    }
    {
        o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
    }
    int o_off_7 = o_row_base + m_local_r_3 * 128 + (12 + qj) * 8;
    reinterpret_cast<int4*>(O + o_off_7)[0] = reinterpret_cast<int4*>(o_vec)[0];

    // Cleanup
    __syncthreads();
}

} // extern "C"
