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
#define SMEM_FLAG_OFF 50176
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_K_SMEM_OFF 17408
#define SMEM_K_SMEM_STAGE_BYTES 16384
#define SMEM_K_SMEM_STRIDE 16384
#define SMEM_VT_SMEM_OFF 33792
#define SMEM_VT_SMEM_STAGE_BYTES 16384
#define SMEM_VT_SMEM_STRIDE 16384
#define SMEM_TOTAL 50304
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

__global__ __launch_bounds__(128, 4) void
kernel_cake_vsa_sm90_0ef5b67f3dc93b066ede(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, __nv_bfloat16* __restrict__ O, const CakeParamArray<int16_t, 1750> plan, int seqlen_q, int seqlen_k, float scale_log2, __nv_bfloat16* __restrict__ Wo, float* __restrict__ Ws, unsigned int* __restrict__ Wc)
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
    #define v_full0_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int* flag = reinterpret_cast<int*>(smem_raw + 50176);
    const int flag_addr = smem + 50176;
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int k_smem_addr = smem + 17408;
    __nv_bfloat16* vt_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int vt_smem_addr = smem + 33792;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 3 barriers)
    // Mbarriers at smem_raw[0..24)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // k_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // v_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Vt))) : "memory"); }
    int item = bid;
    int mb = seqlen_q / 64;
    int plan_base = item * 3;
    int meta = plan[plan_base];
    int cnt = meta & 15;
    int tile = plan[plan_base + 1];
    int head = tile / mb;
    int qb = tile - head * mb;
    int q_row = head * seqlen_q + qb * 64;
    int kv_base = head * seqlen_k;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(q_full_addr, 16384);
            tma_3d_gmem2smem(q_smem_addr, (&Q), 0, q_row, 0, q_full_addr);
            if (cnt > 0) {
                int blk = plan[plan_base + 2];
                int blk_row = kv_base + blk * 64;
                mbarrier_arrive_expect_tx(k_full0_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr, (&K), 0, blk_row, 0, k_full0_addr);
                mbarrier_arrive_expect_tx(v_full0_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr, (&Vt), 0, 0, blk_row / 8, 0, v_full0_addr);
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
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
    row_sum0 += _shfl_xor_4;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
    row_sum0 += _shfl_xor_5;
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
    row_sum1 += _shfl_xor_6;
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
    row_sum1 += _shfl_xor_7;
    int qj = lane & 3;
    int do_store = 1;
    int nsplit = meta >> 10;
    if (nsplit > 1) {
        unsigned int w_vec[4];
        int item_base = item * 8192 + tid * 8;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(d_o[0 + 0], d_o[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(d_o[0 + 2], d_o[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(d_o[0 + 4], d_o[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(d_o[0 + 6], d_o[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(Wo + item_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(d_o[8 + 0], d_o[8 + 1]);
            _pk[1] = __floats2bfloat162_rn(d_o[8 + 2], d_o[8 + 3]);
            _pk[2] = __floats2bfloat162_rn(d_o[8 + 4], d_o[8 + 5]);
            _pk[3] = __floats2bfloat162_rn(d_o[8 + 6], d_o[8 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(Wo + (item_base + 1024)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(d_o[16 + 0], d_o[16 + 1]);
            _pk[1] = __floats2bfloat162_rn(d_o[16 + 2], d_o[16 + 3]);
            _pk[2] = __floats2bfloat162_rn(d_o[16 + 4], d_o[16 + 5]);
            _pk[3] = __floats2bfloat162_rn(d_o[16 + 6], d_o[16 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(Wo + (item_base + 2048)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(d_o[24 + 0], d_o[24 + 1]);
            _pk[1] = __floats2bfloat162_rn(d_o[24 + 2], d_o[24 + 3]);
            _pk[2] = __floats2bfloat162_rn(d_o[24 + 4], d_o[24 + 5]);
            _pk[3] = __floats2bfloat162_rn(d_o[24 + 6], d_o[24 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(Wo + (item_base + 3072)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(d_o[32 + 0], d_o[32 + 1]);
            _pk[1] = __floats2bfloat162_rn(d_o[32 + 2], d_o[32 + 3]);
            _pk[2] = __floats2bfloat162_rn(d_o[32 + 4], d_o[32 + 5]);
            _pk[3] = __floats2bfloat162_rn(d_o[32 + 6], d_o[32 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(Wo + (item_base + 4096)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(d_o[40 + 0], d_o[40 + 1]);
            _pk[1] = __floats2bfloat162_rn(d_o[40 + 2], d_o[40 + 3]);
            _pk[2] = __floats2bfloat162_rn(d_o[40 + 4], d_o[40 + 5]);
            _pk[3] = __floats2bfloat162_rn(d_o[40 + 6], d_o[40 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(Wo + (item_base + 5120)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(d_o[48 + 0], d_o[48 + 1]);
            _pk[1] = __floats2bfloat162_rn(d_o[48 + 2], d_o[48 + 3]);
            _pk[2] = __floats2bfloat162_rn(d_o[48 + 4], d_o[48 + 5]);
            _pk[3] = __floats2bfloat162_rn(d_o[48 + 6], d_o[48 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(Wo + (item_base + 6144)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(d_o[56 + 0], d_o[56 + 1]);
            _pk[1] = __floats2bfloat162_rn(d_o[56 + 2], d_o[56 + 3]);
            _pk[2] = __floats2bfloat162_rn(d_o[56 + 4], d_o[56 + 5]);
            _pk[3] = __floats2bfloat162_rn(d_o[56 + 6], d_o[56 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(Wo + (item_base + 7168)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int stats_base = item * 128;
        if (qj == 0) {
            *(reinterpret_cast<float*>(Ws + (stats_base + m0_local * 2)) + (0)) = row_max0;
            *(reinterpret_cast<float*>(Ws + (stats_base + m0_local * 2 + 1)) + (0)) = row_sum0;
            *(reinterpret_cast<float*>(Ws + (stats_base + m1_local * 2)) + (0)) = row_max1;
            *(reinterpret_cast<float*>(Ws + (stats_base + m1_local * 2 + 1)) + (0)) = row_sum1;
        }
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        if (tid == 0) {
            unsigned int _atomic_old_0;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                : "=r"(_atomic_old_0) : "l"(&Wc[tile]), "r"(static_cast<uint32_t>(1)) : "memory");
            unsigned int old_count = _atomic_old_0;
            flag[0] = ((old_count + 1 == (unsigned int)nsplit) ? 1 : 0);
        }
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        do_store = flag[0];
        if (do_store != 0) {
            int my_split = meta >> 4 & 63;
            int first_item = item - my_split;
            float merged_max0 = row_max0;
            float merged_max1 = row_max1;
            #pragma unroll 1
            for (int j = 0; j < nsplit; j++) {
                if (j != my_split) {
                    int other_stats = (first_item + j) * 128;
                    float _max_38 = max_noftz(merged_max0, Ws[other_stats + m0_local * 2]);
                    merged_max0 = _max_38;
                    float _max_39 = max_noftz(merged_max1, Ws[other_stats + m1_local * 2]);
                    merged_max1 = _max_39;
                }
            }
            float _exp2_34 = approx_exp2(row_max0 - merged_max0);
            float weight0 = _exp2_34;
            float _exp2_35 = approx_exp2(row_max1 - merged_max1);
            float weight1 = _exp2_35;
            d_o[0] = d_o[0] * weight0;
            d_o[1] = d_o[1] * weight0;
            d_o[4] = d_o[4] * weight0;
            d_o[5] = d_o[5] * weight0;
            d_o[8] = d_o[8] * weight0;
            d_o[9] = d_o[9] * weight0;
            d_o[12] = d_o[12] * weight0;
            d_o[13] = d_o[13] * weight0;
            d_o[16] = d_o[16] * weight0;
            d_o[17] = d_o[17] * weight0;
            d_o[20] = d_o[20] * weight0;
            d_o[21] = d_o[21] * weight0;
            d_o[24] = d_o[24] * weight0;
            d_o[25] = d_o[25] * weight0;
            d_o[28] = d_o[28] * weight0;
            d_o[29] = d_o[29] * weight0;
            d_o[32] = d_o[32] * weight0;
            d_o[33] = d_o[33] * weight0;
            d_o[36] = d_o[36] * weight0;
            d_o[37] = d_o[37] * weight0;
            d_o[40] = d_o[40] * weight0;
            d_o[41] = d_o[41] * weight0;
            d_o[44] = d_o[44] * weight0;
            d_o[45] = d_o[45] * weight0;
            d_o[48] = d_o[48] * weight0;
            d_o[49] = d_o[49] * weight0;
            d_o[52] = d_o[52] * weight0;
            d_o[53] = d_o[53] * weight0;
            d_o[56] = d_o[56] * weight0;
            d_o[57] = d_o[57] * weight0;
            d_o[60] = d_o[60] * weight0;
            d_o[61] = d_o[61] * weight0;
            d_o[2] = d_o[2] * weight1;
            d_o[3] = d_o[3] * weight1;
            d_o[6] = d_o[6] * weight1;
            d_o[7] = d_o[7] * weight1;
            d_o[10] = d_o[10] * weight1;
            d_o[11] = d_o[11] * weight1;
            d_o[14] = d_o[14] * weight1;
            d_o[15] = d_o[15] * weight1;
            d_o[18] = d_o[18] * weight1;
            d_o[19] = d_o[19] * weight1;
            d_o[22] = d_o[22] * weight1;
            d_o[23] = d_o[23] * weight1;
            d_o[26] = d_o[26] * weight1;
            d_o[27] = d_o[27] * weight1;
            d_o[30] = d_o[30] * weight1;
            d_o[31] = d_o[31] * weight1;
            d_o[34] = d_o[34] * weight1;
            d_o[35] = d_o[35] * weight1;
            d_o[38] = d_o[38] * weight1;
            d_o[39] = d_o[39] * weight1;
            d_o[42] = d_o[42] * weight1;
            d_o[43] = d_o[43] * weight1;
            d_o[46] = d_o[46] * weight1;
            d_o[47] = d_o[47] * weight1;
            d_o[50] = d_o[50] * weight1;
            d_o[51] = d_o[51] * weight1;
            d_o[54] = d_o[54] * weight1;
            d_o[55] = d_o[55] * weight1;
            d_o[58] = d_o[58] * weight1;
            d_o[59] = d_o[59] * weight1;
            d_o[62] = d_o[62] * weight1;
            d_o[63] = d_o[63] * weight1;
            row_sum0 = row_sum0 * weight0;
            row_sum1 = row_sum1 * weight1;
            #pragma unroll 1
            for (int j_1 = 0; j_1 < nsplit; j_1++) {
                if (j_1 != my_split) {
                    int other_item = first_item + j_1;
                    int other_stats_1 = other_item * 128;
                    float _exp2_36 = approx_exp2(Ws[other_stats_1 + m0_local * 2] - merged_max0);
                    float other_w0 = _exp2_36;
                    float _exp2_37 = approx_exp2(Ws[other_stats_1 + m1_local * 2] - merged_max1);
                    float other_w1 = _exp2_37;
                    float _fma_0 = __fmaf_rn(Ws[other_stats_1 + m0_local * 2 + 1], other_w0, row_sum0);
                    row_sum0 = _fma_0;
                    float _fma_1 = __fmaf_rn(Ws[other_stats_1 + m1_local * 2 + 1], other_w1, row_sum1);
                    row_sum1 = _fma_1;
                    int other_base = other_item * 8192 + tid * 8;
                    float _vec_load_0[8];
                    {
                        const uint4* _vptr_5 = reinterpret_cast<const uint4*>(Wo + other_base + 0);
                        uint4 _vld_5[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_5[_blk] = _vptr_5[_blk];
                            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_5[_pair]));
                            }
                        }
                    }
                    float _fma_2 = __fmaf_rn(_vec_load_0[0], ((0) ? other_w1 : other_w0), d_o[0]);
                    d_o[0] = _fma_2;
                    float _fma_3 = __fmaf_rn(_vec_load_0[1], ((0) ? other_w1 : other_w0), d_o[1]);
                    d_o[1] = _fma_3;
                    float _fma_4 = __fmaf_rn(_vec_load_0[2], ((1) ? other_w1 : other_w0), d_o[2]);
                    d_o[2] = _fma_4;
                    float _fma_5 = __fmaf_rn(_vec_load_0[3], ((1) ? other_w1 : other_w0), d_o[3]);
                    d_o[3] = _fma_5;
                    float _fma_6 = __fmaf_rn(_vec_load_0[4], ((0) ? other_w1 : other_w0), d_o[4]);
                    d_o[4] = _fma_6;
                    float _fma_7 = __fmaf_rn(_vec_load_0[5], ((0) ? other_w1 : other_w0), d_o[5]);
                    d_o[5] = _fma_7;
                    float _fma_8 = __fmaf_rn(_vec_load_0[6], ((1) ? other_w1 : other_w0), d_o[6]);
                    d_o[6] = _fma_8;
                    float _fma_9 = __fmaf_rn(_vec_load_0[7], ((1) ? other_w1 : other_w0), d_o[7]);
                    d_o[7] = _fma_9;
                    float _vec_load_1[8];
                    {
                        const uint4* _vptr_6 = reinterpret_cast<const uint4*>(Wo + (other_base + 1024) + 0);
                        uint4 _vld_6[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_6[_blk] = _vptr_6[_blk];
                            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_6[_pair]));
                            }
                        }
                    }
                    float _fma_10 = __fmaf_rn(_vec_load_1[0], ((0) ? other_w1 : other_w0), d_o[8]);
                    d_o[8] = _fma_10;
                    float _fma_11 = __fmaf_rn(_vec_load_1[1], ((0) ? other_w1 : other_w0), d_o[9]);
                    d_o[9] = _fma_11;
                    float _fma_12 = __fmaf_rn(_vec_load_1[2], ((1) ? other_w1 : other_w0), d_o[10]);
                    d_o[10] = _fma_12;
                    float _fma_13 = __fmaf_rn(_vec_load_1[3], ((1) ? other_w1 : other_w0), d_o[11]);
                    d_o[11] = _fma_13;
                    float _fma_14 = __fmaf_rn(_vec_load_1[4], ((0) ? other_w1 : other_w0), d_o[12]);
                    d_o[12] = _fma_14;
                    float _fma_15 = __fmaf_rn(_vec_load_1[5], ((0) ? other_w1 : other_w0), d_o[13]);
                    d_o[13] = _fma_15;
                    float _fma_16 = __fmaf_rn(_vec_load_1[6], ((1) ? other_w1 : other_w0), d_o[14]);
                    d_o[14] = _fma_16;
                    float _fma_17 = __fmaf_rn(_vec_load_1[7], ((1) ? other_w1 : other_w0), d_o[15]);
                    d_o[15] = _fma_17;
                    float _vec_load_2[8];
                    {
                        const uint4* _vptr_7 = reinterpret_cast<const uint4*>(Wo + (other_base + 2048) + 0);
                        uint4 _vld_7[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_7[_blk] = _vptr_7[_blk];
                            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_7[_pair]));
                            }
                        }
                    }
                    float _fma_18 = __fmaf_rn(_vec_load_2[0], ((0) ? other_w1 : other_w0), d_o[16]);
                    d_o[16] = _fma_18;
                    float _fma_19 = __fmaf_rn(_vec_load_2[1], ((0) ? other_w1 : other_w0), d_o[17]);
                    d_o[17] = _fma_19;
                    float _fma_20 = __fmaf_rn(_vec_load_2[2], ((1) ? other_w1 : other_w0), d_o[18]);
                    d_o[18] = _fma_20;
                    float _fma_21 = __fmaf_rn(_vec_load_2[3], ((1) ? other_w1 : other_w0), d_o[19]);
                    d_o[19] = _fma_21;
                    float _fma_22 = __fmaf_rn(_vec_load_2[4], ((0) ? other_w1 : other_w0), d_o[20]);
                    d_o[20] = _fma_22;
                    float _fma_23 = __fmaf_rn(_vec_load_2[5], ((0) ? other_w1 : other_w0), d_o[21]);
                    d_o[21] = _fma_23;
                    float _fma_24 = __fmaf_rn(_vec_load_2[6], ((1) ? other_w1 : other_w0), d_o[22]);
                    d_o[22] = _fma_24;
                    float _fma_25 = __fmaf_rn(_vec_load_2[7], ((1) ? other_w1 : other_w0), d_o[23]);
                    d_o[23] = _fma_25;
                    float _vec_load_3[8];
                    {
                        const uint4* _vptr_8 = reinterpret_cast<const uint4*>(Wo + (other_base + 3072) + 0);
                        uint4 _vld_8[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_8[_blk] = _vptr_8[_blk];
                            uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_8[_pair]));
                            }
                        }
                    }
                    float _fma_26 = __fmaf_rn(_vec_load_3[0], ((0) ? other_w1 : other_w0), d_o[24]);
                    d_o[24] = _fma_26;
                    float _fma_27 = __fmaf_rn(_vec_load_3[1], ((0) ? other_w1 : other_w0), d_o[25]);
                    d_o[25] = _fma_27;
                    float _fma_28 = __fmaf_rn(_vec_load_3[2], ((1) ? other_w1 : other_w0), d_o[26]);
                    d_o[26] = _fma_28;
                    float _fma_29 = __fmaf_rn(_vec_load_3[3], ((1) ? other_w1 : other_w0), d_o[27]);
                    d_o[27] = _fma_29;
                    float _fma_30 = __fmaf_rn(_vec_load_3[4], ((0) ? other_w1 : other_w0), d_o[28]);
                    d_o[28] = _fma_30;
                    float _fma_31 = __fmaf_rn(_vec_load_3[5], ((0) ? other_w1 : other_w0), d_o[29]);
                    d_o[29] = _fma_31;
                    float _fma_32 = __fmaf_rn(_vec_load_3[6], ((1) ? other_w1 : other_w0), d_o[30]);
                    d_o[30] = _fma_32;
                    float _fma_33 = __fmaf_rn(_vec_load_3[7], ((1) ? other_w1 : other_w0), d_o[31]);
                    d_o[31] = _fma_33;
                    float _vec_load_4[8];
                    {
                        const uint4* _vptr_9 = reinterpret_cast<const uint4*>(Wo + (other_base + 4096) + 0);
                        uint4 _vld_9[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_9[_blk] = _vptr_9[_blk];
                            uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_9[_pair]));
                            }
                        }
                    }
                    float _fma_34 = __fmaf_rn(_vec_load_4[0], ((0) ? other_w1 : other_w0), d_o[32]);
                    d_o[32] = _fma_34;
                    float _fma_35 = __fmaf_rn(_vec_load_4[1], ((0) ? other_w1 : other_w0), d_o[33]);
                    d_o[33] = _fma_35;
                    float _fma_36 = __fmaf_rn(_vec_load_4[2], ((1) ? other_w1 : other_w0), d_o[34]);
                    d_o[34] = _fma_36;
                    float _fma_37 = __fmaf_rn(_vec_load_4[3], ((1) ? other_w1 : other_w0), d_o[35]);
                    d_o[35] = _fma_37;
                    float _fma_38 = __fmaf_rn(_vec_load_4[4], ((0) ? other_w1 : other_w0), d_o[36]);
                    d_o[36] = _fma_38;
                    float _fma_39 = __fmaf_rn(_vec_load_4[5], ((0) ? other_w1 : other_w0), d_o[37]);
                    d_o[37] = _fma_39;
                    float _fma_40 = __fmaf_rn(_vec_load_4[6], ((1) ? other_w1 : other_w0), d_o[38]);
                    d_o[38] = _fma_40;
                    float _fma_41 = __fmaf_rn(_vec_load_4[7], ((1) ? other_w1 : other_w0), d_o[39]);
                    d_o[39] = _fma_41;
                    float _vec_load_5[8];
                    {
                        const uint4* _vptr_10 = reinterpret_cast<const uint4*>(Wo + (other_base + 5120) + 0);
                        uint4 _vld_10[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_10[_blk] = _vptr_10[_blk];
                            uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_10[_pair]));
                            }
                        }
                    }
                    float _fma_42 = __fmaf_rn(_vec_load_5[0], ((0) ? other_w1 : other_w0), d_o[40]);
                    d_o[40] = _fma_42;
                    float _fma_43 = __fmaf_rn(_vec_load_5[1], ((0) ? other_w1 : other_w0), d_o[41]);
                    d_o[41] = _fma_43;
                    float _fma_44 = __fmaf_rn(_vec_load_5[2], ((1) ? other_w1 : other_w0), d_o[42]);
                    d_o[42] = _fma_44;
                    float _fma_45 = __fmaf_rn(_vec_load_5[3], ((1) ? other_w1 : other_w0), d_o[43]);
                    d_o[43] = _fma_45;
                    float _fma_46 = __fmaf_rn(_vec_load_5[4], ((0) ? other_w1 : other_w0), d_o[44]);
                    d_o[44] = _fma_46;
                    float _fma_47 = __fmaf_rn(_vec_load_5[5], ((0) ? other_w1 : other_w0), d_o[45]);
                    d_o[45] = _fma_47;
                    float _fma_48 = __fmaf_rn(_vec_load_5[6], ((1) ? other_w1 : other_w0), d_o[46]);
                    d_o[46] = _fma_48;
                    float _fma_49 = __fmaf_rn(_vec_load_5[7], ((1) ? other_w1 : other_w0), d_o[47]);
                    d_o[47] = _fma_49;
                    float _vec_load_6[8];
                    {
                        const uint4* _vptr_11 = reinterpret_cast<const uint4*>(Wo + (other_base + 6144) + 0);
                        uint4 _vld_11[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_11[_blk] = _vptr_11[_blk];
                            uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_11[_pair]));
                            }
                        }
                    }
                    float _fma_50 = __fmaf_rn(_vec_load_6[0], ((0) ? other_w1 : other_w0), d_o[48]);
                    d_o[48] = _fma_50;
                    float _fma_51 = __fmaf_rn(_vec_load_6[1], ((0) ? other_w1 : other_w0), d_o[49]);
                    d_o[49] = _fma_51;
                    float _fma_52 = __fmaf_rn(_vec_load_6[2], ((1) ? other_w1 : other_w0), d_o[50]);
                    d_o[50] = _fma_52;
                    float _fma_53 = __fmaf_rn(_vec_load_6[3], ((1) ? other_w1 : other_w0), d_o[51]);
                    d_o[51] = _fma_53;
                    float _fma_54 = __fmaf_rn(_vec_load_6[4], ((0) ? other_w1 : other_w0), d_o[52]);
                    d_o[52] = _fma_54;
                    float _fma_55 = __fmaf_rn(_vec_load_6[5], ((0) ? other_w1 : other_w0), d_o[53]);
                    d_o[53] = _fma_55;
                    float _fma_56 = __fmaf_rn(_vec_load_6[6], ((1) ? other_w1 : other_w0), d_o[54]);
                    d_o[54] = _fma_56;
                    float _fma_57 = __fmaf_rn(_vec_load_6[7], ((1) ? other_w1 : other_w0), d_o[55]);
                    d_o[55] = _fma_57;
                    float _vec_load_7[8];
                    {
                        const uint4* _vptr_12 = reinterpret_cast<const uint4*>(Wo + (other_base + 7168) + 0);
                        uint4 _vld_12[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_12[_blk] = _vptr_12[_blk];
                            uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_12[_pair]));
                            }
                        }
                    }
                    float _fma_58 = __fmaf_rn(_vec_load_7[0], ((0) ? other_w1 : other_w0), d_o[56]);
                    d_o[56] = _fma_58;
                    float _fma_59 = __fmaf_rn(_vec_load_7[1], ((0) ? other_w1 : other_w0), d_o[57]);
                    d_o[57] = _fma_59;
                    float _fma_60 = __fmaf_rn(_vec_load_7[2], ((1) ? other_w1 : other_w0), d_o[58]);
                    d_o[58] = _fma_60;
                    float _fma_61 = __fmaf_rn(_vec_load_7[3], ((1) ? other_w1 : other_w0), d_o[59]);
                    d_o[59] = _fma_61;
                    float _fma_62 = __fmaf_rn(_vec_load_7[4], ((0) ? other_w1 : other_w0), d_o[60]);
                    d_o[60] = _fma_62;
                    float _fma_63 = __fmaf_rn(_vec_load_7[5], ((0) ? other_w1 : other_w0), d_o[61]);
                    d_o[61] = _fma_63;
                    float _fma_64 = __fmaf_rn(_vec_load_7[6], ((1) ? other_w1 : other_w0), d_o[62]);
                    d_o[62] = _fma_64;
                    float _fma_65 = __fmaf_rn(_vec_load_7[7], ((1) ? other_w1 : other_w0), d_o[63]);
                    d_o[63] = _fma_65;
                }
            }
            if (tid == 0) {
                *(reinterpret_cast<unsigned int*>(Wc + tile) + (0)) = 0;
            }
        }
    }
    if (do_store != 0) {
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
        int qj1 = qj & 1;
        int qj2 = qj & 2;
        unsigned int o_vec[4];
        unsigned int o_tmp[4];
        int o_row_base = q_row * 128;
        int m_local_r = ((1) ? m0_local : m1_local);
        __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(d_o[0], d_o[1]));
        o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_16)[0];
        __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(d_o[4], d_o[5]));
        o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_17)[0];
        __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(d_o[8], d_o[9]));
        o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_18)[0];
        __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(d_o[12], d_o[13]));
        o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_19)[0];
        unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
        o_tmp[0] = _shfl_xor_8;
        unsigned int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
        o_tmp[1] = _shfl_xor_9;
        unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
        o_tmp[2] = _shfl_xor_10;
        unsigned int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
        o_tmp[3] = _shfl_xor_11;
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
        unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
        o_tmp[0] = _shfl_xor_12;
        unsigned int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
        o_tmp[1] = _shfl_xor_13;
        unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
        o_tmp[2] = _shfl_xor_14;
        unsigned int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
        o_tmp[3] = _shfl_xor_15;
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
        __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(d_o[16], d_o[17]));
        o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_20)[0];
        __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(d_o[20], d_o[21]));
        o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_21)[0];
        __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(d_o[24], d_o[25]));
        o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_22)[0];
        __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(d_o[28], d_o[29]));
        o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_23)[0];
        unsigned int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
        o_tmp[0] = _shfl_xor_16;
        unsigned int _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
        o_tmp[1] = _shfl_xor_17;
        unsigned int _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
        o_tmp[2] = _shfl_xor_18;
        unsigned int _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
        o_tmp[3] = _shfl_xor_19;
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
        unsigned int _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
        o_tmp[0] = _shfl_xor_20;
        unsigned int _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
        o_tmp[1] = _shfl_xor_21;
        unsigned int _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
        o_tmp[2] = _shfl_xor_22;
        unsigned int _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
        o_tmp[3] = _shfl_xor_23;
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
        __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(d_o[32], d_o[33]));
        o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_24)[0];
        __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(d_o[36], d_o[37]));
        o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_25)[0];
        __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(d_o[40], d_o[41]));
        o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_26)[0];
        __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(d_o[44], d_o[45]));
        o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_27)[0];
        unsigned int _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
        o_tmp[0] = _shfl_xor_24;
        unsigned int _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
        o_tmp[1] = _shfl_xor_25;
        unsigned int _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
        o_tmp[2] = _shfl_xor_26;
        unsigned int _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
        o_tmp[3] = _shfl_xor_27;
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
        unsigned int _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
        o_tmp[0] = _shfl_xor_28;
        unsigned int _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
        o_tmp[1] = _shfl_xor_29;
        unsigned int _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
        o_tmp[2] = _shfl_xor_30;
        unsigned int _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
        o_tmp[3] = _shfl_xor_31;
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
        __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(d_o[48], d_o[49]));
        o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_28)[0];
        __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(d_o[52], d_o[53]));
        o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_29)[0];
        __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(d_o[56], d_o[57]));
        o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_30)[0];
        __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(d_o[60], d_o[61]));
        o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_31)[0];
        unsigned int _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
        o_tmp[0] = _shfl_xor_32;
        unsigned int _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
        o_tmp[1] = _shfl_xor_33;
        unsigned int _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
        o_tmp[2] = _shfl_xor_34;
        unsigned int _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
        o_tmp[3] = _shfl_xor_35;
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
        unsigned int _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
        o_tmp[0] = _shfl_xor_36;
        unsigned int _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
        o_tmp[1] = _shfl_xor_37;
        unsigned int _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
        o_tmp[2] = _shfl_xor_38;
        unsigned int _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
        o_tmp[3] = _shfl_xor_39;
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
        __nv_bfloat162 _bf16x2_32 = __float22bfloat162_rn(make_float2(d_o[2], d_o[3]));
        o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_32)[0];
        __nv_bfloat162 _bf16x2_33 = __float22bfloat162_rn(make_float2(d_o[6], d_o[7]));
        o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_33)[0];
        __nv_bfloat162 _bf16x2_34 = __float22bfloat162_rn(make_float2(d_o[10], d_o[11]));
        o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_34)[0];
        __nv_bfloat162 _bf16x2_35 = __float22bfloat162_rn(make_float2(d_o[14], d_o[15]));
        o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_35)[0];
        unsigned int _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
        o_tmp[0] = _shfl_xor_40;
        unsigned int _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
        o_tmp[1] = _shfl_xor_41;
        unsigned int _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
        o_tmp[2] = _shfl_xor_42;
        unsigned int _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
        o_tmp[3] = _shfl_xor_43;
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
        unsigned int _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
        o_tmp[0] = _shfl_xor_44;
        unsigned int _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
        o_tmp[1] = _shfl_xor_45;
        unsigned int _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
        o_tmp[2] = _shfl_xor_46;
        unsigned int _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
        o_tmp[3] = _shfl_xor_47;
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
        __nv_bfloat162 _bf16x2_36 = __float22bfloat162_rn(make_float2(d_o[18], d_o[19]));
        o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_36)[0];
        __nv_bfloat162 _bf16x2_37 = __float22bfloat162_rn(make_float2(d_o[22], d_o[23]));
        o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_37)[0];
        __nv_bfloat162 _bf16x2_38 = __float22bfloat162_rn(make_float2(d_o[26], d_o[27]));
        o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_38)[0];
        __nv_bfloat162 _bf16x2_39 = __float22bfloat162_rn(make_float2(d_o[30], d_o[31]));
        o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_39)[0];
        unsigned int _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
        o_tmp[0] = _shfl_xor_48;
        unsigned int _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
        o_tmp[1] = _shfl_xor_49;
        unsigned int _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
        o_tmp[2] = _shfl_xor_50;
        unsigned int _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
        o_tmp[3] = _shfl_xor_51;
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
        unsigned int _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
        o_tmp[0] = _shfl_xor_52;
        unsigned int _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
        o_tmp[1] = _shfl_xor_53;
        unsigned int _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
        o_tmp[2] = _shfl_xor_54;
        unsigned int _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
        o_tmp[3] = _shfl_xor_55;
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
        __nv_bfloat162 _bf16x2_40 = __float22bfloat162_rn(make_float2(d_o[34], d_o[35]));
        o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_40)[0];
        __nv_bfloat162 _bf16x2_41 = __float22bfloat162_rn(make_float2(d_o[38], d_o[39]));
        o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_41)[0];
        __nv_bfloat162 _bf16x2_42 = __float22bfloat162_rn(make_float2(d_o[42], d_o[43]));
        o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_42)[0];
        __nv_bfloat162 _bf16x2_43 = __float22bfloat162_rn(make_float2(d_o[46], d_o[47]));
        o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_43)[0];
        unsigned int _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
        o_tmp[0] = _shfl_xor_56;
        unsigned int _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
        o_tmp[1] = _shfl_xor_57;
        unsigned int _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
        o_tmp[2] = _shfl_xor_58;
        unsigned int _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
        o_tmp[3] = _shfl_xor_59;
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
        unsigned int _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
        o_tmp[0] = _shfl_xor_60;
        unsigned int _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
        o_tmp[1] = _shfl_xor_61;
        unsigned int _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
        o_tmp[2] = _shfl_xor_62;
        unsigned int _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
        o_tmp[3] = _shfl_xor_63;
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
        __nv_bfloat162 _bf16x2_44 = __float22bfloat162_rn(make_float2(d_o[50], d_o[51]));
        o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_44)[0];
        __nv_bfloat162 _bf16x2_45 = __float22bfloat162_rn(make_float2(d_o[54], d_o[55]));
        o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_45)[0];
        __nv_bfloat162 _bf16x2_46 = __float22bfloat162_rn(make_float2(d_o[58], d_o[59]));
        o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_46)[0];
        __nv_bfloat162 _bf16x2_47 = __float22bfloat162_rn(make_float2(d_o[62], d_o[63]));
        o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_47)[0];
        unsigned int _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
        o_tmp[0] = _shfl_xor_64;
        unsigned int _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
        o_tmp[1] = _shfl_xor_65;
        unsigned int _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
        o_tmp[2] = _shfl_xor_66;
        unsigned int _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
        o_tmp[3] = _shfl_xor_67;
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
        unsigned int _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
        o_tmp[0] = _shfl_xor_68;
        unsigned int _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
        o_tmp[1] = _shfl_xor_69;
        unsigned int _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
        o_tmp[2] = _shfl_xor_70;
        unsigned int _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
        o_tmp[3] = _shfl_xor_71;
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
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
