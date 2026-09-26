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

#define CAKE_INF CUDART_INF_F
#define NUM_K_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 3
#define NUM_Q_PIPE_STAGES 2
#define NUM_META_PIPE_STAGES 2
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_K_SMEM_OFF 33792
#define SMEM_K_SMEM_STAGE_BYTES 32768
#define SMEM_K_SMEM_STRIDE 32768
#define SMEM_VT_SMEM_A_OFF 132096
#define SMEM_VT_SMEM_A_STAGE_BYTES 16384
#define SMEM_VT_SMEM_A_STRIDE 32768
#define SMEM_VT_SMEM_B_OFF 148480
#define SMEM_VT_SMEM_B_STAGE_BYTES 16384
#define SMEM_VT_SMEM_B_STRIDE 32768
#define SMEM_META_SMEM_OFF 230400
#define SMEM_META_SMEM_STAGE_BYTES 1152
#define SMEM_META_SMEM_STRIDE 1152
#define SMEM_MERGE_O_OFF 17408
#define SMEM_MERGE_O_STAGE_BYTES 16384
#define SMEM_MERGE_O_STRIDE 16384
#define SMEM_MERGE_ML_OFF 231552
#define SMEM_MERGE_ML_STAGE_BYTES 512
#define SMEM_MERGE_ML_STRIDE 512
#define SMEM_SMEM_V7_OFF 148480
#define SMEM_SMEM_V7_STAGE_BYTES 16384
#define SMEM_SMEM_V7_STRIDE 16384
#define SMEM_SMEM_V8_OFF 181248
#define SMEM_SMEM_V8_STAGE_BYTES 16384
#define SMEM_SMEM_V8_STRIDE 16384
#define SMEM_SMEM_V9_OFF 214016
#define SMEM_SMEM_V9_STAGE_BYTES 16384
#define SMEM_SMEM_V9_STRIDE 16384
#define SMEM_TOTAL 232064
#define THREADS 384

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

__global__ __launch_bounds__(384, 1) void
kernel_cake_vsa_sm90_0144b5f123596d98e5f5(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, __nv_bfloat16* __restrict__ O, int* __restrict__ meta, int tile_stride, int seqlen_q, int seqlen_k, float scale_log2, int* __restrict__ dbg, unsigned long long* __restrict__ tl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_ready_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define k_full_addr (mbar_base + 24)
    #define v_full_addr (mbar_base + 48)
    #define k_empty_addr (mbar_base + 72)
    #define v_empty_addr (mbar_base + 96)
    #define meta_full_addr (mbar_base + 120)
    #define meta_empty_addr (mbar_base + 136)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int k_smem_addr = smem + 33792;
    __nv_bfloat16* vt_smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int vt_smem_a_addr = smem + 132096;
    __nv_bfloat16* vt_smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int vt_smem_b_addr = smem + 148480;
    int* meta_smem = reinterpret_cast<int*>(smem_raw + 230400);
    const int meta_smem_addr = smem + 230400;
    float* merge_o = reinterpret_cast<float*>(smem_raw + 17408);
    const int merge_o_addr = smem + 17408;
    float* merge_ml = reinterpret_cast<float*>(smem_raw + 231552);
    const int merge_ml_addr = smem + 231552;
    float* smem_v7 = reinterpret_cast<float*>(smem_raw + 148480);
    const int smem_v7_addr = smem + 148480;
    float* smem_v8 = reinterpret_cast<float*>(smem_raw + 181248);
    const int smem_v8_addr = smem + 181248;
    float* smem_v9 = reinterpret_cast<float*>(smem_raw + 214016);
    const int smem_v9_addr = smem + 214016;
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Vt))) : "memory");
        }
    }
    __syncwarp();

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 19 barriers)
    // Mbarriers at smem_raw[0..152)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // q_empty: 1 barriers, init_count=8
            mbarrier_init(smem + 16, 8);
            // --- pipeline 'k_pipe' ---
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // --- pipeline 'k_pipe' ---
            // k_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 72, 8);
            mbarrier_init(smem + 80, 8);
            mbarrier_init(smem + 88, 8);
            // --- pipeline 'v_pipe' ---
            // v_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 96, 8);
            mbarrier_init(smem + 104, 8);
            mbarrier_init(smem + 112, 8);
            // --- pipeline 'meta_pipe' ---
            // meta_full: 2 barriers, init_count=64
            mbarrier_init(smem + 120, 64);
            mbarrier_init(smem + 128, 64);
            // meta_empty: 2 barriers, init_count=10
            mbarrier_init(smem + 136, 10);
            mbarrier_init(smem + 144, 10);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: producer ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
        { // producer_main
            int cta = bid;
            int row0 = cta * tile_stride;
            if (warp == 1) {
                asm volatile("barrier.sync 8, 384;" ::: "memory");
            } else {
                asm volatile("barrier.arrive 8, 384;" ::: "memory");
            }
            if (warp >= 2) {
                int st_t = (warp - 2) * 32 + lane;
                int nt_st_ld = meta[row0 * 144 + 7];
                int slot = 0;
                int mrow = row0 * 144;
                int w0 = meta[mrow + st_t];
                int w1 = meta[mrow + st_t + 64];
                int w2 = 0;
                if (st_t + 128 < 144) {
                    w2 = meta[mrow + st_t + 128];
                }
                mbarrier_wait(meta_empty_addr + (slot) * 8, 1);
                if (warp == 2) {
                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, w0, 0);
                    int head = _shfl_0;
                    int _shfl_1 = __shfl_sync(0xFFFFFFFF, w0, 1);
                    int qb0 = _shfl_1;
                    int _shfl_2 = __shfl_sync(0xFFFFFFFF, w0, 2);
                    int qb1 = _shfl_2;
                    int _shfl_3 = __shfl_sync(0xFFFFFFFF, w0, 3);
                    int mode_w = _shfl_3;
                    if (elect_sync()) {
                        int q_row0 = head * seqlen_q + qb0 * 64;
                        mbarrier_arrive_expect_tx(q_ready_addr, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(q_smem_addr), "l"((&Q)), "r"(0), "r"(q_row0), "r"(0),
                               "r"(q_ready_addr), "l"(0x12F0000000000000ULL) : "memory");
                        if (mode_w != 1) {
                            int q_row1 = head * seqlen_q + qb1 * 64;
                            mbarrier_arrive_expect_tx(q_ready_addr + 8, 16384);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(q_smem_addr + 16384), "l"((&Q)), "r"(0), "r"(q_row1), "r"(0),
                                   "r"(q_ready_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                meta_smem[slot * 144 + st_t] = w0;
                meta_smem[slot * 144 + st_t + 64] = w1;
                if (st_t + 128 < 144) {
                    meta_smem[slot * 144 + st_t + 128] = w2;
                }
                mbarrier_arrive(meta_full_addr + (slot) * 8);
                int _shfl_4 = __shfl_sync(0xFFFFFFFF, nt_st_ld, 0);
                int nt_st = _shfl_4;
                #pragma unroll 1
                for (int ti = 1; ti < nt_st; ti++) {
                    int slot_0 = ti & 1;
                    int mrow_1 = (row0 + ti) * 144;
                    int w0_2 = meta[mrow_1 + st_t];
                    int w1_3 = meta[mrow_1 + st_t + 64];
                    int w2_4 = 0;
                    if (st_t + 128 < 144) {
                        w2_4 = meta[mrow_1 + st_t + 128];
                    }
                    mbarrier_wait(meta_empty_addr + (slot_0) * 8, (ti >> 1) + 1 & 1);
                    if (warp == 2) {
                        int _shfl_5 = __shfl_sync(0xFFFFFFFF, w0_2, 0);
                        int head_1 = _shfl_5;
                        int _shfl_6 = __shfl_sync(0xFFFFFFFF, w0_2, 1);
                        int qb0_1 = _shfl_6;
                        int _shfl_7 = __shfl_sync(0xFFFFFFFF, w0_2, 2);
                        int qb1_1 = _shfl_7;
                        int _shfl_8 = __shfl_sync(0xFFFFFFFF, w0_2, 3);
                        int mode_w_1 = _shfl_8;
                        if (elect_sync()) {
                            if (ti > 0) {
                                mbarrier_wait(q_empty_addr, ti - 1 & 1);
                            }
                            int q_row0_1 = head_1 * seqlen_q + qb0_1 * 64;
                            mbarrier_arrive_expect_tx(q_ready_addr, 16384);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(q_smem_addr), "l"((&Q)), "r"(0), "r"(q_row0_1), "r"(0),
                                   "r"(q_ready_addr), "l"(0x12F0000000000000ULL) : "memory");
                            if (mode_w_1 != 1) {
                                int q_row1_1 = head_1 * seqlen_q + qb1_1 * 64;
                                mbarrier_arrive_expect_tx(q_ready_addr + 8, 16384);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(q_smem_addr + 16384), "l"((&Q)), "r"(0), "r"(q_row1_1), "r"(0),
                                       "r"(q_ready_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
                            }
                        }
                    }
                    meta_smem[slot_0 * 144 + st_t] = w0_2;
                    meta_smem[slot_0 * 144 + st_t + 64] = w1_3;
                    if (st_t + 128 < 144) {
                        meta_smem[slot_0 * 144 + st_t + 128] = w2_4;
                    }
                    mbarrier_arrive(meta_full_addr + (slot_0) * 8);
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    int gk = 0;
                    mbarrier_wait(meta_full_addr, 0);
                    int nt_k = meta_smem[7];
                    #pragma unroll 1
                    for (int ti_1 = 0; ti_1 < nt_k; ti_1++) {
                        int slot_k = ti_1 & 1;
                        int mb_k = slot_k * 144;
                        if (ti_1 > 0) {
                            mbarrier_wait(meta_full_addr + (slot_k) * 8, ti_1 >> 1 & 1);
                        }
                        int head_k = meta_smem[mb_k];
                        int n_seq_k = meta_smem[mb_k + 4];
                        int kv_base = head_k * seqlen_k;
                        #pragma unroll 1
                        for (int i = 0; i < n_seq_k; i++) {
                            int p = gk + i;
                            int stage = p % 3;
                            mbarrier_wait(k_empty_addr + (stage) * 8, p / 3 + 1 & 1);
                            int sw = meta_smem[mb_k + 8 + i];
                            int blk_a = sw << 16 >> 16;
                            int blk_b = sw >> 16;
                            int kv_row_a = kv_base + blk_a * 64;
                            int kv_row_b = kv_base + blk_b * 64;
                            int has_b = 1;
                            if (blk_b < 0) {
                                has_b = 0;
                            }
                            int k_dst = k_smem_addr + (unsigned int)(stage * 32768);
                            mbarrier_arrive_expect_tx(k_full_addr + (stage) * 8, 16384 * (1 + has_b));
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(k_dst), "l"((&K)), "r"(0), "r"(kv_row_a), "r"(0),
                                   "r"(k_full_addr + (stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(k_dst + 16384), "l"((&K)), "r"(0), "r"(kv_row_a), "r"(1),
                                   "r"(k_full_addr + (stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            if (has_b != 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(k_dst + 8192), "l"((&K)), "r"(0), "r"(kv_row_b), "r"(0),
                                       "r"(k_full_addr + (stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(k_dst + 16384 + 8192), "l"((&K)), "r"(0), "r"(kv_row_b), "r"(1),
                                       "r"(k_full_addr + (stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                        }
                        gk += n_seq_k;
                        mbarrier_arrive(meta_empty_addr + (slot_k) * 8);
                    }
                }
            }
            if (warp == 1) {
                if (elect_sync()) {
                    int gv = 0;
                    mbarrier_wait(meta_full_addr, 0);
                    int nt_v = meta_smem[7];
                    #pragma unroll 1
                    for (int ti_2 = 0; ti_2 < nt_v; ti_2++) {
                        int slot_v = ti_2 & 1;
                        int mb_v = slot_v * 144;
                        if (ti_2 > 0) {
                            mbarrier_wait(meta_full_addr + (slot_v) * 8, ti_2 >> 1 & 1);
                        }
                        int head_v = meta_smem[mb_v];
                        int n_seq_v = meta_smem[mb_v + 4];
                        int kv_base_v = head_v * seqlen_k;
                        #pragma unroll 1
                        for (int i_1 = 0; i_1 < n_seq_v; i_1++) {
                            int pv = gv + i_1;
                            int stage_v = pv % 3;
                            mbarrier_wait(v_empty_addr + (stage_v) * 8, pv / 3 + 1 & 1);
                            int svw = meta_smem[mb_v + 8 + i_1];
                            int vblk_a = svw << 16 >> 16;
                            int vblk_b = svw >> 16;
                            int v_row_a = kv_base_v + vblk_a * 64;
                            int v_row_b = kv_base_v + vblk_b * 64;
                            int v_has_b = 1;
                            if (vblk_b < 0) {
                                v_has_b = 0;
                            }
                            mbarrier_arrive_expect_tx(v_full_addr + (stage_v) * 8, 16384 * (1 + v_has_b));
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(vt_smem_a_addr + (unsigned int)(stage_v * 32768)), "l"((&Vt)), "r"(0), "r"(0), "r"(v_row_a / 8), "r"(0),
                                   "r"(v_full_addr + (stage_v) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            if (v_has_b != 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(vt_smem_b_addr + (unsigned int)(stage_v * 32768)), "l"((&Vt)), "r"(0), "r"(0), "r"(v_row_b / 8), "r"(0),
                                       "r"(v_full_addr + (stage_v) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                        }
                        gv += n_seq_v;
                        mbarrier_arrive(meta_empty_addr + (slot_v) * 8);
                    }
                }
            }
        }
    }
    // ---- Role: main ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 240;");
        { // main_main
            int cta_m = bid;
            int warp_ld = warp - 4;
            int _shfl_9 = __shfl_sync(0xFFFFFFFF, warp_ld, 0);
            int consumer_warp = _shfl_9;
            int cwg = consumer_warp / 4;
            int warp_in_wg = consumer_warp % 4;
            int tid_wg = warp_in_wg * 32 + lane;
            int m0_local = warp_in_wg * 16 + lane / 4;
            int m1_local = m0_local + 8;
            int quad = warp_in_wg * 8 + lane / 4;
            int tid_c = cwg * 128 + tid_wg;
            smem_v7[tid_c] = 0.0f;
            smem_v7[256 + tid_c] = 0.0f;
            smem_v7[512 + tid_c] = 0.0f;
            smem_v7[768 + tid_c] = 0.0f;
            smem_v7[1024 + tid_c] = 0.0f;
            smem_v7[1280 + tid_c] = 0.0f;
            smem_v7[1536 + tid_c] = 0.0f;
            smem_v7[1792 + tid_c] = 0.0f;
            smem_v7[2048 + tid_c] = 0.0f;
            smem_v7[2304 + tid_c] = 0.0f;
            smem_v7[2560 + tid_c] = 0.0f;
            smem_v7[2816 + tid_c] = 0.0f;
            smem_v7[3072 + tid_c] = 0.0f;
            smem_v7[3328 + tid_c] = 0.0f;
            smem_v7[3584 + tid_c] = 0.0f;
            smem_v7[3840 + tid_c] = 0.0f;
            smem_v8[tid_c] = 0.0f;
            smem_v8[256 + tid_c] = 0.0f;
            smem_v8[512 + tid_c] = 0.0f;
            smem_v8[768 + tid_c] = 0.0f;
            smem_v8[1024 + tid_c] = 0.0f;
            smem_v8[1280 + tid_c] = 0.0f;
            smem_v8[1536 + tid_c] = 0.0f;
            smem_v8[1792 + tid_c] = 0.0f;
            smem_v8[2048 + tid_c] = 0.0f;
            smem_v8[2304 + tid_c] = 0.0f;
            smem_v8[2560 + tid_c] = 0.0f;
            smem_v8[2816 + tid_c] = 0.0f;
            smem_v8[3072 + tid_c] = 0.0f;
            smem_v8[3328 + tid_c] = 0.0f;
            smem_v8[3584 + tid_c] = 0.0f;
            smem_v8[3840 + tid_c] = 0.0f;
            smem_v9[tid_c] = 0.0f;
            smem_v9[256 + tid_c] = 0.0f;
            smem_v9[512 + tid_c] = 0.0f;
            smem_v9[768 + tid_c] = 0.0f;
            smem_v9[1024 + tid_c] = 0.0f;
            smem_v9[1280 + tid_c] = 0.0f;
            smem_v9[1536 + tid_c] = 0.0f;
            smem_v9[1792 + tid_c] = 0.0f;
            smem_v9[2048 + tid_c] = 0.0f;
            smem_v9[2304 + tid_c] = 0.0f;
            smem_v9[2560 + tid_c] = 0.0f;
            smem_v9[2816 + tid_c] = 0.0f;
            smem_v9[3072 + tid_c] = 0.0f;
            smem_v9[3328 + tid_c] = 0.0f;
            smem_v9[3584 + tid_c] = 0.0f;
            smem_v9[3840 + tid_c] = 0.0f;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.arrive 8, 384;" ::: "memory");
            float d_o[64];
            float d_qk[64];
            unsigned int p_bf16[32];
            float poly_tf[2];
            int poly_ti[2];
            unsigned int q_frag[32];
            float row_max0 = -CAKE_INF;
            float row_max1 = -CAKE_INF;
            float row_sum0 = 0.0f;
            float row_sum1 = 0.0f;
            int q_ld_row = warp_in_wg * 16 + lane % 8 + 8 * (lane / 8 % 2);
            int q_ld_col_lane = 8 * (lane / 16);
            int gbase = 0;
            int prev = -1;
            mbarrier_wait(meta_full_addr, 0);
            int _shfl_10 = __shfl_sync(0xFFFFFFFF, meta_smem[7], 0);
            int nt_m = _shfl_10;
            #pragma unroll 1
            for (int ti_3 = 0; ti_3 < nt_m; ti_3++) {
                int slot_m = ti_3 & 1;
                int mbase = slot_m * 144;
                if (ti_3 > 0) {
                    mbarrier_wait(meta_full_addr + (slot_m) * 8, ti_3 >> 1 & 1);
                }
                int _shfl_11 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase], 0);
                int head_m = _shfl_11;
                int _shfl_12 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 1], 0);
                int qb0_m = _shfl_12;
                int _shfl_13 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 2], 0);
                int qb1_m = _shfl_13;
                int _shfl_14 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 4], 0);
                int n_seq_m = _shfl_14;
                int _shfl_15 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 5 + cwg], 0);
                int n_own = _shfl_15;
                int _shfl_16 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 5 + 1 - cwg], 0);
                int n_other = _shfl_16;
                int _shfl_17 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 3], 0);
                int mode_m = _shfl_17;
                int is_split = mode_m == 1;
                int my_qb = qb0_m + cwg * (qb1_m - qb0_m);
                int qs = ((is_split != 0) ? 0 : cwg);
                int own_base_w = mbase + 76 + cwg * 34;
                d_o[0] = 0.0f;
                d_o[1] = 0.0f;
                d_o[2] = 0.0f;
                d_o[3] = 0.0f;
                d_o[4] = 0.0f;
                d_o[5] = 0.0f;
                d_o[6] = 0.0f;
                d_o[7] = 0.0f;
                d_o[8] = 0.0f;
                d_o[9] = 0.0f;
                d_o[10] = 0.0f;
                d_o[11] = 0.0f;
                d_o[12] = 0.0f;
                d_o[13] = 0.0f;
                d_o[14] = 0.0f;
                d_o[15] = 0.0f;
                d_o[16] = 0.0f;
                d_o[17] = 0.0f;
                d_o[18] = 0.0f;
                d_o[19] = 0.0f;
                d_o[20] = 0.0f;
                d_o[21] = 0.0f;
                d_o[22] = 0.0f;
                d_o[23] = 0.0f;
                d_o[24] = 0.0f;
                d_o[25] = 0.0f;
                d_o[26] = 0.0f;
                d_o[27] = 0.0f;
                d_o[28] = 0.0f;
                d_o[29] = 0.0f;
                d_o[30] = 0.0f;
                d_o[31] = 0.0f;
                d_o[32] = 0.0f;
                d_o[33] = 0.0f;
                d_o[34] = 0.0f;
                d_o[35] = 0.0f;
                d_o[36] = 0.0f;
                d_o[37] = 0.0f;
                d_o[38] = 0.0f;
                d_o[39] = 0.0f;
                d_o[40] = 0.0f;
                d_o[41] = 0.0f;
                d_o[42] = 0.0f;
                d_o[43] = 0.0f;
                d_o[44] = 0.0f;
                d_o[45] = 0.0f;
                d_o[46] = 0.0f;
                d_o[47] = 0.0f;
                d_o[48] = 0.0f;
                d_o[49] = 0.0f;
                d_o[50] = 0.0f;
                d_o[51] = 0.0f;
                d_o[52] = 0.0f;
                d_o[53] = 0.0f;
                d_o[54] = 0.0f;
                d_o[55] = 0.0f;
                d_o[56] = 0.0f;
                d_o[57] = 0.0f;
                d_o[58] = 0.0f;
                d_o[59] = 0.0f;
                d_o[60] = 0.0f;
                d_o[61] = 0.0f;
                d_o[62] = 0.0f;
                d_o[63] = 0.0f;
                row_max0 = -CAKE_INF;
                row_max1 = -CAKE_INF;
                row_sum0 = 0.0f;
                row_sum1 = 0.0f;
                mbarrier_wait(q_ready_addr + (qs) * 8, ti_3 & 1);
                int q_half_base = q_smem_addr + (unsigned int)(qs * 16384) + (unsigned int)(q_ld_row * 128);
                int q_col_bytes = q_ld_col_lane * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[0]), "=r"(q_frag[1]), "=r"(q_frag[2]), "=r"(q_frag[3])
                    : "r"((q_half_base + (q_col_bytes ^ (q_half_base >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_0 = q_smem_addr + (unsigned int)(qs * 16384) + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_1 = (16 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[4]), "=r"(q_frag[5]), "=r"(q_frag[6]), "=r"(q_frag[7])
                    : "r"((q_half_base_0 + (q_col_bytes_1 ^ (q_half_base_0 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_2 = q_smem_addr + (unsigned int)(qs * 16384) + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_3 = (32 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[8]), "=r"(q_frag[9]), "=r"(q_frag[10]), "=r"(q_frag[11])
                    : "r"((q_half_base_2 + (q_col_bytes_3 ^ (q_half_base_2 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_4 = q_smem_addr + (unsigned int)(qs * 16384) + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_5 = (48 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[12]), "=r"(q_frag[13]), "=r"(q_frag[14]), "=r"(q_frag[15])
                    : "r"((q_half_base_4 + (q_col_bytes_5 ^ (q_half_base_4 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_6 = q_smem_addr + (unsigned int)(qs * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_7 = q_ld_col_lane * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[16]), "=r"(q_frag[17]), "=r"(q_frag[18]), "=r"(q_frag[19])
                    : "r"((q_half_base_6 + (q_col_bytes_7 ^ (q_half_base_6 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_8 = q_smem_addr + (unsigned int)(qs * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_9 = (16 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[20]), "=r"(q_frag[21]), "=r"(q_frag[22]), "=r"(q_frag[23])
                    : "r"((q_half_base_8 + (q_col_bytes_9 ^ (q_half_base_8 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_10 = q_smem_addr + (unsigned int)(qs * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_11 = (32 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[24]), "=r"(q_frag[25]), "=r"(q_frag[26]), "=r"(q_frag[27])
                    : "r"((q_half_base_10 + (q_col_bytes_11 ^ (q_half_base_10 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_12 = q_smem_addr + (unsigned int)(qs * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_13 = (48 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[28]), "=r"(q_frag[29]), "=r"(q_frag[30]), "=r"(q_frag[31])
                    : "r"((q_half_base_12 + (q_col_bytes_13 ^ (q_half_base_12 >> 7 & 7) << 4)))
                    : "memory");
                if (elect_sync()) {
                    mbarrier_arrive(q_empty_addr);
                }
                int cur_pos = 0;
                int cur_has2 = 1;
                if (n_own > 0) {
                    int w_e = meta_smem[own_base_w];
                    int e0 = w_e & 65535;
                    int pos0 = gbase + (e0 >> 3);
                    int has2_0 = e0 >> 2 & 1;
                    #pragma unroll 1
                    for (int p_1 = prev + 1; p_1 < pos0; p_1++) {
                        mbarrier_wait(k_full_addr + (p_1 % 3) * 8, p_1 / 3 & 1);
                        if (elect_sync()) {
                            mbarrier_arrive(k_empty_addr + (p_1 % 3) * 8);
                        }
                        mbarrier_wait(v_full_addr + (p_1 % 3) * 8, p_1 / 3 & 1);
                        if (elect_sync()) {
                            mbarrier_arrive(v_empty_addr + (p_1 % 3) * 8);
                        }
                    }
                    cur_pos = pos0;
                    cur_has2 = has2_0;
                    int stage_0 = pos0 % 3;
                    mbarrier_wait(k_full_addr + (stage_0) * 8, pos0 / 3 & 1);
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_0 = (((uint64_t)(((k_smem_addr + (unsigned int)(stage_0 * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "l"(_wgmma_b_0_0)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[4]), "r"(q_frag[(4) + 1]), "r"(q_frag[(4) + 2]), "r"(q_frag[(4) + 3]), "l"(_wgmma_b_0_0 + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[8]), "r"(q_frag[(8) + 1]), "r"(q_frag[(8) + 2]), "r"(q_frag[(8) + 3]), "l"(_wgmma_b_0_0 + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[12]), "r"(q_frag[(12) + 1]), "r"(q_frag[(12) + 2]), "r"(q_frag[(12) + 3]), "l"(_wgmma_b_0_0 + 6)
                        : "memory");
                    uint64_t _wgmma_desc_1 = (((uint64_t)(((k_smem_addr + (unsigned int)(stage_0 * 32768) + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[16]), "r"(q_frag[(16) + 1]), "r"(q_frag[(16) + 2]), "r"(q_frag[(16) + 3]), "l"(_wgmma_b_0_1)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[20]), "r"(q_frag[(20) + 1]), "r"(q_frag[(20) + 2]), "r"(q_frag[(20) + 3]), "l"(_wgmma_b_0_1 + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[24]), "r"(q_frag[(24) + 1]), "r"(q_frag[(24) + 2]), "r"(q_frag[(24) + 3]), "l"(_wgmma_b_0_1 + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[28]), "r"(q_frag[(28) + 1]), "r"(q_frag[(28) + 2]), "r"(q_frag[(28) + 3]), "l"(_wgmma_b_0_1 + 6)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(k_empty_addr + (pos0 % 3) * 8);
                    }
                    float f_max0 = -CAKE_INF;
                    float f_max1 = -CAKE_INF;
                    float m0 = -CAKE_INF;
                    float m1 = -CAKE_INF;
                    if (scale_log2 >= 0.0f) {
                        float _max_0 = max_noftz(d_qk[0], d_qk[1]);
                        float _max_1 = max_noftz(d_qk[4], d_qk[5]);
                        float _max_2 = max_noftz(d_qk[8], d_qk[9]);
                        float _max_3 = max_noftz(d_qk[12], d_qk[13]);
                        float _max_4 = max_noftz(d_qk[16], d_qk[17]);
                        float _max_5 = max_noftz(d_qk[20], d_qk[21]);
                        float _max_6 = max_noftz(d_qk[24], d_qk[25]);
                        float _max_7 = max_noftz(d_qk[28], d_qk[29]);
                        float _max_8 = max_noftz(_max_0, _max_1);
                        float _max_9 = max_noftz(_max_2, _max_3);
                        float _max_10 = max_noftz(_max_4, _max_5);
                        float _max_11 = max_noftz(_max_6, _max_7);
                        float _max_12 = max_noftz(_max_8, _max_9);
                        float _max_13 = max_noftz(_max_10, _max_11);
                        float _max_14 = max_noftz(_max_12, _max_13);
                        float _max_15 = max_noftz(d_qk[2], d_qk[3]);
                        float _max_16 = max_noftz(d_qk[6], d_qk[7]);
                        float _max_17 = max_noftz(d_qk[10], d_qk[11]);
                        float _max_18 = max_noftz(d_qk[14], d_qk[15]);
                        float _max_19 = max_noftz(d_qk[18], d_qk[19]);
                        float _max_20 = max_noftz(d_qk[22], d_qk[23]);
                        float _max_21 = max_noftz(d_qk[26], d_qk[27]);
                        float _max_22 = max_noftz(d_qk[30], d_qk[31]);
                        float _max_23 = max_noftz(_max_15, _max_16);
                        float _max_24 = max_noftz(_max_17, _max_18);
                        float _max_25 = max_noftz(_max_19, _max_20);
                        float _max_26 = max_noftz(_max_21, _max_22);
                        float _max_27 = max_noftz(_max_23, _max_24);
                        float _max_28 = max_noftz(_max_25, _max_26);
                        float _max_29 = max_noftz(_max_27, _max_28);
                        float _max_30 = max_noftz(d_qk[32], d_qk[33]);
                        float _max_31 = max_noftz(d_qk[36], d_qk[37]);
                        float _max_32 = max_noftz(d_qk[40], d_qk[41]);
                        float _max_33 = max_noftz(d_qk[44], d_qk[45]);
                        float _max_34 = max_noftz(d_qk[48], d_qk[49]);
                        float _max_35 = max_noftz(d_qk[52], d_qk[53]);
                        float _max_36 = max_noftz(d_qk[56], d_qk[57]);
                        float _max_37 = max_noftz(d_qk[60], d_qk[61]);
                        float _max_38 = max_noftz(_max_30, _max_31);
                        float _max_39 = max_noftz(_max_32, _max_33);
                        float _max_40 = max_noftz(_max_34, _max_35);
                        float _max_41 = max_noftz(_max_36, _max_37);
                        float _max_42 = max_noftz(_max_38, _max_39);
                        float _max_43 = max_noftz(_max_40, _max_41);
                        float _max_44 = max_noftz(_max_42, _max_43);
                        float _max_45 = max_noftz(d_qk[34], d_qk[35]);
                        float _max_46 = max_noftz(d_qk[38], d_qk[39]);
                        float _max_47 = max_noftz(d_qk[42], d_qk[43]);
                        float _max_48 = max_noftz(d_qk[46], d_qk[47]);
                        float _max_49 = max_noftz(d_qk[50], d_qk[51]);
                        float _max_50 = max_noftz(d_qk[54], d_qk[55]);
                        float _max_51 = max_noftz(d_qk[58], d_qk[59]);
                        float _max_52 = max_noftz(d_qk[62], d_qk[63]);
                        float _max_53 = max_noftz(_max_45, _max_46);
                        float _max_54 = max_noftz(_max_47, _max_48);
                        float _max_55 = max_noftz(_max_49, _max_50);
                        float _max_56 = max_noftz(_max_51, _max_52);
                        float _max_57 = max_noftz(_max_53, _max_54);
                        float _max_58 = max_noftz(_max_55, _max_56);
                        float _max_59 = max_noftz(_max_57, _max_58);
                        float _max_60 = max_noftz(_max_14, _max_44);
                        m0 = ((has2_0 != 0) ? _max_60 : _max_14);
                        float _max_61 = max_noftz(_max_29, _max_59);
                        m1 = ((has2_0 != 0) ? _max_61 : _max_29);
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, m0, 2);
                        float _max_62 = max_noftz(m0, _shfl_xor_0);
                        m0 = _max_62;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, m0, 1);
                        float _max_63 = max_noftz(m0, _shfl_xor_1);
                        m0 = _max_63;
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, m1, 2);
                        float _max_64 = max_noftz(m1, _shfl_xor_2);
                        m1 = _max_64;
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, m1, 1);
                        float _max_65 = max_noftz(m1, _shfl_xor_3);
                        m1 = _max_65;
                    } else {
                        float _min_0 = fminf(d_qk[0], d_qk[1]);
                        float _min_1 = fminf(d_qk[4], d_qk[5]);
                        float _min_2 = fminf(d_qk[8], d_qk[9]);
                        float _min_3 = fminf(d_qk[12], d_qk[13]);
                        float _min_4 = fminf(d_qk[16], d_qk[17]);
                        float _min_5 = fminf(d_qk[20], d_qk[21]);
                        float _min_6 = fminf(d_qk[24], d_qk[25]);
                        float _min_7 = fminf(d_qk[28], d_qk[29]);
                        float _min_8 = fminf(_min_0, _min_1);
                        float _min_9 = fminf(_min_2, _min_3);
                        float _min_10 = fminf(_min_4, _min_5);
                        float _min_11 = fminf(_min_6, _min_7);
                        float _min_12 = fminf(_min_8, _min_9);
                        float _min_13 = fminf(_min_10, _min_11);
                        float _min_14 = fminf(_min_12, _min_13);
                        float _min_15 = fminf(d_qk[2], d_qk[3]);
                        float _min_16 = fminf(d_qk[6], d_qk[7]);
                        float _min_17 = fminf(d_qk[10], d_qk[11]);
                        float _min_18 = fminf(d_qk[14], d_qk[15]);
                        float _min_19 = fminf(d_qk[18], d_qk[19]);
                        float _min_20 = fminf(d_qk[22], d_qk[23]);
                        float _min_21 = fminf(d_qk[26], d_qk[27]);
                        float _min_22 = fminf(d_qk[30], d_qk[31]);
                        float _min_23 = fminf(_min_15, _min_16);
                        float _min_24 = fminf(_min_17, _min_18);
                        float _min_25 = fminf(_min_19, _min_20);
                        float _min_26 = fminf(_min_21, _min_22);
                        float _min_27 = fminf(_min_23, _min_24);
                        float _min_28 = fminf(_min_25, _min_26);
                        float _min_29 = fminf(_min_27, _min_28);
                        float _min_30 = fminf(d_qk[32], d_qk[33]);
                        float _min_31 = fminf(d_qk[36], d_qk[37]);
                        float _min_32 = fminf(d_qk[40], d_qk[41]);
                        float _min_33 = fminf(d_qk[44], d_qk[45]);
                        float _min_34 = fminf(d_qk[48], d_qk[49]);
                        float _min_35 = fminf(d_qk[52], d_qk[53]);
                        float _min_36 = fminf(d_qk[56], d_qk[57]);
                        float _min_37 = fminf(d_qk[60], d_qk[61]);
                        float _min_38 = fminf(_min_30, _min_31);
                        float _min_39 = fminf(_min_32, _min_33);
                        float _min_40 = fminf(_min_34, _min_35);
                        float _min_41 = fminf(_min_36, _min_37);
                        float _min_42 = fminf(_min_38, _min_39);
                        float _min_43 = fminf(_min_40, _min_41);
                        float _min_44 = fminf(_min_42, _min_43);
                        float _min_45 = fminf(d_qk[34], d_qk[35]);
                        float _min_46 = fminf(d_qk[38], d_qk[39]);
                        float _min_47 = fminf(d_qk[42], d_qk[43]);
                        float _min_48 = fminf(d_qk[46], d_qk[47]);
                        float _min_49 = fminf(d_qk[50], d_qk[51]);
                        float _min_50 = fminf(d_qk[54], d_qk[55]);
                        float _min_51 = fminf(d_qk[58], d_qk[59]);
                        float _min_52 = fminf(d_qk[62], d_qk[63]);
                        float _min_53 = fminf(_min_45, _min_46);
                        float _min_54 = fminf(_min_47, _min_48);
                        float _min_55 = fminf(_min_49, _min_50);
                        float _min_56 = fminf(_min_51, _min_52);
                        float _min_57 = fminf(_min_53, _min_54);
                        float _min_58 = fminf(_min_55, _min_56);
                        float _min_59 = fminf(_min_57, _min_58);
                        float _min_60 = fminf(_min_14, _min_44);
                        m0 = ((has2_0 != 0) ? _min_60 : _min_14);
                        float _min_61 = fminf(_min_29, _min_59);
                        m1 = ((has2_0 != 0) ? _min_61 : _min_29);
                        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, m0, 2);
                        float _min_62 = fminf(m0, _shfl_xor_4);
                        m0 = _min_62;
                        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, m0, 1);
                        float _min_63 = fminf(m0, _shfl_xor_5);
                        m0 = _min_63;
                        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, m1, 2);
                        float _min_64 = fminf(m1, _shfl_xor_6);
                        m1 = _min_64;
                        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, m1, 1);
                        float _min_65 = fminf(m1, _shfl_xor_7);
                        m1 = _min_65;
                    }
                    f_max0 = m0;
                    f_max1 = m1;
                    row_max0 = f_max0 * scale_log2;
                    row_max1 = f_max1 * scale_log2;
                    float f_sum0 = 0.0f;
                    float f_sum1 = 0.0f;
                    float sa0 = 0.0f;
                    float sa1 = 0.0f;
                    float sb0 = 0.0f;
                    float sb1 = 0.0f;
                    {
                        float _exp2_0 = approx_exp2(d_qk[0] * scale_log2 - row_max0);
                        float _exp2_1 = approx_exp2(d_qk[1] * scale_log2 - row_max0);
                        d_qk[0] = _exp2_0;
                        d_qk[1] = _exp2_1;
                        sa0 += _exp2_0 + _exp2_1;
                        float _exp2_2 = approx_exp2(d_qk[2] * scale_log2 - row_max1);
                        float _exp2_3 = approx_exp2(d_qk[3] * scale_log2 - row_max1);
                        d_qk[2] = _exp2_2;
                        d_qk[3] = _exp2_3;
                        sa1 += _exp2_2 + _exp2_3;
                        float _exp2_4 = approx_exp2(d_qk[4] * scale_log2 - row_max0);
                        float _exp2_5 = approx_exp2(d_qk[5] * scale_log2 - row_max0);
                        d_qk[4] = _exp2_4;
                        d_qk[5] = _exp2_5;
                        sa0 += _exp2_4 + _exp2_5;
                        float _exp2_6 = approx_exp2(d_qk[6] * scale_log2 - row_max1);
                        float _exp2_7 = approx_exp2(d_qk[7] * scale_log2 - row_max1);
                        d_qk[6] = _exp2_6;
                        d_qk[7] = _exp2_7;
                        sa1 += _exp2_6 + _exp2_7;
                        float _exp2_8 = approx_exp2(d_qk[8] * scale_log2 - row_max0);
                        float _exp2_9 = approx_exp2(d_qk[9] * scale_log2 - row_max0);
                        d_qk[8] = _exp2_8;
                        d_qk[9] = _exp2_9;
                        sa0 += _exp2_8 + _exp2_9;
                        float _exp2_10 = approx_exp2(d_qk[10] * scale_log2 - row_max1);
                        float _exp2_11 = approx_exp2(d_qk[11] * scale_log2 - row_max1);
                        d_qk[10] = _exp2_10;
                        d_qk[11] = _exp2_11;
                        sa1 += _exp2_10 + _exp2_11;
                        float _exp2_12 = approx_exp2(d_qk[12] * scale_log2 - row_max0);
                        float _exp2_13 = approx_exp2(d_qk[13] * scale_log2 - row_max0);
                        d_qk[12] = _exp2_12;
                        d_qk[13] = _exp2_13;
                        sa0 += _exp2_12 + _exp2_13;
                        float _exp2_14 = approx_exp2(d_qk[14] * scale_log2 - row_max1);
                        float _exp2_15 = approx_exp2(d_qk[15] * scale_log2 - row_max1);
                        d_qk[14] = _exp2_14;
                        d_qk[15] = _exp2_15;
                        sa1 += _exp2_14 + _exp2_15;
                        float _exp2_16 = approx_exp2(d_qk[16] * scale_log2 - row_max0);
                        float _exp2_17 = approx_exp2(d_qk[17] * scale_log2 - row_max0);
                        d_qk[16] = _exp2_16;
                        d_qk[17] = _exp2_17;
                        sa0 += _exp2_16 + _exp2_17;
                        float _exp2_18 = approx_exp2(d_qk[18] * scale_log2 - row_max1);
                        float _exp2_19 = approx_exp2(d_qk[19] * scale_log2 - row_max1);
                        d_qk[18] = _exp2_18;
                        d_qk[19] = _exp2_19;
                        sa1 += _exp2_18 + _exp2_19;
                        float _exp2_20 = approx_exp2(d_qk[20] * scale_log2 - row_max0);
                        float _exp2_21 = approx_exp2(d_qk[21] * scale_log2 - row_max0);
                        d_qk[20] = _exp2_20;
                        d_qk[21] = _exp2_21;
                        sa0 += _exp2_20 + _exp2_21;
                        float _exp2_22 = approx_exp2(d_qk[22] * scale_log2 - row_max1);
                        float _exp2_23 = approx_exp2(d_qk[23] * scale_log2 - row_max1);
                        d_qk[22] = _exp2_22;
                        d_qk[23] = _exp2_23;
                        sa1 += _exp2_22 + _exp2_23;
                        float _exp2_24 = approx_exp2(d_qk[24] * scale_log2 - row_max0);
                        float _exp2_25 = approx_exp2(d_qk[25] * scale_log2 - row_max0);
                        d_qk[24] = _exp2_24;
                        d_qk[25] = _exp2_25;
                        sa0 += _exp2_24 + _exp2_25;
                        float _exp2_26 = approx_exp2(d_qk[26] * scale_log2 - row_max1);
                        float _exp2_27 = approx_exp2(d_qk[27] * scale_log2 - row_max1);
                        d_qk[26] = _exp2_26;
                        d_qk[27] = _exp2_27;
                        sa1 += _exp2_26 + _exp2_27;
                        float _exp2_28 = approx_exp2(d_qk[28] * scale_log2 - row_max0);
                        float _exp2_29 = approx_exp2(d_qk[29] * scale_log2 - row_max0);
                        d_qk[28] = _exp2_28;
                        d_qk[29] = _exp2_29;
                        sa0 += _exp2_28 + _exp2_29;
                        float _exp2_30 = approx_exp2(d_qk[30] * scale_log2 - row_max1);
                        float _exp2_31 = approx_exp2(d_qk[31] * scale_log2 - row_max1);
                        d_qk[30] = _exp2_30;
                        d_qk[31] = _exp2_31;
                        sa1 += _exp2_30 + _exp2_31;
                        float _exp2_32 = approx_exp2(d_qk[32] * scale_log2 - row_max0);
                        float _exp2_33 = approx_exp2(d_qk[33] * scale_log2 - row_max0);
                        d_qk[32] = _exp2_32;
                        d_qk[33] = _exp2_33;
                        sb0 += _exp2_32 + _exp2_33;
                        float _exp2_34 = approx_exp2(d_qk[34] * scale_log2 - row_max1);
                        float _exp2_35 = approx_exp2(d_qk[35] * scale_log2 - row_max1);
                        d_qk[34] = _exp2_34;
                        d_qk[35] = _exp2_35;
                        sb1 += _exp2_34 + _exp2_35;
                        float _exp2_36 = approx_exp2(d_qk[36] * scale_log2 - row_max0);
                        float _exp2_37 = approx_exp2(d_qk[37] * scale_log2 - row_max0);
                        d_qk[36] = _exp2_36;
                        d_qk[37] = _exp2_37;
                        sb0 += _exp2_36 + _exp2_37;
                        float _exp2_38 = approx_exp2(d_qk[38] * scale_log2 - row_max1);
                        float _exp2_39 = approx_exp2(d_qk[39] * scale_log2 - row_max1);
                        d_qk[38] = _exp2_38;
                        d_qk[39] = _exp2_39;
                        sb1 += _exp2_38 + _exp2_39;
                        float _exp2_40 = approx_exp2(d_qk[40] * scale_log2 - row_max0);
                        float _exp2_41 = approx_exp2(d_qk[41] * scale_log2 - row_max0);
                        d_qk[40] = _exp2_40;
                        d_qk[41] = _exp2_41;
                        sb0 += _exp2_40 + _exp2_41;
                        float _exp2_42 = approx_exp2(d_qk[42] * scale_log2 - row_max1);
                        float _exp2_43 = approx_exp2(d_qk[43] * scale_log2 - row_max1);
                        d_qk[42] = _exp2_42;
                        d_qk[43] = _exp2_43;
                        sb1 += _exp2_42 + _exp2_43;
                        float _exp2_44 = approx_exp2(d_qk[44] * scale_log2 - row_max0);
                        float _exp2_45 = approx_exp2(d_qk[45] * scale_log2 - row_max0);
                        d_qk[44] = _exp2_44;
                        d_qk[45] = _exp2_45;
                        sb0 += _exp2_44 + _exp2_45;
                        float _exp2_46 = approx_exp2(d_qk[46] * scale_log2 - row_max1);
                        float _exp2_47 = approx_exp2(d_qk[47] * scale_log2 - row_max1);
                        d_qk[46] = _exp2_46;
                        d_qk[47] = _exp2_47;
                        sb1 += _exp2_46 + _exp2_47;
                        float _exp2_48 = approx_exp2(d_qk[48] * scale_log2 - row_max0);
                        float _exp2_49 = approx_exp2(d_qk[49] * scale_log2 - row_max0);
                        d_qk[48] = _exp2_48;
                        d_qk[49] = _exp2_49;
                        sb0 += _exp2_48 + _exp2_49;
                        float _exp2_50 = approx_exp2(d_qk[50] * scale_log2 - row_max1);
                        float _exp2_51 = approx_exp2(d_qk[51] * scale_log2 - row_max1);
                        d_qk[50] = _exp2_50;
                        d_qk[51] = _exp2_51;
                        sb1 += _exp2_50 + _exp2_51;
                        float _exp2_52 = approx_exp2(d_qk[52] * scale_log2 - row_max0);
                        float _exp2_53 = approx_exp2(d_qk[53] * scale_log2 - row_max0);
                        d_qk[52] = _exp2_52;
                        d_qk[53] = _exp2_53;
                        sb0 += _exp2_52 + _exp2_53;
                        float _exp2_54 = approx_exp2(d_qk[54] * scale_log2 - row_max1);
                        float _exp2_55 = approx_exp2(d_qk[55] * scale_log2 - row_max1);
                        d_qk[54] = _exp2_54;
                        d_qk[55] = _exp2_55;
                        sb1 += _exp2_54 + _exp2_55;
                        float _exp2_56 = approx_exp2(d_qk[56] * scale_log2 - row_max0);
                        float _exp2_57 = approx_exp2(d_qk[57] * scale_log2 - row_max0);
                        d_qk[56] = _exp2_56;
                        d_qk[57] = _exp2_57;
                        sb0 += _exp2_56 + _exp2_57;
                        float _exp2_58 = approx_exp2(d_qk[58] * scale_log2 - row_max1);
                        float _exp2_59 = approx_exp2(d_qk[59] * scale_log2 - row_max1);
                        d_qk[58] = _exp2_58;
                        d_qk[59] = _exp2_59;
                        sb1 += _exp2_58 + _exp2_59;
                        float _exp2_60 = approx_exp2(d_qk[60] * scale_log2 - row_max0);
                        float _exp2_61 = approx_exp2(d_qk[61] * scale_log2 - row_max0);
                        d_qk[60] = _exp2_60;
                        d_qk[61] = _exp2_61;
                        sb0 += _exp2_60 + _exp2_61;
                        float _exp2_62 = approx_exp2(d_qk[62] * scale_log2 - row_max1);
                        float _exp2_63 = approx_exp2(d_qk[63] * scale_log2 - row_max1);
                        d_qk[62] = _exp2_62;
                        d_qk[63] = _exp2_63;
                        sb1 += _exp2_62 + _exp2_63;
                    }
                    f_sum0 += sa0 + ((has2_0 != 0) ? sb0 : 0.0f);
                    f_sum1 += sa1 + ((has2_0 != 0) ? sb1 : 0.0f);
                    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, f_sum0, 2);
                    f_sum0 += _shfl_xor_8;
                    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, f_sum0, 1);
                    f_sum0 += _shfl_xor_9;
                    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, f_sum1, 2);
                    f_sum1 += _shfl_xor_10;
                    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, f_sum1, 1);
                    f_sum1 += _shfl_xor_11;
                    row_sum0 = f_sum0;
                    row_sum1 = f_sum1;
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
                    __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(d_qk[32], d_qk[33]));
                    p_bf16[16] = reinterpret_cast<unsigned int*>(&_bf16x2_16)[0];
                    __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(d_qk[34], d_qk[35]));
                    p_bf16[17] = reinterpret_cast<unsigned int*>(&_bf16x2_17)[0];
                    __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(d_qk[36], d_qk[37]));
                    p_bf16[18] = reinterpret_cast<unsigned int*>(&_bf16x2_18)[0];
                    __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(d_qk[38], d_qk[39]));
                    p_bf16[19] = reinterpret_cast<unsigned int*>(&_bf16x2_19)[0];
                    __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(d_qk[40], d_qk[41]));
                    p_bf16[20] = reinterpret_cast<unsigned int*>(&_bf16x2_20)[0];
                    __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(d_qk[42], d_qk[43]));
                    p_bf16[21] = reinterpret_cast<unsigned int*>(&_bf16x2_21)[0];
                    __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(d_qk[44], d_qk[45]));
                    p_bf16[22] = reinterpret_cast<unsigned int*>(&_bf16x2_22)[0];
                    __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(d_qk[46], d_qk[47]));
                    p_bf16[23] = reinterpret_cast<unsigned int*>(&_bf16x2_23)[0];
                    __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(d_qk[48], d_qk[49]));
                    p_bf16[24] = reinterpret_cast<unsigned int*>(&_bf16x2_24)[0];
                    __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(d_qk[50], d_qk[51]));
                    p_bf16[25] = reinterpret_cast<unsigned int*>(&_bf16x2_25)[0];
                    __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(d_qk[52], d_qk[53]));
                    p_bf16[26] = reinterpret_cast<unsigned int*>(&_bf16x2_26)[0];
                    __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(d_qk[54], d_qk[55]));
                    p_bf16[27] = reinterpret_cast<unsigned int*>(&_bf16x2_27)[0];
                    __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(d_qk[56], d_qk[57]));
                    p_bf16[28] = reinterpret_cast<unsigned int*>(&_bf16x2_28)[0];
                    __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(d_qk[58], d_qk[59]));
                    p_bf16[29] = reinterpret_cast<unsigned int*>(&_bf16x2_29)[0];
                    __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(d_qk[60], d_qk[61]));
                    p_bf16[30] = reinterpret_cast<unsigned int*>(&_bf16x2_30)[0];
                    __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(d_qk[62], d_qk[63]));
                    p_bf16[31] = reinterpret_cast<unsigned int*>(&_bf16x2_31)[0];
                    if (has2_0 == 0) {
                        p_bf16[16] = 0;
                        p_bf16[17] = 0;
                        p_bf16[18] = 0;
                        p_bf16[19] = 0;
                        p_bf16[20] = 0;
                        p_bf16[21] = 0;
                        p_bf16[22] = 0;
                        p_bf16[23] = 0;
                        p_bf16[24] = 0;
                        p_bf16[25] = 0;
                        p_bf16[26] = 0;
                        p_bf16[27] = 0;
                        p_bf16[28] = 0;
                        p_bf16[29] = 0;
                        p_bf16[30] = 0;
                        p_bf16[31] = 0;
                    }
                    int n_loop = n_own - 1;
                    #pragma unroll 1
                    for (int n = 0; n < n_loop; n++) {
                        int cur_stage = cur_pos % 3;
                        int cur_ph = cur_pos / 3 & 1;
                        int w_e_0 = meta_smem[own_base_w + (n + 1 >> 1)];
                        int en = w_e_0 >> (n + 1 & 1) * 16 & 65535;
                        int nxt_pos = gbase + (en >> 3);
                        int nxt_has2 = en >> 2 & 1;
                        #pragma unroll 1
                        for (int p_2 = cur_pos + 1; p_2 < nxt_pos; p_2++) {
                            mbarrier_wait(k_full_addr + (p_2 % 3) * 8, p_2 / 3 & 1);
                            if (elect_sync()) {
                                mbarrier_arrive(k_empty_addr + (p_2 % 3) * 8);
                            }
                            mbarrier_wait(v_full_addr + (p_2 % 3) * 8, p_2 / 3 & 1);
                            if (elect_sync()) {
                                mbarrier_arrive(v_empty_addr + (p_2 % 3) * 8);
                            }
                        }
                        int nxt_stage = nxt_pos % 3;
                        mbarrier_wait(k_full_addr + (nxt_stage) * 8, nxt_pos / 3 & 1);
                        mbarrier_wait(v_full_addr + (cur_stage) * 8, cur_ph);
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_2 = (((uint64_t)(((k_smem_addr + (unsigned int)(nxt_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "l"(_wgmma_b_0_2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[4]), "r"(q_frag[(4) + 1]), "r"(q_frag[(4) + 2]), "r"(q_frag[(4) + 3]), "l"(_wgmma_b_0_2 + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[8]), "r"(q_frag[(8) + 1]), "r"(q_frag[(8) + 2]), "r"(q_frag[(8) + 3]), "l"(_wgmma_b_0_2 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[12]), "r"(q_frag[(12) + 1]), "r"(q_frag[(12) + 2]), "r"(q_frag[(12) + 3]), "l"(_wgmma_b_0_2 + 6)
                            : "memory");
                        uint64_t _wgmma_desc_3 = (((uint64_t)(((k_smem_addr + (unsigned int)(nxt_stage * 32768) + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[16]), "r"(q_frag[(16) + 1]), "r"(q_frag[(16) + 2]), "r"(q_frag[(16) + 3]), "l"(_wgmma_b_0_3)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[20]), "r"(q_frag[(20) + 1]), "r"(q_frag[(20) + 2]), "r"(q_frag[(20) + 3]), "l"(_wgmma_b_0_3 + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[24]), "r"(q_frag[(24) + 1]), "r"(q_frag[(24) + 2]), "r"(q_frag[(24) + 3]), "l"(_wgmma_b_0_3 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[28]), "r"(q_frag[(28) + 1]), "r"(q_frag[(28) + 2]), "r"(q_frag[(28) + 3]), "l"(_wgmma_b_0_3 + 6)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_4 = (((uint64_t)(((vt_smem_a_addr + (unsigned int)(cur_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
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
                        uint64_t _wgmma_desc_5 = (((uint64_t)(((vt_smem_b_addr + (unsigned int)(cur_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_5 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_5 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_5);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[16]), "r"(p_bf16[(16) + 1]), "r"(p_bf16[(16) + 2]), "r"(p_bf16[(16) + 3]), "l"(_wgmma_b_0_5)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[20]), "r"(p_bf16[(20) + 1]), "r"(p_bf16[(20) + 2]), "r"(p_bf16[(20) + 3]), "l"(_wgmma_b_0_5 + 128)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[24]), "r"(p_bf16[(24) + 1]), "r"(p_bf16[(24) + 2]), "r"(p_bf16[(24) + 3]), "l"(_wgmma_b_0_5 + 256)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[28]), "r"(p_bf16[(28) + 1]), "r"(p_bf16[(28) + 2]), "r"(p_bf16[(28) + 3]), "l"(_wgmma_b_0_5 + 384)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                        if (elect_sync()) {
                            mbarrier_arrive(k_empty_addr + (nxt_pos % 3) * 8);
                        }
                        float new_max0 = -CAKE_INF;
                        float new_max1 = -CAKE_INF;
                        {
                            float m0_0 = -CAKE_INF;
                            float m1_1 = -CAKE_INF;
                            if (scale_log2 >= 0.0f) {
                                float _max_66 = max_noftz(d_qk[0], d_qk[1]);
                                float _max_67 = max_noftz(d_qk[4], d_qk[5]);
                                float _max_68 = max_noftz(d_qk[8], d_qk[9]);
                                float _max_69 = max_noftz(d_qk[12], d_qk[13]);
                                float _max_70 = max_noftz(d_qk[16], d_qk[17]);
                                float _max_71 = max_noftz(d_qk[20], d_qk[21]);
                                float _max_72 = max_noftz(d_qk[24], d_qk[25]);
                                float _max_73 = max_noftz(d_qk[28], d_qk[29]);
                                float _max_74 = max_noftz(_max_66, _max_67);
                                float _max_75 = max_noftz(_max_68, _max_69);
                                float _max_76 = max_noftz(_max_70, _max_71);
                                float _max_77 = max_noftz(_max_72, _max_73);
                                float _max_78 = max_noftz(_max_74, _max_75);
                                float _max_79 = max_noftz(_max_76, _max_77);
                                float _max_80 = max_noftz(_max_78, _max_79);
                                float _max_81 = max_noftz(d_qk[2], d_qk[3]);
                                float _max_82 = max_noftz(d_qk[6], d_qk[7]);
                                float _max_83 = max_noftz(d_qk[10], d_qk[11]);
                                float _max_84 = max_noftz(d_qk[14], d_qk[15]);
                                float _max_85 = max_noftz(d_qk[18], d_qk[19]);
                                float _max_86 = max_noftz(d_qk[22], d_qk[23]);
                                float _max_87 = max_noftz(d_qk[26], d_qk[27]);
                                float _max_88 = max_noftz(d_qk[30], d_qk[31]);
                                float _max_89 = max_noftz(_max_81, _max_82);
                                float _max_90 = max_noftz(_max_83, _max_84);
                                float _max_91 = max_noftz(_max_85, _max_86);
                                float _max_92 = max_noftz(_max_87, _max_88);
                                float _max_93 = max_noftz(_max_89, _max_90);
                                float _max_94 = max_noftz(_max_91, _max_92);
                                float _max_95 = max_noftz(_max_93, _max_94);
                                float _max_96 = max_noftz(d_qk[32], d_qk[33]);
                                float _max_97 = max_noftz(d_qk[36], d_qk[37]);
                                float _max_98 = max_noftz(d_qk[40], d_qk[41]);
                                float _max_99 = max_noftz(d_qk[44], d_qk[45]);
                                float _max_100 = max_noftz(d_qk[48], d_qk[49]);
                                float _max_101 = max_noftz(d_qk[52], d_qk[53]);
                                float _max_102 = max_noftz(d_qk[56], d_qk[57]);
                                float _max_103 = max_noftz(d_qk[60], d_qk[61]);
                                float _max_104 = max_noftz(_max_96, _max_97);
                                float _max_105 = max_noftz(_max_98, _max_99);
                                float _max_106 = max_noftz(_max_100, _max_101);
                                float _max_107 = max_noftz(_max_102, _max_103);
                                float _max_108 = max_noftz(_max_104, _max_105);
                                float _max_109 = max_noftz(_max_106, _max_107);
                                float _max_110 = max_noftz(_max_108, _max_109);
                                float _max_111 = max_noftz(d_qk[34], d_qk[35]);
                                float _max_112 = max_noftz(d_qk[38], d_qk[39]);
                                float _max_113 = max_noftz(d_qk[42], d_qk[43]);
                                float _max_114 = max_noftz(d_qk[46], d_qk[47]);
                                float _max_115 = max_noftz(d_qk[50], d_qk[51]);
                                float _max_116 = max_noftz(d_qk[54], d_qk[55]);
                                float _max_117 = max_noftz(d_qk[58], d_qk[59]);
                                float _max_118 = max_noftz(d_qk[62], d_qk[63]);
                                float _max_119 = max_noftz(_max_111, _max_112);
                                float _max_120 = max_noftz(_max_113, _max_114);
                                float _max_121 = max_noftz(_max_115, _max_116);
                                float _max_122 = max_noftz(_max_117, _max_118);
                                float _max_123 = max_noftz(_max_119, _max_120);
                                float _max_124 = max_noftz(_max_121, _max_122);
                                float _max_125 = max_noftz(_max_123, _max_124);
                                float _max_126 = max_noftz(_max_80, _max_110);
                                m0_0 = ((nxt_has2 != 0) ? _max_126 : _max_80);
                                float _max_127 = max_noftz(_max_95, _max_125);
                                m1_1 = ((nxt_has2 != 0) ? _max_127 : _max_95);
                                float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, m0_0, 2);
                                float _max_128 = max_noftz(m0_0, _shfl_xor_12);
                                m0_0 = _max_128;
                                float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, m0_0, 1);
                                float _max_129 = max_noftz(m0_0, _shfl_xor_13);
                                m0_0 = _max_129;
                                float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, m1_1, 2);
                                float _max_130 = max_noftz(m1_1, _shfl_xor_14);
                                m1_1 = _max_130;
                                float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, m1_1, 1);
                                float _max_131 = max_noftz(m1_1, _shfl_xor_15);
                                m1_1 = _max_131;
                            } else {
                                float _min_66 = fminf(d_qk[0], d_qk[1]);
                                float _min_67 = fminf(d_qk[4], d_qk[5]);
                                float _min_68 = fminf(d_qk[8], d_qk[9]);
                                float _min_69 = fminf(d_qk[12], d_qk[13]);
                                float _min_70 = fminf(d_qk[16], d_qk[17]);
                                float _min_71 = fminf(d_qk[20], d_qk[21]);
                                float _min_72 = fminf(d_qk[24], d_qk[25]);
                                float _min_73 = fminf(d_qk[28], d_qk[29]);
                                float _min_74 = fminf(_min_66, _min_67);
                                float _min_75 = fminf(_min_68, _min_69);
                                float _min_76 = fminf(_min_70, _min_71);
                                float _min_77 = fminf(_min_72, _min_73);
                                float _min_78 = fminf(_min_74, _min_75);
                                float _min_79 = fminf(_min_76, _min_77);
                                float _min_80 = fminf(_min_78, _min_79);
                                float _min_81 = fminf(d_qk[2], d_qk[3]);
                                float _min_82 = fminf(d_qk[6], d_qk[7]);
                                float _min_83 = fminf(d_qk[10], d_qk[11]);
                                float _min_84 = fminf(d_qk[14], d_qk[15]);
                                float _min_85 = fminf(d_qk[18], d_qk[19]);
                                float _min_86 = fminf(d_qk[22], d_qk[23]);
                                float _min_87 = fminf(d_qk[26], d_qk[27]);
                                float _min_88 = fminf(d_qk[30], d_qk[31]);
                                float _min_89 = fminf(_min_81, _min_82);
                                float _min_90 = fminf(_min_83, _min_84);
                                float _min_91 = fminf(_min_85, _min_86);
                                float _min_92 = fminf(_min_87, _min_88);
                                float _min_93 = fminf(_min_89, _min_90);
                                float _min_94 = fminf(_min_91, _min_92);
                                float _min_95 = fminf(_min_93, _min_94);
                                float _min_96 = fminf(d_qk[32], d_qk[33]);
                                float _min_97 = fminf(d_qk[36], d_qk[37]);
                                float _min_98 = fminf(d_qk[40], d_qk[41]);
                                float _min_99 = fminf(d_qk[44], d_qk[45]);
                                float _min_100 = fminf(d_qk[48], d_qk[49]);
                                float _min_101 = fminf(d_qk[52], d_qk[53]);
                                float _min_102 = fminf(d_qk[56], d_qk[57]);
                                float _min_103 = fminf(d_qk[60], d_qk[61]);
                                float _min_104 = fminf(_min_96, _min_97);
                                float _min_105 = fminf(_min_98, _min_99);
                                float _min_106 = fminf(_min_100, _min_101);
                                float _min_107 = fminf(_min_102, _min_103);
                                float _min_108 = fminf(_min_104, _min_105);
                                float _min_109 = fminf(_min_106, _min_107);
                                float _min_110 = fminf(_min_108, _min_109);
                                float _min_111 = fminf(d_qk[34], d_qk[35]);
                                float _min_112 = fminf(d_qk[38], d_qk[39]);
                                float _min_113 = fminf(d_qk[42], d_qk[43]);
                                float _min_114 = fminf(d_qk[46], d_qk[47]);
                                float _min_115 = fminf(d_qk[50], d_qk[51]);
                                float _min_116 = fminf(d_qk[54], d_qk[55]);
                                float _min_117 = fminf(d_qk[58], d_qk[59]);
                                float _min_118 = fminf(d_qk[62], d_qk[63]);
                                float _min_119 = fminf(_min_111, _min_112);
                                float _min_120 = fminf(_min_113, _min_114);
                                float _min_121 = fminf(_min_115, _min_116);
                                float _min_122 = fminf(_min_117, _min_118);
                                float _min_123 = fminf(_min_119, _min_120);
                                float _min_124 = fminf(_min_121, _min_122);
                                float _min_125 = fminf(_min_123, _min_124);
                                float _min_126 = fminf(_min_80, _min_110);
                                m0_0 = ((nxt_has2 != 0) ? _min_126 : _min_80);
                                float _min_127 = fminf(_min_95, _min_125);
                                m1_1 = ((nxt_has2 != 0) ? _min_127 : _min_95);
                                float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, m0_0, 2);
                                float _min_128 = fminf(m0_0, _shfl_xor_16);
                                m0_0 = _min_128;
                                float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, m0_0, 1);
                                float _min_129 = fminf(m0_0, _shfl_xor_17);
                                m0_0 = _min_129;
                                float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, m1_1, 2);
                                float _min_130 = fminf(m1_1, _shfl_xor_18);
                                m1_1 = _min_130;
                                float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, m1_1, 1);
                                float _min_131 = fminf(m1_1, _shfl_xor_19);
                                m1_1 = _min_131;
                            }
                            new_max0 = m0_0;
                            new_max1 = m1_1;
                        }
                        float merged_max0 = new_max0 * scale_log2;
                        float merged_max1 = new_max1 * scale_log2;
                        float _max_132 = max_noftz(merged_max0, row_max0);
                        merged_max0 = _max_132;
                        float _max_133 = max_noftz(merged_max1, row_max1);
                        merged_max1 = _max_133;
                        float _exp2_64 = approx_exp2(row_max0 - merged_max0);
                        float _exp2_65 = approx_exp2(row_max1 - merged_max1);
                        row_max0 = merged_max0;
                        row_max1 = merged_max1;
                        float new_sum0 = 0.0f;
                        float new_sum1 = 0.0f;
                        {
                            float sa0_0 = 0.0f;
                            float sa1_1 = 0.0f;
                            float sb0_2 = 0.0f;
                            float sb1_3 = 0.0f;
                            {
                                float _exp2_66 = approx_exp2(d_qk[0] * scale_log2 - merged_max0);
                                float _exp2_67 = approx_exp2(d_qk[1] * scale_log2 - merged_max0);
                                d_qk[0] = _exp2_66;
                                d_qk[1] = _exp2_67;
                                sa0_0 += _exp2_66 + _exp2_67;
                                float _exp2_68 = approx_exp2(d_qk[2] * scale_log2 - merged_max1);
                                float _exp2_69 = approx_exp2(d_qk[3] * scale_log2 - merged_max1);
                                d_qk[2] = _exp2_68;
                                d_qk[3] = _exp2_69;
                                sa1_1 += _exp2_68 + _exp2_69;
                                float _exp2_70 = approx_exp2(d_qk[4] * scale_log2 - merged_max0);
                                float _exp2_71 = approx_exp2(d_qk[5] * scale_log2 - merged_max0);
                                d_qk[4] = _exp2_70;
                                d_qk[5] = _exp2_71;
                                sa0_0 += _exp2_70 + _exp2_71;
                                float _exp2_72 = approx_exp2(d_qk[6] * scale_log2 - merged_max1);
                                float _exp2_73 = approx_exp2(d_qk[7] * scale_log2 - merged_max1);
                                d_qk[6] = _exp2_72;
                                d_qk[7] = _exp2_73;
                                sa1_1 += _exp2_72 + _exp2_73;
                                float _exp2_74 = approx_exp2(d_qk[8] * scale_log2 - merged_max0);
                                float _exp2_75 = approx_exp2(d_qk[9] * scale_log2 - merged_max0);
                                d_qk[8] = _exp2_74;
                                d_qk[9] = _exp2_75;
                                sa0_0 += _exp2_74 + _exp2_75;
                                float _exp2_76 = approx_exp2(d_qk[10] * scale_log2 - merged_max1);
                                float _exp2_77 = approx_exp2(d_qk[11] * scale_log2 - merged_max1);
                                d_qk[10] = _exp2_76;
                                d_qk[11] = _exp2_77;
                                sa1_1 += _exp2_76 + _exp2_77;
                                float _exp2_78 = approx_exp2(d_qk[12] * scale_log2 - merged_max0);
                                float _exp2_79 = approx_exp2(d_qk[13] * scale_log2 - merged_max0);
                                d_qk[12] = _exp2_78;
                                d_qk[13] = _exp2_79;
                                sa0_0 += _exp2_78 + _exp2_79;
                                float _exp2_80 = approx_exp2(d_qk[14] * scale_log2 - merged_max1);
                                float _exp2_81 = approx_exp2(d_qk[15] * scale_log2 - merged_max1);
                                d_qk[14] = _exp2_80;
                                d_qk[15] = _exp2_81;
                                sa1_1 += _exp2_80 + _exp2_81;
                                float _exp2_82 = approx_exp2(d_qk[16] * scale_log2 - merged_max0);
                                float _exp2_83 = approx_exp2(d_qk[17] * scale_log2 - merged_max0);
                                d_qk[16] = _exp2_82;
                                d_qk[17] = _exp2_83;
                                sa0_0 += _exp2_82 + _exp2_83;
                                float _exp2_84 = approx_exp2(d_qk[18] * scale_log2 - merged_max1);
                                float _exp2_85 = approx_exp2(d_qk[19] * scale_log2 - merged_max1);
                                d_qk[18] = _exp2_84;
                                d_qk[19] = _exp2_85;
                                sa1_1 += _exp2_84 + _exp2_85;
                                float _exp2_86 = approx_exp2(d_qk[20] * scale_log2 - merged_max0);
                                float _exp2_87 = approx_exp2(d_qk[21] * scale_log2 - merged_max0);
                                d_qk[20] = _exp2_86;
                                d_qk[21] = _exp2_87;
                                sa0_0 += _exp2_86 + _exp2_87;
                                float _exp2_88 = approx_exp2(d_qk[22] * scale_log2 - merged_max1);
                                float _exp2_89 = approx_exp2(d_qk[23] * scale_log2 - merged_max1);
                                d_qk[22] = _exp2_88;
                                d_qk[23] = _exp2_89;
                                sa1_1 += _exp2_88 + _exp2_89;
                                float _exp2_90 = approx_exp2(d_qk[24] * scale_log2 - merged_max0);
                                float _exp2_91 = approx_exp2(d_qk[25] * scale_log2 - merged_max0);
                                d_qk[24] = _exp2_90;
                                d_qk[25] = _exp2_91;
                                sa0_0 += _exp2_90 + _exp2_91;
                                float _exp2_92 = approx_exp2(d_qk[26] * scale_log2 - merged_max1);
                                float _exp2_93 = approx_exp2(d_qk[27] * scale_log2 - merged_max1);
                                d_qk[26] = _exp2_92;
                                d_qk[27] = _exp2_93;
                                sa1_1 += _exp2_92 + _exp2_93;
                                float _exp2_94 = approx_exp2(d_qk[28] * scale_log2 - merged_max0);
                                float _exp2_95 = approx_exp2(d_qk[29] * scale_log2 - merged_max0);
                                d_qk[28] = _exp2_94;
                                d_qk[29] = _exp2_95;
                                sa0_0 += _exp2_94 + _exp2_95;
                                float _exp2_96 = approx_exp2(d_qk[30] * scale_log2 - merged_max1);
                                float _exp2_97 = approx_exp2(d_qk[31] * scale_log2 - merged_max1);
                                d_qk[30] = _exp2_96;
                                d_qk[31] = _exp2_97;
                                sa1_1 += _exp2_96 + _exp2_97;
                                float _exp2_98 = approx_exp2(d_qk[32] * scale_log2 - merged_max0);
                                float _exp2_99 = approx_exp2(d_qk[33] * scale_log2 - merged_max0);
                                d_qk[32] = _exp2_98;
                                d_qk[33] = _exp2_99;
                                sb0_2 += _exp2_98 + _exp2_99;
                                float _exp2_100 = approx_exp2(d_qk[34] * scale_log2 - merged_max1);
                                float _exp2_101 = approx_exp2(d_qk[35] * scale_log2 - merged_max1);
                                d_qk[34] = _exp2_100;
                                d_qk[35] = _exp2_101;
                                sb1_3 += _exp2_100 + _exp2_101;
                                float _exp2_102 = approx_exp2(d_qk[36] * scale_log2 - merged_max0);
                                float _exp2_103 = approx_exp2(d_qk[37] * scale_log2 - merged_max0);
                                d_qk[36] = _exp2_102;
                                d_qk[37] = _exp2_103;
                                sb0_2 += _exp2_102 + _exp2_103;
                                float _exp2_104 = approx_exp2(d_qk[38] * scale_log2 - merged_max1);
                                float _exp2_105 = approx_exp2(d_qk[39] * scale_log2 - merged_max1);
                                d_qk[38] = _exp2_104;
                                d_qk[39] = _exp2_105;
                                sb1_3 += _exp2_104 + _exp2_105;
                                float _exp2_106 = approx_exp2(d_qk[40] * scale_log2 - merged_max0);
                                float _exp2_107 = approx_exp2(d_qk[41] * scale_log2 - merged_max0);
                                d_qk[40] = _exp2_106;
                                d_qk[41] = _exp2_107;
                                sb0_2 += _exp2_106 + _exp2_107;
                                float _exp2_108 = approx_exp2(d_qk[42] * scale_log2 - merged_max1);
                                float _exp2_109 = approx_exp2(d_qk[43] * scale_log2 - merged_max1);
                                d_qk[42] = _exp2_108;
                                d_qk[43] = _exp2_109;
                                sb1_3 += _exp2_108 + _exp2_109;
                                float _exp2_110 = approx_exp2(d_qk[44] * scale_log2 - merged_max0);
                                float _exp2_111 = approx_exp2(d_qk[45] * scale_log2 - merged_max0);
                                d_qk[44] = _exp2_110;
                                d_qk[45] = _exp2_111;
                                sb0_2 += _exp2_110 + _exp2_111;
                                float _exp2_112 = approx_exp2(d_qk[46] * scale_log2 - merged_max1);
                                float _exp2_113 = approx_exp2(d_qk[47] * scale_log2 - merged_max1);
                                d_qk[46] = _exp2_112;
                                d_qk[47] = _exp2_113;
                                sb1_3 += _exp2_112 + _exp2_113;
                                float _exp2_114 = approx_exp2(d_qk[48] * scale_log2 - merged_max0);
                                float _exp2_115 = approx_exp2(d_qk[49] * scale_log2 - merged_max0);
                                d_qk[48] = _exp2_114;
                                d_qk[49] = _exp2_115;
                                sb0_2 += _exp2_114 + _exp2_115;
                                float _exp2_116 = approx_exp2(d_qk[50] * scale_log2 - merged_max1);
                                float _exp2_117 = approx_exp2(d_qk[51] * scale_log2 - merged_max1);
                                d_qk[50] = _exp2_116;
                                d_qk[51] = _exp2_117;
                                sb1_3 += _exp2_116 + _exp2_117;
                                float _exp2_118 = approx_exp2(d_qk[52] * scale_log2 - merged_max0);
                                float _exp2_119 = approx_exp2(d_qk[53] * scale_log2 - merged_max0);
                                d_qk[52] = _exp2_118;
                                d_qk[53] = _exp2_119;
                                sb0_2 += _exp2_118 + _exp2_119;
                                float _exp2_120 = approx_exp2(d_qk[54] * scale_log2 - merged_max1);
                                float _exp2_121 = approx_exp2(d_qk[55] * scale_log2 - merged_max1);
                                d_qk[54] = _exp2_120;
                                d_qk[55] = _exp2_121;
                                sb1_3 += _exp2_120 + _exp2_121;
                                float _exp2_122 = approx_exp2(d_qk[56] * scale_log2 - merged_max0);
                                float _exp2_123 = approx_exp2(d_qk[57] * scale_log2 - merged_max0);
                                d_qk[56] = _exp2_122;
                                d_qk[57] = _exp2_123;
                                sb0_2 += _exp2_122 + _exp2_123;
                                float _exp2_124 = approx_exp2(d_qk[58] * scale_log2 - merged_max1);
                                float _exp2_125 = approx_exp2(d_qk[59] * scale_log2 - merged_max1);
                                d_qk[58] = _exp2_124;
                                d_qk[59] = _exp2_125;
                                sb1_3 += _exp2_124 + _exp2_125;
                                float _exp2_126 = approx_exp2(d_qk[60] * scale_log2 - merged_max0);
                                float _exp2_127 = approx_exp2(d_qk[61] * scale_log2 - merged_max0);
                                d_qk[60] = _exp2_126;
                                d_qk[61] = _exp2_127;
                                sb0_2 += _exp2_126 + _exp2_127;
                                float _exp2_128 = approx_exp2(d_qk[62] * scale_log2 - merged_max1);
                                float _exp2_129 = approx_exp2(d_qk[63] * scale_log2 - merged_max1);
                                d_qk[62] = _exp2_128;
                                d_qk[63] = _exp2_129;
                                sb1_3 += _exp2_128 + _exp2_129;
                            }
                            new_sum0 += sa0_0 + ((nxt_has2 != 0) ? sb0_2 : 0.0f);
                            new_sum1 += sa1_1 + ((nxt_has2 != 0) ? sb1_3 : 0.0f);
                        }
                        {
                            float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, new_sum0, 2);
                            new_sum0 += _shfl_xor_20;
                            float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, new_sum0, 1);
                            new_sum0 += _shfl_xor_21;
                            float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, new_sum1, 2);
                            new_sum1 += _shfl_xor_22;
                            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, new_sum1, 1);
                            new_sum1 += _shfl_xor_23;
                        }
                        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                        if (elect_sync()) {
                            mbarrier_arrive(v_empty_addr + (cur_pos % 3) * 8);
                        }
                        int _vote_0 = __any_sync(0xFFFFFFFF, _exp2_64 != 1.0f || _exp2_65 != 1.0f);
                        int need_rescale = _vote_0;
                        if (need_rescale != 0) {
                            d_o[0] = d_o[0] * _exp2_64;
                            d_o[1] = d_o[1] * _exp2_64;
                            d_o[4] = d_o[4] * _exp2_64;
                            d_o[5] = d_o[5] * _exp2_64;
                            d_o[8] = d_o[8] * _exp2_64;
                            d_o[9] = d_o[9] * _exp2_64;
                            d_o[12] = d_o[12] * _exp2_64;
                            d_o[13] = d_o[13] * _exp2_64;
                            d_o[16] = d_o[16] * _exp2_64;
                            d_o[17] = d_o[17] * _exp2_64;
                            d_o[20] = d_o[20] * _exp2_64;
                            d_o[21] = d_o[21] * _exp2_64;
                            d_o[24] = d_o[24] * _exp2_64;
                            d_o[25] = d_o[25] * _exp2_64;
                            d_o[28] = d_o[28] * _exp2_64;
                            d_o[29] = d_o[29] * _exp2_64;
                            d_o[32] = d_o[32] * _exp2_64;
                            d_o[33] = d_o[33] * _exp2_64;
                            d_o[36] = d_o[36] * _exp2_64;
                            d_o[37] = d_o[37] * _exp2_64;
                            d_o[40] = d_o[40] * _exp2_64;
                            d_o[41] = d_o[41] * _exp2_64;
                            d_o[44] = d_o[44] * _exp2_64;
                            d_o[45] = d_o[45] * _exp2_64;
                            d_o[48] = d_o[48] * _exp2_64;
                            d_o[49] = d_o[49] * _exp2_64;
                            d_o[52] = d_o[52] * _exp2_64;
                            d_o[53] = d_o[53] * _exp2_64;
                            d_o[56] = d_o[56] * _exp2_64;
                            d_o[57] = d_o[57] * _exp2_64;
                            d_o[60] = d_o[60] * _exp2_64;
                            d_o[61] = d_o[61] * _exp2_64;
                            d_o[2] = d_o[2] * _exp2_65;
                            d_o[3] = d_o[3] * _exp2_65;
                            d_o[6] = d_o[6] * _exp2_65;
                            d_o[7] = d_o[7] * _exp2_65;
                            d_o[10] = d_o[10] * _exp2_65;
                            d_o[11] = d_o[11] * _exp2_65;
                            d_o[14] = d_o[14] * _exp2_65;
                            d_o[15] = d_o[15] * _exp2_65;
                            d_o[18] = d_o[18] * _exp2_65;
                            d_o[19] = d_o[19] * _exp2_65;
                            d_o[22] = d_o[22] * _exp2_65;
                            d_o[23] = d_o[23] * _exp2_65;
                            d_o[26] = d_o[26] * _exp2_65;
                            d_o[27] = d_o[27] * _exp2_65;
                            d_o[30] = d_o[30] * _exp2_65;
                            d_o[31] = d_o[31] * _exp2_65;
                            d_o[34] = d_o[34] * _exp2_65;
                            d_o[35] = d_o[35] * _exp2_65;
                            d_o[38] = d_o[38] * _exp2_65;
                            d_o[39] = d_o[39] * _exp2_65;
                            d_o[42] = d_o[42] * _exp2_65;
                            d_o[43] = d_o[43] * _exp2_65;
                            d_o[46] = d_o[46] * _exp2_65;
                            d_o[47] = d_o[47] * _exp2_65;
                            d_o[50] = d_o[50] * _exp2_65;
                            d_o[51] = d_o[51] * _exp2_65;
                            d_o[54] = d_o[54] * _exp2_65;
                            d_o[55] = d_o[55] * _exp2_65;
                            d_o[58] = d_o[58] * _exp2_65;
                            d_o[59] = d_o[59] * _exp2_65;
                            d_o[62] = d_o[62] * _exp2_65;
                            d_o[63] = d_o[63] * _exp2_65;
                        }
                        row_sum0 = row_sum0 * _exp2_64 + new_sum0;
                        row_sum1 = row_sum1 * _exp2_65 + new_sum1;
                        {
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
                            __nv_bfloat162 _bf16x2_48 = __float22bfloat162_rn(make_float2(d_qk[32], d_qk[33]));
                            p_bf16[16] = reinterpret_cast<unsigned int*>(&_bf16x2_48)[0];
                            __nv_bfloat162 _bf16x2_49 = __float22bfloat162_rn(make_float2(d_qk[34], d_qk[35]));
                            p_bf16[17] = reinterpret_cast<unsigned int*>(&_bf16x2_49)[0];
                            __nv_bfloat162 _bf16x2_50 = __float22bfloat162_rn(make_float2(d_qk[36], d_qk[37]));
                            p_bf16[18] = reinterpret_cast<unsigned int*>(&_bf16x2_50)[0];
                            __nv_bfloat162 _bf16x2_51 = __float22bfloat162_rn(make_float2(d_qk[38], d_qk[39]));
                            p_bf16[19] = reinterpret_cast<unsigned int*>(&_bf16x2_51)[0];
                            __nv_bfloat162 _bf16x2_52 = __float22bfloat162_rn(make_float2(d_qk[40], d_qk[41]));
                            p_bf16[20] = reinterpret_cast<unsigned int*>(&_bf16x2_52)[0];
                            __nv_bfloat162 _bf16x2_53 = __float22bfloat162_rn(make_float2(d_qk[42], d_qk[43]));
                            p_bf16[21] = reinterpret_cast<unsigned int*>(&_bf16x2_53)[0];
                            __nv_bfloat162 _bf16x2_54 = __float22bfloat162_rn(make_float2(d_qk[44], d_qk[45]));
                            p_bf16[22] = reinterpret_cast<unsigned int*>(&_bf16x2_54)[0];
                            __nv_bfloat162 _bf16x2_55 = __float22bfloat162_rn(make_float2(d_qk[46], d_qk[47]));
                            p_bf16[23] = reinterpret_cast<unsigned int*>(&_bf16x2_55)[0];
                            __nv_bfloat162 _bf16x2_56 = __float22bfloat162_rn(make_float2(d_qk[48], d_qk[49]));
                            p_bf16[24] = reinterpret_cast<unsigned int*>(&_bf16x2_56)[0];
                            __nv_bfloat162 _bf16x2_57 = __float22bfloat162_rn(make_float2(d_qk[50], d_qk[51]));
                            p_bf16[25] = reinterpret_cast<unsigned int*>(&_bf16x2_57)[0];
                            __nv_bfloat162 _bf16x2_58 = __float22bfloat162_rn(make_float2(d_qk[52], d_qk[53]));
                            p_bf16[26] = reinterpret_cast<unsigned int*>(&_bf16x2_58)[0];
                            __nv_bfloat162 _bf16x2_59 = __float22bfloat162_rn(make_float2(d_qk[54], d_qk[55]));
                            p_bf16[27] = reinterpret_cast<unsigned int*>(&_bf16x2_59)[0];
                            __nv_bfloat162 _bf16x2_60 = __float22bfloat162_rn(make_float2(d_qk[56], d_qk[57]));
                            p_bf16[28] = reinterpret_cast<unsigned int*>(&_bf16x2_60)[0];
                            __nv_bfloat162 _bf16x2_61 = __float22bfloat162_rn(make_float2(d_qk[58], d_qk[59]));
                            p_bf16[29] = reinterpret_cast<unsigned int*>(&_bf16x2_61)[0];
                            __nv_bfloat162 _bf16x2_62 = __float22bfloat162_rn(make_float2(d_qk[60], d_qk[61]));
                            p_bf16[30] = reinterpret_cast<unsigned int*>(&_bf16x2_62)[0];
                            __nv_bfloat162 _bf16x2_63 = __float22bfloat162_rn(make_float2(d_qk[62], d_qk[63]));
                            p_bf16[31] = reinterpret_cast<unsigned int*>(&_bf16x2_63)[0];
                            if (nxt_has2 == 0) {
                                p_bf16[16] = 0;
                                p_bf16[17] = 0;
                                p_bf16[18] = 0;
                                p_bf16[19] = 0;
                                p_bf16[20] = 0;
                                p_bf16[21] = 0;
                                p_bf16[22] = 0;
                                p_bf16[23] = 0;
                                p_bf16[24] = 0;
                                p_bf16[25] = 0;
                                p_bf16[26] = 0;
                                p_bf16[27] = 0;
                                p_bf16[28] = 0;
                                p_bf16[29] = 0;
                                p_bf16[30] = 0;
                                p_bf16[31] = 0;
                            }
                        }
                        prev = nxt_pos;
                        cur_pos = nxt_pos;
                        cur_has2 = nxt_has2;
                    }
                    int last_stage = cur_pos % 3;
                    mbarrier_wait(v_full_addr + (last_stage) * 8, cur_pos / 3 & 1);
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_6 = (((uint64_t)(((vt_smem_a_addr + (unsigned int)(last_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_6 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_6 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_6);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_b_0_6)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_b_0_6 + 128)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_b_0_6 + 256)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_b_0_6 + 384)
                        : "memory");
                    uint64_t _wgmma_desc_7 = (((uint64_t)(((vt_smem_b_addr + (unsigned int)(last_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_7 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_7 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_7);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[16]), "r"(p_bf16[(16) + 1]), "r"(p_bf16[(16) + 2]), "r"(p_bf16[(16) + 3]), "l"(_wgmma_b_0_7)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[20]), "r"(p_bf16[(20) + 1]), "r"(p_bf16[(20) + 2]), "r"(p_bf16[(20) + 3]), "l"(_wgmma_b_0_7 + 128)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[24]), "r"(p_bf16[(24) + 1]), "r"(p_bf16[(24) + 2]), "r"(p_bf16[(24) + 3]), "l"(_wgmma_b_0_7 + 256)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[28]), "r"(p_bf16[(28) + 1]), "r"(p_bf16[(28) + 2]), "r"(p_bf16[(28) + 3]), "l"(_wgmma_b_0_7 + 384)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(v_empty_addr + (cur_pos % 3) * 8);
                    }
                    prev = cur_pos;
                }
                int tile_end = gbase + n_seq_m;
                int tail_lo = prev + 1;
                #pragma unroll 1
                for (int p_3 = tail_lo; p_3 < tile_end; p_3++) {
                    mbarrier_wait(k_full_addr + (p_3 % 3) * 8, p_3 / 3 & 1);
                    if (elect_sync()) {
                        mbarrier_arrive(k_empty_addr + (p_3 % 3) * 8);
                    }
                    mbarrier_wait(v_full_addr + (p_3 % 3) * 8, p_3 / 3 & 1);
                    if (elect_sync()) {
                        mbarrier_arrive(v_empty_addr + (p_3 % 3) * 8);
                    }
                }
                prev = tile_end - 1;
                gbase = tile_end;
                int store_wg = 1;
                int partner_idle = n_other == 0;
                int self_idle = n_own == 0;
                int skip_merge = ((partner_idle != 0) ? 1 : ((self_idle != 0) ? 1 : 0));
                if (is_split != 0) {
                    if (skip_merge != 0) {
                        store_wg = ((self_idle != 0) ? 0 : 1);
                    }
                }
                int do_merge = ((is_split != 0) ? 1 : 0);
                if (skip_merge != 0) {
                    do_merge = 0;
                }
                float a0 = 1.0f;
                float a1 = 1.0f;
                float b0 = 0.0f;
                float b1 = 0.0f;
                if (do_merge != 0) {
                    asm volatile("barrier.sync 9, 256;" ::: "memory");
                    if (cwg == 1) {
                        merge_o[tid_wg] = d_o[0];
                        merge_o[128 + tid_wg] = d_o[1];
                        merge_o[256 + tid_wg] = d_o[2];
                        merge_o[384 + tid_wg] = d_o[3];
                        merge_o[512 + tid_wg] = d_o[4];
                        merge_o[640 + tid_wg] = d_o[5];
                        merge_o[768 + tid_wg] = d_o[6];
                        merge_o[896 + tid_wg] = d_o[7];
                        merge_o[1024 + tid_wg] = d_o[8];
                        merge_o[1152 + tid_wg] = d_o[9];
                        merge_o[1280 + tid_wg] = d_o[10];
                        merge_o[1408 + tid_wg] = d_o[11];
                        merge_o[1536 + tid_wg] = d_o[12];
                        merge_o[1664 + tid_wg] = d_o[13];
                        merge_o[1792 + tid_wg] = d_o[14];
                        merge_o[1920 + tid_wg] = d_o[15];
                        merge_o[2048 + tid_wg] = d_o[16];
                        merge_o[2176 + tid_wg] = d_o[17];
                        merge_o[2304 + tid_wg] = d_o[18];
                        merge_o[2432 + tid_wg] = d_o[19];
                        merge_o[2560 + tid_wg] = d_o[20];
                        merge_o[2688 + tid_wg] = d_o[21];
                        merge_o[2816 + tid_wg] = d_o[22];
                        merge_o[2944 + tid_wg] = d_o[23];
                        merge_o[3072 + tid_wg] = d_o[24];
                        merge_o[3200 + tid_wg] = d_o[25];
                        merge_o[3328 + tid_wg] = d_o[26];
                        merge_o[3456 + tid_wg] = d_o[27];
                        merge_o[3584 + tid_wg] = d_o[28];
                        merge_o[3712 + tid_wg] = d_o[29];
                        merge_o[3840 + tid_wg] = d_o[30];
                        merge_o[3968 + tid_wg] = d_o[31];
                        if ((lane & 3) == 0) {
                            merge_ml[quad * 4] = row_max0;
                            merge_ml[quad * 4 + 1] = row_max1;
                            merge_ml[quad * 4 + 2] = row_sum0;
                            merge_ml[quad * 4 + 3] = row_sum1;
                        }
                        store_wg = 0;
                    }
                    asm volatile("barrier.sync 9, 256;" ::: "memory");
                    if (cwg == 0) {
                        float pm0 = merge_ml[quad * 4];
                        float pm1 = merge_ml[quad * 4 + 1];
                        float pl0 = merge_ml[quad * 4 + 2];
                        float pl1 = merge_ml[quad * 4 + 3];
                        float _max_134 = max_noftz(row_max0, pm0);
                        float mm0 = _max_134;
                        float _max_135 = max_noftz(row_max1, pm1);
                        float mm1 = _max_135;
                        float _exp2_130 = approx_exp2(row_max0 - mm0);
                        a0 = ((row_max0 == -CAKE_INF) ? 0.0f : _exp2_130);
                        float _exp2_131 = approx_exp2(row_max1 - mm1);
                        a1 = ((row_max1 == -CAKE_INF) ? 0.0f : _exp2_131);
                        float _exp2_132 = approx_exp2(pm0 - mm0);
                        b0 = ((pm0 == -CAKE_INF) ? 0.0f : _exp2_132);
                        float _exp2_133 = approx_exp2(pm1 - mm1);
                        b1 = ((pm1 == -CAKE_INF) ? 0.0f : _exp2_133);
                        float po = merge_o[tid_wg];
                        {
                            d_o[0] = d_o[0] * a0 + po * b0;
                        }
                        float po_0 = merge_o[128 + tid_wg];
                        {
                            d_o[1] = d_o[1] * a0 + po_0 * b0;
                        }
                        float po_1 = merge_o[256 + tid_wg];
                        {
                            d_o[2] = d_o[2] * a1 + po_1 * b1;
                        }
                        float po_2 = merge_o[384 + tid_wg];
                        {
                            d_o[3] = d_o[3] * a1 + po_2 * b1;
                        }
                        float po_3 = merge_o[512 + tid_wg];
                        {
                            d_o[4] = d_o[4] * a0 + po_3 * b0;
                        }
                        float po_4 = merge_o[640 + tid_wg];
                        {
                            d_o[5] = d_o[5] * a0 + po_4 * b0;
                        }
                        float po_5 = merge_o[768 + tid_wg];
                        {
                            d_o[6] = d_o[6] * a1 + po_5 * b1;
                        }
                        float po_6 = merge_o[896 + tid_wg];
                        {
                            d_o[7] = d_o[7] * a1 + po_6 * b1;
                        }
                        float po_7 = merge_o[1024 + tid_wg];
                        {
                            d_o[8] = d_o[8] * a0 + po_7 * b0;
                        }
                        float po_8 = merge_o[1152 + tid_wg];
                        {
                            d_o[9] = d_o[9] * a0 + po_8 * b0;
                        }
                        float po_9 = merge_o[1280 + tid_wg];
                        {
                            d_o[10] = d_o[10] * a1 + po_9 * b1;
                        }
                        float po_10 = merge_o[1408 + tid_wg];
                        {
                            d_o[11] = d_o[11] * a1 + po_10 * b1;
                        }
                        float po_11 = merge_o[1536 + tid_wg];
                        {
                            d_o[12] = d_o[12] * a0 + po_11 * b0;
                        }
                        float po_12 = merge_o[1664 + tid_wg];
                        {
                            d_o[13] = d_o[13] * a0 + po_12 * b0;
                        }
                        float po_13 = merge_o[1792 + tid_wg];
                        {
                            d_o[14] = d_o[14] * a1 + po_13 * b1;
                        }
                        float po_14 = merge_o[1920 + tid_wg];
                        {
                            d_o[15] = d_o[15] * a1 + po_14 * b1;
                        }
                        float po_15 = merge_o[2048 + tid_wg];
                        {
                            d_o[16] = d_o[16] * a0 + po_15 * b0;
                        }
                        float po_16 = merge_o[2176 + tid_wg];
                        {
                            d_o[17] = d_o[17] * a0 + po_16 * b0;
                        }
                        float po_17 = merge_o[2304 + tid_wg];
                        {
                            d_o[18] = d_o[18] * a1 + po_17 * b1;
                        }
                        float po_18 = merge_o[2432 + tid_wg];
                        {
                            d_o[19] = d_o[19] * a1 + po_18 * b1;
                        }
                        float po_19 = merge_o[2560 + tid_wg];
                        {
                            d_o[20] = d_o[20] * a0 + po_19 * b0;
                        }
                        float po_20 = merge_o[2688 + tid_wg];
                        {
                            d_o[21] = d_o[21] * a0 + po_20 * b0;
                        }
                        float po_21 = merge_o[2816 + tid_wg];
                        {
                            d_o[22] = d_o[22] * a1 + po_21 * b1;
                        }
                        float po_22 = merge_o[2944 + tid_wg];
                        {
                            d_o[23] = d_o[23] * a1 + po_22 * b1;
                        }
                        float po_23 = merge_o[3072 + tid_wg];
                        {
                            d_o[24] = d_o[24] * a0 + po_23 * b0;
                        }
                        float po_24 = merge_o[3200 + tid_wg];
                        {
                            d_o[25] = d_o[25] * a0 + po_24 * b0;
                        }
                        float po_25 = merge_o[3328 + tid_wg];
                        {
                            d_o[26] = d_o[26] * a1 + po_25 * b1;
                        }
                        float po_26 = merge_o[3456 + tid_wg];
                        {
                            d_o[27] = d_o[27] * a1 + po_26 * b1;
                        }
                        float po_27 = merge_o[3584 + tid_wg];
                        {
                            d_o[28] = d_o[28] * a0 + po_27 * b0;
                        }
                        float po_28 = merge_o[3712 + tid_wg];
                        {
                            d_o[29] = d_o[29] * a0 + po_28 * b0;
                        }
                        float po_29 = merge_o[3840 + tid_wg];
                        {
                            d_o[30] = d_o[30] * a1 + po_29 * b1;
                        }
                        float po_30 = merge_o[3968 + tid_wg];
                        {
                            d_o[31] = d_o[31] * a1 + po_30 * b1;
                        }
                        row_sum0 = row_sum0 * a0 + pl0 * b0;
                        row_sum1 = row_sum1 * a1 + pl1 * b1;
                    }
                    asm volatile("barrier.sync 9, 256;" ::: "memory");
                    if (cwg == 1) {
                        merge_o[tid_wg] = d_o[32];
                        merge_o[128 + tid_wg] = d_o[33];
                        merge_o[256 + tid_wg] = d_o[34];
                        merge_o[384 + tid_wg] = d_o[35];
                        merge_o[512 + tid_wg] = d_o[36];
                        merge_o[640 + tid_wg] = d_o[37];
                        merge_o[768 + tid_wg] = d_o[38];
                        merge_o[896 + tid_wg] = d_o[39];
                        merge_o[1024 + tid_wg] = d_o[40];
                        merge_o[1152 + tid_wg] = d_o[41];
                        merge_o[1280 + tid_wg] = d_o[42];
                        merge_o[1408 + tid_wg] = d_o[43];
                        merge_o[1536 + tid_wg] = d_o[44];
                        merge_o[1664 + tid_wg] = d_o[45];
                        merge_o[1792 + tid_wg] = d_o[46];
                        merge_o[1920 + tid_wg] = d_o[47];
                        merge_o[2048 + tid_wg] = d_o[48];
                        merge_o[2176 + tid_wg] = d_o[49];
                        merge_o[2304 + tid_wg] = d_o[50];
                        merge_o[2432 + tid_wg] = d_o[51];
                        merge_o[2560 + tid_wg] = d_o[52];
                        merge_o[2688 + tid_wg] = d_o[53];
                        merge_o[2816 + tid_wg] = d_o[54];
                        merge_o[2944 + tid_wg] = d_o[55];
                        merge_o[3072 + tid_wg] = d_o[56];
                        merge_o[3200 + tid_wg] = d_o[57];
                        merge_o[3328 + tid_wg] = d_o[58];
                        merge_o[3456 + tid_wg] = d_o[59];
                        merge_o[3584 + tid_wg] = d_o[60];
                        merge_o[3712 + tid_wg] = d_o[61];
                        merge_o[3840 + tid_wg] = d_o[62];
                        merge_o[3968 + tid_wg] = d_o[63];
                    }
                    asm volatile("barrier.sync 9, 256;" ::: "memory");
                    if (cwg == 0) {
                        float po_h = merge_o[tid_wg];
                        {
                            d_o[32] = d_o[32] * a0 + po_h * b0;
                        }
                        float po_h_0 = merge_o[128 + tid_wg];
                        {
                            d_o[33] = d_o[33] * a0 + po_h_0 * b0;
                        }
                        float po_h_1 = merge_o[256 + tid_wg];
                        {
                            d_o[34] = d_o[34] * a1 + po_h_1 * b1;
                        }
                        float po_h_2 = merge_o[384 + tid_wg];
                        {
                            d_o[35] = d_o[35] * a1 + po_h_2 * b1;
                        }
                        float po_h_3 = merge_o[512 + tid_wg];
                        {
                            d_o[36] = d_o[36] * a0 + po_h_3 * b0;
                        }
                        float po_h_4 = merge_o[640 + tid_wg];
                        {
                            d_o[37] = d_o[37] * a0 + po_h_4 * b0;
                        }
                        float po_h_5 = merge_o[768 + tid_wg];
                        {
                            d_o[38] = d_o[38] * a1 + po_h_5 * b1;
                        }
                        float po_h_6 = merge_o[896 + tid_wg];
                        {
                            d_o[39] = d_o[39] * a1 + po_h_6 * b1;
                        }
                        float po_h_7 = merge_o[1024 + tid_wg];
                        {
                            d_o[40] = d_o[40] * a0 + po_h_7 * b0;
                        }
                        float po_h_8 = merge_o[1152 + tid_wg];
                        {
                            d_o[41] = d_o[41] * a0 + po_h_8 * b0;
                        }
                        float po_h_9 = merge_o[1280 + tid_wg];
                        {
                            d_o[42] = d_o[42] * a1 + po_h_9 * b1;
                        }
                        float po_h_10 = merge_o[1408 + tid_wg];
                        {
                            d_o[43] = d_o[43] * a1 + po_h_10 * b1;
                        }
                        float po_h_11 = merge_o[1536 + tid_wg];
                        {
                            d_o[44] = d_o[44] * a0 + po_h_11 * b0;
                        }
                        float po_h_12 = merge_o[1664 + tid_wg];
                        {
                            d_o[45] = d_o[45] * a0 + po_h_12 * b0;
                        }
                        float po_h_13 = merge_o[1792 + tid_wg];
                        {
                            d_o[46] = d_o[46] * a1 + po_h_13 * b1;
                        }
                        float po_h_14 = merge_o[1920 + tid_wg];
                        {
                            d_o[47] = d_o[47] * a1 + po_h_14 * b1;
                        }
                        float po_h_15 = merge_o[2048 + tid_wg];
                        {
                            d_o[48] = d_o[48] * a0 + po_h_15 * b0;
                        }
                        float po_h_16 = merge_o[2176 + tid_wg];
                        {
                            d_o[49] = d_o[49] * a0 + po_h_16 * b0;
                        }
                        float po_h_17 = merge_o[2304 + tid_wg];
                        {
                            d_o[50] = d_o[50] * a1 + po_h_17 * b1;
                        }
                        float po_h_18 = merge_o[2432 + tid_wg];
                        {
                            d_o[51] = d_o[51] * a1 + po_h_18 * b1;
                        }
                        float po_h_19 = merge_o[2560 + tid_wg];
                        {
                            d_o[52] = d_o[52] * a0 + po_h_19 * b0;
                        }
                        float po_h_20 = merge_o[2688 + tid_wg];
                        {
                            d_o[53] = d_o[53] * a0 + po_h_20 * b0;
                        }
                        float po_h_21 = merge_o[2816 + tid_wg];
                        {
                            d_o[54] = d_o[54] * a1 + po_h_21 * b1;
                        }
                        float po_h_22 = merge_o[2944 + tid_wg];
                        {
                            d_o[55] = d_o[55] * a1 + po_h_22 * b1;
                        }
                        float po_h_23 = merge_o[3072 + tid_wg];
                        {
                            d_o[56] = d_o[56] * a0 + po_h_23 * b0;
                        }
                        float po_h_24 = merge_o[3200 + tid_wg];
                        {
                            d_o[57] = d_o[57] * a0 + po_h_24 * b0;
                        }
                        float po_h_25 = merge_o[3328 + tid_wg];
                        {
                            d_o[58] = d_o[58] * a1 + po_h_25 * b1;
                        }
                        float po_h_26 = merge_o[3456 + tid_wg];
                        {
                            d_o[59] = d_o[59] * a1 + po_h_26 * b1;
                        }
                        float po_h_27 = merge_o[3584 + tid_wg];
                        {
                            d_o[60] = d_o[60] * a0 + po_h_27 * b0;
                        }
                        float po_h_28 = merge_o[3712 + tid_wg];
                        {
                            d_o[61] = d_o[61] * a0 + po_h_28 * b0;
                        }
                        float po_h_29 = merge_o[3840 + tid_wg];
                        {
                            d_o[62] = d_o[62] * a1 + po_h_29 * b1;
                        }
                        float po_h_30 = merge_o[3968 + tid_wg];
                        {
                            d_o[63] = d_o[63] * a1 + po_h_30 * b1;
                        }
                    }
                }
                if (store_wg != 0) {
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
                    int o_row_base = (head_m * seqlen_q + my_qb * 64) * 128;
                    int m_local_r = ((1) ? m0_local : m1_local);
                    __nv_bfloat162 _bf16x2_64 = __float22bfloat162_rn(make_float2(d_o[0], d_o[1]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_64)[0];
                    __nv_bfloat162 _bf16x2_65 = __float22bfloat162_rn(make_float2(d_o[4], d_o[5]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_65)[0];
                    __nv_bfloat162 _bf16x2_66 = __float22bfloat162_rn(make_float2(d_o[8], d_o[9]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_66)[0];
                    __nv_bfloat162 _bf16x2_67 = __float22bfloat162_rn(make_float2(d_o[12], d_o[13]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_67)[0];
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
                    int o_off = o_row_base + m_local_r * 128 + qj * 8;
                    reinterpret_cast<int4*>(O + o_off)[0] = reinterpret_cast<int4*>(o_vec)[0];
                    __nv_bfloat162 _bf16x2_68 = __float22bfloat162_rn(make_float2(d_o[16], d_o[17]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_68)[0];
                    __nv_bfloat162 _bf16x2_69 = __float22bfloat162_rn(make_float2(d_o[20], d_o[21]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_69)[0];
                    __nv_bfloat162 _bf16x2_70 = __float22bfloat162_rn(make_float2(d_o[24], d_o[25]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_70)[0];
                    __nv_bfloat162 _bf16x2_71 = __float22bfloat162_rn(make_float2(d_o[28], d_o[29]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_71)[0];
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
                    int o_off_0 = o_row_base + m_local_r * 128 + (4 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_0)[0] = reinterpret_cast<int4*>(o_vec)[0];
                    __nv_bfloat162 _bf16x2_72 = __float22bfloat162_rn(make_float2(d_o[32], d_o[33]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_72)[0];
                    __nv_bfloat162 _bf16x2_73 = __float22bfloat162_rn(make_float2(d_o[36], d_o[37]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_73)[0];
                    __nv_bfloat162 _bf16x2_74 = __float22bfloat162_rn(make_float2(d_o[40], d_o[41]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_74)[0];
                    __nv_bfloat162 _bf16x2_75 = __float22bfloat162_rn(make_float2(d_o[44], d_o[45]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_75)[0];
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
                    int o_off_1 = o_row_base + m_local_r * 128 + (8 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_1)[0] = reinterpret_cast<int4*>(o_vec)[0];
                    __nv_bfloat162 _bf16x2_76 = __float22bfloat162_rn(make_float2(d_o[48], d_o[49]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_76)[0];
                    __nv_bfloat162 _bf16x2_77 = __float22bfloat162_rn(make_float2(d_o[52], d_o[53]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_77)[0];
                    __nv_bfloat162 _bf16x2_78 = __float22bfloat162_rn(make_float2(d_o[56], d_o[57]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_78)[0];
                    __nv_bfloat162 _bf16x2_79 = __float22bfloat162_rn(make_float2(d_o[60], d_o[61]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_79)[0];
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
                    int o_off_2 = o_row_base + m_local_r * 128 + (12 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_2)[0] = reinterpret_cast<int4*>(o_vec)[0];
                    int m_local_r_3 = ((0) ? m0_local : m1_local);
                    __nv_bfloat162 _bf16x2_80 = __float22bfloat162_rn(make_float2(d_o[2], d_o[3]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_80)[0];
                    __nv_bfloat162 _bf16x2_81 = __float22bfloat162_rn(make_float2(d_o[6], d_o[7]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_81)[0];
                    __nv_bfloat162 _bf16x2_82 = __float22bfloat162_rn(make_float2(d_o[10], d_o[11]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_82)[0];
                    __nv_bfloat162 _bf16x2_83 = __float22bfloat162_rn(make_float2(d_o[14], d_o[15]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_83)[0];
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
                    int o_off_4 = o_row_base + m_local_r_3 * 128 + qj * 8;
                    reinterpret_cast<int4*>(O + o_off_4)[0] = reinterpret_cast<int4*>(o_vec)[0];
                    __nv_bfloat162 _bf16x2_84 = __float22bfloat162_rn(make_float2(d_o[18], d_o[19]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_84)[0];
                    __nv_bfloat162 _bf16x2_85 = __float22bfloat162_rn(make_float2(d_o[22], d_o[23]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_85)[0];
                    __nv_bfloat162 _bf16x2_86 = __float22bfloat162_rn(make_float2(d_o[26], d_o[27]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_86)[0];
                    __nv_bfloat162 _bf16x2_87 = __float22bfloat162_rn(make_float2(d_o[30], d_o[31]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_87)[0];
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
                    int o_off_5 = o_row_base + m_local_r_3 * 128 + (4 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_5)[0] = reinterpret_cast<int4*>(o_vec)[0];
                    __nv_bfloat162 _bf16x2_88 = __float22bfloat162_rn(make_float2(d_o[34], d_o[35]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_88)[0];
                    __nv_bfloat162 _bf16x2_89 = __float22bfloat162_rn(make_float2(d_o[38], d_o[39]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_89)[0];
                    __nv_bfloat162 _bf16x2_90 = __float22bfloat162_rn(make_float2(d_o[42], d_o[43]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_90)[0];
                    __nv_bfloat162 _bf16x2_91 = __float22bfloat162_rn(make_float2(d_o[46], d_o[47]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_91)[0];
                    unsigned int _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                    o_tmp[0] = _shfl_xor_72;
                    unsigned int _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                    o_tmp[1] = _shfl_xor_73;
                    unsigned int _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                    o_tmp[2] = _shfl_xor_74;
                    unsigned int _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                    o_tmp[3] = _shfl_xor_75;
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
                    unsigned int _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                    o_tmp[0] = _shfl_xor_76;
                    unsigned int _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                    o_tmp[1] = _shfl_xor_77;
                    unsigned int _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                    o_tmp[2] = _shfl_xor_78;
                    unsigned int _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                    o_tmp[3] = _shfl_xor_79;
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
                    __nv_bfloat162 _bf16x2_92 = __float22bfloat162_rn(make_float2(d_o[50], d_o[51]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bf16x2_92)[0];
                    __nv_bfloat162 _bf16x2_93 = __float22bfloat162_rn(make_float2(d_o[54], d_o[55]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bf16x2_93)[0];
                    __nv_bfloat162 _bf16x2_94 = __float22bfloat162_rn(make_float2(d_o[58], d_o[59]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bf16x2_94)[0];
                    __nv_bfloat162 _bf16x2_95 = __float22bfloat162_rn(make_float2(d_o[62], d_o[63]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bf16x2_95)[0];
                    unsigned int _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                    o_tmp[0] = _shfl_xor_80;
                    unsigned int _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                    o_tmp[1] = _shfl_xor_81;
                    unsigned int _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                    o_tmp[2] = _shfl_xor_82;
                    unsigned int _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                    o_tmp[3] = _shfl_xor_83;
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
                    unsigned int _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                    o_tmp[0] = _shfl_xor_84;
                    unsigned int _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                    o_tmp[1] = _shfl_xor_85;
                    unsigned int _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                    o_tmp[2] = _shfl_xor_86;
                    unsigned int _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                    o_tmp[3] = _shfl_xor_87;
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
                if (elect_sync()) {
                    mbarrier_arrive(meta_empty_addr + (slot_m) * 8);
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
