/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_RAW_PIPE_STAGES 2
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 8192
#define SMEM_SMEM_QD_STRIDE 8192
#define SMEM_SMEM_Q_RAW_OFF 17408
#define SMEM_SMEM_Q_RAW_STAGE_BYTES 4096
#define SMEM_SMEM_Q_RAW_STRIDE 4096
#define SMEM_SMEM_KD_OFF 9216
#define SMEM_SMEM_KD_STAGE_BYTES 8192
#define SMEM_SMEM_KD_STRIDE 8192
#define SMEM_SMEM_K_RAW_OFF 25600
#define SMEM_SMEM_K_RAW_STAGE_BYTES 4096
#define SMEM_SMEM_K_RAW_STRIDE 4096
#define SMEM_SMEM_KI_OFF 17408
#define SMEM_SMEM_KI_STAGE_BYTES 8192
#define SMEM_SMEM_KI_STRIDE 8192
#define SMEM_SMEM_GATE_RAW_OFF 21504
#define SMEM_SMEM_GATE_RAW_STAGE_BYTES 4096
#define SMEM_SMEM_GATE_RAW_STRIDE 4096
#define SMEM_SMEM_W_OUT_OFF 17408
#define SMEM_SMEM_W_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_W_OUT_STRIDE 8192
#define SMEM_SMEM_QK_PLAIN_OFF 29696
#define SMEM_SMEM_QK_PLAIN_STAGE_BYTES 1024
#define SMEM_SMEM_QK_PLAIN_STRIDE 1024
#define SMEM_SMEM_BETA_RAW_OFF 32384
#define SMEM_SMEM_BETA_RAW_STAGE_BYTES 256
#define SMEM_SMEM_BETA_RAW_STRIDE 256
#define SMEM_SMEM_GATE_OFF 1024
#define SMEM_SMEM_GATE_STAGE_BYTES 8192
#define SMEM_SMEM_GATE_STRIDE 8192
#define SMEM_SMEM_GATE_TOTAL_OFF 31744
#define SMEM_SMEM_GATE_TOTAL_STAGE_BYTES 512
#define SMEM_SMEM_GATE_TOTAL_STRIDE 512
#define SMEM_SMEM_BETA_OFF 32256
#define SMEM_SMEM_BETA_STAGE_BYTES 64
#define SMEM_SMEM_BETA_STRIDE 64
#define SMEM_SMEM_ABT_OFF 30720
#define SMEM_SMEM_ABT_STAGE_BYTES 1024
#define SMEM_SMEM_ABT_STRIDE 1024
#define SMEM_TOTAL 32768
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

// Match CUTLASS ClusterBarrier::wait: a large suspendTimeHint lets the hardware
// take the blocking phase-check slowpath instead of spinning on TRYWAIT misses.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
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


__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_5d(
    const void *tmap, int x, int y, int z, int w, int v, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4, %5}], [%6];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(smem_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(128, 6) void
kernel_cake_kda_tf32_4c99177c07ace7591e287d6d59191c460b0038d21d7697a719b133f5316847ff(__nv_bfloat16* __restrict__ q, CakeTensorMap const* q_tma, __nv_bfloat16* __restrict__ k, CakeTensorMap const* k_tma, __nv_bfloat16* __restrict__ raw_gate, CakeTensorMap const* raw_gate_tma, __nv_bfloat16* __restrict__ beta_logits, float* __restrict__ beta_active_f32, CakeTensorMap const* beta_logits_tma, float* __restrict__ a_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ cu_chunks, int* __restrict__ chunk_to_seq, float* __restrict__ ws_qd, CakeTensorMap const* ws_qd_tma, float* __restrict__ ws_kd, CakeTensorMap const* ws_kd_tma, float* __restrict__ ws_w, CakeTensorMap const* ws_w_tma, float* __restrict__ ws_qk_t, float* __restrict__ ws_diag, int total_chunks, int num_heads, float gate_lower_bound, long long beta_token_stride)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define gate_raw_full_addr (mbar_base + 0)
    #define qk_raw_full_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (warp == 3) {
        uint64_t __cake_tensormap_acquire_addr = (uint64_t)(q_tma);
        if (lane == 1) __cake_tensormap_acquire_addr = (uint64_t)(k_tma);
        if (lane == 2) __cake_tensormap_acquire_addr = (uint64_t)(raw_gate_tma);
        if (lane == 3) __cake_tensormap_acquire_addr = (uint64_t)(beta_logits_tma);
        if (lane == 4) __cake_tensormap_acquire_addr = (uint64_t)(ws_qd_tma);
        if (lane == 5) __cake_tensormap_acquire_addr = (uint64_t)(ws_kd_tma);
        if (lane == 6) __cake_tensormap_acquire_addr = (uint64_t)(ws_w_tma);
        if (lane < 7) {
            asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"(__cake_tensormap_acquire_addr) : "memory");
        }
    }


    // Kernel setup ops
    float* smem_qd = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_qd_addr = smem + 1024;
    __nv_bfloat16* smem_q_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_q_raw_addr = smem + 17408;
    float* smem_kd = reinterpret_cast<float*>(smem_raw + 9216);
    const int smem_kd_addr = smem + 9216;
    __nv_bfloat16* smem_k_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int smem_k_raw_addr = smem + 25600;
    float* smem_ki = reinterpret_cast<float*>(smem_raw + 17408);
    const int smem_ki_addr = smem + 17408;
    __nv_bfloat16* smem_gate_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 21504);
    const int smem_gate_raw_addr = smem + 21504;
    float* smem_w_out = reinterpret_cast<float*>(smem_raw + 17408);
    const int smem_w_out_addr = smem + 17408;
    float* smem_qk_plain = reinterpret_cast<float*>(smem_raw + 29696);
    const int smem_qk_plain_addr = smem + 29696;
    __nv_bfloat16* smem_beta_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32384);
    const int smem_beta_raw_addr = smem + 32384;
    float* smem_gate = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_gate_addr = smem + 1024;
    float* smem_gate_total = reinterpret_cast<float*>(smem_raw + 31744);
    const int smem_gate_total_addr = smem + 31744;
    float* smem_beta = reinterpret_cast<float*>(smem_raw + 32256);
    const int smem_beta_addr = smem + 32256;
    float* smem_abt = reinterpret_cast<float*>(smem_raw + 30720);
    const int smem_abt_addr = smem + 30720;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        // --- pipeline 'raw_pipe' ---
        // gate_raw_full: 2 barriers, init_count=1
        // qk_raw_full: 2 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 0 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    int linear_cta = blockIdx.x;
    int base_ctas_per_head = gridDim.x / num_heads;
    int extra_heads = gridDim.x % num_heads;
    int extra_span = (base_ctas_per_head + 1) * extra_heads;
    int head_idx = 0;
    int cta_rank_in_head = 0;
    int ctas_for_head = base_ctas_per_head;
    if (linear_cta < extra_span) {
        ctas_for_head = base_ctas_per_head + 1;
        head_idx = linear_cta / ctas_for_head;
        cta_rank_in_head = linear_cta % ctas_for_head;
    } else {
        head_idx = extra_heads + (linear_cta - extra_span) / ctas_for_head;
        cta_rank_in_head = (linear_cta - extra_span) % ctas_for_head;
    }
    int chunk_lo = cta_rank_in_head * total_chunks / ctas_for_head;
    int chunk_hi = (cta_rank_in_head + 1) * total_chunks / ctas_for_head;
    int my_chunks = chunk_hi - chunk_lo;
    int col = tid;
    float _exp2_0 = approx_exp2(a_log[head_idx] * 1.4426950408889634f);
    float gate_rate = _exp2_0;
    float gate_rate_half = gate_rate * 0.5f;
    float gate_bias = dt_bias[head_idx * 128 + col];
    float gate_half_scale = gate_lower_bound * 0.7213475204444817f;
    int raw_epoch = 0;
    long long current_token_base = 0;
    long long current_eos = 0;
    if (my_chunks > 0) {
        int first_seq = chunk_to_seq[chunk_lo];
        int first_local_chunk = chunk_lo - cu_chunks[first_seq];
        long long first_bos = cu_seqlens[first_seq];
        current_eos = cu_seqlens[first_seq + 1];
        current_token_base = first_bos + (long long)(first_local_chunk * 16);
    }
    #pragma unroll 1
    for (int cta_chunk = 0; cta_chunk < my_chunks; cta_chunk++) {
        int gchunk = chunk_lo + cta_chunk;
        long long token_base = current_token_base;
        long long eos = current_eos;
        int chunk_is_full = ((eos >= token_base + 16) ? 1 : 0);
        unsigned int raw_stage = (unsigned int)raw_epoch & 1;
        unsigned int raw_phase = (unsigned int)raw_epoch / 2 & 1;
        if (cta_chunk == 0) {
            if (warp == 0) {
                if (elect_sync()) {
                    int gate_tx_bytes = 4096;
                    mbarrier_arrive_expect_tx(gate_raw_full_addr + (raw_stage) * 8, gate_tx_bytes);
                    tma_3d_gmem2smem(smem_gate_raw_addr, raw_gate_tma, 0, head_idx, (int)token_base, gate_raw_full_addr + (raw_stage) * 8);
                    mbarrier_arrive_expect_tx(qk_raw_full_addr + (raw_stage) * 8, 8192);
                    tma_4d_gmem2smem(smem_q_raw_addr, q_tma, 0, (int)token_base, head_idx, 0, qk_raw_full_addr + (raw_stage) * 8);
                    tma_4d_gmem2smem(smem_k_raw_addr, k_tma, 0, (int)token_base, head_idx, 0, qk_raw_full_addr + (raw_stage) * 8);
                }
            }
        }
        float beta_value = 0.0f;
        if (tid < 16) {
            {
                long long beta_token = token_base + (long long)tid;
                if (beta_token < eos) {
                    long long beta_index = beta_token * beta_token_stride + (long long)head_idx;
                    {
                        beta_value = beta_active_f32[beta_index];
                    }
                }
            }
        }
        mbarrier_wait(gate_raw_full_addr + (raw_stage) * 8, raw_phase);
        if (tid < 16) {
            smem_beta[tid] = beta_value;
        }
        if (chunk_is_full == 0) {
            mbarrier_wait(qk_raw_full_addr + (raw_stage) * 8, raw_phase);
            int tail_row = warp * 4 + lane / 8;
            int tail_lane_in_row = lane % 8;
            if (eos <= token_base + (long long)tail_row) {
                float tail_zero[8];
                tail_zero[0] = 0.0f;
                tail_zero[1] = 0.0f;
                tail_zero[2] = 0.0f;
                tail_zero[3] = 0.0f;
                tail_zero[4] = 0.0f;
                tail_zero[5] = 0.0f;
                tail_zero[6] = 0.0f;
                tail_zero[7] = 0.0f;
                #pragma unroll
                for (int dim_half = 0; dim_half < 2; dim_half++) {
                    int tail_segment = dim_half * 8 + tail_lane_in_row;
                    unsigned int packed[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tail_zero[_lp*2 + 0], tail_zero[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word = 0; word < 4; word++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_q_raw_addr + (unsigned int)(tail_segment * 8 / 64 * 2048 + tail_row * 128 + tail_segment * 8 % 64 * 2 ^ (tail_segment * 8 / 64 * 2048 + tail_row * 128 + tail_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word * 4)), "r"((packed[word])));
                    }
                    unsigned int packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tail_zero[_lp*2 + 0], tail_zero[_lp*2+1 + 0]));
                        packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_1 = 0; word_1 < 4; word_1++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_k_raw_addr + (unsigned int)(tail_segment * 8 / 64 * 2048 + tail_row * 128 + tail_segment * 8 % 64 * 2 ^ (tail_segment * 8 / 64 * 2048 + tail_row * 128 + tail_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_1 * 4)), "r"((packed_0[word_1])));
                    }
                }
            }
            #pragma unroll
            for (int tail_gate_row = 0; tail_gate_row < 16; tail_gate_row++) {
                if (eos <= token_base + (long long)tail_gate_row) {
                    smem_gate_raw[tail_gate_row * 128 + col] = 0.0f;
                }
            }
            __syncthreads();
        }
        float prefix_log2 = 0.0f;
        float gate_decay[16];
        if (chunk_is_full != 0) {
            #pragma unroll
            for (int row = 0; row < 16; row++) {
                float _cvt_f32_0 = __bfloat162float(smem_gate_raw[row * 128 + col]);
                float gate_raw_value = _cvt_f32_0;
                float gate_arg = gate_rate_half * (gate_raw_value + gate_bias);
                float _tanh_approx_3;
                asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(gate_arg));
                float _fma_0 = __fmaf_rn(_tanh_approx_3, gate_half_scale, gate_half_scale);
                float gate_increment = _fma_0;
                prefix_log2 += gate_increment;
                gate_decay[row] = prefix_log2;
            }
        } else {
            #pragma unroll
            for (int row_1 = 0; row_1 < 16; row_1++) {
                float gate_increment_1 = 0.0f;
                if (eos > token_base + (long long)row_1) {
                    float _cvt_f32_1 = __bfloat162float(smem_gate_raw[row_1 * 128 + col]);
                    float gate_raw_value_1 = _cvt_f32_1;
                    float gate_arg_1 = gate_rate_half * (gate_raw_value_1 + gate_bias);
                    float _tanh_approx_4;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_4) : "f"(gate_arg_1));
                    float _fma_1 = __fmaf_rn(_tanh_approx_4, gate_half_scale, gate_half_scale);
                    gate_increment_1 = _fma_1;
                }
                prefix_log2 += gate_increment_1;
                gate_decay[row_1] = prefix_log2;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int row_2 = 0; row_2 < 16; row_2++) {
            float _exp2_1 = approx_exp2(gate_decay[row_2]);
            gate_decay[row_2] = _exp2_1;
            smem_gate[((smem_gate_addr + (unsigned int)(col / 32 * 2048 + row_2 * 128 + col % 32 * 4 ^ (col / 32 * 2048 + row_2 * 128 + col % 32 * 4 >> 7 & 7) << 4)) - smem_gate_addr) / 4] = gate_decay[row_2];
        }
        float total_decay = gate_decay[15];
        smem_gate_total[col] = total_decay;
        ws_diag[((long long)head_idx * (long long)total_chunks + (long long)gchunk) * 128 + (long long)col] = total_decay;
        __syncthreads();
        if (chunk_is_full != 0) {
            mbarrier_wait(qk_raw_full_addr + (raw_stage) * 8, raw_phase);
        }
        int row_3 = warp * 4 + lane / 8;
        int lane_in_row = lane % 8;
        float q_raw[16];
        float k_raw[16];
        unsigned int q_raw_packed[8];
        unsigned int k_raw_packed[8];
        q_raw[0] = 0.0f;
        q_raw[1] = 0.0f;
        q_raw[2] = 0.0f;
        q_raw[3] = 0.0f;
        q_raw[4] = 0.0f;
        q_raw[5] = 0.0f;
        q_raw[6] = 0.0f;
        q_raw[7] = 0.0f;
        q_raw[8] = 0.0f;
        q_raw[9] = 0.0f;
        q_raw[10] = 0.0f;
        q_raw[11] = 0.0f;
        q_raw[12] = 0.0f;
        q_raw[13] = 0.0f;
        q_raw[14] = 0.0f;
        q_raw[15] = 0.0f;
        k_raw[0] = 0.0f;
        k_raw[1] = 0.0f;
        k_raw[2] = 0.0f;
        k_raw[3] = 0.0f;
        k_raw[4] = 0.0f;
        k_raw[5] = 0.0f;
        k_raw[6] = 0.0f;
        k_raw[7] = 0.0f;
        k_raw[8] = 0.0f;
        k_raw[9] = 0.0f;
        k_raw[10] = 0.0f;
        k_raw[11] = 0.0f;
        k_raw[12] = 0.0f;
        k_raw[13] = 0.0f;
        k_raw[14] = 0.0f;
        k_raw[15] = 0.0f;
        #pragma unroll
        for (int dim_half_1 = 0; dim_half_1 < 2; dim_half_1++) {
            int segment = dim_half_1 * 8 + lane_in_row;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&q_raw_packed[dim_half_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&q_raw_packed[(dim_half_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_raw_packed[(dim_half_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_raw_packed[(dim_half_1 * 4) + 3]))
                : "r"((smem_q_raw_addr + (unsigned int)(segment * 8 / 64 * 2048 + row_3 * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row_3 * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&k_raw_packed[dim_half_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&k_raw_packed[(dim_half_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_raw_packed[(dim_half_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_raw_packed[(dim_half_1 * 4) + 3]))
                : "r"((smem_k_raw_addr + (unsigned int)(segment * 8 / 64 * 2048 + row_3 * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row_3 * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
        }
        float q_raw_packed_f32[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&q_raw_packed_f32[_pair * 2])[0]), "=f"((&q_raw_packed_f32[_pair * 2])[1])
                : "r"(q_raw_packed[_pair]));
        }
        float k_raw_packed_f32[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&k_raw_packed_f32[_pair * 2])[0]), "=f"((&k_raw_packed_f32[_pair * 2])[1])
                : "r"(k_raw_packed[_pair]));
        }
        #pragma unroll
        for (int elem = 0; elem < 16; elem++) {
            q_raw[elem] = q_raw_packed_f32[elem];
            k_raw[elem] = k_raw_packed_f32[elem];
        }
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        #pragma unroll
        for (int elem_pair = 0; elem_pair < 8; elem_pair++) {
            float _bf16x2_dot_f32_0;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_bf16x2_dot_f32_0) : "r"(q_raw_packed[elem_pair]), "r"(q_raw_packed[elem_pair]), "f"(q_sum));
            q_sum = _bf16x2_dot_f32_0;
            float _bf16x2_dot_f32_1;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_bf16x2_dot_f32_1) : "r"(k_raw_packed[elem_pair]), "r"(k_raw_packed[elem_pair]), "f"(k_sum));
            k_sum = _bf16x2_dot_f32_1;
        }
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 4);
        q_sum += _shfl_xor_0;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 4);
        k_sum += _shfl_xor_1;
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 2);
        q_sum += _shfl_xor_2;
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 2);
        k_sum += _shfl_xor_3;
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 1);
        q_sum += _shfl_xor_4;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 1);
        k_sum += _shfl_xor_5;
        float _rsqrt_0 = rsqrtf(q_sum + 1e-06f);
        float q_inv = _rsqrt_0;
        float _rsqrt_1 = rsqrtf(k_sum + 1e-06f);
        float k_inv_norm = _rsqrt_1;
        __syncthreads();
        float kk_acc[8];
        #pragma unroll
        for (int dim_half_2 = 0; dim_half_2 < 2; dim_half_2++) {
            int segment_1 = dim_half_2 * 8 + lane_in_row;
            int reg_base = dim_half_2 * 8;
            float qd_values[8];
            float kd_values[8];
            float ki_values[8];
            float2 _f2_0 = make_float2(q_inv, q_inv);
            float2 q_norm_pair = _f2_0;
            float2 _f2_1 = make_float2(k_inv_norm, k_inv_norm);
            float2 k_norm_pair = _f2_1;
            #pragma unroll
            for (int elem_pair_1 = 0; elem_pair_1 < 4; elem_pair_1++) {
                int elem0 = elem_pair_1 * 2;
                int elem1 = elem0 + 1;
                int this_col0 = segment_1 * 8 + elem0;
                int this_col1 = this_col0 + 1;
                float decay0 = smem_gate[((smem_gate_addr + (unsigned int)(this_col0 / 32 * 2048 + row_3 * 128 + this_col0 % 32 * 4 ^ (this_col0 / 32 * 2048 + row_3 * 128 + this_col0 % 32 * 4 >> 7 & 7) << 4)) - smem_gate_addr) / 4];
                float decay1 = smem_gate[((smem_gate_addr + (unsigned int)(this_col1 / 32 * 2048 + row_3 * 128 + this_col1 % 32 * 4 ^ (this_col1 / 32 * 2048 + row_3 * 128 + this_col1 % 32 * 4 >> 7 & 7) << 4)) - smem_gate_addr) / 4];
                float _rcp_0 = approx_rcp(decay0);
                float inv_decay0 = _rcp_0;
                float _rcp_1 = approx_rcp(decay1);
                float inv_decay1 = _rcp_1;
                float2 _f2_2 = make_float2(q_raw[reg_base + elem0], q_raw[reg_base + elem1]);
                float2 raw_q_pair = _f2_2;
                float2 _f2_3 = make_float2(k_raw[reg_base + elem0], k_raw[reg_base + elem1]);
                float2 raw_k_pair = _f2_3;
                float2 _mul_f32x2_0;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&raw_q_pair), "l"(*(const unsigned long long*)&q_norm_pair));
                float2 q_value_pair = _mul_f32x2_0;
                float2 _mul_f32x2_1;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&raw_k_pair), "l"(*(const unsigned long long*)&k_norm_pair));
                float2 k_value_pair = _mul_f32x2_1;
                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(q_value_pair.x);
                float _cvt_f32_2 = __bfloat162float(_cvt_bf16_0);
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(q_value_pair.y);
                float _cvt_f32_3 = __bfloat162float(_cvt_bf16_1);
                float2 _f2_4 = make_float2(_cvt_f32_2, _cvt_f32_3);
                q_value_pair = _f2_4;
                __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(k_value_pair.x);
                float _cvt_f32_4 = __bfloat162float(_cvt_bf16_2);
                __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(k_value_pair.y);
                float _cvt_f32_5 = __bfloat162float(_cvt_bf16_3);
                float2 _f2_5 = make_float2(_cvt_f32_4, _cvt_f32_5);
                k_value_pair = _f2_5;
                float2 _f2_6 = make_float2(decay0, decay1);
                float2 decay_pair = _f2_6;
                float2 _f2_7 = make_float2(inv_decay0, inv_decay1);
                float2 inv_decay_pair = _f2_7;
                float2 _mul_f32x2_2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&q_value_pair), "l"(*(const unsigned long long*)&decay_pair));
                float2 qd_pair = _mul_f32x2_2;
                float2 _mul_f32x2_3;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&k_value_pair), "l"(*(const unsigned long long*)&decay_pair));
                float2 kd_pair = _mul_f32x2_3;
                float2 _mul_f32x2_4;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&k_value_pair), "l"(*(const unsigned long long*)&inv_decay_pair));
                float2 ki_pair = _mul_f32x2_4;
                qd_values[elem0] = qd_pair.x;
                qd_values[elem1] = qd_pair.y;
                kd_values[elem0] = kd_pair.x;
                kd_values[elem1] = kd_pair.y;
                ki_values[elem0] = ki_pair.x;
                ki_values[elem1] = ki_pair.y;
            }
            unsigned int words[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                words[_lp] = __float_as_uint(qd_values[_lp + 0]);
            }
            #pragma unroll
            for (int vector = 0; vector < 2; vector++) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"((smem_qd_addr + (unsigned int)((segment_1 * 8 + vector * 4) / 32 * 2048 + row_3 * 128 + (segment_1 * 8 + vector * 4) % 32 * 4 ^ ((segment_1 * 8 + vector * 4) / 32 * 2048 + row_3 * 128 + (segment_1 * 8 + vector * 4) % 32 * 4 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&words[vector * 4])), "r"(*reinterpret_cast<uint32_t*>(&words[(vector * 4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words[(vector * 4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words[(vector * 4) + 3])));
            }
            unsigned int words_0[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                words_0[_lp] = __float_as_uint(kd_values[_lp + 0]);
            }
            #pragma unroll
            for (int vector_1 = 0; vector_1 < 2; vector_1++) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"((smem_kd_addr + (unsigned int)((segment_1 * 8 + vector_1 * 4) / 32 * 2048 + row_3 * 128 + (segment_1 * 8 + vector_1 * 4) % 32 * 4 ^ ((segment_1 * 8 + vector_1 * 4) / 32 * 2048 + row_3 * 128 + (segment_1 * 8 + vector_1 * 4) % 32 * 4 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&words_0[vector_1 * 4])), "r"(*reinterpret_cast<uint32_t*>(&words_0[(vector_1 * 4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_0[(vector_1 * 4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_0[(vector_1 * 4) + 3])));
            }
            unsigned int words_1[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                words_1[_lp] = __float_as_uint(ki_values[_lp + 0]);
            }
            #pragma unroll
            for (int vector_2 = 0; vector_2 < 2; vector_2++) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"((smem_ki_addr + (unsigned int)((segment_1 * 8 + vector_2 * 4) / 32 * 2048 + row_3 * 128 + (segment_1 * 8 + vector_2 * 4) % 32 * 4 ^ ((segment_1 * 8 + vector_2 * 4) / 32 * 2048 + row_3 * 128 + (segment_1 * 8 + vector_2 * 4) % 32 * 4 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&words_1[vector_2 * 4])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(vector_2 * 4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(vector_2 * 4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(vector_2 * 4) + 3])));
            }
            __syncthreads();
            if (dim_half_2 == 0) {
                if (warp == 0) {
                    float a_values[4];
                    float b_values[2];
                    unsigned int a_regs[4];
                    unsigned int b_regs[2];
                    #pragma unroll
                    for (int kk = 0; kk < 8; kk++) {
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            int ar = lane / 4 + (unsigned int)(i % 2 * 8);
                            int ak = (unsigned int)(kk * 8) + lane % 4 + (unsigned int)(i / 2 * 4);
                            a_values[i] = smem_kd[((smem_kd_addr + (unsigned int)(ak / 32 * 2048 + ar * 128 + ak % 32 * 4 ^ (ak / 32 * 2048 + ar * 128 + ak % 32 * 4 >> 7 & 7) << 4)) - smem_kd_addr) / 4];
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            a_regs[_lp] = __float_as_uint(a_values[_lp + 0]);
                        }
                        #pragma unroll
                        for (int nn = 0; nn < 2; nn++) {
                            #pragma unroll
                            for (int i_1 = 0; i_1 < 2; i_1++) {
                                int bk = (unsigned int)(kk * 8) + lane % 4 + (unsigned int)(i_1 * 4);
                                int bn = (unsigned int)(nn * 8) + lane / 4;
                                {
                                    b_values[i_1] = smem_ki[((smem_ki_addr + (unsigned int)(bk / 32 * 2048 + bn * 128 + bk % 32 * 4 ^ (bk / 32 * 2048 + bn * 128 + bk % 32 * 4 >> 7 & 7) << 4)) - smem_ki_addr) / 4];
                                }
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                b_regs[_lp] = __float_as_uint(b_values[_lp + 0]);
                            }
                            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"((kk_acc + nn * 4)[0]), "=f"((kk_acc + nn * 4)[1]), "=f"((kk_acc + nn * 4)[2]), "=f"((kk_acc + nn * 4)[3])
                                : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]), "r"(b_regs[0]), "r"(b_regs[1]), "f"(((((kk == 0) ? 1 : 0)) ? 0.0f : (kk_acc + nn * 4)[0])), "f"(((((kk == 0) ? 1 : 0)) ? 0.0f : (kk_acc + nn * 4)[1])), "f"(((((kk == 0) ? 1 : 0)) ? 0.0f : (kk_acc + nn * 4)[2])), "f"(((((kk == 0) ? 1 : 0)) ? 0.0f : (kk_acc + nn * 4)[3])));
                        }
                    }
                }
            }
        }
        if (warp == 0) {
            float a_values_1[4];
            float b_values_1[2];
            unsigned int a_regs_1[4];
            unsigned int b_regs_1[2];
            #pragma unroll
            for (int kk_1 = 0; kk_1 < 8; kk_1++) {
                #pragma unroll
                for (int i_2 = 0; i_2 < 4; i_2++) {
                    int ar_1 = lane / 4 + (unsigned int)(i_2 % 2 * 8);
                    int ak_1 = (unsigned int)(64 + kk_1 * 8) + lane % 4 + (unsigned int)(i_2 / 2 * 4);
                    a_values_1[i_2] = smem_kd[((smem_kd_addr + (unsigned int)(ak_1 / 32 * 2048 + ar_1 * 128 + ak_1 % 32 * 4 ^ (ak_1 / 32 * 2048 + ar_1 * 128 + ak_1 % 32 * 4 >> 7 & 7) << 4)) - smem_kd_addr) / 4];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    a_regs_1[_lp] = __float_as_uint(a_values_1[_lp + 0]);
                }
                #pragma unroll
                for (int nn_1 = 0; nn_1 < 2; nn_1++) {
                    #pragma unroll
                    for (int i_3 = 0; i_3 < 2; i_3++) {
                        int bk_1 = (unsigned int)(64 + kk_1 * 8) + lane % 4 + (unsigned int)(i_3 * 4);
                        int bn_1 = (unsigned int)(nn_1 * 8) + lane / 4;
                        {
                            b_values_1[i_3] = smem_ki[((smem_ki_addr + (unsigned int)(bk_1 / 32 * 2048 + bn_1 * 128 + bk_1 % 32 * 4 ^ (bk_1 / 32 * 2048 + bn_1 * 128 + bk_1 % 32 * 4 >> 7 & 7) << 4)) - smem_ki_addr) / 4];
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        b_regs_1[_lp] = __float_as_uint(b_values_1[_lp + 0]);
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"((kk_acc + nn_1 * 4)[0]), "=f"((kk_acc + nn_1 * 4)[1]), "=f"((kk_acc + nn_1 * 4)[2]), "=f"((kk_acc + nn_1 * 4)[3])
                        : "r"(a_regs_1[0]), "r"(a_regs_1[1]), "r"(a_regs_1[2]), "r"(a_regs_1[3]), "r"(b_regs_1[0]), "r"(b_regs_1[1]), "f"(((((kk_1 == 0) ? 0 : 0)) ? 0.0f : (kk_acc + nn_1 * 4)[0])), "f"(((((kk_1 == 0) ? 0 : 0)) ? 0.0f : (kk_acc + nn_1 * 4)[1])), "f"(((((kk_1 == 0) ? 0 : 0)) ? 0.0f : (kk_acc + nn_1 * 4)[2])), "f"(((((kk_1 == 0) ? 0 : 0)) ? 0.0f : (kk_acc + nn_1 * 4)[3])));
                }
            }
            int row0 = lane / 4;
            int row1 = row0 + 8;
            int col0 = lane % 4 * 2;
            float beta0 = smem_beta[row0];
            float beta1 = smem_beta[row1];
            float l_values[8];
            l_values[0] = 0.0f;
            l_values[1] = 0.0f;
            l_values[2] = 0.0f;
            l_values[3] = 0.0f;
            l_values[4] = 0.0f;
            l_values[5] = 0.0f;
            l_values[6] = 0.0f;
            l_values[7] = 0.0f;
            if (row0 > col0) {
                l_values[0] = kk_acc[0] * beta0;
            }
            if (row0 > col0 + 1) {
                l_values[1] = kk_acc[1] * beta0;
            }
            if (row1 > col0) {
                l_values[2] = kk_acc[2] * beta1;
            }
            if (row1 > col0 + 1) {
                l_values[3] = kk_acc[3] * beta1;
            }
            if (row0 > col0 + 8) {
                l_values[4] = kk_acc[4] * beta0;
            }
            if (row0 > col0 + 9) {
                l_values[5] = kk_acc[5] * beta0;
            }
            if (row1 > col0 + 8) {
                l_values[6] = kk_acc[6] * beta1;
            }
            if (row1 > col0 + 9) {
                l_values[7] = kk_acc[7] * beta1;
            }
            float inverse_low[2];
            float inverse_high[2];
            #pragma unroll
            for (int word_2 = 0; word_2 < 2; word_2++) {
                inverse_low[word_2] = 0.0f;
                if (lane / 4 == lane % 4 * 2 + (unsigned int)word_2) {
                    inverse_low[word_2] = 1.0f;
                }
            }
            if (lane % 4 == 0 && lane / 4 > 0) {
                inverse_low[0] = -l_values[0];
            }
            #pragma unroll
            for (int inner = 1; inner < 7; inner++) {
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, (l_values + 0)[inner % 2], lane / 4 * 4 + (unsigned int)(inner / 2));
                float coeff = _shfl_0;
                float _shfl_1 = __shfl_sync(0xFFFFFFFF, inverse_low[0], (unsigned int)(inner * 4) + lane % 4);
                float prior0 = _shfl_1;
                float _shfl_2 = __shfl_sync(0xFFFFFFFF, inverse_low[1], (unsigned int)(inner * 4) + lane % 4);
                float prior1 = _shfl_2;
                if (lane / 4 > (unsigned int)inner) {
                    float _fma_2 = __fmaf_rn(-coeff, prior0, inverse_low[0]);
                    inverse_low[0] = _fma_2;
                    float _fma_3 = __fmaf_rn(-coeff, prior1, inverse_low[1]);
                    inverse_low[1] = _fma_3;
                }
            }
            #pragma unroll
            for (int word_3 = 0; word_3 < 2; word_3++) {
                inverse_high[word_3] = 0.0f;
                if (lane / 4 == lane % 4 * 2 + (unsigned int)word_3) {
                    inverse_high[word_3] = 1.0f;
                }
            }
            if (lane % 4 == 0 && lane / 4 > 0) {
                inverse_high[0] = -l_values[6];
            }
            #pragma unroll
            for (int inner_1 = 1; inner_1 < 7; inner_1++) {
                float _shfl_3 = __shfl_sync(0xFFFFFFFF, (l_values + 6)[inner_1 % 2], lane / 4 * 4 + (unsigned int)(inner_1 / 2));
                float coeff_1 = _shfl_3;
                float _shfl_4 = __shfl_sync(0xFFFFFFFF, inverse_high[0], (unsigned int)(inner_1 * 4) + lane % 4);
                float prior0_1 = _shfl_4;
                float _shfl_5 = __shfl_sync(0xFFFFFFFF, inverse_high[1], (unsigned int)(inner_1 * 4) + lane % 4);
                float prior1_1 = _shfl_5;
                if (lane / 4 > (unsigned int)inner_1) {
                    float _fma_4 = __fmaf_rn(-coeff_1, prior0_1, inverse_high[0]);
                    inverse_high[0] = _fma_4;
                    float _fma_5 = __fmaf_rn(-coeff_1, prior1_1, inverse_high[1]);
                    inverse_high[1] = _fma_5;
                }
            }
            float inverse_cross_tmp[2];
            float inverse_cross[2];
            unsigned int inv8_lhs_words[2];
            unsigned int inv8_rhs_words[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                inv8_lhs_words[_lp] = __float_as_uint(inverse_high[_lp + 0]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                inv8_rhs_words[_lp] = __float_as_uint(l_values[_lp + 2]);
            }
            float inv8_a[4];
            float inv8_b[2];
            unsigned int inv8_a_words[4];
            unsigned int inv8_b_words[2];
            float inv8_acc[4];
            #pragma unroll
            for (int word_4 = 0; word_4 < 2; word_4++) {
                int col_0 = lane % 4 + (unsigned int)(word_4 * 4);
                int a_lane = lane / 4 * 4 + (unsigned int)(col_0 / 2);
                float _shfl_6 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_lhs_words)[0], a_lane);
                float a_low = _shfl_6;
                float _shfl_7 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_lhs_words)[1], a_lane);
                float a_high = _shfl_7;
                float a_value = a_low;
                if ((lane & 1) != 0) {
                    a_value = a_high;
                }
                inv8_a[word_4 * 2] = a_value;
                inv8_a[word_4 * 2 + 1] = a_value;
                int b_lane = (unsigned int)(col_0 * 4) + lane / 4 / 2;
                float _shfl_8 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_rhs_words)[0], b_lane);
                float b_low = _shfl_8;
                float _shfl_9 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_rhs_words)[1], b_lane);
                float b_high = _shfl_9;
                inv8_b[word_4] = b_low;
                if ((lane / 4 & 1) != 0) {
                    inv8_b[word_4] = b_high;
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                inv8_a_words[_lp] = __float_as_uint(inv8_a[_lp + 0]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                inv8_b_words[_lp] = __float_as_uint(inv8_b[_lp + 0]);
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(inv8_acc[0]), "=f"(inv8_acc[1]), "=f"(inv8_acc[2]), "=f"(inv8_acc[3])
                : "r"(inv8_a_words[0]), "r"(inv8_a_words[1]), "r"(inv8_a_words[2]), "r"(inv8_a_words[3]), "r"(inv8_b_words[0]), "r"(inv8_b_words[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            inverse_cross_tmp[0] = inv8_acc[0];
            inverse_cross_tmp[1] = inv8_acc[1];
            unsigned int inv8_lhs_words_0[2];
            unsigned int inv8_rhs_words_1[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                inv8_lhs_words_0[_lp] = __float_as_uint(inverse_cross_tmp[_lp + 0]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                inv8_rhs_words_1[_lp] = __float_as_uint(inverse_low[_lp + 0]);
            }
            float inv8_a_2[4];
            float inv8_b_3[2];
            unsigned int inv8_a_words_4[4];
            unsigned int inv8_b_words_5[2];
            float inv8_acc_6[4];
            #pragma unroll
            for (int word_5 = 0; word_5 < 2; word_5++) {
                int col_0_1 = lane % 4 + (unsigned int)(word_5 * 4);
                int a_lane_1 = lane / 4 * 4 + (unsigned int)(col_0_1 / 2);
                float _shfl_10 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_lhs_words_0)[0], a_lane_1);
                float a_low_1 = _shfl_10;
                float _shfl_11 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_lhs_words_0)[1], a_lane_1);
                float a_high_1 = _shfl_11;
                float a_value_1 = a_low_1;
                if ((lane & 1) != 0) {
                    a_value_1 = a_high_1;
                }
                inv8_a_2[word_5 * 2] = a_value_1;
                inv8_a_2[word_5 * 2 + 1] = a_value_1;
                int b_lane_1 = (unsigned int)(col_0_1 * 4) + lane / 4 / 2;
                float _shfl_12 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_rhs_words_1)[0], b_lane_1);
                float b_low_1 = _shfl_12;
                float _shfl_13 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_rhs_words_1)[1], b_lane_1);
                float b_high_1 = _shfl_13;
                inv8_b_3[word_5] = b_low_1;
                if ((lane / 4 & 1) != 0) {
                    inv8_b_3[word_5] = b_high_1;
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                inv8_a_words_4[_lp] = __float_as_uint(inv8_a_2[_lp + 0]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                inv8_b_words_5[_lp] = __float_as_uint(inv8_b_3[_lp + 0]);
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(inv8_acc_6[0]), "=f"(inv8_acc_6[1]), "=f"(inv8_acc_6[2]), "=f"(inv8_acc_6[3])
                : "r"(inv8_a_words_4[0]), "r"(inv8_a_words_4[1]), "r"(inv8_a_words_4[2]), "r"(inv8_a_words_4[3]), "r"(inv8_b_words_5[0]), "r"(inv8_b_words_5[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            inverse_cross[0] = inv8_acc_6[0];
            inverse_cross[1] = inv8_acc_6[1];
            float inverse[8];
            inverse[0] = 0.0f;
            inverse[1] = 0.0f;
            inverse[2] = 0.0f;
            inverse[3] = 0.0f;
            inverse[4] = 0.0f;
            inverse[5] = 0.0f;
            inverse[6] = 0.0f;
            inverse[7] = 0.0f;
            inverse[0] = inverse_low[0];
            inverse[1] = inverse_low[1];
            inverse[6] = inverse_high[0];
            inverse[7] = inverse_high[1];
            inverse[2] = -inverse_cross[0];
            inverse[3] = -inverse_cross[1];
            float result[8];
            #pragma unroll
            for (int word_6 = 0; word_6 < 8; word_6++) {
                int col_0_2 = lane % 4 * 2 + (unsigned int)(word_6 % 2) + (unsigned int)(word_6 / 4 * 8);
                float beta_col = smem_beta[col_0_2];
                result[word_6] = inverse[word_6] * beta_col;
            }
            unsigned int inverse_packed[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                inverse_packed[_lp] = __float_as_uint(result[_lp + 0]);
            }
            #pragma unroll
            for (int word_7 = 0; word_7 < 8; word_7++) {
                int row_0 = lane / 4 + (unsigned int)(word_7 % 4 / 2 * 8);
                int col_1 = lane % 4 * 2 + (unsigned int)(word_7 % 2) + (unsigned int)(word_7 / 4 * 8);
                smem_abt[((smem_abt_addr + (unsigned int)(row_0 / 16 * 1024 + col_1 * 64 + row_0 % 16 * 4 ^ (row_0 / 16 * 1024 + col_1 * 64 + row_0 % 16 * 4 >> 7 & 3) << 4)) - smem_abt_addr) / 4] = reinterpret_cast<float*>(inverse_packed)[word_7];
            }
            __syncwarp();
        }
        if (warp == 2) {
            float qk_acc[8];
            float a_values_2[4];
            float b_values_2[2];
            unsigned int a_regs_2[4];
            unsigned int b_regs_2[2];
            #pragma unroll
            for (int kk_2 = 0; kk_2 < 16; kk_2++) {
                #pragma unroll
                for (int i_4 = 0; i_4 < 4; i_4++) {
                    int ar_2 = lane / 4 + (unsigned int)(i_4 % 2 * 8);
                    int ak_2 = (unsigned int)(kk_2 * 8) + lane % 4 + (unsigned int)(i_4 / 2 * 4);
                    a_values_2[i_4] = smem_qd[((smem_qd_addr + (unsigned int)(ak_2 / 32 * 2048 + ar_2 * 128 + ak_2 % 32 * 4 ^ (ak_2 / 32 * 2048 + ar_2 * 128 + ak_2 % 32 * 4 >> 7 & 7) << 4)) - smem_qd_addr) / 4];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    a_regs_2[_lp] = __float_as_uint(a_values_2[_lp + 0]);
                }
                #pragma unroll
                for (int nn_2 = 0; nn_2 < 2; nn_2++) {
                    #pragma unroll
                    for (int i_5 = 0; i_5 < 2; i_5++) {
                        int bk_2 = (unsigned int)(kk_2 * 8) + lane % 4 + (unsigned int)(i_5 * 4);
                        int bn_2 = (unsigned int)(nn_2 * 8) + lane / 4;
                        {
                            b_values_2[i_5] = smem_ki[((smem_ki_addr + (unsigned int)(bk_2 / 32 * 2048 + bn_2 * 128 + bk_2 % 32 * 4 ^ (bk_2 / 32 * 2048 + bn_2 * 128 + bk_2 % 32 * 4 >> 7 & 7) << 4)) - smem_ki_addr) / 4];
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        b_regs_2[_lp] = __float_as_uint(b_values_2[_lp + 0]);
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"((qk_acc + nn_2 * 4)[0]), "=f"((qk_acc + nn_2 * 4)[1]), "=f"((qk_acc + nn_2 * 4)[2]), "=f"((qk_acc + nn_2 * 4)[3])
                        : "r"(a_regs_2[0]), "r"(a_regs_2[1]), "r"(a_regs_2[2]), "r"(a_regs_2[3]), "r"(b_regs_2[0]), "r"(b_regs_2[1]), "f"(((((kk_2 == 0) ? 1 : 0)) ? 0.0f : (qk_acc + nn_2 * 4)[0])), "f"(((((kk_2 == 0) ? 1 : 0)) ? 0.0f : (qk_acc + nn_2 * 4)[1])), "f"(((((kk_2 == 0) ? 1 : 0)) ? 0.0f : (qk_acc + nn_2 * 4)[2])), "f"(((((kk_2 == 0) ? 1 : 0)) ? 0.0f : (qk_acc + nn_2 * 4)[3])));
                }
            }
            int row0_1 = lane / 4;
            int row1_1 = row0_1 + 8;
            int col0_1 = lane % 4 * 2;
            float qk_values[8];
            qk_values[0] = 0.0f;
            qk_values[1] = 0.0f;
            qk_values[2] = 0.0f;
            qk_values[3] = 0.0f;
            qk_values[4] = 0.0f;
            qk_values[5] = 0.0f;
            qk_values[6] = 0.0f;
            qk_values[7] = 0.0f;
            if (row0_1 >= col0_1) {
                qk_values[0] = qk_acc[0];
            }
            if (row0_1 >= col0_1 + 1) {
                qk_values[1] = qk_acc[1];
            }
            if (row1_1 >= col0_1) {
                qk_values[2] = qk_acc[2];
            }
            if (row1_1 >= col0_1 + 1) {
                qk_values[3] = qk_acc[3];
            }
            if (row0_1 >= col0_1 + 8) {
                qk_values[4] = qk_acc[4];
            }
            if (row0_1 >= col0_1 + 9) {
                qk_values[5] = qk_acc[5];
            }
            if (row1_1 >= col0_1 + 8) {
                qk_values[6] = qk_acc[6];
            }
            if (row1_1 >= col0_1 + 9) {
                qk_values[7] = qk_acc[7];
            }
            #pragma unroll
            for (int i_6 = 0; i_6 < 8; i_6++) {
                int row_0_1 = lane / 4 + (unsigned int)(i_6 % 4 / 2 * 8);
                int col_1_1 = lane % 4 * 2 + (unsigned int)(i_6 % 2) + (unsigned int)(i_6 / 4 * 8);
                smem_qk_plain[((smem_qk_plain_addr + (unsigned int)(col_1_1 / 16 * 1024 + row_0_1 * 64 + col_1_1 % 16 * 4 ^ (col_1_1 / 16 * 1024 + row_0_1 * 64 + col_1_1 % 16 * 4 >> 7 & 3) << 4)) - smem_qk_plain_addr) / 4] = qk_values[i_6];
            }
            __syncwarp();
        }
        __syncthreads();
        long long qk_ws_base = ((long long)head_idx * (long long)total_chunks + (long long)gchunk) * 16 * 16;
        if (warp == 2) {
            float qk_fold_acc[8];
            float a_values_3[4];
            float b_values_3[2];
            unsigned int a_regs_3[4];
            unsigned int b_regs_3[2];
            #pragma unroll
            for (int kk_3 = 0; kk_3 < 2; kk_3++) {
                #pragma unroll
                for (int i_7 = 0; i_7 < 4; i_7++) {
                    int ar_3 = lane / 4 + (unsigned int)(i_7 % 2 * 8);
                    int ak_3 = (unsigned int)(kk_3 * 8) + lane % 4 + (unsigned int)(i_7 / 2 * 4);
                    a_values_3[i_7] = smem_abt[((smem_abt_addr + (unsigned int)(ak_3 / 16 * 1024 + ar_3 * 64 + ak_3 % 16 * 4 ^ (ak_3 / 16 * 1024 + ar_3 * 64 + ak_3 % 16 * 4 >> 7 & 3) << 4)) - smem_abt_addr) / 4];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    a_regs_3[_lp] = __float_as_uint(a_values_3[_lp + 0]);
                }
                #pragma unroll
                for (int nn_3 = 0; nn_3 < 2; nn_3++) {
                    #pragma unroll
                    for (int i_8 = 0; i_8 < 2; i_8++) {
                        int bk_3 = (unsigned int)(kk_3 * 8) + lane % 4 + (unsigned int)(i_8 * 4);
                        int bn_3 = (unsigned int)(nn_3 * 8) + lane / 4;
                        {
                            b_values_3[i_8] = smem_qk_plain[((smem_qk_plain_addr + (unsigned int)(bk_3 / 16 * 1024 + bn_3 * 64 + bk_3 % 16 * 4 ^ (bk_3 / 16 * 1024 + bn_3 * 64 + bk_3 % 16 * 4 >> 7 & 3) << 4)) - smem_qk_plain_addr) / 4];
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        b_regs_3[_lp] = __float_as_uint(b_values_3[_lp + 0]);
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"((qk_fold_acc + nn_3 * 4)[0]), "=f"((qk_fold_acc + nn_3 * 4)[1]), "=f"((qk_fold_acc + nn_3 * 4)[2]), "=f"((qk_fold_acc + nn_3 * 4)[3])
                        : "r"(a_regs_3[0]), "r"(a_regs_3[1]), "r"(a_regs_3[2]), "r"(a_regs_3[3]), "r"(b_regs_3[0]), "r"(b_regs_3[1]), "f"(((((kk_3 == 0) ? 1 : 0)) ? 0.0f : (qk_fold_acc + nn_3 * 4)[0])), "f"(((((kk_3 == 0) ? 1 : 0)) ? 0.0f : (qk_fold_acc + nn_3 * 4)[1])), "f"(((((kk_3 == 0) ? 1 : 0)) ? 0.0f : (qk_fold_acc + nn_3 * 4)[2])), "f"(((((kk_3 == 0) ? 1 : 0)) ? 0.0f : (qk_fold_acc + nn_3 * 4)[3])));
                }
            }
            unsigned int rounded[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                rounded[_lp] = __float_as_uint(qk_fold_acc[_lp + 0]);
            }
            #pragma unroll
            for (int i_9 = 0; i_9 < 8; i_9++) {
                int row_0_2 = lane / 4 + (unsigned int)(i_9 % 4 / 2 * 8);
                int col_1_2 = lane % 4 * 2 + (unsigned int)(i_9 % 2) + (unsigned int)(i_9 / 4 * 8);
                ws_qk_t[qk_ws_base + (long long)col_1_2 * 16 + (long long)row_0_2] = reinterpret_cast<float*>(rounded)[i_9];
            }
        }
        if (warp == 1 || warp == 3) {
            int fold_slot = 0;
            int w_group_base = 0;
            if (warp == 3) {
                fold_slot = 2;
                w_group_base = 4;
            }
            #pragma unroll
            for (int local_group = 0; local_group < 4; local_group++) {
                int w_group = w_group_base + local_group;
                float w_fold_acc[8];
                float a_values_4[4];
                float b_values_4[2];
                unsigned int a_regs_4[4];
                unsigned int b_regs_4[2];
                #pragma unroll
                for (int kk_4 = 0; kk_4 < 2; kk_4++) {
                    #pragma unroll
                    for (int i_10 = 0; i_10 < 4; i_10++) {
                        int ar_4 = lane / 4 + (unsigned int)(i_10 % 2 * 8);
                        int ak_4 = (unsigned int)(kk_4 * 8) + lane % 4 + (unsigned int)(i_10 / 2 * 4);
                        a_values_4[i_10] = smem_abt[((smem_abt_addr + (unsigned int)(ak_4 / 16 * 1024 + ar_4 * 64 + ak_4 % 16 * 4 ^ (ak_4 / 16 * 1024 + ar_4 * 64 + ak_4 % 16 * 4 >> 7 & 3) << 4)) - smem_abt_addr) / 4];
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        a_regs_4[_lp] = __float_as_uint(a_values_4[_lp + 0]);
                    }
                    #pragma unroll
                    for (int nn_4 = 0; nn_4 < 2; nn_4++) {
                        #pragma unroll
                        for (int i_11 = 0; i_11 < 2; i_11++) {
                            int bk_4 = (unsigned int)(kk_4 * 8) + lane % 4 + (unsigned int)(i_11 * 4);
                            int bn_4 = (unsigned int)(w_group * 16 + nn_4 * 8) + lane / 4;
                            {
                                b_values_4[i_11] = smem_ki[((smem_ki_addr + (unsigned int)(bn_4 / 32 * 2048 + bk_4 * 128 + bn_4 % 32 * 4 ^ (bn_4 / 32 * 2048 + bk_4 * 128 + bn_4 % 32 * 4 >> 7 & 7) << 4)) - smem_ki_addr) / 4];
                            }
                        }
                        {
                            float total = smem_gate_total[(unsigned int)(w_group * 16 + nn_4 * 8) + lane / 4];
                            float2 _f2_16 = make_float2(b_values_4[0], b_values_4[1]);
                            float2 _f2_17 = make_float2(total, total);
                            float2 _mul_f32x2_9;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&_f2_16), "l"(*(const unsigned long long*)&_f2_17));
                            float2 scaled = _mul_f32x2_9;
                            b_values_4[0] = scaled.x;
                            b_values_4[1] = scaled.y;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            b_regs_4[_lp] = __float_as_uint(b_values_4[_lp + 0]);
                        }
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"((w_fold_acc + nn_4 * 4)[0]), "=f"((w_fold_acc + nn_4 * 4)[1]), "=f"((w_fold_acc + nn_4 * 4)[2]), "=f"((w_fold_acc + nn_4 * 4)[3])
                            : "r"(a_regs_4[0]), "r"(a_regs_4[1]), "r"(a_regs_4[2]), "r"(a_regs_4[3]), "r"(b_regs_4[0]), "r"(b_regs_4[1]), "f"(((((kk_4 == 0) ? 1 : 0)) ? 0.0f : (w_fold_acc + nn_4 * 4)[0])), "f"(((((kk_4 == 0) ? 1 : 0)) ? 0.0f : (w_fold_acc + nn_4 * 4)[1])), "f"(((((kk_4 == 0) ? 1 : 0)) ? 0.0f : (w_fold_acc + nn_4 * 4)[2])), "f"(((((kk_4 == 0) ? 1 : 0)) ? 0.0f : (w_fold_acc + nn_4 * 4)[3])));
                    }
                }
                #pragma unroll
                for (int word_8 = 0; word_8 < 8; word_8++) {
                    int row_0_3 = lane / 4 + (unsigned int)(word_8 % 4 / 2 * 8);
                    int col_1_3 = (unsigned int)(w_group * 16) + lane % 4 * 2 + (unsigned int)(word_8 % 2) + (unsigned int)(word_8 / 4 * 8);
                    ws_w[((long long)head_idx * (long long)total_chunks + (long long)gchunk) * 128 * 16 + (long long)col_1_3 * 16 + (long long)row_0_3] = w_fold_acc[word_8];
                }
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        raw_epoch += 1;
        if (my_chunks > cta_chunk + 1) {
            int next_gchunk = gchunk + 1;
            int next_seq = chunk_to_seq[next_gchunk];
            int next_local_chunk = next_gchunk - cu_chunks[next_seq];
            long long next_bos = cu_seqlens[next_seq];
            long long next_eos = cu_seqlens[next_seq + 1];
            long long next_token_base = next_bos + (long long)(next_local_chunk * 16);
            unsigned int next_raw_stage = (unsigned int)raw_epoch & 1;
            if (warp == 0) {
                if (elect_sync()) {
                    int gate_tx_bytes_1 = 4096;
                    mbarrier_arrive_expect_tx(gate_raw_full_addr + (next_raw_stage) * 8, gate_tx_bytes_1);
                    tma_3d_gmem2smem(smem_gate_raw_addr, raw_gate_tma, 0, head_idx, (int)next_token_base, gate_raw_full_addr + (next_raw_stage) * 8);
                    mbarrier_arrive_expect_tx(qk_raw_full_addr + (next_raw_stage) * 8, 8192);
                    tma_4d_gmem2smem(smem_q_raw_addr, q_tma, 0, (int)next_token_base, head_idx, 0, qk_raw_full_addr + (next_raw_stage) * 8);
                    tma_4d_gmem2smem(smem_k_raw_addr, k_tma, 0, (int)next_token_base, head_idx, 0, qk_raw_full_addr + (next_raw_stage) * 8);
                }
            }
            current_token_base = next_token_base;
            current_eos = next_eos;
        }
        if (warp == 0) {
            if (elect_sync()) {
                tma_store_5d(ws_qd_tma, 0, gchunk * 16, head_idx, 0, 0, smem_qd_addr);
                tma_store_5d(ws_kd_tma, 0, gchunk * 16, head_idx, 0, 0, smem_kd_addr);
            }
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
        __syncthreads();
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
