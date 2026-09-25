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
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_MAX_OFF 1024
#define SMEM_SMEM_MAX_STAGE_BYTES 512
#define SMEM_SMEM_MAX_STRIDE 512
#define SMEM_SMEM_SUM_OFF 1536
#define SMEM_SMEM_SUM_STAGE_BYTES 512
#define SMEM_SMEM_SUM_STRIDE 512
#define SMEM_SMEM_Q_OFF 2048
#define SMEM_SMEM_Q_STAGE_BYTES 4352
#define SMEM_SMEM_Q_STRIDE 4352
#define SMEM_SMEM_STAT_OFF 6400
#define SMEM_SMEM_STAT_STAGE_BYTES 96
#define SMEM_SMEM_STAT_STRIDE 96
#define SMEM_SMEM_ZONE_OFF 6528
#define SMEM_SMEM_ZONE_STAGE_BYTES 6144
#define SMEM_SMEM_ZONE_STRIDE 6144
#define SMEM_SMEM_TILES_OFF 12672
#define SMEM_SMEM_TILES_STAGE_BYTES 69632
#define SMEM_SMEM_TILES_STRIDE 69632
#define SMEM_SMEM_OPART_OFF 12672
#define SMEM_SMEM_OPART_STAGE_BYTES 69632
#define SMEM_SMEM_OPART_STRIDE 69632
#define SMEM_TOTAL 82304
#define THREADS 256

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


__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}


__device__ __forceinline__ uint32_t mapa_to_rank(uint32_t local_addr, uint32_t rank) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote) : "r"(local_addr), "r"(rank));
    return remote;
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

extern "C" {

__global__ __launch_bounds__(256, 1) __cluster_dims__(4,1,1) void
kernel_cake_msa_nvfp4_decode_ea637f6d9fac2e60b40a(__nv_bfloat16* __restrict__ Q, uint8_t* __restrict__ K, uint8_t* __restrict__ K_scale, uint8_t* __restrict__ V, uint8_t* __restrict__ V_scale, __nv_bfloat16* __restrict__ O, float* __restrict__ msa_lse, int* __restrict__ kv_indices, int* __restrict__ kv_indptr, int* __restrict__ task_kind, int* __restrict__ task_request, int* __restrict__ task_kv_head, int total_q, int seqlen_q, int num_q_heads, int num_kv_heads, float softmax_scale_log2, float output_scale, int msa_max_pages, int k_page_stride, int k_head_stride, int ks_page_stride, int ks_head_stride, int v_page_stride, int v_head_stride, int vs_page_stride, int vs_head_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define part_ready_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* smem_max = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_max_addr = smem + 1024;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 1536);
    const int smem_sum_addr = smem + 1536;
    unsigned int* smem_q = reinterpret_cast<unsigned int*>(smem_raw + 2048);
    const int smem_q_addr = smem + 2048;
    float* smem_stat = reinterpret_cast<float*>(smem_raw + 6400);
    const int smem_stat_addr = smem + 6400;
    float* smem_zone = reinterpret_cast<float*>(smem_raw + 6528);
    const int smem_zone_addr = smem + 6528;
    unsigned int* smem_tiles = reinterpret_cast<unsigned int*>(smem_raw + 12672);
    const int smem_tiles_addr = smem + 12672;
    float* smem_opart = reinterpret_cast<float*>(smem_raw + 12672);
    const int smem_opart_addr = smem + 12672;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // part_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // === Task calls (dependency order) ===
    unsigned int total_items = total_q * num_kv_heads;
    int group_size = num_q_heads / num_kv_heads;
    int warp_0 = warp;
    int lane_1 = lane;
    int tid_2 = tid;
    int rank = cta_rank;
    unsigned int cluster_x = blockIdx.x / 4;
    unsigned int num_clusters_x = gridDim.x / 4;
    int tok0 = warp_0 * 16;
    int slice_addr = smem_tiles_addr + (unsigned int)(warp_0 * 8704);
    int wbase = warp_0 * 2176;
    int r0 = lane_1 / 4;
    int quad = lane_1 % 4;
    int h_e = tid_2 / 16;
    int g_e = tid_2 % 16;
    int owner = h_e / 4;
    int hl = h_e % 4;
    int slot = ((rank > owner) ? rank - 1 : rank);
    int q_addr = smem_q_addr;
    float s_acc[8];
    float o_acc[64];
    unsigned int a_frag[4];
    unsigned int b_frag[4];
    unsigned int v_frag[4];
    unsigned int raw_k[8];
    int sw_k[2];
    unsigned int raw_v[8];
    int sw_v[4];
    float eacc[8];
    float ov[8];
    int item_par = 0;
    #pragma unroll 1
    for (unsigned int work_idx = cluster_x; work_idx < total_items; work_idx += num_clusters_x) {
        int item = (int)work_idx;
        int query = item / num_kv_heads;
        int kv_head = item % num_kv_heads;
        int q_row0 = query * num_q_heads + kv_head * group_size;
        int page_l = 0;
        int valid_l = 0;
        int pg = 0;
        int vl = 0;
        if (lane_1 < 16) {
            int tok_f = 0;
            int batch = query / seqlen_q;
            int query_in_batch = query - batch * seqlen_q;
            int selected_block = task_kind[(kv_head * total_q + query) * 16 + lane_1];
            int kv_len = task_kv_head[batch];
            int valid_cols = 0;
            if (selected_block >= 0) {
                int block_start = selected_block * 128;
                valid_cols = kv_len - block_start;
                if (valid_cols > 128) {
                    valid_cols = 128;
                }
                if (valid_cols < 0) {
                    valid_cols = 0;
                }
                {
                    int query_position = kv_len - seqlen_q + query_in_batch;
                    int causal_cols = query_position - block_start + 1;
                    if (valid_cols > causal_cols) {
                        valid_cols = causal_cols;
                    }
                    if (valid_cols < 0) {
                        valid_cols = 0;
                    }
                }
            }
            int token_base = 0;
            int page_head = 0;
            {
                int physical_page = 0;
                if (selected_block >= 0) {
                    physical_page = kv_indices[batch * msa_max_pages + selected_block];
                    if (physical_page < 0) {
                        valid_cols = 0;
                        physical_page = 0;
                    }
                }
                page_head = physical_page * num_kv_heads + kv_head;
            }
            tok_f = token_base;
            pg = page_head;
            vl = valid_cols;
        }
        page_l = pg;
        valid_l = vl;
        int pick = -1;
        int seen = 0;
        int vs = 0;
        int _shfl_0 = __shfl_sync(0xFFFFFFFF, valid_l, 0);
        vs = _shfl_0;
        if (vs > 0) {
            if (seen == rank) {
                pick = 0;
            }
            seen = seen + 1;
        }
        int _shfl_1 = __shfl_sync(0xFFFFFFFF, valid_l, 1);
        vs = _shfl_1;
        if (vs > 0) {
            if (seen == rank) {
                pick = 1;
            }
            seen = seen + 1;
        }
        int _shfl_2 = __shfl_sync(0xFFFFFFFF, valid_l, 2);
        vs = _shfl_2;
        if (vs > 0) {
            if (seen == rank) {
                pick = 2;
            }
            seen = seen + 1;
        }
        int _shfl_3 = __shfl_sync(0xFFFFFFFF, valid_l, 3);
        vs = _shfl_3;
        if (vs > 0) {
            if (seen == rank) {
                pick = 3;
            }
            seen = seen + 1;
        }
        int _shfl_4 = __shfl_sync(0xFFFFFFFF, valid_l, 4);
        vs = _shfl_4;
        if (vs > 0) {
            if (seen == rank) {
                pick = 4;
            }
            seen = seen + 1;
        }
        int _shfl_5 = __shfl_sync(0xFFFFFFFF, valid_l, 5);
        vs = _shfl_5;
        if (vs > 0) {
            if (seen == rank) {
                pick = 5;
            }
            seen = seen + 1;
        }
        int _shfl_6 = __shfl_sync(0xFFFFFFFF, valid_l, 6);
        vs = _shfl_6;
        if (vs > 0) {
            if (seen == rank) {
                pick = 6;
            }
            seen = seen + 1;
        }
        int _shfl_7 = __shfl_sync(0xFFFFFFFF, valid_l, 7);
        vs = _shfl_7;
        if (vs > 0) {
            if (seen == rank) {
                pick = 7;
            }
            seen = seen + 1;
        }
        int _shfl_8 = __shfl_sync(0xFFFFFFFF, valid_l, 8);
        vs = _shfl_8;
        if (vs > 0) {
            if (seen == rank) {
                pick = 8;
            }
            seen = seen + 1;
        }
        int _shfl_9 = __shfl_sync(0xFFFFFFFF, valid_l, 9);
        vs = _shfl_9;
        if (vs > 0) {
            if (seen == rank) {
                pick = 9;
            }
            seen = seen + 1;
        }
        int _shfl_10 = __shfl_sync(0xFFFFFFFF, valid_l, 10);
        vs = _shfl_10;
        if (vs > 0) {
            if (seen == rank) {
                pick = 10;
            }
            seen = seen + 1;
        }
        int _shfl_11 = __shfl_sync(0xFFFFFFFF, valid_l, 11);
        vs = _shfl_11;
        if (vs > 0) {
            if (seen == rank) {
                pick = 11;
            }
            seen = seen + 1;
        }
        int _shfl_12 = __shfl_sync(0xFFFFFFFF, valid_l, 12);
        vs = _shfl_12;
        if (vs > 0) {
            if (seen == rank) {
                pick = 12;
            }
            seen = seen + 1;
        }
        int _shfl_13 = __shfl_sync(0xFFFFFFFF, valid_l, 13);
        vs = _shfl_13;
        if (vs > 0) {
            if (seen == rank) {
                pick = 13;
            }
            seen = seen + 1;
        }
        int _shfl_14 = __shfl_sync(0xFFFFFFFF, valid_l, 14);
        vs = _shfl_14;
        if (vs > 0) {
            if (seen == rank) {
                pick = 14;
            }
            seen = seen + 1;
        }
        int _shfl_15 = __shfl_sync(0xFFFFFFFF, valid_l, 15);
        vs = _shfl_15;
        if (vs > 0) {
            if (seen == rank) {
                pick = 15;
            }
            seen = seen + 1;
        }
        int my_slot = pick;
        int slot_c = ((my_slot >= 0) ? my_slot : 0);
        int _shfl_16 = __shfl_sync(0xFFFFFFFF, page_l, slot_c);
        int page_c = _shfl_16;
        int _shfl_17 = __shfl_sync(0xFFFFFFFF, valid_l, slot_c);
        int vc = _shfl_17;
        int valid_cols_1 = ((my_slot >= 0) ? vc : 0);
        int vloc = valid_cols_1 - tok0;
        int active = ((vloc > 0) ? 1 : 0);
        int phys = page_c / num_kv_heads;
        int ph_head = page_c - phys * num_kv_heads;
        long long phys64 = (long long)phys;
        long long k_off = phys64 * (long long)k_page_stride + (long long)(ph_head * k_head_stride);
        long long ks_off = phys64 * (long long)ks_page_stride + (long long)(ph_head * ks_head_stride);
        long long v_off = phys64 * (long long)v_page_stride + (long long)(ph_head * v_head_stride);
        long long vs_off = phys64 * (long long)vs_page_stride + (long long)(ph_head * vs_head_stride);
        if (active != 0) {
            {
                const uint4* _ivptr_0 = reinterpret_cast<const uint4*>(K + k_off + (long long)(((tok0 + lane_1 % 8) * 4 + lane_1 / 8) * 16));
                uint4 _ivld_0;
                _ivld_0 = *_ivptr_0;
                raw_k[0 + 0] = _ivld_0.x;
                raw_k[0 + 1] = _ivld_0.y;
                raw_k[0 + 2] = _ivld_0.z;
                raw_k[0 + 3] = _ivld_0.w;
            }
            {
                sw_k[0] = *reinterpret_cast<const int*>(K_scale + ks_off + (long long)((tok0 + lane_1 % 8) * 8 + lane_1 / 8 / 2 * 4));
            }
            {
                const uint4* _ivptr_1 = reinterpret_cast<const uint4*>(K + k_off + (long long)(((tok0 + (8 + lane_1 % 8)) * 4 + lane_1 / 8) * 16));
                uint4 _ivld_1;
                _ivld_1 = *_ivptr_1;
                raw_k[4 + 0] = _ivld_1.x;
                raw_k[4 + 1] = _ivld_1.y;
                raw_k[4 + 2] = _ivld_1.z;
                raw_k[4 + 3] = _ivld_1.w;
            }
            {
                sw_k[1] = *reinterpret_cast<const int*>(K_scale + ks_off + (long long)((tok0 + (8 + lane_1 % 8)) * 8 + lane_1 / 8 / 2 * 4));
            }
            {
                const uint4* _ivptr_2 = reinterpret_cast<const uint4*>(V + v_off + (long long)(((tok0 + lane_1 % 8) * 4 + lane_1 / 8) * 16));
                uint4 _ivld_2;
                _ivld_2 = *_ivptr_2;
                raw_v[0 + 0] = _ivld_2.x;
                raw_v[0 + 1] = _ivld_2.y;
                raw_v[0 + 2] = _ivld_2.z;
                raw_v[0 + 3] = _ivld_2.w;
            }
            {
                const int2* _ivptr_3 = reinterpret_cast<const int2*>(V_scale + vs_off + (long long)(((tok0 + lane_1 % 8) / 4 * 4 + lane_1 / 8) * 8));
                int2 _ivld_3;
                _ivld_3 = *_ivptr_3;
                sw_v[0 + 0] = _ivld_3.x;
                sw_v[0 + 1] = _ivld_3.y;
            }
            {
                const uint4* _ivptr_4 = reinterpret_cast<const uint4*>(V + v_off + (long long)(((tok0 + (8 + lane_1 % 8)) * 4 + lane_1 / 8) * 16));
                uint4 _ivld_4;
                _ivld_4 = *_ivptr_4;
                raw_v[4 + 0] = _ivld_4.x;
                raw_v[4 + 1] = _ivld_4.y;
                raw_v[4 + 2] = _ivld_4.z;
                raw_v[4 + 3] = _ivld_4.w;
            }
            {
                const int2* _ivptr_5 = reinterpret_cast<const int2*>(V_scale + vs_off + (long long)(((tok0 + (8 + lane_1 % 8)) / 4 * 4 + lane_1 / 8) * 8));
                int2 _ivld_5;
                _ivld_5 = *_ivptr_5;
                sw_v[2 + 0] = _ivld_5.x;
                sw_v[2 + 1] = _ivld_5.y;
            }
        }
        int qrow = ((group_size > tid_2 / 16) ? tid_2 / 16 : group_size - 1);
        int _vec_load_0[4];
        {
            const int4* _ivptr_6 = reinterpret_cast<const int4*>(Q + (q_row0 + qrow) * 128 + tid_2 % 16 * 8);
            int4 _ivld_6;
            _ivld_6 = *_ivptr_6;
            _vec_load_0[0 + 0] = _ivld_6.x;
            _vec_load_0[0 + 1] = _ivld_6.y;
            _vec_load_0[0 + 2] = _ivld_6.z;
            _vec_load_0[0 + 3] = _ivld_6.w;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(q_addr + tid_2 / 16 * 272 + tid_2 % 16 * 16), "r"(*reinterpret_cast<uint32_t*>(&_vec_load_0[0])), "r"(*reinterpret_cast<uint32_t*>(&_vec_load_0[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_vec_load_0[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_vec_load_0[(0) + 3])));
        __syncthreads();
        s_acc[0] = 0.0f;
        s_acc[1] = 0.0f;
        s_acc[2] = 0.0f;
        s_acc[3] = 0.0f;
        s_acc[4] = 0.0f;
        s_acc[5] = 0.0f;
        s_acc[6] = 0.0f;
        s_acc[7] = 0.0f;
        float m0 = -CAKE_INF;
        float m1 = -CAKE_INF;
        float l0 = 0.0f;
        float l1 = 0.0f;
        o_acc[0] = 0.0f;
        o_acc[1] = 0.0f;
        o_acc[2] = 0.0f;
        o_acc[3] = 0.0f;
        o_acc[4] = 0.0f;
        o_acc[5] = 0.0f;
        o_acc[6] = 0.0f;
        o_acc[7] = 0.0f;
        o_acc[8] = 0.0f;
        o_acc[9] = 0.0f;
        o_acc[10] = 0.0f;
        o_acc[11] = 0.0f;
        o_acc[12] = 0.0f;
        o_acc[13] = 0.0f;
        o_acc[14] = 0.0f;
        o_acc[15] = 0.0f;
        o_acc[16] = 0.0f;
        o_acc[17] = 0.0f;
        o_acc[18] = 0.0f;
        o_acc[19] = 0.0f;
        o_acc[20] = 0.0f;
        o_acc[21] = 0.0f;
        o_acc[22] = 0.0f;
        o_acc[23] = 0.0f;
        o_acc[24] = 0.0f;
        o_acc[25] = 0.0f;
        o_acc[26] = 0.0f;
        o_acc[27] = 0.0f;
        o_acc[28] = 0.0f;
        o_acc[29] = 0.0f;
        o_acc[30] = 0.0f;
        o_acc[31] = 0.0f;
        o_acc[32] = 0.0f;
        o_acc[33] = 0.0f;
        o_acc[34] = 0.0f;
        o_acc[35] = 0.0f;
        o_acc[36] = 0.0f;
        o_acc[37] = 0.0f;
        o_acc[38] = 0.0f;
        o_acc[39] = 0.0f;
        o_acc[40] = 0.0f;
        o_acc[41] = 0.0f;
        o_acc[42] = 0.0f;
        o_acc[43] = 0.0f;
        o_acc[44] = 0.0f;
        o_acc[45] = 0.0f;
        o_acc[46] = 0.0f;
        o_acc[47] = 0.0f;
        o_acc[48] = 0.0f;
        o_acc[49] = 0.0f;
        o_acc[50] = 0.0f;
        o_acc[51] = 0.0f;
        o_acc[52] = 0.0f;
        o_acc[53] = 0.0f;
        o_acc[54] = 0.0f;
        o_acc[55] = 0.0f;
        o_acc[56] = 0.0f;
        o_acc[57] = 0.0f;
        o_acc[58] = 0.0f;
        o_acc[59] = 0.0f;
        o_acc[60] = 0.0f;
        o_acc[61] = 0.0f;
        o_acc[62] = 0.0f;
        o_acc[63] = 0.0f;
        if (active != 0) {
            uint32_t _fp4_dequant_block16_bf16_0[8];
            {
                uint32_t _scale_byte = ((uint32_t)((uint8_t)(sw_k[0] >> lane_1 / 8 % 2 * 16 & 255))) & 0xFFu;
                uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                uint32_t _scale_bf16x2;
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_bf16x2) : "h"(_scale_e4m3x2));
                #else
                uint32_t _f16x2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                float _f0;
                float _f1;
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_f1), "f"(_f0));
                #endif
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[0])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[0]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[0])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[1]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[0])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[2]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[0])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[3]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[1])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[4]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[1])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[5]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[1])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[6]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[1])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[7]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + lane_1 % 8 * 272 + lane_1 / 8 * 64), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[0])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[(0) + 3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + lane_1 % 8 * 272 + lane_1 / 8 * 64 + 16), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[4])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_0[(4) + 3])));
            uint32_t _fp4_dequant_block16_bf16_1[8];
            {
                uint32_t _scale_byte = ((uint32_t)((uint8_t)(sw_k[0] >> lane_1 / 8 % 2 * 16 + 8 & 255))) & 0xFFu;
                uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                uint32_t _scale_bf16x2;
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_bf16x2) : "h"(_scale_e4m3x2));
                #else
                uint32_t _f16x2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                float _f0;
                float _f1;
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_f1), "f"(_f0));
                #endif
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[2])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[0]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[2])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[1]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[2])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[2]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[2])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[3]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[3])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[4]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[3])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[5]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[3])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[6]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[3])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_1[7]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + lane_1 % 8 * 272 + lane_1 / 8 * 64 + 32), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[0])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[(0) + 3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + lane_1 % 8 * 272 + lane_1 / 8 * 64 + 48), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[4])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_1[(4) + 3])));
            uint32_t _fp4_dequant_block16_bf16_2[8];
            {
                uint32_t _scale_byte = ((uint32_t)((uint8_t)(sw_k[1] >> lane_1 / 8 % 2 * 16 & 255))) & 0xFFu;
                uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                uint32_t _scale_bf16x2;
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_bf16x2) : "h"(_scale_e4m3x2));
                #else
                uint32_t _f16x2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                float _f0;
                float _f1;
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_f1), "f"(_f0));
                #endif
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[4])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[0]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[4])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[1]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[4])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[2]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[4])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[3]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[5])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[4]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[5])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[5]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[5])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[6]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[5])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_2[7]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + (8 + lane_1 % 8) * 272 + lane_1 / 8 * 64), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[0])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[(0) + 3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + (8 + lane_1 % 8) * 272 + lane_1 / 8 * 64 + 16), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[4])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_2[(4) + 3])));
            uint32_t _fp4_dequant_block16_bf16_3[8];
            {
                uint32_t _scale_byte = ((uint32_t)((uint8_t)(sw_k[1] >> lane_1 / 8 % 2 * 16 + 8 & 255))) & 0xFFu;
                uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                uint32_t _scale_bf16x2;
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_bf16x2) : "h"(_scale_e4m3x2));
                #else
                uint32_t _f16x2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                float _f0;
                float _f1;
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_f1), "f"(_f0));
                #endif
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[6])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[0]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[6])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[1]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[6])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[2]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[6])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[3]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[7])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[4]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[7])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[5]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[7])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[6]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_k[7])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_3[7]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + (8 + lane_1 % 8) * 272 + lane_1 / 8 * 64 + 32), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[0])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[(0) + 3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + (8 + lane_1 % 8) * 272 + lane_1 / 8 * 64 + 48), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[4])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_3[(4) + 3])));
            __syncwarp();
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(q_addr + lane_1 % 16 * 272 + lane_1 / 16 * 16)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 16 * 8) * 272 + lane_1 / 8 % 2 * 16)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(q_addr + lane_1 % 16 * 272 + 32 + lane_1 / 16 * 16)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 16 * 8) * 272 + 32 + lane_1 / 8 % 2 * 16)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(q_addr + lane_1 % 16 * 272 + 64 + lane_1 / 16 * 16)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 16 * 8) * 272 + 64 + lane_1 / 8 % 2 * 16)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(q_addr + lane_1 % 16 * 272 + 96 + lane_1 / 16 * 16)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 16 * 8) * 272 + 96 + lane_1 / 8 % 2 * 16)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(q_addr + lane_1 % 16 * 272 + 128 + lane_1 / 16 * 16)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 16 * 8) * 272 + 128 + lane_1 / 8 % 2 * 16)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(q_addr + lane_1 % 16 * 272 + 160 + lane_1 / 16 * 16)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 16 * 8) * 272 + 160 + lane_1 / 8 % 2 * 16)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(q_addr + lane_1 % 16 * 272 + 192 + lane_1 / 16 * 16)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 16 * 8) * 272 + 192 + lane_1 / 8 % 2 * 16)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(q_addr + lane_1 % 16 * 272 + 224 + lane_1 / 16 * 16)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 16 * 8) * 272 + 224 + lane_1 / 8 % 2 * 16)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
            float fm0 = -CAKE_INF;
            float fm1 = -CAKE_INF;
            s_acc[0] = ((vloc > quad * 2) ? s_acc[0] : -CAKE_INF);
            float _fmax_0 = fmaxf(fm0, s_acc[0]);
            fm0 = _fmax_0;
            s_acc[1] = ((vloc > quad * 2 + 1) ? s_acc[1] : -CAKE_INF);
            float _fmax_1 = fmaxf(fm0, s_acc[1]);
            fm0 = _fmax_1;
            s_acc[2] = ((vloc > quad * 2) ? s_acc[2] : -CAKE_INF);
            float _fmax_2 = fmaxf(fm1, s_acc[2]);
            fm1 = _fmax_2;
            s_acc[3] = ((vloc > quad * 2 + 1) ? s_acc[3] : -CAKE_INF);
            float _fmax_3 = fmaxf(fm1, s_acc[3]);
            fm1 = _fmax_3;
            s_acc[4] = ((vloc > 8 + quad * 2) ? s_acc[4] : -CAKE_INF);
            float _fmax_4 = fmaxf(fm0, s_acc[4]);
            fm0 = _fmax_4;
            s_acc[5] = ((vloc > 8 + quad * 2 + 1) ? s_acc[5] : -CAKE_INF);
            float _fmax_5 = fmaxf(fm0, s_acc[5]);
            fm0 = _fmax_5;
            s_acc[6] = ((vloc > 8 + quad * 2) ? s_acc[6] : -CAKE_INF);
            float _fmax_6 = fmaxf(fm1, s_acc[6]);
            fm1 = _fmax_6;
            s_acc[7] = ((vloc > 8 + quad * 2 + 1) ? s_acc[7] : -CAKE_INF);
            float _fmax_7 = fmaxf(fm1, s_acc[7]);
            fm1 = _fmax_7;
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, fm0, 1);
            float _fmax_8 = fmaxf(fm0, _shfl_xor_0);
            fm0 = _fmax_8;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, fm0, 2);
            float _fmax_9 = fmaxf(fm0, _shfl_xor_1);
            fm0 = _fmax_9;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, fm1, 1);
            float _fmax_10 = fmaxf(fm1, _shfl_xor_2);
            fm1 = _fmax_10;
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, fm1, 2);
            float _fmax_11 = fmaxf(fm1, _shfl_xor_3);
            fm1 = _fmax_11;
            m0 = fm0;
            m1 = fm1;
            float neg0 = ((m0 == -CAKE_INF) ? 0.0f : -(m0 * softmax_scale_log2));
            float neg1 = ((m1 == -CAKE_INF) ? 0.0f : -(m1 * softmax_scale_log2));
            float fl0 = 0.0f;
            float fl1 = 0.0f;
            float _fma_0 = __fmaf_rn(s_acc[0], softmax_scale_log2, neg0);
            float _exp2_0 = approx_exp2(_fma_0);
            s_acc[0] = _exp2_0;
            fl0 = fl0 + s_acc[0];
            float _fma_1 = __fmaf_rn(s_acc[1], softmax_scale_log2, neg0);
            float _exp2_1 = approx_exp2(_fma_1);
            s_acc[1] = _exp2_1;
            fl0 = fl0 + s_acc[1];
            float _fma_2 = __fmaf_rn(s_acc[2], softmax_scale_log2, neg1);
            float _exp2_2 = approx_exp2(_fma_2);
            s_acc[2] = _exp2_2;
            fl1 = fl1 + s_acc[2];
            float _fma_3 = __fmaf_rn(s_acc[3], softmax_scale_log2, neg1);
            float _exp2_3 = approx_exp2(_fma_3);
            s_acc[3] = _exp2_3;
            fl1 = fl1 + s_acc[3];
            float _fma_4 = __fmaf_rn(s_acc[4], softmax_scale_log2, neg0);
            float _exp2_4 = approx_exp2(_fma_4);
            s_acc[4] = _exp2_4;
            fl0 = fl0 + s_acc[4];
            float _fma_5 = __fmaf_rn(s_acc[5], softmax_scale_log2, neg0);
            float _exp2_5 = approx_exp2(_fma_5);
            s_acc[5] = _exp2_5;
            fl0 = fl0 + s_acc[5];
            float _fma_6 = __fmaf_rn(s_acc[6], softmax_scale_log2, neg1);
            float _exp2_6 = approx_exp2(_fma_6);
            s_acc[6] = _exp2_6;
            fl1 = fl1 + s_acc[6];
            float _fma_7 = __fmaf_rn(s_acc[7], softmax_scale_log2, neg1);
            float _exp2_7 = approx_exp2(_fma_7);
            s_acc[7] = _exp2_7;
            fl1 = fl1 + s_acc[7];
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, fl0, 1);
            fl0 = fl0 + _shfl_xor_4;
            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, fl0, 2);
            fl0 = fl0 + _shfl_xor_5;
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, fl1, 1);
            fl1 = fl1 + _shfl_xor_6;
            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, fl1, 2);
            fl1 = fl1 + _shfl_xor_7;
            l0 = fl0;
            l1 = fl1;
            __syncwarp();
            uint32_t _fp4_dequant_block16_bf16_4[8];
            {
                uint32_t _scale_byte = ((uint32_t)((uint8_t)(sw_v[0] >> (tok0 + lane_1 % 8) % 4 * 8 & 255))) & 0xFFu;
                uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                uint32_t _scale_bf16x2;
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_bf16x2) : "h"(_scale_e4m3x2));
                #else
                uint32_t _f16x2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                float _f0;
                float _f1;
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_f1), "f"(_f0));
                #endif
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[0])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_4[0]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[0])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_4[1]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[0])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_4[2]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[0])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_4[3]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[1])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_4[4]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[1])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_4[5]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[1])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_4[6]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[1])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_4[7]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + lane_1 % 8 * 272 + lane_1 / 8 * 64), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_4[0])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_4[(0) + 3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + lane_1 % 8 * 272 + lane_1 / 8 * 64 + 16), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_4[4])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_4[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_4[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_4[(4) + 3])));
            uint32_t _fp4_dequant_block16_bf16_5[8];
            {
                uint32_t _scale_byte = ((uint32_t)((uint8_t)(sw_v[1] >> (tok0 + lane_1 % 8) % 4 * 8 & 255))) & 0xFFu;
                uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                uint32_t _scale_bf16x2;
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_bf16x2) : "h"(_scale_e4m3x2));
                #else
                uint32_t _f16x2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                float _f0;
                float _f1;
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_f1), "f"(_f0));
                #endif
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[2])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_5[0]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[2])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_5[1]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[2])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_5[2]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[2])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_5[3]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[3])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_5[4]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[3])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_5[5]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[3])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_5[6]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[3])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_5[7]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + lane_1 % 8 * 272 + lane_1 / 8 * 64 + 32), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_5[0])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_5[(0) + 3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + lane_1 % 8 * 272 + lane_1 / 8 * 64 + 48), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_5[4])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_5[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_5[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_5[(4) + 3])));
            uint32_t _fp4_dequant_block16_bf16_6[8];
            {
                uint32_t _scale_byte = ((uint32_t)((uint8_t)(sw_v[2] >> (tok0 + (8 + lane_1 % 8)) % 4 * 8 & 255))) & 0xFFu;
                uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                uint32_t _scale_bf16x2;
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_bf16x2) : "h"(_scale_e4m3x2));
                #else
                uint32_t _f16x2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                float _f0;
                float _f1;
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_f1), "f"(_f0));
                #endif
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[4])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_6[0]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[4])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_6[1]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[4])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_6[2]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[4])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_6[3]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[5])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_6[4]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[5])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_6[5]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[5])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_6[6]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[5])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_6[7]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + (8 + lane_1 % 8) * 272 + lane_1 / 8 * 64), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_6[0])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_6[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_6[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_6[(0) + 3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + (8 + lane_1 % 8) * 272 + lane_1 / 8 * 64 + 16), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_6[4])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_6[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_6[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_6[(4) + 3])));
            uint32_t _fp4_dequant_block16_bf16_7[8];
            {
                uint32_t _scale_byte = ((uint32_t)((uint8_t)(sw_v[3] >> (tok0 + (8 + lane_1 % 8)) % 4 * 8 & 255))) & 0xFFu;
                uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                uint32_t _scale_bf16x2;
                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_bf16x2) : "h"(_scale_e4m3x2));
                #else
                uint32_t _f16x2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                float _f0;
                float _f1;
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_f1), "f"(_f0));
                #endif
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[6])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_7[0]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[6])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_7[1]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[6])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_7[2]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[6])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_7[3]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[7])) >> 0))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_7[4]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[7])) >> 8))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_7[5]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[7])) >> 16))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_7[6]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(raw_v[7])) >> 24))) & 0xFFu);
                    uint32_t _fp4_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_7[7]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + (8 + lane_1 % 8) * 272 + lane_1 / 8 * 64 + 32), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_7[0])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_7[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_7[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_7[(0) + 3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(slice_addr + (8 + lane_1 % 8) * 272 + lane_1 / 8 * 64 + 48), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_7[4])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_7[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_7[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_fp4_dequant_block16_bf16_7[(4) + 3])));
            __syncwarp();
            #pragma unroll
            for (int _lip = 0; _lip < 1; _lip++) {
                __nv_bfloat162 _lip_bf2 = __float22bfloat162_rn(make_float2(s_acc[0 + _lip*2], s_acc[0 + _lip*2 + 1]));
                uint32_t _lip_word = *(uint32_t*)&_lip_bf2;
                s_acc[0 + _lip*2] = __uint_as_float(_lip_word);
            }
            #pragma unroll
            for (int _lip = 0; _lip < 1; _lip++) {
                __nv_bfloat162 _lip_bf2 = __float22bfloat162_rn(make_float2(s_acc[2 + _lip*2], s_acc[2 + _lip*2 + 1]));
                uint32_t _lip_word = *(uint32_t*)&_lip_bf2;
                s_acc[2 + _lip*2] = __uint_as_float(_lip_word);
            }
            #pragma unroll
            for (int _lip = 0; _lip < 1; _lip++) {
                __nv_bfloat162 _lip_bf2 = __float22bfloat162_rn(make_float2(s_acc[4 + _lip*2], s_acc[4 + _lip*2 + 1]));
                uint32_t _lip_word = *(uint32_t*)&_lip_bf2;
                s_acc[4 + _lip*2] = __uint_as_float(_lip_word);
            }
            #pragma unroll
            for (int _lip = 0; _lip < 1; _lip++) {
                __nv_bfloat162 _lip_bf2 = __float22bfloat162_rn(make_float2(s_acc[6 + _lip*2], s_acc[6 + _lip*2 + 1]));
                uint32_t _lip_word = *(uint32_t*)&_lip_bf2;
                s_acc[6 + _lip*2] = __uint_as_float(_lip_word);
            }
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 8 % 2 * 8) * 272 + lane_1 / 16 * 8 * 2)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 8 % 2 * 8) * 272 + (16 + lane_1 / 16 * 8) * 2)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 8 % 2 * 8) * 272 + (32 + lane_1 / 16 * 8) * 2)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 8 % 2 * 8) * 272 + (48 + lane_1 / 16 * 8) * 2)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 8 % 2 * 8) * 272 + (64 + lane_1 / 16 * 8) * 2)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 8 % 2 * 8) * 272 + (80 + lane_1 / 16 * 8) * 2)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 8 % 2 * 8) * 272 + (96 + lane_1 / 16 * 8) * 2)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(slice_addr + (lane_1 % 8 + lane_1 / 8 % 2 * 8) * 272 + (112 + lane_1 / 16 * 8) * 2)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(*reinterpret_cast<uint32_t*>(&s_acc[0])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[2])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[4])), "r"(*reinterpret_cast<uint32_t*>(&s_acc[6])), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        }
        __syncwarp();
        smem_opart[wbase + r0 * 136 + quad * 2] = o_acc[0];
        smem_opart[wbase + r0 * 136 + quad * 2 + 1] = o_acc[1];
        smem_opart[wbase + r0 * 136 + quad * 2 + 1088] = o_acc[2];
        smem_opart[wbase + r0 * 136 + quad * 2 + 1088 + 1] = o_acc[3];
        smem_opart[wbase + r0 * 136 + 8 + quad * 2] = o_acc[4];
        smem_opart[wbase + r0 * 136 + 8 + quad * 2 + 1] = o_acc[5];
        smem_opart[wbase + r0 * 136 + 8 + quad * 2 + 1088] = o_acc[6];
        smem_opart[wbase + r0 * 136 + 8 + quad * 2 + 1088 + 1] = o_acc[7];
        smem_opart[wbase + r0 * 136 + 16 + quad * 2] = o_acc[8];
        smem_opart[wbase + r0 * 136 + 16 + quad * 2 + 1] = o_acc[9];
        smem_opart[wbase + r0 * 136 + 16 + quad * 2 + 1088] = o_acc[10];
        smem_opart[wbase + r0 * 136 + 16 + quad * 2 + 1088 + 1] = o_acc[11];
        smem_opart[wbase + r0 * 136 + 24 + quad * 2] = o_acc[12];
        smem_opart[wbase + r0 * 136 + 24 + quad * 2 + 1] = o_acc[13];
        smem_opart[wbase + r0 * 136 + 24 + quad * 2 + 1088] = o_acc[14];
        smem_opart[wbase + r0 * 136 + 24 + quad * 2 + 1088 + 1] = o_acc[15];
        smem_opart[wbase + r0 * 136 + 32 + quad * 2] = o_acc[16];
        smem_opart[wbase + r0 * 136 + 32 + quad * 2 + 1] = o_acc[17];
        smem_opart[wbase + r0 * 136 + 32 + quad * 2 + 1088] = o_acc[18];
        smem_opart[wbase + r0 * 136 + 32 + quad * 2 + 1088 + 1] = o_acc[19];
        smem_opart[wbase + r0 * 136 + 40 + quad * 2] = o_acc[20];
        smem_opart[wbase + r0 * 136 + 40 + quad * 2 + 1] = o_acc[21];
        smem_opart[wbase + r0 * 136 + 40 + quad * 2 + 1088] = o_acc[22];
        smem_opart[wbase + r0 * 136 + 40 + quad * 2 + 1088 + 1] = o_acc[23];
        smem_opart[wbase + r0 * 136 + 48 + quad * 2] = o_acc[24];
        smem_opart[wbase + r0 * 136 + 48 + quad * 2 + 1] = o_acc[25];
        smem_opart[wbase + r0 * 136 + 48 + quad * 2 + 1088] = o_acc[26];
        smem_opart[wbase + r0 * 136 + 48 + quad * 2 + 1088 + 1] = o_acc[27];
        smem_opart[wbase + r0 * 136 + 56 + quad * 2] = o_acc[28];
        smem_opart[wbase + r0 * 136 + 56 + quad * 2 + 1] = o_acc[29];
        smem_opart[wbase + r0 * 136 + 56 + quad * 2 + 1088] = o_acc[30];
        smem_opart[wbase + r0 * 136 + 56 + quad * 2 + 1088 + 1] = o_acc[31];
        smem_opart[wbase + r0 * 136 + 64 + quad * 2] = o_acc[32];
        smem_opart[wbase + r0 * 136 + 64 + quad * 2 + 1] = o_acc[33];
        smem_opart[wbase + r0 * 136 + 64 + quad * 2 + 1088] = o_acc[34];
        smem_opart[wbase + r0 * 136 + 64 + quad * 2 + 1088 + 1] = o_acc[35];
        smem_opart[wbase + r0 * 136 + 72 + quad * 2] = o_acc[36];
        smem_opart[wbase + r0 * 136 + 72 + quad * 2 + 1] = o_acc[37];
        smem_opart[wbase + r0 * 136 + 72 + quad * 2 + 1088] = o_acc[38];
        smem_opart[wbase + r0 * 136 + 72 + quad * 2 + 1088 + 1] = o_acc[39];
        smem_opart[wbase + r0 * 136 + 80 + quad * 2] = o_acc[40];
        smem_opart[wbase + r0 * 136 + 80 + quad * 2 + 1] = o_acc[41];
        smem_opart[wbase + r0 * 136 + 80 + quad * 2 + 1088] = o_acc[42];
        smem_opart[wbase + r0 * 136 + 80 + quad * 2 + 1088 + 1] = o_acc[43];
        smem_opart[wbase + r0 * 136 + 88 + quad * 2] = o_acc[44];
        smem_opart[wbase + r0 * 136 + 88 + quad * 2 + 1] = o_acc[45];
        smem_opart[wbase + r0 * 136 + 88 + quad * 2 + 1088] = o_acc[46];
        smem_opart[wbase + r0 * 136 + 88 + quad * 2 + 1088 + 1] = o_acc[47];
        smem_opart[wbase + r0 * 136 + 96 + quad * 2] = o_acc[48];
        smem_opart[wbase + r0 * 136 + 96 + quad * 2 + 1] = o_acc[49];
        smem_opart[wbase + r0 * 136 + 96 + quad * 2 + 1088] = o_acc[50];
        smem_opart[wbase + r0 * 136 + 96 + quad * 2 + 1088 + 1] = o_acc[51];
        smem_opart[wbase + r0 * 136 + 104 + quad * 2] = o_acc[52];
        smem_opart[wbase + r0 * 136 + 104 + quad * 2 + 1] = o_acc[53];
        smem_opart[wbase + r0 * 136 + 104 + quad * 2 + 1088] = o_acc[54];
        smem_opart[wbase + r0 * 136 + 104 + quad * 2 + 1088 + 1] = o_acc[55];
        smem_opart[wbase + r0 * 136 + 112 + quad * 2] = o_acc[56];
        smem_opart[wbase + r0 * 136 + 112 + quad * 2 + 1] = o_acc[57];
        smem_opart[wbase + r0 * 136 + 112 + quad * 2 + 1088] = o_acc[58];
        smem_opart[wbase + r0 * 136 + 112 + quad * 2 + 1088 + 1] = o_acc[59];
        smem_opart[wbase + r0 * 136 + 120 + quad * 2] = o_acc[60];
        smem_opart[wbase + r0 * 136 + 120 + quad * 2 + 1] = o_acc[61];
        smem_opart[wbase + r0 * 136 + 120 + quad * 2 + 1088] = o_acc[62];
        smem_opart[wbase + r0 * 136 + 120 + quad * 2 + 1088 + 1] = o_acc[63];
        if (quad == 0) {
            smem_max[warp_0 * 16 + r0] = m0;
            smem_max[warp_0 * 16 + r0 + 8] = m1;
            smem_sum[warp_0 * 16 + r0] = l0;
            smem_sum[warp_0 * 16 + r0 + 8] = l1;
        }
        if (tid_2 == 0) {
            mbarrier_arrive_expect_tx(part_ready_addr, 6240);
        }
        __syncthreads();
        float hm = 0.0f;
        float hs = 0.0f;
        float wm[8];
        float ws[8];
        float ehm = -CAKE_INF;
        wm[0] = smem_max[h_e];
        ws[0] = smem_sum[h_e];
        float _fmax_12 = fmaxf(ehm, wm[0]);
        ehm = _fmax_12;
        wm[1] = smem_max[16 + h_e];
        ws[1] = smem_sum[16 + h_e];
        float _fmax_13 = fmaxf(ehm, wm[1]);
        ehm = _fmax_13;
        wm[2] = smem_max[32 + h_e];
        ws[2] = smem_sum[32 + h_e];
        float _fmax_14 = fmaxf(ehm, wm[2]);
        ehm = _fmax_14;
        wm[3] = smem_max[48 + h_e];
        ws[3] = smem_sum[48 + h_e];
        float _fmax_15 = fmaxf(ehm, wm[3]);
        ehm = _fmax_15;
        wm[4] = smem_max[64 + h_e];
        ws[4] = smem_sum[64 + h_e];
        float _fmax_16 = fmaxf(ehm, wm[4]);
        ehm = _fmax_16;
        wm[5] = smem_max[80 + h_e];
        ws[5] = smem_sum[80 + h_e];
        float _fmax_17 = fmaxf(ehm, wm[5]);
        ehm = _fmax_17;
        wm[6] = smem_max[96 + h_e];
        ws[6] = smem_sum[96 + h_e];
        float _fmax_18 = fmaxf(ehm, wm[6]);
        ehm = _fmax_18;
        wm[7] = smem_max[112 + h_e];
        ws[7] = smem_sum[112 + h_e];
        float _fmax_19 = fmaxf(ehm, wm[7]);
        ehm = _fmax_19;
        float ehs = 0.0f;
        float wgt = 0.0f;
        unsigned int ew[8];
        float eb = 0.0f;
        eacc[0] = 0.0f;
        eacc[1] = 0.0f;
        eacc[2] = 0.0f;
        eacc[3] = 0.0f;
        eacc[4] = 0.0f;
        eacc[5] = 0.0f;
        eacc[6] = 0.0f;
        eacc[7] = 0.0f;
        float _exp2_8 = approx_exp2((wm[0] - ehm) * softmax_scale_log2);
        wgt = ((wm[0] > -CAKE_INF) ? _exp2_8 : 0.0f);
        float _fma_8 = __fmaf_rn(wgt, ws[0], ehs);
        ehs = _fma_8;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[0])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[4])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 16));
        eb = reinterpret_cast<float*>(&ew[0])[0];
        float _fma_9 = __fmaf_rn(wgt, eb, eacc[0]);
        eacc[0] = _fma_9;
        eb = reinterpret_cast<float*>(&ew[1])[0];
        float _fma_10 = __fmaf_rn(wgt, eb, eacc[1]);
        eacc[1] = _fma_10;
        eb = reinterpret_cast<float*>(&ew[2])[0];
        float _fma_11 = __fmaf_rn(wgt, eb, eacc[2]);
        eacc[2] = _fma_11;
        eb = reinterpret_cast<float*>(&ew[3])[0];
        float _fma_12 = __fmaf_rn(wgt, eb, eacc[3]);
        eacc[3] = _fma_12;
        eb = reinterpret_cast<float*>(&ew[4])[0];
        float _fma_13 = __fmaf_rn(wgt, eb, eacc[4]);
        eacc[4] = _fma_13;
        eb = reinterpret_cast<float*>(&ew[5])[0];
        float _fma_14 = __fmaf_rn(wgt, eb, eacc[5]);
        eacc[5] = _fma_14;
        eb = reinterpret_cast<float*>(&ew[6])[0];
        float _fma_15 = __fmaf_rn(wgt, eb, eacc[6]);
        eacc[6] = _fma_15;
        eb = reinterpret_cast<float*>(&ew[7])[0];
        float _fma_16 = __fmaf_rn(wgt, eb, eacc[7]);
        eacc[7] = _fma_16;
        float _exp2_9 = approx_exp2((wm[1] - ehm) * softmax_scale_log2);
        wgt = ((wm[1] > -CAKE_INF) ? _exp2_9 : 0.0f);
        float _fma_17 = __fmaf_rn(wgt, ws[1], ehs);
        ehs = _fma_17;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[0])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 8704));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[4])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 8704 + 16));
        eb = reinterpret_cast<float*>(&ew[0])[0];
        float _fma_18 = __fmaf_rn(wgt, eb, eacc[0]);
        eacc[0] = _fma_18;
        eb = reinterpret_cast<float*>(&ew[1])[0];
        float _fma_19 = __fmaf_rn(wgt, eb, eacc[1]);
        eacc[1] = _fma_19;
        eb = reinterpret_cast<float*>(&ew[2])[0];
        float _fma_20 = __fmaf_rn(wgt, eb, eacc[2]);
        eacc[2] = _fma_20;
        eb = reinterpret_cast<float*>(&ew[3])[0];
        float _fma_21 = __fmaf_rn(wgt, eb, eacc[3]);
        eacc[3] = _fma_21;
        eb = reinterpret_cast<float*>(&ew[4])[0];
        float _fma_22 = __fmaf_rn(wgt, eb, eacc[4]);
        eacc[4] = _fma_22;
        eb = reinterpret_cast<float*>(&ew[5])[0];
        float _fma_23 = __fmaf_rn(wgt, eb, eacc[5]);
        eacc[5] = _fma_23;
        eb = reinterpret_cast<float*>(&ew[6])[0];
        float _fma_24 = __fmaf_rn(wgt, eb, eacc[6]);
        eacc[6] = _fma_24;
        eb = reinterpret_cast<float*>(&ew[7])[0];
        float _fma_25 = __fmaf_rn(wgt, eb, eacc[7]);
        eacc[7] = _fma_25;
        float _exp2_10 = approx_exp2((wm[2] - ehm) * softmax_scale_log2);
        wgt = ((wm[2] > -CAKE_INF) ? _exp2_10 : 0.0f);
        float _fma_26 = __fmaf_rn(wgt, ws[2], ehs);
        ehs = _fma_26;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[0])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 17408));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[4])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 17408 + 16));
        eb = reinterpret_cast<float*>(&ew[0])[0];
        float _fma_27 = __fmaf_rn(wgt, eb, eacc[0]);
        eacc[0] = _fma_27;
        eb = reinterpret_cast<float*>(&ew[1])[0];
        float _fma_28 = __fmaf_rn(wgt, eb, eacc[1]);
        eacc[1] = _fma_28;
        eb = reinterpret_cast<float*>(&ew[2])[0];
        float _fma_29 = __fmaf_rn(wgt, eb, eacc[2]);
        eacc[2] = _fma_29;
        eb = reinterpret_cast<float*>(&ew[3])[0];
        float _fma_30 = __fmaf_rn(wgt, eb, eacc[3]);
        eacc[3] = _fma_30;
        eb = reinterpret_cast<float*>(&ew[4])[0];
        float _fma_31 = __fmaf_rn(wgt, eb, eacc[4]);
        eacc[4] = _fma_31;
        eb = reinterpret_cast<float*>(&ew[5])[0];
        float _fma_32 = __fmaf_rn(wgt, eb, eacc[5]);
        eacc[5] = _fma_32;
        eb = reinterpret_cast<float*>(&ew[6])[0];
        float _fma_33 = __fmaf_rn(wgt, eb, eacc[6]);
        eacc[6] = _fma_33;
        eb = reinterpret_cast<float*>(&ew[7])[0];
        float _fma_34 = __fmaf_rn(wgt, eb, eacc[7]);
        eacc[7] = _fma_34;
        float _exp2_11 = approx_exp2((wm[3] - ehm) * softmax_scale_log2);
        wgt = ((wm[3] > -CAKE_INF) ? _exp2_11 : 0.0f);
        float _fma_35 = __fmaf_rn(wgt, ws[3], ehs);
        ehs = _fma_35;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[0])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 26112));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[4])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 26112 + 16));
        eb = reinterpret_cast<float*>(&ew[0])[0];
        float _fma_36 = __fmaf_rn(wgt, eb, eacc[0]);
        eacc[0] = _fma_36;
        eb = reinterpret_cast<float*>(&ew[1])[0];
        float _fma_37 = __fmaf_rn(wgt, eb, eacc[1]);
        eacc[1] = _fma_37;
        eb = reinterpret_cast<float*>(&ew[2])[0];
        float _fma_38 = __fmaf_rn(wgt, eb, eacc[2]);
        eacc[2] = _fma_38;
        eb = reinterpret_cast<float*>(&ew[3])[0];
        float _fma_39 = __fmaf_rn(wgt, eb, eacc[3]);
        eacc[3] = _fma_39;
        eb = reinterpret_cast<float*>(&ew[4])[0];
        float _fma_40 = __fmaf_rn(wgt, eb, eacc[4]);
        eacc[4] = _fma_40;
        eb = reinterpret_cast<float*>(&ew[5])[0];
        float _fma_41 = __fmaf_rn(wgt, eb, eacc[5]);
        eacc[5] = _fma_41;
        eb = reinterpret_cast<float*>(&ew[6])[0];
        float _fma_42 = __fmaf_rn(wgt, eb, eacc[6]);
        eacc[6] = _fma_42;
        eb = reinterpret_cast<float*>(&ew[7])[0];
        float _fma_43 = __fmaf_rn(wgt, eb, eacc[7]);
        eacc[7] = _fma_43;
        float _exp2_12 = approx_exp2((wm[4] - ehm) * softmax_scale_log2);
        wgt = ((wm[4] > -CAKE_INF) ? _exp2_12 : 0.0f);
        float _fma_44 = __fmaf_rn(wgt, ws[4], ehs);
        ehs = _fma_44;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[0])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 34816));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[4])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 34816 + 16));
        eb = reinterpret_cast<float*>(&ew[0])[0];
        float _fma_45 = __fmaf_rn(wgt, eb, eacc[0]);
        eacc[0] = _fma_45;
        eb = reinterpret_cast<float*>(&ew[1])[0];
        float _fma_46 = __fmaf_rn(wgt, eb, eacc[1]);
        eacc[1] = _fma_46;
        eb = reinterpret_cast<float*>(&ew[2])[0];
        float _fma_47 = __fmaf_rn(wgt, eb, eacc[2]);
        eacc[2] = _fma_47;
        eb = reinterpret_cast<float*>(&ew[3])[0];
        float _fma_48 = __fmaf_rn(wgt, eb, eacc[3]);
        eacc[3] = _fma_48;
        eb = reinterpret_cast<float*>(&ew[4])[0];
        float _fma_49 = __fmaf_rn(wgt, eb, eacc[4]);
        eacc[4] = _fma_49;
        eb = reinterpret_cast<float*>(&ew[5])[0];
        float _fma_50 = __fmaf_rn(wgt, eb, eacc[5]);
        eacc[5] = _fma_50;
        eb = reinterpret_cast<float*>(&ew[6])[0];
        float _fma_51 = __fmaf_rn(wgt, eb, eacc[6]);
        eacc[6] = _fma_51;
        eb = reinterpret_cast<float*>(&ew[7])[0];
        float _fma_52 = __fmaf_rn(wgt, eb, eacc[7]);
        eacc[7] = _fma_52;
        float _exp2_13 = approx_exp2((wm[5] - ehm) * softmax_scale_log2);
        wgt = ((wm[5] > -CAKE_INF) ? _exp2_13 : 0.0f);
        float _fma_53 = __fmaf_rn(wgt, ws[5], ehs);
        ehs = _fma_53;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[0])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 43520));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[4])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 43520 + 16));
        eb = reinterpret_cast<float*>(&ew[0])[0];
        float _fma_54 = __fmaf_rn(wgt, eb, eacc[0]);
        eacc[0] = _fma_54;
        eb = reinterpret_cast<float*>(&ew[1])[0];
        float _fma_55 = __fmaf_rn(wgt, eb, eacc[1]);
        eacc[1] = _fma_55;
        eb = reinterpret_cast<float*>(&ew[2])[0];
        float _fma_56 = __fmaf_rn(wgt, eb, eacc[2]);
        eacc[2] = _fma_56;
        eb = reinterpret_cast<float*>(&ew[3])[0];
        float _fma_57 = __fmaf_rn(wgt, eb, eacc[3]);
        eacc[3] = _fma_57;
        eb = reinterpret_cast<float*>(&ew[4])[0];
        float _fma_58 = __fmaf_rn(wgt, eb, eacc[4]);
        eacc[4] = _fma_58;
        eb = reinterpret_cast<float*>(&ew[5])[0];
        float _fma_59 = __fmaf_rn(wgt, eb, eacc[5]);
        eacc[5] = _fma_59;
        eb = reinterpret_cast<float*>(&ew[6])[0];
        float _fma_60 = __fmaf_rn(wgt, eb, eacc[6]);
        eacc[6] = _fma_60;
        eb = reinterpret_cast<float*>(&ew[7])[0];
        float _fma_61 = __fmaf_rn(wgt, eb, eacc[7]);
        eacc[7] = _fma_61;
        float _exp2_14 = approx_exp2((wm[6] - ehm) * softmax_scale_log2);
        wgt = ((wm[6] > -CAKE_INF) ? _exp2_14 : 0.0f);
        float _fma_62 = __fmaf_rn(wgt, ws[6], ehs);
        ehs = _fma_62;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[0])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 52224));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[4])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 52224 + 16));
        eb = reinterpret_cast<float*>(&ew[0])[0];
        float _fma_63 = __fmaf_rn(wgt, eb, eacc[0]);
        eacc[0] = _fma_63;
        eb = reinterpret_cast<float*>(&ew[1])[0];
        float _fma_64 = __fmaf_rn(wgt, eb, eacc[1]);
        eacc[1] = _fma_64;
        eb = reinterpret_cast<float*>(&ew[2])[0];
        float _fma_65 = __fmaf_rn(wgt, eb, eacc[2]);
        eacc[2] = _fma_65;
        eb = reinterpret_cast<float*>(&ew[3])[0];
        float _fma_66 = __fmaf_rn(wgt, eb, eacc[3]);
        eacc[3] = _fma_66;
        eb = reinterpret_cast<float*>(&ew[4])[0];
        float _fma_67 = __fmaf_rn(wgt, eb, eacc[4]);
        eacc[4] = _fma_67;
        eb = reinterpret_cast<float*>(&ew[5])[0];
        float _fma_68 = __fmaf_rn(wgt, eb, eacc[5]);
        eacc[5] = _fma_68;
        eb = reinterpret_cast<float*>(&ew[6])[0];
        float _fma_69 = __fmaf_rn(wgt, eb, eacc[6]);
        eacc[6] = _fma_69;
        eb = reinterpret_cast<float*>(&ew[7])[0];
        float _fma_70 = __fmaf_rn(wgt, eb, eacc[7]);
        eacc[7] = _fma_70;
        float _exp2_15 = approx_exp2((wm[7] - ehm) * softmax_scale_log2);
        wgt = ((wm[7] > -CAKE_INF) ? _exp2_15 : 0.0f);
        float _fma_71 = __fmaf_rn(wgt, ws[7], ehs);
        ehs = _fma_71;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[0])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(0) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 60928));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ew[4])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ew[(4) + 3]))
            : "r"(smem_tiles_addr + (unsigned int)((h_e * 136 + g_e * 8) * 4) + 60928 + 16));
        eb = reinterpret_cast<float*>(&ew[0])[0];
        float _fma_72 = __fmaf_rn(wgt, eb, eacc[0]);
        eacc[0] = _fma_72;
        eb = reinterpret_cast<float*>(&ew[1])[0];
        float _fma_73 = __fmaf_rn(wgt, eb, eacc[1]);
        eacc[1] = _fma_73;
        eb = reinterpret_cast<float*>(&ew[2])[0];
        float _fma_74 = __fmaf_rn(wgt, eb, eacc[2]);
        eacc[2] = _fma_74;
        eb = reinterpret_cast<float*>(&ew[3])[0];
        float _fma_75 = __fmaf_rn(wgt, eb, eacc[3]);
        eacc[3] = _fma_75;
        eb = reinterpret_cast<float*>(&ew[4])[0];
        float _fma_76 = __fmaf_rn(wgt, eb, eacc[4]);
        eacc[4] = _fma_76;
        eb = reinterpret_cast<float*>(&ew[5])[0];
        float _fma_77 = __fmaf_rn(wgt, eb, eacc[5]);
        eacc[5] = _fma_77;
        eb = reinterpret_cast<float*>(&ew[6])[0];
        float _fma_78 = __fmaf_rn(wgt, eb, eacc[6]);
        eacc[6] = _fma_78;
        eb = reinterpret_cast<float*>(&ew[7])[0];
        float _fma_79 = __fmaf_rn(wgt, eb, eacc[7]);
        eacc[7] = _fma_79;
        hm = ehm;
        hs = ehs;
        if (owner != rank) {
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(smem_zone_addr + (unsigned int)((slot * 4 + hl) * 512) + (unsigned int)(g_e * 32)), "r"(owner));
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(part_ready_addr), "r"(owner));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0), "r"(__float_as_uint(eacc[0])), "r"(__float_as_uint(eacc[1])), "r"(__float_as_uint(eacc[2])), "r"(__float_as_uint(eacc[3])), "r"(_mapa_1) : "memory");
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 16), "r"(__float_as_uint(eacc[4])), "r"(__float_as_uint(eacc[5])), "r"(__float_as_uint(eacc[6])), "r"(__float_as_uint(eacc[7])), "r"(_mapa_1) : "memory");
            if (g_e == 0) {
                uint32_t _mapa_2;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_2) : "r"(smem_stat_addr + (unsigned int)((slot * 4 + hl) * 8)), "r"(owner));
                asm volatile(
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                    :: "r"(_mapa_2), "f"(hm), "r"(_mapa_1) : "memory");
                asm volatile(
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                    :: "r"(_mapa_2 + 4), "f"(hs), "r"(_mapa_1) : "memory");
            }
        } else {
            mbarrier_wait_cluster_hint(part_ready_addr, item_par, 10000000);
            float tm = 0.0f;
            float ts = 0.0f;
            float pm[3];
            float ps[3];
            float tm_0 = hm;
            pm[0] = smem_stat[hl * 2];
            ps[0] = smem_stat[hl * 2 + 1];
            float _fmax_20 = fmaxf(tm_0, pm[0]);
            tm_0 = _fmax_20;
            pm[1] = smem_stat[(4 + hl) * 2];
            ps[1] = smem_stat[(4 + hl) * 2 + 1];
            float _fmax_21 = fmaxf(tm_0, pm[1]);
            tm_0 = _fmax_21;
            pm[2] = smem_stat[(8 + hl) * 2];
            ps[2] = smem_stat[(8 + hl) * 2 + 1];
            float _fmax_22 = fmaxf(tm_0, pm[2]);
            tm_0 = _fmax_22;
            float _exp2_16 = approx_exp2((hm - tm_0) * softmax_scale_log2);
            float w0 = ((hm > -CAKE_INF) ? _exp2_16 : 0.0f);
            float ts_1 = w0 * hs;
            ov[0] = w0 * eacc[0];
            ov[1] = w0 * eacc[1];
            ov[2] = w0 * eacc[2];
            ov[3] = w0 * eacc[3];
            ov[4] = w0 * eacc[4];
            ov[5] = w0 * eacc[5];
            ov[6] = w0 * eacc[6];
            ov[7] = w0 * eacc[7];
            float wp = 0.0f;
            unsigned int zw[8];
            float zb = 0.0f;
            float _exp2_17 = approx_exp2((pm[0] - tm_0) * softmax_scale_log2);
            wp = ((pm[0] > -CAKE_INF) ? _exp2_17 : 0.0f);
            float _fma_80 = __fmaf_rn(wp, ps[0], ts_1);
            ts_1 = _fma_80;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&zw[0])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(0) + 3]))
                : "r"(smem_zone_addr + (unsigned int)(hl * 512) + (unsigned int)(g_e * 32)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&zw[4])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(4) + 3]))
                : "r"(smem_zone_addr + (unsigned int)(hl * 512) + (unsigned int)(g_e * 32) + 16));
            zb = reinterpret_cast<float*>(&zw[0])[0];
            float _fma_81 = __fmaf_rn(wp, zb, ov[0]);
            ov[0] = _fma_81;
            zb = reinterpret_cast<float*>(&zw[1])[0];
            float _fma_82 = __fmaf_rn(wp, zb, ov[1]);
            ov[1] = _fma_82;
            zb = reinterpret_cast<float*>(&zw[2])[0];
            float _fma_83 = __fmaf_rn(wp, zb, ov[2]);
            ov[2] = _fma_83;
            zb = reinterpret_cast<float*>(&zw[3])[0];
            float _fma_84 = __fmaf_rn(wp, zb, ov[3]);
            ov[3] = _fma_84;
            zb = reinterpret_cast<float*>(&zw[4])[0];
            float _fma_85 = __fmaf_rn(wp, zb, ov[4]);
            ov[4] = _fma_85;
            zb = reinterpret_cast<float*>(&zw[5])[0];
            float _fma_86 = __fmaf_rn(wp, zb, ov[5]);
            ov[5] = _fma_86;
            zb = reinterpret_cast<float*>(&zw[6])[0];
            float _fma_87 = __fmaf_rn(wp, zb, ov[6]);
            ov[6] = _fma_87;
            zb = reinterpret_cast<float*>(&zw[7])[0];
            float _fma_88 = __fmaf_rn(wp, zb, ov[7]);
            ov[7] = _fma_88;
            float _exp2_18 = approx_exp2((pm[1] - tm_0) * softmax_scale_log2);
            wp = ((pm[1] > -CAKE_INF) ? _exp2_18 : 0.0f);
            float _fma_89 = __fmaf_rn(wp, ps[1], ts_1);
            ts_1 = _fma_89;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&zw[0])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(0) + 3]))
                : "r"(smem_zone_addr + (unsigned int)((4 + hl) * 512) + (unsigned int)(g_e * 32)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&zw[4])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(4) + 3]))
                : "r"(smem_zone_addr + (unsigned int)((4 + hl) * 512) + (unsigned int)(g_e * 32) + 16));
            zb = reinterpret_cast<float*>(&zw[0])[0];
            float _fma_90 = __fmaf_rn(wp, zb, ov[0]);
            ov[0] = _fma_90;
            zb = reinterpret_cast<float*>(&zw[1])[0];
            float _fma_91 = __fmaf_rn(wp, zb, ov[1]);
            ov[1] = _fma_91;
            zb = reinterpret_cast<float*>(&zw[2])[0];
            float _fma_92 = __fmaf_rn(wp, zb, ov[2]);
            ov[2] = _fma_92;
            zb = reinterpret_cast<float*>(&zw[3])[0];
            float _fma_93 = __fmaf_rn(wp, zb, ov[3]);
            ov[3] = _fma_93;
            zb = reinterpret_cast<float*>(&zw[4])[0];
            float _fma_94 = __fmaf_rn(wp, zb, ov[4]);
            ov[4] = _fma_94;
            zb = reinterpret_cast<float*>(&zw[5])[0];
            float _fma_95 = __fmaf_rn(wp, zb, ov[5]);
            ov[5] = _fma_95;
            zb = reinterpret_cast<float*>(&zw[6])[0];
            float _fma_96 = __fmaf_rn(wp, zb, ov[6]);
            ov[6] = _fma_96;
            zb = reinterpret_cast<float*>(&zw[7])[0];
            float _fma_97 = __fmaf_rn(wp, zb, ov[7]);
            ov[7] = _fma_97;
            float _exp2_19 = approx_exp2((pm[2] - tm_0) * softmax_scale_log2);
            wp = ((pm[2] > -CAKE_INF) ? _exp2_19 : 0.0f);
            float _fma_98 = __fmaf_rn(wp, ps[2], ts_1);
            ts_1 = _fma_98;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&zw[0])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(0) + 3]))
                : "r"(smem_zone_addr + (unsigned int)((8 + hl) * 512) + (unsigned int)(g_e * 32)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&zw[4])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zw[(4) + 3]))
                : "r"(smem_zone_addr + (unsigned int)((8 + hl) * 512) + (unsigned int)(g_e * 32) + 16));
            zb = reinterpret_cast<float*>(&zw[0])[0];
            float _fma_99 = __fmaf_rn(wp, zb, ov[0]);
            ov[0] = _fma_99;
            zb = reinterpret_cast<float*>(&zw[1])[0];
            float _fma_100 = __fmaf_rn(wp, zb, ov[1]);
            ov[1] = _fma_100;
            zb = reinterpret_cast<float*>(&zw[2])[0];
            float _fma_101 = __fmaf_rn(wp, zb, ov[2]);
            ov[2] = _fma_101;
            zb = reinterpret_cast<float*>(&zw[3])[0];
            float _fma_102 = __fmaf_rn(wp, zb, ov[3]);
            ov[3] = _fma_102;
            zb = reinterpret_cast<float*>(&zw[4])[0];
            float _fma_103 = __fmaf_rn(wp, zb, ov[4]);
            ov[4] = _fma_103;
            zb = reinterpret_cast<float*>(&zw[5])[0];
            float _fma_104 = __fmaf_rn(wp, zb, ov[5]);
            ov[5] = _fma_104;
            zb = reinterpret_cast<float*>(&zw[6])[0];
            float _fma_105 = __fmaf_rn(wp, zb, ov[6]);
            ov[6] = _fma_105;
            zb = reinterpret_cast<float*>(&zw[7])[0];
            float _fma_106 = __fmaf_rn(wp, zb, ov[7]);
            ov[7] = _fma_106;
            float _rcp_0 = approx_rcp(ts_1);
            float tinv = ((ts_1 > 0.0f) ? output_scale * _rcp_0 : 0.0f);
            ov[0] = ov[0] * tinv;
            ov[1] = ov[1] * tinv;
            ov[2] = ov[2] * tinv;
            ov[3] = ov[3] * tinv;
            ov[4] = ov[4] * tinv;
            ov[5] = ov[5] * tinv;
            ov[6] = ov[6] * tinv;
            ov[7] = ov[7] * tinv;
            tm = tm_0;
            ts = ts_1;
            if (h_e < group_size) {
                int o_off = (q_row0 + h_e) * 128 + g_e * 8;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(ov[0 + 0], ov[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(ov[0 + 2], ov[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(ov[0 + 4], ov[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(ov[0 + 6], ov[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + o_off))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                if (g_e == 0) {
                    float lse = -CAKE_INF;
                    if (ts > 0.0f) {
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(ts));
                        lse = tm * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f;
                    }
                    *(reinterpret_cast<float*>(msa_lse + (q_row0 + h_e)) + (0)) = lse;
                }
            }
        }
        item_par = 1 - item_par;
        if (total_items > num_clusters_x) {
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
}

} // extern "C"
