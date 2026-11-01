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

#define CAKE_INF CUDART_INF_F
#define NUM_PIPE_STAGES 2
#define SMEM_SW_OFF 1024
#define SMEM_SW_STAGE_BYTES 14336
#define SMEM_SW_STRIDE 14336
#define SMEM_SX_OFF 9216
#define SMEM_SX_STAGE_BYTES 4096
#define SMEM_SX_STRIDE 14336
#define SMEM_RED_OFF 29696
#define SMEM_RED_STAGE_BYTES 1024
#define SMEM_RED_STRIDE 1024
#define SMEM_TOTAL 30720
#define THREADS 160
#define LAUNCH_MIN_BLOCKS 1

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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(160, LAUNCH_MIN_BLOCKS) __cluster_dims__(3,1,1) void
kernel_cake_tinygemm2_wgmma_218f7ee3cce31592eb67(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ bias, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 16)
    #define red_full_addr (mbar_base + 32)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 3;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 3;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* sw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sw_addr = smem + 1024;
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int sx_addr = smem + 9216;
    float* red = reinterpret_cast<float*>(smem_raw + 29696);
    const int red_addr = smem + 29696;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&X))) : "memory"); }

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 5 barriers)
    // Mbarriers at smem_raw[0..40)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // empty: 2 barriers, init_count=4
            mbarrier_init(smem + 16, 4);
            mbarrier_init(smem + 24, 4);
            // red_full: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // Kernel post-init ops
    int rank = cta_rank;
    int n0 = blockIdx.x / 3 * 8;
    int m0 = blockIdx.y * 16;
    int rounds = (K + 1023) / 1024;
    int nslices = ((rounds * 4) < ((K + 63) / 64 - 12 * (rounds - 1)) ? (rounds * 4) : ((K + 63) / 64 - 12 * (rounds - 1)));
    int nstages = (nslices + 4 - 1) / 4;

    // ---- Role: mma ----
    if (warp <= 3) {
        { // mma_main
            float acc_0[4] = {0};
            int g = lane / 4;
            int t = lane % 4;
            int _min_0 = ((m0 + g) < (M - 1) ? (m0 + g) : (M - 1));
            float bvr_0 = bias[_min_0];
            int _min_1 = ((m0 + g + 8) < (M - 1) ? (m0 + g + 8) : (M - 1));
            float bvr_1 = bias[_min_1];
            int _base = make_warp_uniform((unsigned int)smem);
            unsigned int base = _base;
            int slot = 0;
            asm volatile("" : "+f"(acc_0[0]) :: "memory");
            asm volatile("" : "+f"(acc_0[1]) :: "memory");
            asm volatile("" : "+f"(acc_0[2]) :: "memory");
            asm volatile("" : "+f"(acc_0[3]) :: "memory");
            unsigned int _phase_full = 0;
            mbarrier_wait(full_addr + (slot) * 8, _phase_full);
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            unsigned int sb = base + (unsigned int)(slot * 14336);
            uint64_t d_b_0 = (((uint64_t)(((sb + (9216u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t d_a_0 = (((uint64_t)(((sb + (1024u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                : "l"(d_a_0), "l"(d_b_0)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                : "l"(d_a_0 + 2), "l"(d_b_0 + 2)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                : "l"(d_a_0 + 4), "l"(d_b_0 + 4)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                : "l"(d_a_0 + 6), "l"(d_b_0 + 6)
                : "memory");
            if (1 < nslices) {
                uint64_t d_b_1 = (((uint64_t)(((sb + (9216u + 1024u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_1 = (((uint64_t)(((sb + (1024u + 2048u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_1), "l"(d_b_1)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_1 + 2), "l"(d_b_1 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_1 + 4), "l"(d_b_1 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_1 + 6), "l"(d_b_1 + 6)
                    : "memory");
            }
            if (2 < nslices) {
                uint64_t d_b_2 = (((uint64_t)(((sb + (9216u + 2048u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_2 = (((uint64_t)(((sb + (1024u + 4096u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_2), "l"(d_b_2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_2 + 2), "l"(d_b_2 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_2 + 4), "l"(d_b_2 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_2 + 6), "l"(d_b_2 + 6)
                    : "memory");
            }
            if (3 < nslices) {
                uint64_t d_b_3 = (((uint64_t)(((sb + (9216u + 3072u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_3 = (((uint64_t)(((sb + (1024u + 6144u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_3), "l"(d_b_3)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_3 + 2), "l"(d_b_3 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_3 + 4), "l"(d_b_3 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_3 + 6), "l"(d_b_3 + 6)
                    : "memory");
            }
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
            asm volatile("" : "+f"(acc_0[0]) :: "memory");
            asm volatile("" : "+f"(acc_0[1]) :: "memory");
            asm volatile("" : "+f"(acc_0[2]) :: "memory");
            asm volatile("" : "+f"(acc_0[3]) :: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            if (warp == 0) {
                if (rank > 0) {
                    uint32_t _cluster_mapa_2;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_2) : "r"(red_addr + (unsigned int)((rank - 1) * 512 + 16 * lane)), "r"(0));
                    uint32_t _mbar_peer_3;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_3) : "r"(red_full_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_2), "r"(__float_as_uint(acc_0[0])), "r"(__float_as_uint(acc_0[1])), "r"(__float_as_uint(acc_0[2])), "r"(__float_as_uint(acc_0[3])), "r"(_mbar_peer_3) : "memory");
                } else {
                    if (lane == 0) {
                        mbarrier_arrive_expect_tx(red_full_addr, 1024);
                    }
                    mbarrier_wait_cluster_hint(red_full_addr, 0, 10000000);
                    float v_e_0 = acc_0[0];
                    v_e_0 = v_e_0 + red[4 * lane];
                    v_e_0 = v_e_0 + red[128 + 4 * lane];
                    float _add_4;
                    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(v_e_0), "f"(0.0f));
                    v_e_0 = _add_4;
                    int m_e_0 = m0 + g;
                    int n_e_0 = n0 + 2 * t;
                    if (m_e_0 < M && n_e_0 < N) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_0 + n_e_0 * M)) + (0)) = (__nv_bfloat16)(v_e_0 + bvr_0);
                    }
                    float v_e_1 = acc_0[1];
                    v_e_1 = v_e_1 + red[4 * lane + 1];
                    v_e_1 = v_e_1 + red[128 + 4 * lane + 1];
                    float _add_5;
                    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(v_e_1), "f"(0.0f));
                    v_e_1 = _add_5;
                    int m_e_1 = m0 + g;
                    int n_e_1 = n0 + 2 * t + 1;
                    if (m_e_1 < M && n_e_1 < N) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_1 + n_e_1 * M)) + (0)) = (__nv_bfloat16)(v_e_1 + bvr_0);
                    }
                    float v_e_2 = acc_0[2];
                    v_e_2 = v_e_2 + red[4 * lane + 2];
                    v_e_2 = v_e_2 + red[128 + 4 * lane + 2];
                    float _add_6;
                    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(v_e_2), "f"(0.0f));
                    v_e_2 = _add_6;
                    int m_e_2 = m0 + g + 8;
                    int n_e_2 = n0 + 2 * t;
                    if (m_e_2 < M && n_e_2 < N) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_2 + n_e_2 * M)) + (0)) = (__nv_bfloat16)(v_e_2 + bvr_1);
                    }
                    float v_e_3 = acc_0[3];
                    v_e_3 = v_e_3 + red[4 * lane + 3];
                    v_e_3 = v_e_3 + red[128 + 4 * lane + 3];
                    float _add_7;
                    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(v_e_3), "f"(0.0f));
                    v_e_3 = _add_7;
                    int m_e_3 = m0 + g + 8;
                    int n_e_3 = n0 + 2 * t + 1;
                    if (m_e_3 < M && n_e_3 < N) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_3 + n_e_3 * M)) + (0)) = (__nv_bfloat16)(v_e_3 + bvr_1);
                    }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(full_addr, nslices * 3072);
                int sl_0 = 0;
                tma_2d_gmem2smem(sw_addr + 0, (&W), (sl_0 / 4 * 16 + 4 * rank + sl_0 % 4) * 64, m0, full_addr);
                if (1 < nslices) {
                    int sl_1 = 1;
                    tma_2d_gmem2smem(sw_addr + 2048, (&W), (sl_1 / 4 * 16 + 4 * rank + sl_1 % 4) * 64, m0, full_addr);
                }
                if (2 < nslices) {
                    int sl_2 = 2;
                    tma_2d_gmem2smem(sw_addr + 4096, (&W), (sl_2 / 4 * 16 + 4 * rank + sl_2 % 4) * 64, m0, full_addr);
                }
                if (3 < nslices) {
                    int sl_3 = 3;
                    tma_2d_gmem2smem(sw_addr + 6144, (&W), (sl_3 / 4 * 16 + 4 * rank + sl_3 % 4) * 64, m0, full_addr);
                }
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            if (lane == 0) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (elect_sync()) {
                int slx_0 = 0;
                tma_2d_gmem2smem(sx_addr + 0, (&X), (slx_0 / 4 * 16 + 4 * rank + slx_0 % 4) * 64, n0, full_addr);
                if (1 < nslices) {
                    int slx_1 = 1;
                    tma_2d_gmem2smem(sx_addr + 1024, (&X), (slx_1 / 4 * 16 + 4 * rank + slx_1 % 4) * 64, n0, full_addr);
                }
                if (2 < nslices) {
                    int slx_2 = 2;
                    tma_2d_gmem2smem(sx_addr + 2048, (&X), (slx_2 / 4 * 16 + 4 * rank + slx_2 % 4) * 64, n0, full_addr);
                }
                if (3 < nslices) {
                    int slx_3 = 3;
                    tma_2d_gmem2smem(sx_addr + 3072, (&X), (slx_3 / 4 * 16 + 4 * rank + slx_3 % 4) * 64, n0, full_addr);
                }
            }
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }
}

} // extern "C"
