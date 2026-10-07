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
#define NUM_MAIN_STAGES 1
#define SMEM_K_S_OFF 1024
#define SMEM_K_S_STAGE_BYTES 16384
#define SMEM_K_S_STRIDE 16384
#define SMEM_Q_S_OFF 33792
#define SMEM_Q_S_STAGE_BYTES 4096
#define SMEM_Q_S_STRIDE 4096
#define SMEM_RED_OFF 37888
#define SMEM_RED_STAGE_BYTES 256
#define SMEM_RED_STRIDE 256
#define SMEM_TOTAL 38144
#define THREADS 128
#define TRACE 0
#define LAUNCH_MIN_BLOCKS 4

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





__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_hopper_msa_eea34cc0811dd62a1c48(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, float* __restrict__ out, int* __restrict__ page_table, int* __restrict__ seqused_k, int* __restrict__ q_offset, int has_qoff, int pt_stride, int max_k_tiles, int total_q, int batch_fast, int evict_first, unsigned long long* __restrict__ trace)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* k_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int k_s_addr = smem + 1024;
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int q_s_addr = smem + 33792;
    float* red = reinterpret_cast<float*>(smem_raw + 37888);
    const int red_addr = smem + 37888;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    unsigned long long stamps[5];
    stamps[0] = 0;
    stamps[1] = 0;
    stamps[2] = 0;
    stamps[3] = 0;
    stamps[4] = 0;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    int b = blockIdx.x;
    int t = blockIdx.y;
    if (batch_fast == 0) {
        b = blockIdx.y;
        t = blockIdx.x;
    }
    int tid_1 = threadIdx.x;
    int lane_0 = lane;
    int warp_1 = warp;
    int g = lane_0 / 4;
    int cq = lane_0 - g * 4;
    int sk = seqused_k[b];
    int page = page_table[b * pt_stride + t];
    int nb = (sk + 127) / 128;
    int qoff = sk - 4;
    if (has_qoff != 0) {
        qoff = q_offset[b];
    }
    float neg = -CAKE_INF;
    int oidx = ((tid_1 - tid_1 / 4 * 4) * max_k_tiles + t) * total_q + b * 4 + tid_1 / 4;
    if (t >= nb) {
        if (tid_1 < 16) {
            out[oidx] = neg;
        }
    }
    uint64_t _wgmma_desc_0 = (((uint64_t)(((k_s_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
    uint64_t _wgmma_desc_1 = (((uint64_t)(((q_s_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
    uint64_t _wgmma_desc_2 = (((uint64_t)(((k_s_addr + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
    uint64_t _wgmma_desc_3 = (((uint64_t)(((k_s_addr + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
    uint64_t _wgmma_desc_4 = (((uint64_t)(((q_s_addr + 2048)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
    uint64_t _wgmma_desc_5 = (((uint64_t)(((k_s_addr + 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0_5 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_5 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_5);
    if (t < nb) {
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(full_addr, 36864);
                int krow = page * 128;
                if (evict_first != 0) {
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(k_s_addr), "l"((&K)), "r"(0), "r"(krow), "r"(0),
                           "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(k_s_addr + 16384), "l"((&K)), "r"(0), "r"(krow + 64), "r"(0),
                           "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
                }
                if (evict_first == 0) {
                    tma_3d_gmem2smem(k_s_addr, (&K), 0, krow, 0, full_addr);
                    tma_3d_gmem2smem(k_s_addr + 16384, (&K), 0, krow + 64, 0, full_addr);
                }
                tma_3d_gmem2smem(q_s_addr, (&Q), 0, b * 16, 0, full_addr);
            }
        }
        int sk1 = sk - 1;
        int limc[4];
        int pos = qoff + 2 * cq / 4;
        if (pos > sk1) {
            pos = sk1;
        }
        limc[0] = pos - t * 128;
        int pos_0 = qoff + (2 * cq + 1) / 4;
        if (pos_0 > sk1) {
            pos_0 = sk1;
        }
        limc[1] = pos_0 - t * 128;
        int pos_1 = qoff + (2 * cq + 8) / 4;
        if (pos_1 > sk1) {
            pos_1 = sk1;
        }
        limc[2] = pos_1 - t * 128;
        int pos_2 = qoff + (2 * cq + 9) / 4;
        if (pos_2 > sk1) {
            pos_2 = sk1;
        }
        limc[3] = pos_2 - t * 128;
        mbarrier_wait(full_addr, 0);
        float acc0[8];
        float acc1[8];
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7])
            : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7])
            : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7])
            : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_1 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7])
            : "l"(_wgmma_a_0_2 + 2), "l"(_wgmma_b_0_1 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7])
            : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_1 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7])
            : "l"(_wgmma_a_0_2 + 4), "l"(_wgmma_b_0_1 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7])
            : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_1 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7])
            : "l"(_wgmma_a_0_2 + 6), "l"(_wgmma_b_0_1 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7])
            : "l"(_wgmma_a_0_3), "l"(_wgmma_b_0_4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7])
            : "l"(_wgmma_a_0_5), "l"(_wgmma_b_0_4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7])
            : "l"(_wgmma_a_0_3 + 2), "l"(_wgmma_b_0_4 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7])
            : "l"(_wgmma_a_0_5 + 2), "l"(_wgmma_b_0_4 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7])
            : "l"(_wgmma_a_0_3 + 4), "l"(_wgmma_b_0_4 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7])
            : "l"(_wgmma_a_0_5 + 4), "l"(_wgmma_b_0_4 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7])
            : "l"(_wgmma_a_0_3 + 6), "l"(_wgmma_b_0_4 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7])
            : "l"(_wgmma_a_0_5 + 6), "l"(_wgmma_b_0_4 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        float cm[4];
        cm[0] = -CAKE_INF;
        cm[1] = -CAKE_INF;
        cm[2] = -CAKE_INF;
        cm[3] = -CAKE_INF;
        int r0 = warp_1 * 16 + g;
        float v0 = -CAKE_INF;
        if (limc[0] >= r0) {
            v0 = acc0[0];
        }
        float v1 = -CAKE_INF;
        if (limc[0] >= r0 + 64) {
            v1 = acc1[0];
        }
        float _max_0 = max_noftz(v0, v1);
        float _max_1 = max_noftz(cm[0], _max_0);
        cm[0] = _max_1;
        float v0_3 = -CAKE_INF;
        if (limc[1] >= r0) {
            v0_3 = acc0[1];
        }
        float v1_4 = -CAKE_INF;
        if (limc[1] >= r0 + 64) {
            v1_4 = acc1[1];
        }
        float _max_2 = max_noftz(v0_3, v1_4);
        float _max_3 = max_noftz(cm[1], _max_2);
        cm[1] = _max_3;
        float v0_5 = -CAKE_INF;
        if (limc[0] >= r0 + 8) {
            v0_5 = acc0[2];
        }
        float v1_6 = -CAKE_INF;
        if (limc[0] >= r0 + 72) {
            v1_6 = acc1[2];
        }
        float _max_4 = max_noftz(v0_5, v1_6);
        float _max_5 = max_noftz(cm[0], _max_4);
        cm[0] = _max_5;
        float v0_7 = -CAKE_INF;
        if (limc[1] >= r0 + 8) {
            v0_7 = acc0[3];
        }
        float v1_8 = -CAKE_INF;
        if (limc[1] >= r0 + 72) {
            v1_8 = acc1[3];
        }
        float _max_6 = max_noftz(v0_7, v1_8);
        float _max_7 = max_noftz(cm[1], _max_6);
        cm[1] = _max_7;
        float v0_9 = -CAKE_INF;
        if (limc[2] >= r0) {
            v0_9 = acc0[4];
        }
        float v1_10 = -CAKE_INF;
        if (limc[2] >= r0 + 64) {
            v1_10 = acc1[4];
        }
        float _max_8 = max_noftz(v0_9, v1_10);
        float _max_9 = max_noftz(cm[2], _max_8);
        cm[2] = _max_9;
        float v0_11 = -CAKE_INF;
        if (limc[3] >= r0) {
            v0_11 = acc0[5];
        }
        float v1_12 = -CAKE_INF;
        if (limc[3] >= r0 + 64) {
            v1_12 = acc1[5];
        }
        float _max_10 = max_noftz(v0_11, v1_12);
        float _max_11 = max_noftz(cm[3], _max_10);
        cm[3] = _max_11;
        float v0_13 = -CAKE_INF;
        if (limc[2] >= r0 + 8) {
            v0_13 = acc0[6];
        }
        float v1_14 = -CAKE_INF;
        if (limc[2] >= r0 + 72) {
            v1_14 = acc1[6];
        }
        float _max_12 = max_noftz(v0_13, v1_14);
        float _max_13 = max_noftz(cm[2], _max_12);
        cm[2] = _max_13;
        float v0_15 = -CAKE_INF;
        if (limc[3] >= r0 + 8) {
            v0_15 = acc0[7];
        }
        float v1_16 = -CAKE_INF;
        if (limc[3] >= r0 + 72) {
            v1_16 = acc1[7];
        }
        float _max_14 = max_noftz(v0_15, v1_16);
        float _max_15 = max_noftz(cm[3], _max_14);
        cm[3] = _max_15;
        float tt = cm[0];
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, tt, 4);
        float _max_16 = max_noftz(tt, _shfl_xor_0);
        tt = _max_16;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, tt, 8);
        float _max_17 = max_noftz(tt, _shfl_xor_1);
        tt = _max_17;
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, tt, 16);
        float _max_18 = max_noftz(tt, _shfl_xor_2);
        tt = _max_18;
        cm[0] = tt;
        float tt_17 = cm[1];
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, tt_17, 4);
        float _max_19 = max_noftz(tt_17, _shfl_xor_3);
        tt_17 = _max_19;
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, tt_17, 8);
        float _max_20 = max_noftz(tt_17, _shfl_xor_4);
        tt_17 = _max_20;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, tt_17, 16);
        float _max_21 = max_noftz(tt_17, _shfl_xor_5);
        tt_17 = _max_21;
        cm[1] = tt_17;
        float tt_18 = cm[2];
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, tt_18, 4);
        float _max_22 = max_noftz(tt_18, _shfl_xor_6);
        tt_18 = _max_22;
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, tt_18, 8);
        float _max_23 = max_noftz(tt_18, _shfl_xor_7);
        tt_18 = _max_23;
        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, tt_18, 16);
        float _max_24 = max_noftz(tt_18, _shfl_xor_8);
        tt_18 = _max_24;
        cm[2] = tt_18;
        float tt_19 = cm[3];
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, tt_19, 4);
        float _max_25 = max_noftz(tt_19, _shfl_xor_9);
        tt_19 = _max_25;
        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, tt_19, 8);
        float _max_26 = max_noftz(tt_19, _shfl_xor_10);
        tt_19 = _max_26;
        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, tt_19, 16);
        float _max_27 = max_noftz(tt_19, _shfl_xor_11);
        tt_19 = _max_27;
        cm[3] = tt_19;
        if (lane_0 < 4) {
            red[2 * lane_0 * 4 + warp_1] = cm[0];
            red[(2 * lane_0 + 1) * 4 + warp_1] = cm[1];
            red[(8 + 2 * lane_0) * 4 + warp_1] = cm[2];
            red[(8 + 2 * lane_0 + 1) * 4 + warp_1] = cm[3];
        }
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        if (tid_1 < 16) {
            float rv[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                : "r"(red_addr + (unsigned int)(tid_1 * 16)));
            float _max_28 = max_noftz(rv[0], rv[1]);
            float _max_29 = max_noftz(rv[2], rv[3]);
            float _max_30 = max_noftz(_max_28, _max_29);
            float o = _max_30;
            out[oidx] = o;
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
