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
#define SMEM_K_S_STAGE_BYTES 8192
#define SMEM_K_S_STRIDE 8192
#define SMEM_Q_S_OFF 17408
#define SMEM_Q_S_STAGE_BYTES 1024
#define SMEM_Q_S_STRIDE 1024
#define SMEM_RED_OFF 18432
#define SMEM_RED_STAGE_BYTES 128
#define SMEM_RED_STRIDE 128
#define SMEM_TOTAL 18560
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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_hopper_msa_40001be54ff9d1ed698d(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, float* __restrict__ out, int* __restrict__ page_table, int* __restrict__ seqused_k, int* __restrict__ q_offset, int has_qoff, int pt_stride, int max_k_tiles, int total_q, int batch_fast, int evict_first, unsigned long long* __restrict__ trace)
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
    uint8_t* k_s = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int k_s_addr = smem + 1024;
    uint8_t* q_s = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int q_s_addr = smem + 17408;
    float* red = reinterpret_cast<float*>(smem_raw + 18432);
    const int red_addr = smem + 18432;

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
    int qoff = sk - 1;
    if (has_qoff != 0) {
        qoff = q_offset[b];
    }
    float neg = -CAKE_INF;
    int oidx = ((tid_1 - tid_1 / 4 * 4) * max_k_tiles + t) * total_q + b + tid_1 / 4;
    if (t >= nb) {
        if (tid_1 < 4) {
            out[oidx] = neg;
        }
    }
    uint64_t _wgmma_desc_0 = (((uint64_t)(((k_s_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
    uint64_t _wgmma_desc_1 = (((uint64_t)(((q_s_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
    uint64_t _wgmma_desc_2 = (((uint64_t)(((k_s_addr + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
    if (t < nb) {
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(full_addr, 17408);
                int krow = page * 128;
                if (evict_first != 0) {
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(k_s_addr), "l"((&K)), "r"(0), "r"(krow),
                           "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(k_s_addr + 8192), "l"((&K)), "r"(0), "r"(krow + 64),
                           "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
                }
                if (evict_first == 0) {
                    tma_2d_gmem2smem(k_s_addr, (&K), 0, krow, full_addr);
                    tma_2d_gmem2smem(k_s_addr + 8192, (&K), 0, krow + 64, full_addr);
                }
                tma_2d_gmem2smem(q_s_addr, (&Q), 0, b * 4, full_addr);
            }
        }
        int sk1 = sk - 1;
        int limc[2];
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
        mbarrier_wait(full_addr, 0);
        float acc0[4];
        float acc1[4];
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 0, 1, 1;\n}\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
            : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 0, 1, 1;\n}\n"
            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
            : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 1, 1, 1;\n}\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
            : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_1 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 1, 1, 1;\n}\n"
            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
            : "l"(_wgmma_a_0_2 + 2), "l"(_wgmma_b_0_1 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 1, 1, 1;\n}\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
            : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_1 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 1, 1, 1;\n}\n"
            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
            : "l"(_wgmma_a_0_2 + 4), "l"(_wgmma_b_0_1 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 1, 1, 1;\n}\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
            : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_1 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 1, 1, 1;\n}\n"
            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
            : "l"(_wgmma_a_0_2 + 6), "l"(_wgmma_b_0_1 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        float cm[2];
        cm[0] = -CAKE_INF;
        cm[1] = -CAKE_INF;
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
        float v0_1 = -CAKE_INF;
        if (limc[1] >= r0) {
            v0_1 = acc0[1];
        }
        float v1_2 = -CAKE_INF;
        if (limc[1] >= r0 + 64) {
            v1_2 = acc1[1];
        }
        float _max_2 = max_noftz(v0_1, v1_2);
        float _max_3 = max_noftz(cm[1], _max_2);
        cm[1] = _max_3;
        float v0_3 = -CAKE_INF;
        if (limc[0] >= r0 + 8) {
            v0_3 = acc0[2];
        }
        float v1_4 = -CAKE_INF;
        if (limc[0] >= r0 + 72) {
            v1_4 = acc1[2];
        }
        float _max_4 = max_noftz(v0_3, v1_4);
        float _max_5 = max_noftz(cm[0], _max_4);
        cm[0] = _max_5;
        float v0_5 = -CAKE_INF;
        if (limc[1] >= r0 + 8) {
            v0_5 = acc0[3];
        }
        float v1_6 = -CAKE_INF;
        if (limc[1] >= r0 + 72) {
            v1_6 = acc1[3];
        }
        float _max_6 = max_noftz(v0_5, v1_6);
        float _max_7 = max_noftz(cm[1], _max_6);
        cm[1] = _max_7;
        float tt = cm[0];
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, tt, 4);
        float _max_8 = max_noftz(tt, _shfl_xor_0);
        tt = _max_8;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, tt, 8);
        float _max_9 = max_noftz(tt, _shfl_xor_1);
        tt = _max_9;
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, tt, 16);
        float _max_10 = max_noftz(tt, _shfl_xor_2);
        tt = _max_10;
        cm[0] = tt;
        float tt_7 = cm[1];
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, tt_7, 4);
        float _max_11 = max_noftz(tt_7, _shfl_xor_3);
        tt_7 = _max_11;
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, tt_7, 8);
        float _max_12 = max_noftz(tt_7, _shfl_xor_4);
        tt_7 = _max_12;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, tt_7, 16);
        float _max_13 = max_noftz(tt_7, _shfl_xor_5);
        tt_7 = _max_13;
        cm[1] = tt_7;
        if (lane_0 < 4) {
            red[2 * lane_0 * 4 + warp_1] = cm[0];
            red[(2 * lane_0 + 1) * 4 + warp_1] = cm[1];
        }
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        if (tid_1 < 4) {
            float rv[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                : "r"(red_addr + (unsigned int)(tid_1 * 16)));
            float _max_14 = max_noftz(rv[0], rv[1]);
            float _max_15 = max_noftz(rv[2], rv[3]);
            float _max_16 = max_noftz(_max_14, _max_15);
            float o = _max_16;
            out[oidx] = o;
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
