/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
static_assert(sizeof(uint64_t) == 8, "Fp4 requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) Fp4TensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(Fp4TensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(Fp4TensorMap64) == 64, "64-aligned tensor-map ABI alignment");

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

#define FP4_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_PAT_OFF 1024
#define SMEM_PAT_STAGE_BYTES 12288
#define SMEM_PAT_STRIDE 12288
#define SMEM_CTL_OFF 13312
#define SMEM_CTL_STAGE_BYTES 16
#define SMEM_CTL_STRIDE 16
#define SMEM_S_AM_OFF 226176
#define SMEM_S_AM_STAGE_BYTES 80
#define SMEM_S_AM_STRIDE 80
#define SMEM_WSM_OFF 13376
#define SMEM_WSM_STAGE_BYTES 212800
#define SMEM_WSM_STRIDE 212800
#define SMEM_WSM16_OFF 13376
#define SMEM_WSM16_STAGE_BYTES 212800
#define SMEM_WSM16_STRIDE 212800
#define SMEM_WSM32_OFF 13376
#define SMEM_WSM32_STAGE_BYTES 212800
#define SMEM_WSM32_STRIDE 212800
#define SMEM_TOTAL 226304
#define THREADS 352
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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}



// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}



__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_fp4_allreduce_tp4_twoshot_nvfp4_fused_bv1_r0_cap16384(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ out, unsigned int* __restrict__ ctrl, int rows, const unsigned long long* __restrict__ workspace)
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
    #define xb_full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    unsigned int* pat = reinterpret_cast<unsigned int*>(smem_raw + SMEM_PAT_OFF);
    const int pat_addr = smem + SMEM_PAT_OFF;
    unsigned int* ctl = reinterpret_cast<unsigned int*>(smem_raw + SMEM_CTL_OFF);
    const int ctl_addr = smem + SMEM_CTL_OFF;
    float* s_am = reinterpret_cast<float*>(smem_raw + SMEM_S_AM_OFF);
    const int s_am_addr = smem + SMEM_S_AM_OFF;
    uint8_t* wsm = reinterpret_cast<uint8_t*>(smem_raw + SMEM_WSM_OFF);
    const int wsm_addr = smem + SMEM_WSM_OFF;
    uint16_t* wsm16 = reinterpret_cast<uint16_t*>(smem_raw + SMEM_WSM16_OFF);
    const int wsm16_addr = smem + SMEM_WSM16_OFF;
    float* wsm32 = reinterpret_cast<float*>(smem_raw + SMEM_WSM32_OFF);
    const int wsm32_addr = smem + SMEM_WSM32_OFF;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // xb_full: 20 barriers, init_count=1
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
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: workers ----
    if (warp <= 9) {
        { // workers_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            long long bid64 = (long long)bid;
            long long g64 = (long long)num_bids;
            long long quads64 = (long long)(rows / 4);
            int r0 = (int)(4 * (bid64 * quads64 / g64));
            int r1 = (int)(4 * ((bid64 + 1) * quads64 / g64));
            int nrows = r1 - r0;
            int warp_0 = (int)warp;
            int lane_1 = (int)lane;
            if (tid == 0) {
                ctl[0] = ctrl[0];
                ctl[1] = ctrl[2];
                ctl[2] = ctrl[3];
            }
            asm volatile("barrier.sync 0, 352;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int call = ctl[0];
            unsigned int buf = call % 3;
            int* out_i32 = reinterpret_cast<int*>(out);
            int f = lane_1 >> 1 & 3;
            int xb_off = warp_0 * 21280;
            int own_off = xb_off + 12288;
            int stg_off = own_off + 6944;
            uint8_t* self_u8 = reinterpret_cast<uint8_t*>(workspace[0]);
            uint8_t* peer_base = reinterpret_cast<uint8_t*>(workspace[0]);
            uint8_t* peer_base_2 = reinterpret_cast<uint8_t*>(workspace[1]);
            uint8_t* peer_base_3 = reinterpret_cast<uint8_t*>(workspace[2]);
            uint8_t* peer_base_4 = reinterpret_cast<uint8_t*>(workspace[3]);
            long long rs_base = (long long)buf * 113770496;
            long long ag_base = rs_base + 56885248;
            int n = 0;
            if (nrows > warp_0) {
                n = (nrows - warp_0 + 10 - 1) / 10;
            }
            if (n > 0 && lane_1 == 0) {
                long long first_row = (long long)(r0 + warp_0);
                int zero_k = 0;
                int stage = 2 * warp_0 + (zero_k & 1);
                int slot = zero_k % 3;
                mbarrier_arrive_expect_tx(xb_full_addr + (stage) * 8, 3456);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(xb_off + slot * 3472), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(xq) + ((unsigned long long)(first_row * 3456) * (unsigned long long)1)), 3456, xb_full_addr + (stage) * 8);
            }
            #pragma unroll 1
            for (int k = 0; k < n; k++) {
                int ob_off = own_off + (k & 1) * 3472;
                ob_off = xb_off + k % 3 * 3472;
                if (k >= 2) {
                    if (lane_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncwarp();
                }
                if (lane_1 == 0 && n > k + 1) {
                    long long next_row = (long long)(r0 + warp_0 + 10 * (k + 1));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int stage_1 = 2 * warp_0 + (k + 1 & 1);
                    int slot_1 = (k + 1) % 3;
                    mbarrier_arrive_expect_tx(xb_full_addr + (stage_1) * 8, 3456);
                    cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(xb_off + slot_1 * 3472), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(xq) + ((unsigned long long)(next_row * 3456) * (unsigned long long)1)), 3456, xb_full_addr + (stage_1) * 8);
                }
                mbarrier_wait(xb_full_addr + (2 * warp_0 + (k & 1)) * 8, k >> 1 & 1);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane_1 == 0) {
                    long long row64 = (long long)(r0 + warp_0 + 10 * k);
                    long long owner = row64 % 4;
                    long long ix64 = row64 / 4;
                    if (owner == 0) {
                        {
                            void* _cpbulk_dst_0 = reinterpret_cast<void*>(peer_base + ((int)rs_base + (int)ix64 * 3072));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_0), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_1 = reinterpret_cast<void*>(peer_base + ((int)rs_base + 50331648 + (int)ix64 * 384));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_1), "r"(wsm_addr + (unsigned int)ob_off + 3072), "r"((uint32_t)(384))
                                : "memory");
                        }
                    }
                    if (owner == 1) {
                        {
                            void* _cpbulk_dst_2 = reinterpret_cast<void*>(peer_base_2 + ((int)rs_base + (int)ix64 * 3072));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_2), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_3 = reinterpret_cast<void*>(peer_base_2 + ((int)rs_base + 50331648 + (int)ix64 * 384));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_3), "r"(wsm_addr + (unsigned int)ob_off + 3072), "r"((uint32_t)(384))
                                : "memory");
                        }
                    }
                    if (owner == 2) {
                        {
                            void* _cpbulk_dst_4 = reinterpret_cast<void*>(peer_base_3 + ((int)rs_base + (int)ix64 * 3072));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_4), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_5 = reinterpret_cast<void*>(peer_base_3 + ((int)rs_base + 50331648 + (int)ix64 * 384));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_5), "r"(wsm_addr + (unsigned int)ob_off + 3072), "r"((uint32_t)(384))
                                : "memory");
                        }
                    }
                    if (owner == 3) {
                        {
                            void* _cpbulk_dst_6 = reinterpret_cast<void*>(peer_base_4 + ((int)rs_base + (int)ix64 * 3072));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_6), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_7 = reinterpret_cast<void*>(peer_base_4 + ((int)rs_base + 50331648 + (int)ix64 * 384));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_7), "r"(wsm_addr + (unsigned int)ob_off + 3072), "r"((uint32_t)(384))
                                : "memory");
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
            }
            int rb0 = r0 + (-(r0 % 4) + 4) % 4;
            int nB = 0;
            if (rb0 < r0 + nrows) {
                nB = (r0 + nrows - 1 - rb0) / 4 + 1;
            }
            if (lane_1 == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
            __syncwarp();
            int g32 = (int)num_bids;
            if (rows / g32 <= 8) {
                asm volatile("barrier.arrive 7, 352;" ::: "memory");
            }
            int nC = nrows - nB;
            int wB = (warp_0 - nrows % 10 + 10) % 10;
            int wC = (warp_0 - (nrows + nB) % 10 + 10) % 10;
            int kb = 0;
            int t = warp_0 >> 1;
            int h = warp_0 & 1;
            int lxb_off = 2 * t * 21280;
            int i0 = (t - (nrows % 10 >> 1) + 5) % 5;
            #pragma unroll 1
            for (int i = i0; i < nB; i += 5) {
                int trow = rb0 + 4 * i;
                long long tix64 = (long long)(trow / 4);
                long long trow64 = (long long)trow;
                int kbp = kb & 1;
                int tob_off = lxb_off + kbp * 3472;
                if (h == 0 && kb >= 2 && lane_1 == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                float g = gs[0];
                float g_0 = gs[1];
                float g_1 = gs[2];
                float g_2 = gs[3];
                int pp = lane_1 + 32 * (3 * h);
                uint32_t _fp4_wire_values_0[16];
                uint32_t _fp4_wire_scales_0[4];
                uint16_t _wire_h16_8_0;
                uint16_t _wire_h16_8_1;
                uint16_t _wire_h16_8_2;
                uint16_t _wire_h16_8_3;
                uint32_t _wire_vpend_8 = 15u;
                uint32_t _wire_spend_8 = 15u;
                do {
                    if (_wire_vpend_8 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)rs_base + (int)tix64 * 3072 + 16 * pp))) : "memory");
                    }
                    if (_wire_vpend_8 & 2u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)rs_base + (4096 + (int)tix64) * 3072 + 16 * pp))) : "memory");
                    }
                    if (_wire_vpend_8 & 4u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[8]), "=r"(_fp4_wire_values_0[9]), "=r"(_fp4_wire_values_0[10]), "=r"(_fp4_wire_values_0[11]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)rs_base + (8192 + (int)tix64) * 3072 + 16 * pp))) : "memory");
                    }
                    if (_wire_vpend_8 & 8u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[12]), "=r"(_fp4_wire_values_0[13]), "=r"(_fp4_wire_values_0[14]), "=r"(_fp4_wire_values_0[15]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)rs_base + (12288 + (int)tix64) * 3072 + 16 * pp))) : "memory");
                    }
                    if (_wire_spend_8 & 1u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_8_0) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)rs_base + 50331648 + (int)tix64 * 384 + 2 * pp))) : "memory");
                        _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_8_0;
                    }
                    if (_wire_spend_8 & 2u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_8_1) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)rs_base + 50331648 + (4096 + (int)tix64) * 384 + 2 * pp))) : "memory");
                        _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_8_1;
                    }
                    if (_wire_spend_8 & 4u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_8_2) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)rs_base + 50331648 + (8192 + (int)tix64) * 384 + 2 * pp))) : "memory");
                        _fp4_wire_scales_0[2] = (uint32_t)_wire_h16_8_2;
                    }
                    if (_wire_spend_8 & 8u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_8_3) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)rs_base + 50331648 + (12288 + (int)tix64) * 384 + 2 * pp))) : "memory");
                        _fp4_wire_scales_0[3] = (uint32_t)_wire_h16_8_3;
                    }
                    if (_wire_vpend_8 & 1u) {
                        if (!((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_8 &= ~1u;
                    }
                    if (_wire_vpend_8 & 2u) {
                        if (!((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_8 &= ~2u;
                    }
                    if (_wire_vpend_8 & 4u) {
                        if (!((((_fp4_wire_values_0[8] & ~((_fp4_wire_values_0[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[9] & ~((_fp4_wire_values_0[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[10] & ~((_fp4_wire_values_0[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[11] & ~((_fp4_wire_values_0[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_8 &= ~4u;
                    }
                    if (_wire_vpend_8 & 8u) {
                        if (!((((_fp4_wire_values_0[12] & ~((_fp4_wire_values_0[12] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[13] & ~((_fp4_wire_values_0[13] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[14] & ~((_fp4_wire_values_0[14] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[15] & ~((_fp4_wire_values_0[15] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_8 &= ~8u;
                    }
                    if (_wire_spend_8 & 1u) {
                        if (!(((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[0] >> 8) == 0xFFu))) _wire_spend_8 &= ~1u;
                    }
                    if (_wire_spend_8 & 2u) {
                        if (!(((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[1] >> 8) == 0xFFu))) _wire_spend_8 &= ~2u;
                    }
                    if (_wire_spend_8 & 4u) {
                        if (!(((_fp4_wire_scales_0[2] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[2] >> 8) == 0xFFu))) _wire_spend_8 &= ~4u;
                    }
                    if (_wire_spend_8 & 8u) {
                        if (!(((_fp4_wire_scales_0[3] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[3] >> 8) == 0xFFu))) _wire_spend_8 &= ~8u;
                    }
                } while ((_wire_vpend_8 | _wire_spend_8) != 0u);
                float acc[32];
                acc[0] = 0.0f;
                acc[1] = 0.0f;
                acc[2] = 0.0f;
                acc[3] = 0.0f;
                acc[4] = 0.0f;
                acc[5] = 0.0f;
                acc[6] = 0.0f;
                acc[7] = 0.0f;
                acc[8] = 0.0f;
                acc[9] = 0.0f;
                acc[10] = 0.0f;
                acc[11] = 0.0f;
                acc[12] = 0.0f;
                acc[13] = 0.0f;
                acc[14] = 0.0f;
                acc[15] = 0.0f;
                acc[16] = 0.0f;
                acc[17] = 0.0f;
                acc[18] = 0.0f;
                acc[19] = 0.0f;
                acc[20] = 0.0f;
                acc[21] = 0.0f;
                acc[22] = 0.0f;
                acc[23] = 0.0f;
                acc[24] = 0.0f;
                acc[25] = 0.0f;
                acc[26] = 0.0f;
                acc[27] = 0.0f;
                acc[28] = 0.0f;
                acc[29] = 0.0f;
                acc[30] = 0.0f;
                acc[31] = 0.0f;
                float2 _fp8x2_decode_0;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_0.x), "=f"(_fp8x2_decode_0.y) : "h"((uint16_t)_fp4_wire_scales_0[0]));
                float _mul_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(_fp8x2_decode_0.x), "f"(g));
                float s0 = _mul_0;
                float _mul_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(_fp8x2_decode_0.y), "f"(g));
                float s1 = _mul_1;
                float2 _f2_0 = make_float2(s0, s0);
                float2 _f2_1 = make_float2(s1, s1);
                float _unpack_fp4_0[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[0]));
                    (&_unpack_fp4_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_0[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_0[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_0[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_0[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_0[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_0[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_0[0]), "=f"(_unpack_fp4_0[1]), "=f"(_unpack_fp4_0[2]), "=f"(_unpack_fp4_0[3]), "=f"(_unpack_fp4_0[4]), "=f"(_unpack_fp4_0[5]), "=f"(_unpack_fp4_0[6]), "=f"(_unpack_fp4_0[7]) : "r"(_fp4_wire_values_0[0]));
                    #endif
                }
                float2 _f2_2 = make_float2(acc[0], acc[1]);
                float2 _f2_3 = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                float2 r = fma_f32x2_rn_noftz(_f2_3, _f2_0, _f2_2);
                acc[0] = r.x;
                acc[1] = r.y;
                float2 _f2_4 = make_float2(acc[2], acc[3]);
                float2 _f2_5 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                float2 r_3 = fma_f32x2_rn_noftz(_f2_5, _f2_0, _f2_4);
                acc[2] = r_3.x;
                acc[3] = r_3.y;
                float2 _f2_6 = make_float2(acc[4], acc[5]);
                float2 _f2_7 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                float2 r_4 = fma_f32x2_rn_noftz(_f2_7, _f2_0, _f2_6);
                acc[4] = r_4.x;
                acc[5] = r_4.y;
                float2 _f2_8 = make_float2(acc[6], acc[7]);
                float2 _f2_9 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                float2 r_5 = fma_f32x2_rn_noftz(_f2_9, _f2_0, _f2_8);
                acc[6] = r_5.x;
                acc[7] = r_5.y;
                float _unpack_fp4_1[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[1 + 0]));
                    (&_unpack_fp4_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_1[0]), "=f"(_unpack_fp4_1[1]), "=f"(_unpack_fp4_1[2]), "=f"(_unpack_fp4_1[3]), "=f"(_unpack_fp4_1[4]), "=f"(_unpack_fp4_1[5]), "=f"(_unpack_fp4_1[6]), "=f"(_unpack_fp4_1[7]) : "r"(_fp4_wire_values_0[1 + 0]));
                    #endif
                }
                float2 _f2_10 = make_float2(acc[8], acc[9]);
                float2 _f2_11 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                float2 r_6 = fma_f32x2_rn_noftz(_f2_11, _f2_0, _f2_10);
                acc[8] = r_6.x;
                acc[9] = r_6.y;
                float2 _f2_12 = make_float2(acc[10], acc[11]);
                float2 _f2_13 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                float2 r_7 = fma_f32x2_rn_noftz(_f2_13, _f2_0, _f2_12);
                acc[10] = r_7.x;
                acc[11] = r_7.y;
                float2 _f2_14 = make_float2(acc[12], acc[13]);
                float2 _f2_15 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                float2 r_8 = fma_f32x2_rn_noftz(_f2_15, _f2_0, _f2_14);
                acc[12] = r_8.x;
                acc[13] = r_8.y;
                float2 _f2_16 = make_float2(acc[14], acc[15]);
                float2 _f2_17 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                float2 r_9 = fma_f32x2_rn_noftz(_f2_17, _f2_0, _f2_16);
                acc[14] = r_9.x;
                acc[15] = r_9.y;
                float _unpack_fp4_2[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[2 + 0]));
                    (&_unpack_fp4_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_2[0]), "=f"(_unpack_fp4_2[1]), "=f"(_unpack_fp4_2[2]), "=f"(_unpack_fp4_2[3]), "=f"(_unpack_fp4_2[4]), "=f"(_unpack_fp4_2[5]), "=f"(_unpack_fp4_2[6]), "=f"(_unpack_fp4_2[7]) : "r"(_fp4_wire_values_0[2 + 0]));
                    #endif
                }
                float2 _f2_18 = make_float2(acc[16], acc[17]);
                float2 _f2_19 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                float2 r_10 = fma_f32x2_rn_noftz(_f2_19, _f2_1, _f2_18);
                acc[16] = r_10.x;
                acc[17] = r_10.y;
                float2 _f2_20 = make_float2(acc[18], acc[19]);
                float2 _f2_21 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                float2 r_11 = fma_f32x2_rn_noftz(_f2_21, _f2_1, _f2_20);
                acc[18] = r_11.x;
                acc[19] = r_11.y;
                float2 _f2_22 = make_float2(acc[20], acc[21]);
                float2 _f2_23 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                float2 r_12 = fma_f32x2_rn_noftz(_f2_23, _f2_1, _f2_22);
                acc[20] = r_12.x;
                acc[21] = r_12.y;
                float2 _f2_24 = make_float2(acc[22], acc[23]);
                float2 _f2_25 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                float2 r_13 = fma_f32x2_rn_noftz(_f2_25, _f2_1, _f2_24);
                acc[22] = r_13.x;
                acc[23] = r_13.y;
                float _unpack_fp4_3[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[3 + 0]));
                    (&_unpack_fp4_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_3[0]), "=f"(_unpack_fp4_3[1]), "=f"(_unpack_fp4_3[2]), "=f"(_unpack_fp4_3[3]), "=f"(_unpack_fp4_3[4]), "=f"(_unpack_fp4_3[5]), "=f"(_unpack_fp4_3[6]), "=f"(_unpack_fp4_3[7]) : "r"(_fp4_wire_values_0[3 + 0]));
                    #endif
                }
                float2 _f2_26 = make_float2(acc[24], acc[25]);
                float2 _f2_27 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                float2 r_14 = fma_f32x2_rn_noftz(_f2_27, _f2_1, _f2_26);
                acc[24] = r_14.x;
                acc[25] = r_14.y;
                float2 _f2_28 = make_float2(acc[26], acc[27]);
                float2 _f2_29 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                float2 r_15 = fma_f32x2_rn_noftz(_f2_29, _f2_1, _f2_28);
                acc[26] = r_15.x;
                acc[27] = r_15.y;
                float2 _f2_30 = make_float2(acc[28], acc[29]);
                float2 _f2_31 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                float2 r_16 = fma_f32x2_rn_noftz(_f2_31, _f2_1, _f2_30);
                acc[28] = r_16.x;
                acc[29] = r_16.y;
                float2 _f2_32 = make_float2(acc[30], acc[31]);
                float2 _f2_33 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                float2 r_17 = fma_f32x2_rn_noftz(_f2_33, _f2_1, _f2_32);
                acc[30] = r_17.x;
                acc[31] = r_17.y;
                float2 _fp8x2_decode_1;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_1.x), "=f"(_fp8x2_decode_1.y) : "h"((uint16_t)_fp4_wire_scales_0[1]));
                float _mul_2;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_fp8x2_decode_1.x), "f"(g_0));
                float s0_18 = _mul_2;
                float _mul_3;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_fp8x2_decode_1.y), "f"(g_0));
                float s1_19 = _mul_3;
                float2 _f2_34 = make_float2(s0_18, s0_18);
                float2 _f2_35 = make_float2(s1_19, s1_19);
                float _unpack_fp4_4[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[4 + 0]));
                    (&_unpack_fp4_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_4[0]), "=f"(_unpack_fp4_4[1]), "=f"(_unpack_fp4_4[2]), "=f"(_unpack_fp4_4[3]), "=f"(_unpack_fp4_4[4]), "=f"(_unpack_fp4_4[5]), "=f"(_unpack_fp4_4[6]), "=f"(_unpack_fp4_4[7]) : "r"(_fp4_wire_values_0[4 + 0]));
                    #endif
                }
                float2 _f2_36 = make_float2(acc[0], acc[1]);
                float2 _f2_37 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                float2 r_20 = fma_f32x2_rn_noftz(_f2_37, _f2_34, _f2_36);
                acc[0] = r_20.x;
                acc[1] = r_20.y;
                float2 _f2_38 = make_float2(acc[2], acc[3]);
                float2 _f2_39 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                float2 r_21 = fma_f32x2_rn_noftz(_f2_39, _f2_34, _f2_38);
                acc[2] = r_21.x;
                acc[3] = r_21.y;
                float2 _f2_40 = make_float2(acc[4], acc[5]);
                float2 _f2_41 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                float2 r_22 = fma_f32x2_rn_noftz(_f2_41, _f2_34, _f2_40);
                acc[4] = r_22.x;
                acc[5] = r_22.y;
                float2 _f2_42 = make_float2(acc[6], acc[7]);
                float2 _f2_43 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                float2 r_23 = fma_f32x2_rn_noftz(_f2_43, _f2_34, _f2_42);
                acc[6] = r_23.x;
                acc[7] = r_23.y;
                float _unpack_fp4_5[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[5 + 0]));
                    (&_unpack_fp4_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_5[0]), "=f"(_unpack_fp4_5[1]), "=f"(_unpack_fp4_5[2]), "=f"(_unpack_fp4_5[3]), "=f"(_unpack_fp4_5[4]), "=f"(_unpack_fp4_5[5]), "=f"(_unpack_fp4_5[6]), "=f"(_unpack_fp4_5[7]) : "r"(_fp4_wire_values_0[5 + 0]));
                    #endif
                }
                float2 _f2_44 = make_float2(acc[8], acc[9]);
                float2 _f2_45 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                float2 r_24 = fma_f32x2_rn_noftz(_f2_45, _f2_34, _f2_44);
                acc[8] = r_24.x;
                acc[9] = r_24.y;
                float2 _f2_46 = make_float2(acc[10], acc[11]);
                float2 _f2_47 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                float2 r_25 = fma_f32x2_rn_noftz(_f2_47, _f2_34, _f2_46);
                acc[10] = r_25.x;
                acc[11] = r_25.y;
                float2 _f2_48 = make_float2(acc[12], acc[13]);
                float2 _f2_49 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                float2 r_26 = fma_f32x2_rn_noftz(_f2_49, _f2_34, _f2_48);
                acc[12] = r_26.x;
                acc[13] = r_26.y;
                float2 _f2_50 = make_float2(acc[14], acc[15]);
                float2 _f2_51 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                float2 r_27 = fma_f32x2_rn_noftz(_f2_51, _f2_34, _f2_50);
                acc[14] = r_27.x;
                acc[15] = r_27.y;
                float _unpack_fp4_6[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[6 + 0]));
                    (&_unpack_fp4_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_6[0]), "=f"(_unpack_fp4_6[1]), "=f"(_unpack_fp4_6[2]), "=f"(_unpack_fp4_6[3]), "=f"(_unpack_fp4_6[4]), "=f"(_unpack_fp4_6[5]), "=f"(_unpack_fp4_6[6]), "=f"(_unpack_fp4_6[7]) : "r"(_fp4_wire_values_0[6 + 0]));
                    #endif
                }
                float2 _f2_52 = make_float2(acc[16], acc[17]);
                float2 _f2_53 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                float2 r_28 = fma_f32x2_rn_noftz(_f2_53, _f2_35, _f2_52);
                acc[16] = r_28.x;
                acc[17] = r_28.y;
                float2 _f2_54 = make_float2(acc[18], acc[19]);
                float2 _f2_55 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                float2 r_29 = fma_f32x2_rn_noftz(_f2_55, _f2_35, _f2_54);
                acc[18] = r_29.x;
                acc[19] = r_29.y;
                float2 _f2_56 = make_float2(acc[20], acc[21]);
                float2 _f2_57 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                float2 r_30 = fma_f32x2_rn_noftz(_f2_57, _f2_35, _f2_56);
                acc[20] = r_30.x;
                acc[21] = r_30.y;
                float2 _f2_58 = make_float2(acc[22], acc[23]);
                float2 _f2_59 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                float2 r_31 = fma_f32x2_rn_noftz(_f2_59, _f2_35, _f2_58);
                acc[22] = r_31.x;
                acc[23] = r_31.y;
                float _unpack_fp4_7[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[7 + 0]));
                    (&_unpack_fp4_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_7[0]), "=f"(_unpack_fp4_7[1]), "=f"(_unpack_fp4_7[2]), "=f"(_unpack_fp4_7[3]), "=f"(_unpack_fp4_7[4]), "=f"(_unpack_fp4_7[5]), "=f"(_unpack_fp4_7[6]), "=f"(_unpack_fp4_7[7]) : "r"(_fp4_wire_values_0[7 + 0]));
                    #endif
                }
                float2 _f2_60 = make_float2(acc[24], acc[25]);
                float2 _f2_61 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                float2 r_32 = fma_f32x2_rn_noftz(_f2_61, _f2_35, _f2_60);
                acc[24] = r_32.x;
                acc[25] = r_32.y;
                float2 _f2_62 = make_float2(acc[26], acc[27]);
                float2 _f2_63 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                float2 r_33 = fma_f32x2_rn_noftz(_f2_63, _f2_35, _f2_62);
                acc[26] = r_33.x;
                acc[27] = r_33.y;
                float2 _f2_64 = make_float2(acc[28], acc[29]);
                float2 _f2_65 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                float2 r_34 = fma_f32x2_rn_noftz(_f2_65, _f2_35, _f2_64);
                acc[28] = r_34.x;
                acc[29] = r_34.y;
                float2 _f2_66 = make_float2(acc[30], acc[31]);
                float2 _f2_67 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                float2 r_35 = fma_f32x2_rn_noftz(_f2_67, _f2_35, _f2_66);
                acc[30] = r_35.x;
                acc[31] = r_35.y;
                float2 _fp8x2_decode_2;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_2.x), "=f"(_fp8x2_decode_2.y) : "h"((uint16_t)_fp4_wire_scales_0[2]));
                float _mul_4;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_fp8x2_decode_2.x), "f"(g_1));
                float s0_36 = _mul_4;
                float _mul_5;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_fp8x2_decode_2.y), "f"(g_1));
                float s1_37 = _mul_5;
                float2 _f2_68 = make_float2(s0_36, s0_36);
                float2 _f2_69 = make_float2(s1_37, s1_37);
                float _unpack_fp4_8[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[8 + 0]));
                    (&_unpack_fp4_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_8[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_8[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_8[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_8[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_8[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_8[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_8[0]), "=f"(_unpack_fp4_8[1]), "=f"(_unpack_fp4_8[2]), "=f"(_unpack_fp4_8[3]), "=f"(_unpack_fp4_8[4]), "=f"(_unpack_fp4_8[5]), "=f"(_unpack_fp4_8[6]), "=f"(_unpack_fp4_8[7]) : "r"(_fp4_wire_values_0[8 + 0]));
                    #endif
                }
                float2 _f2_70 = make_float2(acc[0], acc[1]);
                float2 _f2_71 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                float2 r_38 = fma_f32x2_rn_noftz(_f2_71, _f2_68, _f2_70);
                acc[0] = r_38.x;
                acc[1] = r_38.y;
                float2 _f2_72 = make_float2(acc[2], acc[3]);
                float2 _f2_73 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                float2 r_39 = fma_f32x2_rn_noftz(_f2_73, _f2_68, _f2_72);
                acc[2] = r_39.x;
                acc[3] = r_39.y;
                float2 _f2_74 = make_float2(acc[4], acc[5]);
                float2 _f2_75 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                float2 r_40 = fma_f32x2_rn_noftz(_f2_75, _f2_68, _f2_74);
                acc[4] = r_40.x;
                acc[5] = r_40.y;
                float2 _f2_76 = make_float2(acc[6], acc[7]);
                float2 _f2_77 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                float2 r_41 = fma_f32x2_rn_noftz(_f2_77, _f2_68, _f2_76);
                acc[6] = r_41.x;
                acc[7] = r_41.y;
                float _unpack_fp4_9[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[9 + 0]));
                    (&_unpack_fp4_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_9[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_9[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_9[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_9[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_9[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_9[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_9[0]), "=f"(_unpack_fp4_9[1]), "=f"(_unpack_fp4_9[2]), "=f"(_unpack_fp4_9[3]), "=f"(_unpack_fp4_9[4]), "=f"(_unpack_fp4_9[5]), "=f"(_unpack_fp4_9[6]), "=f"(_unpack_fp4_9[7]) : "r"(_fp4_wire_values_0[9 + 0]));
                    #endif
                }
                float2 _f2_78 = make_float2(acc[8], acc[9]);
                float2 _f2_79 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                float2 r_42 = fma_f32x2_rn_noftz(_f2_79, _f2_68, _f2_78);
                acc[8] = r_42.x;
                acc[9] = r_42.y;
                float2 _f2_80 = make_float2(acc[10], acc[11]);
                float2 _f2_81 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                float2 r_43 = fma_f32x2_rn_noftz(_f2_81, _f2_68, _f2_80);
                acc[10] = r_43.x;
                acc[11] = r_43.y;
                float2 _f2_82 = make_float2(acc[12], acc[13]);
                float2 _f2_83 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                float2 r_44 = fma_f32x2_rn_noftz(_f2_83, _f2_68, _f2_82);
                acc[12] = r_44.x;
                acc[13] = r_44.y;
                float2 _f2_84 = make_float2(acc[14], acc[15]);
                float2 _f2_85 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                float2 r_45 = fma_f32x2_rn_noftz(_f2_85, _f2_68, _f2_84);
                acc[14] = r_45.x;
                acc[15] = r_45.y;
                float _unpack_fp4_10[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[10 + 0]));
                    (&_unpack_fp4_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_10[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_10[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_10[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_10[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_10[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_10[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_10[0]), "=f"(_unpack_fp4_10[1]), "=f"(_unpack_fp4_10[2]), "=f"(_unpack_fp4_10[3]), "=f"(_unpack_fp4_10[4]), "=f"(_unpack_fp4_10[5]), "=f"(_unpack_fp4_10[6]), "=f"(_unpack_fp4_10[7]) : "r"(_fp4_wire_values_0[10 + 0]));
                    #endif
                }
                float2 _f2_86 = make_float2(acc[16], acc[17]);
                float2 _f2_87 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                float2 r_46 = fma_f32x2_rn_noftz(_f2_87, _f2_69, _f2_86);
                acc[16] = r_46.x;
                acc[17] = r_46.y;
                float2 _f2_88 = make_float2(acc[18], acc[19]);
                float2 _f2_89 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                float2 r_47 = fma_f32x2_rn_noftz(_f2_89, _f2_69, _f2_88);
                acc[18] = r_47.x;
                acc[19] = r_47.y;
                float2 _f2_90 = make_float2(acc[20], acc[21]);
                float2 _f2_91 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                float2 r_48 = fma_f32x2_rn_noftz(_f2_91, _f2_69, _f2_90);
                acc[20] = r_48.x;
                acc[21] = r_48.y;
                float2 _f2_92 = make_float2(acc[22], acc[23]);
                float2 _f2_93 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                float2 r_49 = fma_f32x2_rn_noftz(_f2_93, _f2_69, _f2_92);
                acc[22] = r_49.x;
                acc[23] = r_49.y;
                float _unpack_fp4_11[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[11 + 0]));
                    (&_unpack_fp4_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_11[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_11[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_11[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_11[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_11[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_11[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_11[0]), "=f"(_unpack_fp4_11[1]), "=f"(_unpack_fp4_11[2]), "=f"(_unpack_fp4_11[3]), "=f"(_unpack_fp4_11[4]), "=f"(_unpack_fp4_11[5]), "=f"(_unpack_fp4_11[6]), "=f"(_unpack_fp4_11[7]) : "r"(_fp4_wire_values_0[11 + 0]));
                    #endif
                }
                float2 _f2_94 = make_float2(acc[24], acc[25]);
                float2 _f2_95 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                float2 r_50 = fma_f32x2_rn_noftz(_f2_95, _f2_69, _f2_94);
                acc[24] = r_50.x;
                acc[25] = r_50.y;
                float2 _f2_96 = make_float2(acc[26], acc[27]);
                float2 _f2_97 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                float2 r_51 = fma_f32x2_rn_noftz(_f2_97, _f2_69, _f2_96);
                acc[26] = r_51.x;
                acc[27] = r_51.y;
                float2 _f2_98 = make_float2(acc[28], acc[29]);
                float2 _f2_99 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                float2 r_52 = fma_f32x2_rn_noftz(_f2_99, _f2_69, _f2_98);
                acc[28] = r_52.x;
                acc[29] = r_52.y;
                float2 _f2_100 = make_float2(acc[30], acc[31]);
                float2 _f2_101 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                float2 r_53 = fma_f32x2_rn_noftz(_f2_101, _f2_69, _f2_100);
                acc[30] = r_53.x;
                acc[31] = r_53.y;
                float2 _fp8x2_decode_3;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_3.x), "=f"(_fp8x2_decode_3.y) : "h"((uint16_t)_fp4_wire_scales_0[3]));
                float _mul_6;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_fp8x2_decode_3.x), "f"(g_2));
                float s0_54 = _mul_6;
                float _mul_7;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_fp8x2_decode_3.y), "f"(g_2));
                float s1_55 = _mul_7;
                float2 _f2_102 = make_float2(s0_54, s0_54);
                float2 _f2_103 = make_float2(s1_55, s1_55);
                float _unpack_fp4_12[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[12 + 0]));
                    (&_unpack_fp4_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_12[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_12[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_12[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_12[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_12[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_12[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_12[0]), "=f"(_unpack_fp4_12[1]), "=f"(_unpack_fp4_12[2]), "=f"(_unpack_fp4_12[3]), "=f"(_unpack_fp4_12[4]), "=f"(_unpack_fp4_12[5]), "=f"(_unpack_fp4_12[6]), "=f"(_unpack_fp4_12[7]) : "r"(_fp4_wire_values_0[12 + 0]));
                    #endif
                }
                float2 _f2_104 = make_float2(acc[0], acc[1]);
                float2 _f2_105 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                float2 r_56 = fma_f32x2_rn_noftz(_f2_105, _f2_102, _f2_104);
                acc[0] = r_56.x;
                acc[1] = r_56.y;
                float2 _f2_106 = make_float2(acc[2], acc[3]);
                float2 _f2_107 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                float2 r_57 = fma_f32x2_rn_noftz(_f2_107, _f2_102, _f2_106);
                acc[2] = r_57.x;
                acc[3] = r_57.y;
                float2 _f2_108 = make_float2(acc[4], acc[5]);
                float2 _f2_109 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                float2 r_58 = fma_f32x2_rn_noftz(_f2_109, _f2_102, _f2_108);
                acc[4] = r_58.x;
                acc[5] = r_58.y;
                float2 _f2_110 = make_float2(acc[6], acc[7]);
                float2 _f2_111 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                float2 r_59 = fma_f32x2_rn_noftz(_f2_111, _f2_102, _f2_110);
                acc[6] = r_59.x;
                acc[7] = r_59.y;
                float _unpack_fp4_13[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[13 + 0]));
                    (&_unpack_fp4_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_13[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_13[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_13[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_13[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_13[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_13[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_13[0]), "=f"(_unpack_fp4_13[1]), "=f"(_unpack_fp4_13[2]), "=f"(_unpack_fp4_13[3]), "=f"(_unpack_fp4_13[4]), "=f"(_unpack_fp4_13[5]), "=f"(_unpack_fp4_13[6]), "=f"(_unpack_fp4_13[7]) : "r"(_fp4_wire_values_0[13 + 0]));
                    #endif
                }
                float2 _f2_112 = make_float2(acc[8], acc[9]);
                float2 _f2_113 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                float2 r_60 = fma_f32x2_rn_noftz(_f2_113, _f2_102, _f2_112);
                acc[8] = r_60.x;
                acc[9] = r_60.y;
                float2 _f2_114 = make_float2(acc[10], acc[11]);
                float2 _f2_115 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                float2 r_61 = fma_f32x2_rn_noftz(_f2_115, _f2_102, _f2_114);
                acc[10] = r_61.x;
                acc[11] = r_61.y;
                float2 _f2_116 = make_float2(acc[12], acc[13]);
                float2 _f2_117 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                float2 r_62 = fma_f32x2_rn_noftz(_f2_117, _f2_102, _f2_116);
                acc[12] = r_62.x;
                acc[13] = r_62.y;
                float2 _f2_118 = make_float2(acc[14], acc[15]);
                float2 _f2_119 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                float2 r_63 = fma_f32x2_rn_noftz(_f2_119, _f2_102, _f2_118);
                acc[14] = r_63.x;
                acc[15] = r_63.y;
                float _unpack_fp4_14[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[14 + 0]));
                    (&_unpack_fp4_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_14[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_14[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_14[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_14[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_14[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_14[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_14[0]), "=f"(_unpack_fp4_14[1]), "=f"(_unpack_fp4_14[2]), "=f"(_unpack_fp4_14[3]), "=f"(_unpack_fp4_14[4]), "=f"(_unpack_fp4_14[5]), "=f"(_unpack_fp4_14[6]), "=f"(_unpack_fp4_14[7]) : "r"(_fp4_wire_values_0[14 + 0]));
                    #endif
                }
                float2 _f2_120 = make_float2(acc[16], acc[17]);
                float2 _f2_121 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                float2 r_64 = fma_f32x2_rn_noftz(_f2_121, _f2_103, _f2_120);
                acc[16] = r_64.x;
                acc[17] = r_64.y;
                float2 _f2_122 = make_float2(acc[18], acc[19]);
                float2 _f2_123 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                float2 r_65 = fma_f32x2_rn_noftz(_f2_123, _f2_103, _f2_122);
                acc[18] = r_65.x;
                acc[19] = r_65.y;
                float2 _f2_124 = make_float2(acc[20], acc[21]);
                float2 _f2_125 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                float2 r_66 = fma_f32x2_rn_noftz(_f2_125, _f2_103, _f2_124);
                acc[20] = r_66.x;
                acc[21] = r_66.y;
                float2 _f2_126 = make_float2(acc[22], acc[23]);
                float2 _f2_127 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                float2 r_67 = fma_f32x2_rn_noftz(_f2_127, _f2_103, _f2_126);
                acc[22] = r_67.x;
                acc[23] = r_67.y;
                float _unpack_fp4_15[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[15 + 0]));
                    (&_unpack_fp4_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_15[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_15[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_15[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_15[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_15[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_15[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_15[0]), "=f"(_unpack_fp4_15[1]), "=f"(_unpack_fp4_15[2]), "=f"(_unpack_fp4_15[3]), "=f"(_unpack_fp4_15[4]), "=f"(_unpack_fp4_15[5]), "=f"(_unpack_fp4_15[6]), "=f"(_unpack_fp4_15[7]) : "r"(_fp4_wire_values_0[15 + 0]));
                    #endif
                }
                float2 _f2_128 = make_float2(acc[24], acc[25]);
                float2 _f2_129 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                float2 r_68 = fma_f32x2_rn_noftz(_f2_129, _f2_103, _f2_128);
                acc[24] = r_68.x;
                acc[25] = r_68.y;
                float2 _f2_130 = make_float2(acc[26], acc[27]);
                float2 _f2_131 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                float2 r_69 = fma_f32x2_rn_noftz(_f2_131, _f2_103, _f2_130);
                acc[26] = r_69.x;
                acc[27] = r_69.y;
                float2 _f2_132 = make_float2(acc[28], acc[29]);
                float2 _f2_133 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                float2 r_70 = fma_f32x2_rn_noftz(_f2_133, _f2_103, _f2_132);
                acc[28] = r_70.x;
                acc[29] = r_70.y;
                float2 _f2_134 = make_float2(acc[30], acc[31]);
                float2 _f2_135 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                float2 r_71 = fma_f32x2_rn_noftz(_f2_135, _f2_103, _f2_134);
                acc[30] = r_71.x;
                acc[31] = r_71.y;
                int pp_72 = lane_1 + 32 * (3 * h + 1);
                uint32_t _fp4_wire_values_1[16];
                uint32_t _fp4_wire_scales_1[4];
                uint16_t _wire_h16_9_0;
                uint16_t _wire_h16_9_1;
                uint16_t _wire_h16_9_2;
                uint16_t _wire_h16_9_3;
                uint32_t _wire_vpend_9 = 15u;
                uint32_t _wire_spend_9 = 15u;
                do {
                    if (_wire_vpend_9 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[0]), "=r"(_fp4_wire_values_1[1]), "=r"(_fp4_wire_values_1[2]), "=r"(_fp4_wire_values_1[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)rs_base + (int)tix64 * 3072 + 16 * pp_72))) : "memory");
                    }
                    if (_wire_vpend_9 & 2u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[4]), "=r"(_fp4_wire_values_1[5]), "=r"(_fp4_wire_values_1[6]), "=r"(_fp4_wire_values_1[7]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)rs_base + (4096 + (int)tix64) * 3072 + 16 * pp_72))) : "memory");
                    }
                    if (_wire_vpend_9 & 4u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[8]), "=r"(_fp4_wire_values_1[9]), "=r"(_fp4_wire_values_1[10]), "=r"(_fp4_wire_values_1[11]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)rs_base + (8192 + (int)tix64) * 3072 + 16 * pp_72))) : "memory");
                    }
                    if (_wire_vpend_9 & 8u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[12]), "=r"(_fp4_wire_values_1[13]), "=r"(_fp4_wire_values_1[14]), "=r"(_fp4_wire_values_1[15]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)rs_base + (12288 + (int)tix64) * 3072 + 16 * pp_72))) : "memory");
                    }
                    if (_wire_spend_9 & 1u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_9_0) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)rs_base + 50331648 + (int)tix64 * 384 + 2 * pp_72))) : "memory");
                        _fp4_wire_scales_1[0] = (uint32_t)_wire_h16_9_0;
                    }
                    if (_wire_spend_9 & 2u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_9_1) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)rs_base + 50331648 + (4096 + (int)tix64) * 384 + 2 * pp_72))) : "memory");
                        _fp4_wire_scales_1[1] = (uint32_t)_wire_h16_9_1;
                    }
                    if (_wire_spend_9 & 4u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_9_2) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)rs_base + 50331648 + (8192 + (int)tix64) * 384 + 2 * pp_72))) : "memory");
                        _fp4_wire_scales_1[2] = (uint32_t)_wire_h16_9_2;
                    }
                    if (_wire_spend_9 & 8u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_9_3) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)rs_base + 50331648 + (12288 + (int)tix64) * 384 + 2 * pp_72))) : "memory");
                        _fp4_wire_scales_1[3] = (uint32_t)_wire_h16_9_3;
                    }
                    if (_wire_vpend_9 & 1u) {
                        if (!((((_fp4_wire_values_1[0] & ~((_fp4_wire_values_1[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[1] & ~((_fp4_wire_values_1[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[2] & ~((_fp4_wire_values_1[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[3] & ~((_fp4_wire_values_1[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_9 &= ~1u;
                    }
                    if (_wire_vpend_9 & 2u) {
                        if (!((((_fp4_wire_values_1[4] & ~((_fp4_wire_values_1[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[5] & ~((_fp4_wire_values_1[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[6] & ~((_fp4_wire_values_1[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[7] & ~((_fp4_wire_values_1[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_9 &= ~2u;
                    }
                    if (_wire_vpend_9 & 4u) {
                        if (!((((_fp4_wire_values_1[8] & ~((_fp4_wire_values_1[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[9] & ~((_fp4_wire_values_1[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[10] & ~((_fp4_wire_values_1[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[11] & ~((_fp4_wire_values_1[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_9 &= ~4u;
                    }
                    if (_wire_vpend_9 & 8u) {
                        if (!((((_fp4_wire_values_1[12] & ~((_fp4_wire_values_1[12] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[13] & ~((_fp4_wire_values_1[13] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[14] & ~((_fp4_wire_values_1[14] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[15] & ~((_fp4_wire_values_1[15] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_9 &= ~8u;
                    }
                    if (_wire_spend_9 & 1u) {
                        if (!(((_fp4_wire_scales_1[0] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_1[0] >> 8) == 0xFFu))) _wire_spend_9 &= ~1u;
                    }
                    if (_wire_spend_9 & 2u) {
                        if (!(((_fp4_wire_scales_1[1] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_1[1] >> 8) == 0xFFu))) _wire_spend_9 &= ~2u;
                    }
                    if (_wire_spend_9 & 4u) {
                        if (!(((_fp4_wire_scales_1[2] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_1[2] >> 8) == 0xFFu))) _wire_spend_9 &= ~4u;
                    }
                    if (_wire_spend_9 & 8u) {
                        if (!(((_fp4_wire_scales_1[3] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_1[3] >> 8) == 0xFFu))) _wire_spend_9 &= ~8u;
                    }
                } while ((_wire_vpend_9 | _wire_spend_9) != 0u);
                float acc_73[32];
                acc_73[0] = 0.0f;
                acc_73[1] = 0.0f;
                acc_73[2] = 0.0f;
                acc_73[3] = 0.0f;
                acc_73[4] = 0.0f;
                acc_73[5] = 0.0f;
                acc_73[6] = 0.0f;
                acc_73[7] = 0.0f;
                acc_73[8] = 0.0f;
                acc_73[9] = 0.0f;
                acc_73[10] = 0.0f;
                acc_73[11] = 0.0f;
                acc_73[12] = 0.0f;
                acc_73[13] = 0.0f;
                acc_73[14] = 0.0f;
                acc_73[15] = 0.0f;
                acc_73[16] = 0.0f;
                acc_73[17] = 0.0f;
                acc_73[18] = 0.0f;
                acc_73[19] = 0.0f;
                acc_73[20] = 0.0f;
                acc_73[21] = 0.0f;
                acc_73[22] = 0.0f;
                acc_73[23] = 0.0f;
                acc_73[24] = 0.0f;
                acc_73[25] = 0.0f;
                acc_73[26] = 0.0f;
                acc_73[27] = 0.0f;
                acc_73[28] = 0.0f;
                acc_73[29] = 0.0f;
                acc_73[30] = 0.0f;
                acc_73[31] = 0.0f;
                float2 _fp8x2_decode_4;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_4.x), "=f"(_fp8x2_decode_4.y) : "h"((uint16_t)_fp4_wire_scales_1[0]));
                float _mul_8;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_fp8x2_decode_4.x), "f"(g));
                float s0_74 = _mul_8;
                float _mul_9;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_fp8x2_decode_4.y), "f"(g));
                float s1_75 = _mul_9;
                float2 _f2_136 = make_float2(s0_74, s0_74);
                float2 _f2_137 = make_float2(s1_75, s1_75);
                float _unpack_fp4_16[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[0]));
                    (&_unpack_fp4_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_16[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_16[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_16[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_16[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_16[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_16[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_16[0]), "=f"(_unpack_fp4_16[1]), "=f"(_unpack_fp4_16[2]), "=f"(_unpack_fp4_16[3]), "=f"(_unpack_fp4_16[4]), "=f"(_unpack_fp4_16[5]), "=f"(_unpack_fp4_16[6]), "=f"(_unpack_fp4_16[7]) : "r"(_fp4_wire_values_1[0]));
                    #endif
                }
                float2 _f2_138 = make_float2(acc_73[0], acc_73[1]);
                float2 _f2_139 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                float2 r_76 = fma_f32x2_rn_noftz(_f2_139, _f2_136, _f2_138);
                acc_73[0] = r_76.x;
                acc_73[1] = r_76.y;
                float2 _f2_140 = make_float2(acc_73[2], acc_73[3]);
                float2 _f2_141 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                float2 r_77 = fma_f32x2_rn_noftz(_f2_141, _f2_136, _f2_140);
                acc_73[2] = r_77.x;
                acc_73[3] = r_77.y;
                float2 _f2_142 = make_float2(acc_73[4], acc_73[5]);
                float2 _f2_143 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                float2 r_78 = fma_f32x2_rn_noftz(_f2_143, _f2_136, _f2_142);
                acc_73[4] = r_78.x;
                acc_73[5] = r_78.y;
                float2 _f2_144 = make_float2(acc_73[6], acc_73[7]);
                float2 _f2_145 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                float2 r_79 = fma_f32x2_rn_noftz(_f2_145, _f2_136, _f2_144);
                acc_73[6] = r_79.x;
                acc_73[7] = r_79.y;
                float _unpack_fp4_17[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[1 + 0]));
                    (&_unpack_fp4_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_17[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_17[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_17[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_17[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_17[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_17[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_17[0]), "=f"(_unpack_fp4_17[1]), "=f"(_unpack_fp4_17[2]), "=f"(_unpack_fp4_17[3]), "=f"(_unpack_fp4_17[4]), "=f"(_unpack_fp4_17[5]), "=f"(_unpack_fp4_17[6]), "=f"(_unpack_fp4_17[7]) : "r"(_fp4_wire_values_1[1 + 0]));
                    #endif
                }
                float2 _f2_146 = make_float2(acc_73[8], acc_73[9]);
                float2 _f2_147 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                float2 r_80 = fma_f32x2_rn_noftz(_f2_147, _f2_136, _f2_146);
                acc_73[8] = r_80.x;
                acc_73[9] = r_80.y;
                float2 _f2_148 = make_float2(acc_73[10], acc_73[11]);
                float2 _f2_149 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                float2 r_81 = fma_f32x2_rn_noftz(_f2_149, _f2_136, _f2_148);
                acc_73[10] = r_81.x;
                acc_73[11] = r_81.y;
                float2 _f2_150 = make_float2(acc_73[12], acc_73[13]);
                float2 _f2_151 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                float2 r_82 = fma_f32x2_rn_noftz(_f2_151, _f2_136, _f2_150);
                acc_73[12] = r_82.x;
                acc_73[13] = r_82.y;
                float2 _f2_152 = make_float2(acc_73[14], acc_73[15]);
                float2 _f2_153 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                float2 r_83 = fma_f32x2_rn_noftz(_f2_153, _f2_136, _f2_152);
                acc_73[14] = r_83.x;
                acc_73[15] = r_83.y;
                float _unpack_fp4_18[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[2 + 0]));
                    (&_unpack_fp4_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_18[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_18[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_18[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_18[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_18[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_18[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_18[0]), "=f"(_unpack_fp4_18[1]), "=f"(_unpack_fp4_18[2]), "=f"(_unpack_fp4_18[3]), "=f"(_unpack_fp4_18[4]), "=f"(_unpack_fp4_18[5]), "=f"(_unpack_fp4_18[6]), "=f"(_unpack_fp4_18[7]) : "r"(_fp4_wire_values_1[2 + 0]));
                    #endif
                }
                float2 _f2_154 = make_float2(acc_73[16], acc_73[17]);
                float2 _f2_155 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                float2 r_84 = fma_f32x2_rn_noftz(_f2_155, _f2_137, _f2_154);
                acc_73[16] = r_84.x;
                acc_73[17] = r_84.y;
                float2 _f2_156 = make_float2(acc_73[18], acc_73[19]);
                float2 _f2_157 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                float2 r_85 = fma_f32x2_rn_noftz(_f2_157, _f2_137, _f2_156);
                acc_73[18] = r_85.x;
                acc_73[19] = r_85.y;
                float2 _f2_158 = make_float2(acc_73[20], acc_73[21]);
                float2 _f2_159 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                float2 r_86 = fma_f32x2_rn_noftz(_f2_159, _f2_137, _f2_158);
                acc_73[20] = r_86.x;
                acc_73[21] = r_86.y;
                float2 _f2_160 = make_float2(acc_73[22], acc_73[23]);
                float2 _f2_161 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                float2 r_87 = fma_f32x2_rn_noftz(_f2_161, _f2_137, _f2_160);
                acc_73[22] = r_87.x;
                acc_73[23] = r_87.y;
                float _unpack_fp4_19[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[3 + 0]));
                    (&_unpack_fp4_19[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_19[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_19[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_19[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_19[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_19[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_19[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_19[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_19[0]), "=f"(_unpack_fp4_19[1]), "=f"(_unpack_fp4_19[2]), "=f"(_unpack_fp4_19[3]), "=f"(_unpack_fp4_19[4]), "=f"(_unpack_fp4_19[5]), "=f"(_unpack_fp4_19[6]), "=f"(_unpack_fp4_19[7]) : "r"(_fp4_wire_values_1[3 + 0]));
                    #endif
                }
                float2 _f2_162 = make_float2(acc_73[24], acc_73[25]);
                float2 _f2_163 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                float2 r_88 = fma_f32x2_rn_noftz(_f2_163, _f2_137, _f2_162);
                acc_73[24] = r_88.x;
                acc_73[25] = r_88.y;
                float2 _f2_164 = make_float2(acc_73[26], acc_73[27]);
                float2 _f2_165 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                float2 r_89 = fma_f32x2_rn_noftz(_f2_165, _f2_137, _f2_164);
                acc_73[26] = r_89.x;
                acc_73[27] = r_89.y;
                float2 _f2_166 = make_float2(acc_73[28], acc_73[29]);
                float2 _f2_167 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                float2 r_90 = fma_f32x2_rn_noftz(_f2_167, _f2_137, _f2_166);
                acc_73[28] = r_90.x;
                acc_73[29] = r_90.y;
                float2 _f2_168 = make_float2(acc_73[30], acc_73[31]);
                float2 _f2_169 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                float2 r_91 = fma_f32x2_rn_noftz(_f2_169, _f2_137, _f2_168);
                acc_73[30] = r_91.x;
                acc_73[31] = r_91.y;
                float2 _fp8x2_decode_5;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_5.x), "=f"(_fp8x2_decode_5.y) : "h"((uint16_t)_fp4_wire_scales_1[1]));
                float _mul_10;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(_fp8x2_decode_5.x), "f"(g_0));
                float s0_92 = _mul_10;
                float _mul_11;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_fp8x2_decode_5.y), "f"(g_0));
                float s1_93 = _mul_11;
                float2 _f2_170 = make_float2(s0_92, s0_92);
                float2 _f2_171 = make_float2(s1_93, s1_93);
                float _unpack_fp4_20[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[4 + 0]));
                    (&_unpack_fp4_20[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_20[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_20[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_20[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_20[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_20[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_20[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_20[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_20[0]), "=f"(_unpack_fp4_20[1]), "=f"(_unpack_fp4_20[2]), "=f"(_unpack_fp4_20[3]), "=f"(_unpack_fp4_20[4]), "=f"(_unpack_fp4_20[5]), "=f"(_unpack_fp4_20[6]), "=f"(_unpack_fp4_20[7]) : "r"(_fp4_wire_values_1[4 + 0]));
                    #endif
                }
                float2 _f2_172 = make_float2(acc_73[0], acc_73[1]);
                float2 _f2_173 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                float2 r_94 = fma_f32x2_rn_noftz(_f2_173, _f2_170, _f2_172);
                acc_73[0] = r_94.x;
                acc_73[1] = r_94.y;
                float2 _f2_174 = make_float2(acc_73[2], acc_73[3]);
                float2 _f2_175 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                float2 r_95 = fma_f32x2_rn_noftz(_f2_175, _f2_170, _f2_174);
                acc_73[2] = r_95.x;
                acc_73[3] = r_95.y;
                float2 _f2_176 = make_float2(acc_73[4], acc_73[5]);
                float2 _f2_177 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                float2 r_96 = fma_f32x2_rn_noftz(_f2_177, _f2_170, _f2_176);
                acc_73[4] = r_96.x;
                acc_73[5] = r_96.y;
                float2 _f2_178 = make_float2(acc_73[6], acc_73[7]);
                float2 _f2_179 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                float2 r_97 = fma_f32x2_rn_noftz(_f2_179, _f2_170, _f2_178);
                acc_73[6] = r_97.x;
                acc_73[7] = r_97.y;
                float _unpack_fp4_21[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[5 + 0]));
                    (&_unpack_fp4_21[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_21[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_21[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_21[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_21[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_21[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_21[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_21[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_21[0]), "=f"(_unpack_fp4_21[1]), "=f"(_unpack_fp4_21[2]), "=f"(_unpack_fp4_21[3]), "=f"(_unpack_fp4_21[4]), "=f"(_unpack_fp4_21[5]), "=f"(_unpack_fp4_21[6]), "=f"(_unpack_fp4_21[7]) : "r"(_fp4_wire_values_1[5 + 0]));
                    #endif
                }
                float2 _f2_180 = make_float2(acc_73[8], acc_73[9]);
                float2 _f2_181 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                float2 r_98 = fma_f32x2_rn_noftz(_f2_181, _f2_170, _f2_180);
                acc_73[8] = r_98.x;
                acc_73[9] = r_98.y;
                float2 _f2_182 = make_float2(acc_73[10], acc_73[11]);
                float2 _f2_183 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                float2 r_99 = fma_f32x2_rn_noftz(_f2_183, _f2_170, _f2_182);
                acc_73[10] = r_99.x;
                acc_73[11] = r_99.y;
                float2 _f2_184 = make_float2(acc_73[12], acc_73[13]);
                float2 _f2_185 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                float2 r_100 = fma_f32x2_rn_noftz(_f2_185, _f2_170, _f2_184);
                acc_73[12] = r_100.x;
                acc_73[13] = r_100.y;
                float2 _f2_186 = make_float2(acc_73[14], acc_73[15]);
                float2 _f2_187 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                float2 r_101 = fma_f32x2_rn_noftz(_f2_187, _f2_170, _f2_186);
                acc_73[14] = r_101.x;
                acc_73[15] = r_101.y;
                float _unpack_fp4_22[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[6 + 0]));
                    (&_unpack_fp4_22[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_22[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_22[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_22[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_22[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_22[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_22[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_22[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_22[0]), "=f"(_unpack_fp4_22[1]), "=f"(_unpack_fp4_22[2]), "=f"(_unpack_fp4_22[3]), "=f"(_unpack_fp4_22[4]), "=f"(_unpack_fp4_22[5]), "=f"(_unpack_fp4_22[6]), "=f"(_unpack_fp4_22[7]) : "r"(_fp4_wire_values_1[6 + 0]));
                    #endif
                }
                float2 _f2_188 = make_float2(acc_73[16], acc_73[17]);
                float2 _f2_189 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                float2 r_102 = fma_f32x2_rn_noftz(_f2_189, _f2_171, _f2_188);
                acc_73[16] = r_102.x;
                acc_73[17] = r_102.y;
                float2 _f2_190 = make_float2(acc_73[18], acc_73[19]);
                float2 _f2_191 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                float2 r_103 = fma_f32x2_rn_noftz(_f2_191, _f2_171, _f2_190);
                acc_73[18] = r_103.x;
                acc_73[19] = r_103.y;
                float2 _f2_192 = make_float2(acc_73[20], acc_73[21]);
                float2 _f2_193 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                float2 r_104 = fma_f32x2_rn_noftz(_f2_193, _f2_171, _f2_192);
                acc_73[20] = r_104.x;
                acc_73[21] = r_104.y;
                float2 _f2_194 = make_float2(acc_73[22], acc_73[23]);
                float2 _f2_195 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                float2 r_105 = fma_f32x2_rn_noftz(_f2_195, _f2_171, _f2_194);
                acc_73[22] = r_105.x;
                acc_73[23] = r_105.y;
                float _unpack_fp4_23[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[7 + 0]));
                    (&_unpack_fp4_23[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_23[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_23[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_23[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_23[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_23[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_23[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_23[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_23[0]), "=f"(_unpack_fp4_23[1]), "=f"(_unpack_fp4_23[2]), "=f"(_unpack_fp4_23[3]), "=f"(_unpack_fp4_23[4]), "=f"(_unpack_fp4_23[5]), "=f"(_unpack_fp4_23[6]), "=f"(_unpack_fp4_23[7]) : "r"(_fp4_wire_values_1[7 + 0]));
                    #endif
                }
                float2 _f2_196 = make_float2(acc_73[24], acc_73[25]);
                float2 _f2_197 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                float2 r_106 = fma_f32x2_rn_noftz(_f2_197, _f2_171, _f2_196);
                acc_73[24] = r_106.x;
                acc_73[25] = r_106.y;
                float2 _f2_198 = make_float2(acc_73[26], acc_73[27]);
                float2 _f2_199 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                float2 r_107 = fma_f32x2_rn_noftz(_f2_199, _f2_171, _f2_198);
                acc_73[26] = r_107.x;
                acc_73[27] = r_107.y;
                float2 _f2_200 = make_float2(acc_73[28], acc_73[29]);
                float2 _f2_201 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                float2 r_108 = fma_f32x2_rn_noftz(_f2_201, _f2_171, _f2_200);
                acc_73[28] = r_108.x;
                acc_73[29] = r_108.y;
                float2 _f2_202 = make_float2(acc_73[30], acc_73[31]);
                float2 _f2_203 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                float2 r_109 = fma_f32x2_rn_noftz(_f2_203, _f2_171, _f2_202);
                acc_73[30] = r_109.x;
                acc_73[31] = r_109.y;
                float2 _fp8x2_decode_6;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_6.x), "=f"(_fp8x2_decode_6.y) : "h"((uint16_t)_fp4_wire_scales_1[2]));
                float _mul_12;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(_fp8x2_decode_6.x), "f"(g_1));
                float s0_110 = _mul_12;
                float _mul_13;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(_fp8x2_decode_6.y), "f"(g_1));
                float s1_111 = _mul_13;
                float2 _f2_204 = make_float2(s0_110, s0_110);
                float2 _f2_205 = make_float2(s1_111, s1_111);
                float _unpack_fp4_24[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[8 + 0]));
                    (&_unpack_fp4_24[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_24[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_24[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_24[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_24[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_24[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_24[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_24[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_24[0]), "=f"(_unpack_fp4_24[1]), "=f"(_unpack_fp4_24[2]), "=f"(_unpack_fp4_24[3]), "=f"(_unpack_fp4_24[4]), "=f"(_unpack_fp4_24[5]), "=f"(_unpack_fp4_24[6]), "=f"(_unpack_fp4_24[7]) : "r"(_fp4_wire_values_1[8 + 0]));
                    #endif
                }
                float2 _f2_206 = make_float2(acc_73[0], acc_73[1]);
                float2 _f2_207 = make_float2(_unpack_fp4_24[0], _unpack_fp4_24[1]);
                float2 r_112 = fma_f32x2_rn_noftz(_f2_207, _f2_204, _f2_206);
                acc_73[0] = r_112.x;
                acc_73[1] = r_112.y;
                float2 _f2_208 = make_float2(acc_73[2], acc_73[3]);
                float2 _f2_209 = make_float2(_unpack_fp4_24[2], _unpack_fp4_24[3]);
                float2 r_113 = fma_f32x2_rn_noftz(_f2_209, _f2_204, _f2_208);
                acc_73[2] = r_113.x;
                acc_73[3] = r_113.y;
                float2 _f2_210 = make_float2(acc_73[4], acc_73[5]);
                float2 _f2_211 = make_float2(_unpack_fp4_24[4], _unpack_fp4_24[5]);
                float2 r_114 = fma_f32x2_rn_noftz(_f2_211, _f2_204, _f2_210);
                acc_73[4] = r_114.x;
                acc_73[5] = r_114.y;
                float2 _f2_212 = make_float2(acc_73[6], acc_73[7]);
                float2 _f2_213 = make_float2(_unpack_fp4_24[6], _unpack_fp4_24[7]);
                float2 r_115 = fma_f32x2_rn_noftz(_f2_213, _f2_204, _f2_212);
                acc_73[6] = r_115.x;
                acc_73[7] = r_115.y;
                float _unpack_fp4_25[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[9 + 0]));
                    (&_unpack_fp4_25[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_25[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_25[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_25[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_25[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_25[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_25[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_25[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_25[0]), "=f"(_unpack_fp4_25[1]), "=f"(_unpack_fp4_25[2]), "=f"(_unpack_fp4_25[3]), "=f"(_unpack_fp4_25[4]), "=f"(_unpack_fp4_25[5]), "=f"(_unpack_fp4_25[6]), "=f"(_unpack_fp4_25[7]) : "r"(_fp4_wire_values_1[9 + 0]));
                    #endif
                }
                float2 _f2_214 = make_float2(acc_73[8], acc_73[9]);
                float2 _f2_215 = make_float2(_unpack_fp4_25[0], _unpack_fp4_25[1]);
                float2 r_116 = fma_f32x2_rn_noftz(_f2_215, _f2_204, _f2_214);
                acc_73[8] = r_116.x;
                acc_73[9] = r_116.y;
                float2 _f2_216 = make_float2(acc_73[10], acc_73[11]);
                float2 _f2_217 = make_float2(_unpack_fp4_25[2], _unpack_fp4_25[3]);
                float2 r_117 = fma_f32x2_rn_noftz(_f2_217, _f2_204, _f2_216);
                acc_73[10] = r_117.x;
                acc_73[11] = r_117.y;
                float2 _f2_218 = make_float2(acc_73[12], acc_73[13]);
                float2 _f2_219 = make_float2(_unpack_fp4_25[4], _unpack_fp4_25[5]);
                float2 r_118 = fma_f32x2_rn_noftz(_f2_219, _f2_204, _f2_218);
                acc_73[12] = r_118.x;
                acc_73[13] = r_118.y;
                float2 _f2_220 = make_float2(acc_73[14], acc_73[15]);
                float2 _f2_221 = make_float2(_unpack_fp4_25[6], _unpack_fp4_25[7]);
                float2 r_119 = fma_f32x2_rn_noftz(_f2_221, _f2_204, _f2_220);
                acc_73[14] = r_119.x;
                acc_73[15] = r_119.y;
                float _unpack_fp4_26[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[10 + 0]));
                    (&_unpack_fp4_26[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_26[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_26[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_26[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_26[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_26[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_26[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_26[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_26[0]), "=f"(_unpack_fp4_26[1]), "=f"(_unpack_fp4_26[2]), "=f"(_unpack_fp4_26[3]), "=f"(_unpack_fp4_26[4]), "=f"(_unpack_fp4_26[5]), "=f"(_unpack_fp4_26[6]), "=f"(_unpack_fp4_26[7]) : "r"(_fp4_wire_values_1[10 + 0]));
                    #endif
                }
                float2 _f2_222 = make_float2(acc_73[16], acc_73[17]);
                float2 _f2_223 = make_float2(_unpack_fp4_26[0], _unpack_fp4_26[1]);
                float2 r_120 = fma_f32x2_rn_noftz(_f2_223, _f2_205, _f2_222);
                acc_73[16] = r_120.x;
                acc_73[17] = r_120.y;
                float2 _f2_224 = make_float2(acc_73[18], acc_73[19]);
                float2 _f2_225 = make_float2(_unpack_fp4_26[2], _unpack_fp4_26[3]);
                float2 r_121 = fma_f32x2_rn_noftz(_f2_225, _f2_205, _f2_224);
                acc_73[18] = r_121.x;
                acc_73[19] = r_121.y;
                float2 _f2_226 = make_float2(acc_73[20], acc_73[21]);
                float2 _f2_227 = make_float2(_unpack_fp4_26[4], _unpack_fp4_26[5]);
                float2 r_122 = fma_f32x2_rn_noftz(_f2_227, _f2_205, _f2_226);
                acc_73[20] = r_122.x;
                acc_73[21] = r_122.y;
                float2 _f2_228 = make_float2(acc_73[22], acc_73[23]);
                float2 _f2_229 = make_float2(_unpack_fp4_26[6], _unpack_fp4_26[7]);
                float2 r_123 = fma_f32x2_rn_noftz(_f2_229, _f2_205, _f2_228);
                acc_73[22] = r_123.x;
                acc_73[23] = r_123.y;
                float _unpack_fp4_27[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[11 + 0]));
                    (&_unpack_fp4_27[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_27[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_27[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_27[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_27[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_27[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_27[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_27[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_27[0]), "=f"(_unpack_fp4_27[1]), "=f"(_unpack_fp4_27[2]), "=f"(_unpack_fp4_27[3]), "=f"(_unpack_fp4_27[4]), "=f"(_unpack_fp4_27[5]), "=f"(_unpack_fp4_27[6]), "=f"(_unpack_fp4_27[7]) : "r"(_fp4_wire_values_1[11 + 0]));
                    #endif
                }
                float2 _f2_230 = make_float2(acc_73[24], acc_73[25]);
                float2 _f2_231 = make_float2(_unpack_fp4_27[0], _unpack_fp4_27[1]);
                float2 r_124 = fma_f32x2_rn_noftz(_f2_231, _f2_205, _f2_230);
                acc_73[24] = r_124.x;
                acc_73[25] = r_124.y;
                float2 _f2_232 = make_float2(acc_73[26], acc_73[27]);
                float2 _f2_233 = make_float2(_unpack_fp4_27[2], _unpack_fp4_27[3]);
                float2 r_125 = fma_f32x2_rn_noftz(_f2_233, _f2_205, _f2_232);
                acc_73[26] = r_125.x;
                acc_73[27] = r_125.y;
                float2 _f2_234 = make_float2(acc_73[28], acc_73[29]);
                float2 _f2_235 = make_float2(_unpack_fp4_27[4], _unpack_fp4_27[5]);
                float2 r_126 = fma_f32x2_rn_noftz(_f2_235, _f2_205, _f2_234);
                acc_73[28] = r_126.x;
                acc_73[29] = r_126.y;
                float2 _f2_236 = make_float2(acc_73[30], acc_73[31]);
                float2 _f2_237 = make_float2(_unpack_fp4_27[6], _unpack_fp4_27[7]);
                float2 r_127 = fma_f32x2_rn_noftz(_f2_237, _f2_205, _f2_236);
                acc_73[30] = r_127.x;
                acc_73[31] = r_127.y;
                float2 _fp8x2_decode_7;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_7.x), "=f"(_fp8x2_decode_7.y) : "h"((uint16_t)_fp4_wire_scales_1[3]));
                float _mul_14;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(_fp8x2_decode_7.x), "f"(g_2));
                float s0_128 = _mul_14;
                float _mul_15;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(_fp8x2_decode_7.y), "f"(g_2));
                float s1_129 = _mul_15;
                float2 _f2_238 = make_float2(s0_128, s0_128);
                float2 _f2_239 = make_float2(s1_129, s1_129);
                float _unpack_fp4_28[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[12 + 0]));
                    (&_unpack_fp4_28[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_28[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_28[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_28[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_28[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_28[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_28[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_28[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_28[0]), "=f"(_unpack_fp4_28[1]), "=f"(_unpack_fp4_28[2]), "=f"(_unpack_fp4_28[3]), "=f"(_unpack_fp4_28[4]), "=f"(_unpack_fp4_28[5]), "=f"(_unpack_fp4_28[6]), "=f"(_unpack_fp4_28[7]) : "r"(_fp4_wire_values_1[12 + 0]));
                    #endif
                }
                float2 _f2_240 = make_float2(acc_73[0], acc_73[1]);
                float2 _f2_241 = make_float2(_unpack_fp4_28[0], _unpack_fp4_28[1]);
                float2 r_130 = fma_f32x2_rn_noftz(_f2_241, _f2_238, _f2_240);
                acc_73[0] = r_130.x;
                acc_73[1] = r_130.y;
                float2 _f2_242 = make_float2(acc_73[2], acc_73[3]);
                float2 _f2_243 = make_float2(_unpack_fp4_28[2], _unpack_fp4_28[3]);
                float2 r_131 = fma_f32x2_rn_noftz(_f2_243, _f2_238, _f2_242);
                acc_73[2] = r_131.x;
                acc_73[3] = r_131.y;
                float2 _f2_244 = make_float2(acc_73[4], acc_73[5]);
                float2 _f2_245 = make_float2(_unpack_fp4_28[4], _unpack_fp4_28[5]);
                float2 r_132 = fma_f32x2_rn_noftz(_f2_245, _f2_238, _f2_244);
                acc_73[4] = r_132.x;
                acc_73[5] = r_132.y;
                float2 _f2_246 = make_float2(acc_73[6], acc_73[7]);
                float2 _f2_247 = make_float2(_unpack_fp4_28[6], _unpack_fp4_28[7]);
                float2 r_133 = fma_f32x2_rn_noftz(_f2_247, _f2_238, _f2_246);
                acc_73[6] = r_133.x;
                acc_73[7] = r_133.y;
                float _unpack_fp4_29[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[13 + 0]));
                    (&_unpack_fp4_29[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_29[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_29[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_29[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_29[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_29[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_29[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_29[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_29[0]), "=f"(_unpack_fp4_29[1]), "=f"(_unpack_fp4_29[2]), "=f"(_unpack_fp4_29[3]), "=f"(_unpack_fp4_29[4]), "=f"(_unpack_fp4_29[5]), "=f"(_unpack_fp4_29[6]), "=f"(_unpack_fp4_29[7]) : "r"(_fp4_wire_values_1[13 + 0]));
                    #endif
                }
                float2 _f2_248 = make_float2(acc_73[8], acc_73[9]);
                float2 _f2_249 = make_float2(_unpack_fp4_29[0], _unpack_fp4_29[1]);
                float2 r_134 = fma_f32x2_rn_noftz(_f2_249, _f2_238, _f2_248);
                acc_73[8] = r_134.x;
                acc_73[9] = r_134.y;
                float2 _f2_250 = make_float2(acc_73[10], acc_73[11]);
                float2 _f2_251 = make_float2(_unpack_fp4_29[2], _unpack_fp4_29[3]);
                float2 r_135 = fma_f32x2_rn_noftz(_f2_251, _f2_238, _f2_250);
                acc_73[10] = r_135.x;
                acc_73[11] = r_135.y;
                float2 _f2_252 = make_float2(acc_73[12], acc_73[13]);
                float2 _f2_253 = make_float2(_unpack_fp4_29[4], _unpack_fp4_29[5]);
                float2 r_136 = fma_f32x2_rn_noftz(_f2_253, _f2_238, _f2_252);
                acc_73[12] = r_136.x;
                acc_73[13] = r_136.y;
                float2 _f2_254 = make_float2(acc_73[14], acc_73[15]);
                float2 _f2_255 = make_float2(_unpack_fp4_29[6], _unpack_fp4_29[7]);
                float2 r_137 = fma_f32x2_rn_noftz(_f2_255, _f2_238, _f2_254);
                acc_73[14] = r_137.x;
                acc_73[15] = r_137.y;
                float _unpack_fp4_30[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[14 + 0]));
                    (&_unpack_fp4_30[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_30[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_30[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_30[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_30[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_30[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_30[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_30[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_30[0]), "=f"(_unpack_fp4_30[1]), "=f"(_unpack_fp4_30[2]), "=f"(_unpack_fp4_30[3]), "=f"(_unpack_fp4_30[4]), "=f"(_unpack_fp4_30[5]), "=f"(_unpack_fp4_30[6]), "=f"(_unpack_fp4_30[7]) : "r"(_fp4_wire_values_1[14 + 0]));
                    #endif
                }
                float2 _f2_256 = make_float2(acc_73[16], acc_73[17]);
                float2 _f2_257 = make_float2(_unpack_fp4_30[0], _unpack_fp4_30[1]);
                float2 r_138 = fma_f32x2_rn_noftz(_f2_257, _f2_239, _f2_256);
                acc_73[16] = r_138.x;
                acc_73[17] = r_138.y;
                float2 _f2_258 = make_float2(acc_73[18], acc_73[19]);
                float2 _f2_259 = make_float2(_unpack_fp4_30[2], _unpack_fp4_30[3]);
                float2 r_139 = fma_f32x2_rn_noftz(_f2_259, _f2_239, _f2_258);
                acc_73[18] = r_139.x;
                acc_73[19] = r_139.y;
                float2 _f2_260 = make_float2(acc_73[20], acc_73[21]);
                float2 _f2_261 = make_float2(_unpack_fp4_30[4], _unpack_fp4_30[5]);
                float2 r_140 = fma_f32x2_rn_noftz(_f2_261, _f2_239, _f2_260);
                acc_73[20] = r_140.x;
                acc_73[21] = r_140.y;
                float2 _f2_262 = make_float2(acc_73[22], acc_73[23]);
                float2 _f2_263 = make_float2(_unpack_fp4_30[6], _unpack_fp4_30[7]);
                float2 r_141 = fma_f32x2_rn_noftz(_f2_263, _f2_239, _f2_262);
                acc_73[22] = r_141.x;
                acc_73[23] = r_141.y;
                float _unpack_fp4_31[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[15 + 0]));
                    (&_unpack_fp4_31[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_31[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_31[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_31[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_31[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_31[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_31[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_31[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_31[0]), "=f"(_unpack_fp4_31[1]), "=f"(_unpack_fp4_31[2]), "=f"(_unpack_fp4_31[3]), "=f"(_unpack_fp4_31[4]), "=f"(_unpack_fp4_31[5]), "=f"(_unpack_fp4_31[6]), "=f"(_unpack_fp4_31[7]) : "r"(_fp4_wire_values_1[15 + 0]));
                    #endif
                }
                float2 _f2_264 = make_float2(acc_73[24], acc_73[25]);
                float2 _f2_265 = make_float2(_unpack_fp4_31[0], _unpack_fp4_31[1]);
                float2 r_142 = fma_f32x2_rn_noftz(_f2_265, _f2_239, _f2_264);
                acc_73[24] = r_142.x;
                acc_73[25] = r_142.y;
                float2 _f2_266 = make_float2(acc_73[26], acc_73[27]);
                float2 _f2_267 = make_float2(_unpack_fp4_31[2], _unpack_fp4_31[3]);
                float2 r_143 = fma_f32x2_rn_noftz(_f2_267, _f2_239, _f2_266);
                acc_73[26] = r_143.x;
                acc_73[27] = r_143.y;
                float2 _f2_268 = make_float2(acc_73[28], acc_73[29]);
                float2 _f2_269 = make_float2(_unpack_fp4_31[4], _unpack_fp4_31[5]);
                float2 r_144 = fma_f32x2_rn_noftz(_f2_269, _f2_239, _f2_268);
                acc_73[28] = r_144.x;
                acc_73[29] = r_144.y;
                float2 _f2_270 = make_float2(acc_73[30], acc_73[31]);
                float2 _f2_271 = make_float2(_unpack_fp4_31[6], _unpack_fp4_31[7]);
                float2 r_145 = fma_f32x2_rn_noftz(_f2_271, _f2_239, _f2_270);
                acc_73[30] = r_145.x;
                acc_73[31] = r_145.y;
                int pp_146 = lane_1 + 32 * (3 * h + 2);
                uint32_t _fp4_wire_values_2[16];
                uint32_t _fp4_wire_scales_2[4];
                uint16_t _wire_h16_10_0;
                uint16_t _wire_h16_10_1;
                uint16_t _wire_h16_10_2;
                uint16_t _wire_h16_10_3;
                uint32_t _wire_vpend_10 = 15u;
                uint32_t _wire_spend_10 = 15u;
                do {
                    if (_wire_vpend_10 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_2[0]), "=r"(_fp4_wire_values_2[1]), "=r"(_fp4_wire_values_2[2]), "=r"(_fp4_wire_values_2[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)rs_base + (int)tix64 * 3072 + 16 * pp_146))) : "memory");
                    }
                    if (_wire_vpend_10 & 2u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_2[4]), "=r"(_fp4_wire_values_2[5]), "=r"(_fp4_wire_values_2[6]), "=r"(_fp4_wire_values_2[7]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)rs_base + (4096 + (int)tix64) * 3072 + 16 * pp_146))) : "memory");
                    }
                    if (_wire_vpend_10 & 4u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_2[8]), "=r"(_fp4_wire_values_2[9]), "=r"(_fp4_wire_values_2[10]), "=r"(_fp4_wire_values_2[11]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)rs_base + (8192 + (int)tix64) * 3072 + 16 * pp_146))) : "memory");
                    }
                    if (_wire_vpend_10 & 8u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_2[12]), "=r"(_fp4_wire_values_2[13]), "=r"(_fp4_wire_values_2[14]), "=r"(_fp4_wire_values_2[15]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)rs_base + (12288 + (int)tix64) * 3072 + 16 * pp_146))) : "memory");
                    }
                    if (_wire_spend_10 & 1u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_10_0) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)rs_base + 50331648 + (int)tix64 * 384 + 2 * pp_146))) : "memory");
                        _fp4_wire_scales_2[0] = (uint32_t)_wire_h16_10_0;
                    }
                    if (_wire_spend_10 & 2u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_10_1) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)rs_base + 50331648 + (4096 + (int)tix64) * 384 + 2 * pp_146))) : "memory");
                        _fp4_wire_scales_2[1] = (uint32_t)_wire_h16_10_1;
                    }
                    if (_wire_spend_10 & 4u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_10_2) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)rs_base + 50331648 + (8192 + (int)tix64) * 384 + 2 * pp_146))) : "memory");
                        _fp4_wire_scales_2[2] = (uint32_t)_wire_h16_10_2;
                    }
                    if (_wire_spend_10 & 8u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_10_3) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)rs_base + 50331648 + (12288 + (int)tix64) * 384 + 2 * pp_146))) : "memory");
                        _fp4_wire_scales_2[3] = (uint32_t)_wire_h16_10_3;
                    }
                    if (_wire_vpend_10 & 1u) {
                        if (!((((_fp4_wire_values_2[0] & ~((_fp4_wire_values_2[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[1] & ~((_fp4_wire_values_2[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[2] & ~((_fp4_wire_values_2[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[3] & ~((_fp4_wire_values_2[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_10 &= ~1u;
                    }
                    if (_wire_vpend_10 & 2u) {
                        if (!((((_fp4_wire_values_2[4] & ~((_fp4_wire_values_2[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[5] & ~((_fp4_wire_values_2[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[6] & ~((_fp4_wire_values_2[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[7] & ~((_fp4_wire_values_2[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_10 &= ~2u;
                    }
                    if (_wire_vpend_10 & 4u) {
                        if (!((((_fp4_wire_values_2[8] & ~((_fp4_wire_values_2[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[9] & ~((_fp4_wire_values_2[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[10] & ~((_fp4_wire_values_2[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[11] & ~((_fp4_wire_values_2[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_10 &= ~4u;
                    }
                    if (_wire_vpend_10 & 8u) {
                        if (!((((_fp4_wire_values_2[12] & ~((_fp4_wire_values_2[12] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[13] & ~((_fp4_wire_values_2[13] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[14] & ~((_fp4_wire_values_2[14] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[15] & ~((_fp4_wire_values_2[15] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_10 &= ~8u;
                    }
                    if (_wire_spend_10 & 1u) {
                        if (!(((_fp4_wire_scales_2[0] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_2[0] >> 8) == 0xFFu))) _wire_spend_10 &= ~1u;
                    }
                    if (_wire_spend_10 & 2u) {
                        if (!(((_fp4_wire_scales_2[1] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_2[1] >> 8) == 0xFFu))) _wire_spend_10 &= ~2u;
                    }
                    if (_wire_spend_10 & 4u) {
                        if (!(((_fp4_wire_scales_2[2] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_2[2] >> 8) == 0xFFu))) _wire_spend_10 &= ~4u;
                    }
                    if (_wire_spend_10 & 8u) {
                        if (!(((_fp4_wire_scales_2[3] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_2[3] >> 8) == 0xFFu))) _wire_spend_10 &= ~8u;
                    }
                } while ((_wire_vpend_10 | _wire_spend_10) != 0u);
                float acc_147[32];
                acc_147[0] = 0.0f;
                acc_147[1] = 0.0f;
                acc_147[2] = 0.0f;
                acc_147[3] = 0.0f;
                acc_147[4] = 0.0f;
                acc_147[5] = 0.0f;
                acc_147[6] = 0.0f;
                acc_147[7] = 0.0f;
                acc_147[8] = 0.0f;
                acc_147[9] = 0.0f;
                acc_147[10] = 0.0f;
                acc_147[11] = 0.0f;
                acc_147[12] = 0.0f;
                acc_147[13] = 0.0f;
                acc_147[14] = 0.0f;
                acc_147[15] = 0.0f;
                acc_147[16] = 0.0f;
                acc_147[17] = 0.0f;
                acc_147[18] = 0.0f;
                acc_147[19] = 0.0f;
                acc_147[20] = 0.0f;
                acc_147[21] = 0.0f;
                acc_147[22] = 0.0f;
                acc_147[23] = 0.0f;
                acc_147[24] = 0.0f;
                acc_147[25] = 0.0f;
                acc_147[26] = 0.0f;
                acc_147[27] = 0.0f;
                acc_147[28] = 0.0f;
                acc_147[29] = 0.0f;
                acc_147[30] = 0.0f;
                acc_147[31] = 0.0f;
                float2 _fp8x2_decode_8;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_8.x), "=f"(_fp8x2_decode_8.y) : "h"((uint16_t)_fp4_wire_scales_2[0]));
                float _mul_16;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_fp8x2_decode_8.x), "f"(g));
                float s0_148 = _mul_16;
                float _mul_17;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_fp8x2_decode_8.y), "f"(g));
                float s1_149 = _mul_17;
                float2 _f2_272 = make_float2(s0_148, s0_148);
                float2 _f2_273 = make_float2(s1_149, s1_149);
                float _unpack_fp4_32[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[0]));
                    (&_unpack_fp4_32[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_32[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_32[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_32[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_32[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_32[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_32[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_32[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_32[0]), "=f"(_unpack_fp4_32[1]), "=f"(_unpack_fp4_32[2]), "=f"(_unpack_fp4_32[3]), "=f"(_unpack_fp4_32[4]), "=f"(_unpack_fp4_32[5]), "=f"(_unpack_fp4_32[6]), "=f"(_unpack_fp4_32[7]) : "r"(_fp4_wire_values_2[0]));
                    #endif
                }
                float2 _f2_274 = make_float2(acc_147[0], acc_147[1]);
                float2 _f2_275 = make_float2(_unpack_fp4_32[0], _unpack_fp4_32[1]);
                float2 r_150 = fma_f32x2_rn_noftz(_f2_275, _f2_272, _f2_274);
                acc_147[0] = r_150.x;
                acc_147[1] = r_150.y;
                float2 _f2_276 = make_float2(acc_147[2], acc_147[3]);
                float2 _f2_277 = make_float2(_unpack_fp4_32[2], _unpack_fp4_32[3]);
                float2 r_151 = fma_f32x2_rn_noftz(_f2_277, _f2_272, _f2_276);
                acc_147[2] = r_151.x;
                acc_147[3] = r_151.y;
                float2 _f2_278 = make_float2(acc_147[4], acc_147[5]);
                float2 _f2_279 = make_float2(_unpack_fp4_32[4], _unpack_fp4_32[5]);
                float2 r_152 = fma_f32x2_rn_noftz(_f2_279, _f2_272, _f2_278);
                acc_147[4] = r_152.x;
                acc_147[5] = r_152.y;
                float2 _f2_280 = make_float2(acc_147[6], acc_147[7]);
                float2 _f2_281 = make_float2(_unpack_fp4_32[6], _unpack_fp4_32[7]);
                float2 r_153 = fma_f32x2_rn_noftz(_f2_281, _f2_272, _f2_280);
                acc_147[6] = r_153.x;
                acc_147[7] = r_153.y;
                float _unpack_fp4_33[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[1 + 0]));
                    (&_unpack_fp4_33[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_33[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_33[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_33[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_33[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_33[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_33[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_33[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_33[0]), "=f"(_unpack_fp4_33[1]), "=f"(_unpack_fp4_33[2]), "=f"(_unpack_fp4_33[3]), "=f"(_unpack_fp4_33[4]), "=f"(_unpack_fp4_33[5]), "=f"(_unpack_fp4_33[6]), "=f"(_unpack_fp4_33[7]) : "r"(_fp4_wire_values_2[1 + 0]));
                    #endif
                }
                float2 _f2_282 = make_float2(acc_147[8], acc_147[9]);
                float2 _f2_283 = make_float2(_unpack_fp4_33[0], _unpack_fp4_33[1]);
                float2 r_154 = fma_f32x2_rn_noftz(_f2_283, _f2_272, _f2_282);
                acc_147[8] = r_154.x;
                acc_147[9] = r_154.y;
                float2 _f2_284 = make_float2(acc_147[10], acc_147[11]);
                float2 _f2_285 = make_float2(_unpack_fp4_33[2], _unpack_fp4_33[3]);
                float2 r_155 = fma_f32x2_rn_noftz(_f2_285, _f2_272, _f2_284);
                acc_147[10] = r_155.x;
                acc_147[11] = r_155.y;
                float2 _f2_286 = make_float2(acc_147[12], acc_147[13]);
                float2 _f2_287 = make_float2(_unpack_fp4_33[4], _unpack_fp4_33[5]);
                float2 r_156 = fma_f32x2_rn_noftz(_f2_287, _f2_272, _f2_286);
                acc_147[12] = r_156.x;
                acc_147[13] = r_156.y;
                float2 _f2_288 = make_float2(acc_147[14], acc_147[15]);
                float2 _f2_289 = make_float2(_unpack_fp4_33[6], _unpack_fp4_33[7]);
                float2 r_157 = fma_f32x2_rn_noftz(_f2_289, _f2_272, _f2_288);
                acc_147[14] = r_157.x;
                acc_147[15] = r_157.y;
                float _unpack_fp4_34[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[2 + 0]));
                    (&_unpack_fp4_34[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_34[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_34[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_34[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_34[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_34[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_34[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_34[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_34[0]), "=f"(_unpack_fp4_34[1]), "=f"(_unpack_fp4_34[2]), "=f"(_unpack_fp4_34[3]), "=f"(_unpack_fp4_34[4]), "=f"(_unpack_fp4_34[5]), "=f"(_unpack_fp4_34[6]), "=f"(_unpack_fp4_34[7]) : "r"(_fp4_wire_values_2[2 + 0]));
                    #endif
                }
                float2 _f2_290 = make_float2(acc_147[16], acc_147[17]);
                float2 _f2_291 = make_float2(_unpack_fp4_34[0], _unpack_fp4_34[1]);
                float2 r_158 = fma_f32x2_rn_noftz(_f2_291, _f2_273, _f2_290);
                acc_147[16] = r_158.x;
                acc_147[17] = r_158.y;
                float2 _f2_292 = make_float2(acc_147[18], acc_147[19]);
                float2 _f2_293 = make_float2(_unpack_fp4_34[2], _unpack_fp4_34[3]);
                float2 r_159 = fma_f32x2_rn_noftz(_f2_293, _f2_273, _f2_292);
                acc_147[18] = r_159.x;
                acc_147[19] = r_159.y;
                float2 _f2_294 = make_float2(acc_147[20], acc_147[21]);
                float2 _f2_295 = make_float2(_unpack_fp4_34[4], _unpack_fp4_34[5]);
                float2 r_160 = fma_f32x2_rn_noftz(_f2_295, _f2_273, _f2_294);
                acc_147[20] = r_160.x;
                acc_147[21] = r_160.y;
                float2 _f2_296 = make_float2(acc_147[22], acc_147[23]);
                float2 _f2_297 = make_float2(_unpack_fp4_34[6], _unpack_fp4_34[7]);
                float2 r_161 = fma_f32x2_rn_noftz(_f2_297, _f2_273, _f2_296);
                acc_147[22] = r_161.x;
                acc_147[23] = r_161.y;
                float _unpack_fp4_35[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[3 + 0]));
                    (&_unpack_fp4_35[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_35[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_35[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_35[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_35[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_35[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_35[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_35[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_35[0]), "=f"(_unpack_fp4_35[1]), "=f"(_unpack_fp4_35[2]), "=f"(_unpack_fp4_35[3]), "=f"(_unpack_fp4_35[4]), "=f"(_unpack_fp4_35[5]), "=f"(_unpack_fp4_35[6]), "=f"(_unpack_fp4_35[7]) : "r"(_fp4_wire_values_2[3 + 0]));
                    #endif
                }
                float2 _f2_298 = make_float2(acc_147[24], acc_147[25]);
                float2 _f2_299 = make_float2(_unpack_fp4_35[0], _unpack_fp4_35[1]);
                float2 r_162 = fma_f32x2_rn_noftz(_f2_299, _f2_273, _f2_298);
                acc_147[24] = r_162.x;
                acc_147[25] = r_162.y;
                float2 _f2_300 = make_float2(acc_147[26], acc_147[27]);
                float2 _f2_301 = make_float2(_unpack_fp4_35[2], _unpack_fp4_35[3]);
                float2 r_163 = fma_f32x2_rn_noftz(_f2_301, _f2_273, _f2_300);
                acc_147[26] = r_163.x;
                acc_147[27] = r_163.y;
                float2 _f2_302 = make_float2(acc_147[28], acc_147[29]);
                float2 _f2_303 = make_float2(_unpack_fp4_35[4], _unpack_fp4_35[5]);
                float2 r_164 = fma_f32x2_rn_noftz(_f2_303, _f2_273, _f2_302);
                acc_147[28] = r_164.x;
                acc_147[29] = r_164.y;
                float2 _f2_304 = make_float2(acc_147[30], acc_147[31]);
                float2 _f2_305 = make_float2(_unpack_fp4_35[6], _unpack_fp4_35[7]);
                float2 r_165 = fma_f32x2_rn_noftz(_f2_305, _f2_273, _f2_304);
                acc_147[30] = r_165.x;
                acc_147[31] = r_165.y;
                float2 _fp8x2_decode_9;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_9.x), "=f"(_fp8x2_decode_9.y) : "h"((uint16_t)_fp4_wire_scales_2[1]));
                float _mul_18;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(_fp8x2_decode_9.x), "f"(g_0));
                float s0_166 = _mul_18;
                float _mul_19;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_fp8x2_decode_9.y), "f"(g_0));
                float s1_167 = _mul_19;
                float2 _f2_306 = make_float2(s0_166, s0_166);
                float2 _f2_307 = make_float2(s1_167, s1_167);
                float _unpack_fp4_36[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[4 + 0]));
                    (&_unpack_fp4_36[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_36[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_36[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_36[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_36[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_36[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_36[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_36[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_36[0]), "=f"(_unpack_fp4_36[1]), "=f"(_unpack_fp4_36[2]), "=f"(_unpack_fp4_36[3]), "=f"(_unpack_fp4_36[4]), "=f"(_unpack_fp4_36[5]), "=f"(_unpack_fp4_36[6]), "=f"(_unpack_fp4_36[7]) : "r"(_fp4_wire_values_2[4 + 0]));
                    #endif
                }
                float2 _f2_308 = make_float2(acc_147[0], acc_147[1]);
                float2 _f2_309 = make_float2(_unpack_fp4_36[0], _unpack_fp4_36[1]);
                float2 r_168 = fma_f32x2_rn_noftz(_f2_309, _f2_306, _f2_308);
                acc_147[0] = r_168.x;
                acc_147[1] = r_168.y;
                float2 _f2_310 = make_float2(acc_147[2], acc_147[3]);
                float2 _f2_311 = make_float2(_unpack_fp4_36[2], _unpack_fp4_36[3]);
                float2 r_169 = fma_f32x2_rn_noftz(_f2_311, _f2_306, _f2_310);
                acc_147[2] = r_169.x;
                acc_147[3] = r_169.y;
                float2 _f2_312 = make_float2(acc_147[4], acc_147[5]);
                float2 _f2_313 = make_float2(_unpack_fp4_36[4], _unpack_fp4_36[5]);
                float2 r_170 = fma_f32x2_rn_noftz(_f2_313, _f2_306, _f2_312);
                acc_147[4] = r_170.x;
                acc_147[5] = r_170.y;
                float2 _f2_314 = make_float2(acc_147[6], acc_147[7]);
                float2 _f2_315 = make_float2(_unpack_fp4_36[6], _unpack_fp4_36[7]);
                float2 r_171 = fma_f32x2_rn_noftz(_f2_315, _f2_306, _f2_314);
                acc_147[6] = r_171.x;
                acc_147[7] = r_171.y;
                float _unpack_fp4_37[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[5 + 0]));
                    (&_unpack_fp4_37[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_37[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_37[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_37[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_37[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_37[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_37[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_37[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_37[0]), "=f"(_unpack_fp4_37[1]), "=f"(_unpack_fp4_37[2]), "=f"(_unpack_fp4_37[3]), "=f"(_unpack_fp4_37[4]), "=f"(_unpack_fp4_37[5]), "=f"(_unpack_fp4_37[6]), "=f"(_unpack_fp4_37[7]) : "r"(_fp4_wire_values_2[5 + 0]));
                    #endif
                }
                float2 _f2_316 = make_float2(acc_147[8], acc_147[9]);
                float2 _f2_317 = make_float2(_unpack_fp4_37[0], _unpack_fp4_37[1]);
                float2 r_172 = fma_f32x2_rn_noftz(_f2_317, _f2_306, _f2_316);
                acc_147[8] = r_172.x;
                acc_147[9] = r_172.y;
                float2 _f2_318 = make_float2(acc_147[10], acc_147[11]);
                float2 _f2_319 = make_float2(_unpack_fp4_37[2], _unpack_fp4_37[3]);
                float2 r_173 = fma_f32x2_rn_noftz(_f2_319, _f2_306, _f2_318);
                acc_147[10] = r_173.x;
                acc_147[11] = r_173.y;
                float2 _f2_320 = make_float2(acc_147[12], acc_147[13]);
                float2 _f2_321 = make_float2(_unpack_fp4_37[4], _unpack_fp4_37[5]);
                float2 r_174 = fma_f32x2_rn_noftz(_f2_321, _f2_306, _f2_320);
                acc_147[12] = r_174.x;
                acc_147[13] = r_174.y;
                float2 _f2_322 = make_float2(acc_147[14], acc_147[15]);
                float2 _f2_323 = make_float2(_unpack_fp4_37[6], _unpack_fp4_37[7]);
                float2 r_175 = fma_f32x2_rn_noftz(_f2_323, _f2_306, _f2_322);
                acc_147[14] = r_175.x;
                acc_147[15] = r_175.y;
                float _unpack_fp4_38[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[6 + 0]));
                    (&_unpack_fp4_38[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_38[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_38[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_38[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_38[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_38[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_38[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_38[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_38[0]), "=f"(_unpack_fp4_38[1]), "=f"(_unpack_fp4_38[2]), "=f"(_unpack_fp4_38[3]), "=f"(_unpack_fp4_38[4]), "=f"(_unpack_fp4_38[5]), "=f"(_unpack_fp4_38[6]), "=f"(_unpack_fp4_38[7]) : "r"(_fp4_wire_values_2[6 + 0]));
                    #endif
                }
                float2 _f2_324 = make_float2(acc_147[16], acc_147[17]);
                float2 _f2_325 = make_float2(_unpack_fp4_38[0], _unpack_fp4_38[1]);
                float2 r_176 = fma_f32x2_rn_noftz(_f2_325, _f2_307, _f2_324);
                acc_147[16] = r_176.x;
                acc_147[17] = r_176.y;
                float2 _f2_326 = make_float2(acc_147[18], acc_147[19]);
                float2 _f2_327 = make_float2(_unpack_fp4_38[2], _unpack_fp4_38[3]);
                float2 r_177 = fma_f32x2_rn_noftz(_f2_327, _f2_307, _f2_326);
                acc_147[18] = r_177.x;
                acc_147[19] = r_177.y;
                float2 _f2_328 = make_float2(acc_147[20], acc_147[21]);
                float2 _f2_329 = make_float2(_unpack_fp4_38[4], _unpack_fp4_38[5]);
                float2 r_178 = fma_f32x2_rn_noftz(_f2_329, _f2_307, _f2_328);
                acc_147[20] = r_178.x;
                acc_147[21] = r_178.y;
                float2 _f2_330 = make_float2(acc_147[22], acc_147[23]);
                float2 _f2_331 = make_float2(_unpack_fp4_38[6], _unpack_fp4_38[7]);
                float2 r_179 = fma_f32x2_rn_noftz(_f2_331, _f2_307, _f2_330);
                acc_147[22] = r_179.x;
                acc_147[23] = r_179.y;
                float _unpack_fp4_39[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[7 + 0]));
                    (&_unpack_fp4_39[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_39[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_39[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_39[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_39[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_39[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_39[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_39[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_39[0]), "=f"(_unpack_fp4_39[1]), "=f"(_unpack_fp4_39[2]), "=f"(_unpack_fp4_39[3]), "=f"(_unpack_fp4_39[4]), "=f"(_unpack_fp4_39[5]), "=f"(_unpack_fp4_39[6]), "=f"(_unpack_fp4_39[7]) : "r"(_fp4_wire_values_2[7 + 0]));
                    #endif
                }
                float2 _f2_332 = make_float2(acc_147[24], acc_147[25]);
                float2 _f2_333 = make_float2(_unpack_fp4_39[0], _unpack_fp4_39[1]);
                float2 r_180 = fma_f32x2_rn_noftz(_f2_333, _f2_307, _f2_332);
                acc_147[24] = r_180.x;
                acc_147[25] = r_180.y;
                float2 _f2_334 = make_float2(acc_147[26], acc_147[27]);
                float2 _f2_335 = make_float2(_unpack_fp4_39[2], _unpack_fp4_39[3]);
                float2 r_181 = fma_f32x2_rn_noftz(_f2_335, _f2_307, _f2_334);
                acc_147[26] = r_181.x;
                acc_147[27] = r_181.y;
                float2 _f2_336 = make_float2(acc_147[28], acc_147[29]);
                float2 _f2_337 = make_float2(_unpack_fp4_39[4], _unpack_fp4_39[5]);
                float2 r_182 = fma_f32x2_rn_noftz(_f2_337, _f2_307, _f2_336);
                acc_147[28] = r_182.x;
                acc_147[29] = r_182.y;
                float2 _f2_338 = make_float2(acc_147[30], acc_147[31]);
                float2 _f2_339 = make_float2(_unpack_fp4_39[6], _unpack_fp4_39[7]);
                float2 r_183 = fma_f32x2_rn_noftz(_f2_339, _f2_307, _f2_338);
                acc_147[30] = r_183.x;
                acc_147[31] = r_183.y;
                float2 _fp8x2_decode_10;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_10.x), "=f"(_fp8x2_decode_10.y) : "h"((uint16_t)_fp4_wire_scales_2[2]));
                float _mul_20;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(_fp8x2_decode_10.x), "f"(g_1));
                float s0_184 = _mul_20;
                float _mul_21;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(_fp8x2_decode_10.y), "f"(g_1));
                float s1_185 = _mul_21;
                float2 _f2_340 = make_float2(s0_184, s0_184);
                float2 _f2_341 = make_float2(s1_185, s1_185);
                float _unpack_fp4_40[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[8 + 0]));
                    (&_unpack_fp4_40[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_40[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_40[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_40[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_40[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_40[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_40[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_40[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_40[0]), "=f"(_unpack_fp4_40[1]), "=f"(_unpack_fp4_40[2]), "=f"(_unpack_fp4_40[3]), "=f"(_unpack_fp4_40[4]), "=f"(_unpack_fp4_40[5]), "=f"(_unpack_fp4_40[6]), "=f"(_unpack_fp4_40[7]) : "r"(_fp4_wire_values_2[8 + 0]));
                    #endif
                }
                float2 _f2_342 = make_float2(acc_147[0], acc_147[1]);
                float2 _f2_343 = make_float2(_unpack_fp4_40[0], _unpack_fp4_40[1]);
                float2 r_186 = fma_f32x2_rn_noftz(_f2_343, _f2_340, _f2_342);
                acc_147[0] = r_186.x;
                acc_147[1] = r_186.y;
                float2 _f2_344 = make_float2(acc_147[2], acc_147[3]);
                float2 _f2_345 = make_float2(_unpack_fp4_40[2], _unpack_fp4_40[3]);
                float2 r_187 = fma_f32x2_rn_noftz(_f2_345, _f2_340, _f2_344);
                acc_147[2] = r_187.x;
                acc_147[3] = r_187.y;
                float2 _f2_346 = make_float2(acc_147[4], acc_147[5]);
                float2 _f2_347 = make_float2(_unpack_fp4_40[4], _unpack_fp4_40[5]);
                float2 r_188 = fma_f32x2_rn_noftz(_f2_347, _f2_340, _f2_346);
                acc_147[4] = r_188.x;
                acc_147[5] = r_188.y;
                float2 _f2_348 = make_float2(acc_147[6], acc_147[7]);
                float2 _f2_349 = make_float2(_unpack_fp4_40[6], _unpack_fp4_40[7]);
                float2 r_189 = fma_f32x2_rn_noftz(_f2_349, _f2_340, _f2_348);
                acc_147[6] = r_189.x;
                acc_147[7] = r_189.y;
                float _unpack_fp4_41[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[9 + 0]));
                    (&_unpack_fp4_41[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_41[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_41[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_41[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_41[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_41[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_41[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_41[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_41[0]), "=f"(_unpack_fp4_41[1]), "=f"(_unpack_fp4_41[2]), "=f"(_unpack_fp4_41[3]), "=f"(_unpack_fp4_41[4]), "=f"(_unpack_fp4_41[5]), "=f"(_unpack_fp4_41[6]), "=f"(_unpack_fp4_41[7]) : "r"(_fp4_wire_values_2[9 + 0]));
                    #endif
                }
                float2 _f2_350 = make_float2(acc_147[8], acc_147[9]);
                float2 _f2_351 = make_float2(_unpack_fp4_41[0], _unpack_fp4_41[1]);
                float2 r_190 = fma_f32x2_rn_noftz(_f2_351, _f2_340, _f2_350);
                acc_147[8] = r_190.x;
                acc_147[9] = r_190.y;
                float2 _f2_352 = make_float2(acc_147[10], acc_147[11]);
                float2 _f2_353 = make_float2(_unpack_fp4_41[2], _unpack_fp4_41[3]);
                float2 r_191 = fma_f32x2_rn_noftz(_f2_353, _f2_340, _f2_352);
                acc_147[10] = r_191.x;
                acc_147[11] = r_191.y;
                float2 _f2_354 = make_float2(acc_147[12], acc_147[13]);
                float2 _f2_355 = make_float2(_unpack_fp4_41[4], _unpack_fp4_41[5]);
                float2 r_192 = fma_f32x2_rn_noftz(_f2_355, _f2_340, _f2_354);
                acc_147[12] = r_192.x;
                acc_147[13] = r_192.y;
                float2 _f2_356 = make_float2(acc_147[14], acc_147[15]);
                float2 _f2_357 = make_float2(_unpack_fp4_41[6], _unpack_fp4_41[7]);
                float2 r_193 = fma_f32x2_rn_noftz(_f2_357, _f2_340, _f2_356);
                acc_147[14] = r_193.x;
                acc_147[15] = r_193.y;
                float _unpack_fp4_42[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[10 + 0]));
                    (&_unpack_fp4_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_42[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_42[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_42[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_42[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_42[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_42[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_42[0]), "=f"(_unpack_fp4_42[1]), "=f"(_unpack_fp4_42[2]), "=f"(_unpack_fp4_42[3]), "=f"(_unpack_fp4_42[4]), "=f"(_unpack_fp4_42[5]), "=f"(_unpack_fp4_42[6]), "=f"(_unpack_fp4_42[7]) : "r"(_fp4_wire_values_2[10 + 0]));
                    #endif
                }
                float2 _f2_358 = make_float2(acc_147[16], acc_147[17]);
                float2 _f2_359 = make_float2(_unpack_fp4_42[0], _unpack_fp4_42[1]);
                float2 r_194 = fma_f32x2_rn_noftz(_f2_359, _f2_341, _f2_358);
                acc_147[16] = r_194.x;
                acc_147[17] = r_194.y;
                float2 _f2_360 = make_float2(acc_147[18], acc_147[19]);
                float2 _f2_361 = make_float2(_unpack_fp4_42[2], _unpack_fp4_42[3]);
                float2 r_195 = fma_f32x2_rn_noftz(_f2_361, _f2_341, _f2_360);
                acc_147[18] = r_195.x;
                acc_147[19] = r_195.y;
                float2 _f2_362 = make_float2(acc_147[20], acc_147[21]);
                float2 _f2_363 = make_float2(_unpack_fp4_42[4], _unpack_fp4_42[5]);
                float2 r_196 = fma_f32x2_rn_noftz(_f2_363, _f2_341, _f2_362);
                acc_147[20] = r_196.x;
                acc_147[21] = r_196.y;
                float2 _f2_364 = make_float2(acc_147[22], acc_147[23]);
                float2 _f2_365 = make_float2(_unpack_fp4_42[6], _unpack_fp4_42[7]);
                float2 r_197 = fma_f32x2_rn_noftz(_f2_365, _f2_341, _f2_364);
                acc_147[22] = r_197.x;
                acc_147[23] = r_197.y;
                float _unpack_fp4_43[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[11 + 0]));
                    (&_unpack_fp4_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_43[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_43[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_43[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_43[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_43[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_43[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_43[0]), "=f"(_unpack_fp4_43[1]), "=f"(_unpack_fp4_43[2]), "=f"(_unpack_fp4_43[3]), "=f"(_unpack_fp4_43[4]), "=f"(_unpack_fp4_43[5]), "=f"(_unpack_fp4_43[6]), "=f"(_unpack_fp4_43[7]) : "r"(_fp4_wire_values_2[11 + 0]));
                    #endif
                }
                float2 _f2_366 = make_float2(acc_147[24], acc_147[25]);
                float2 _f2_367 = make_float2(_unpack_fp4_43[0], _unpack_fp4_43[1]);
                float2 r_198 = fma_f32x2_rn_noftz(_f2_367, _f2_341, _f2_366);
                acc_147[24] = r_198.x;
                acc_147[25] = r_198.y;
                float2 _f2_368 = make_float2(acc_147[26], acc_147[27]);
                float2 _f2_369 = make_float2(_unpack_fp4_43[2], _unpack_fp4_43[3]);
                float2 r_199 = fma_f32x2_rn_noftz(_f2_369, _f2_341, _f2_368);
                acc_147[26] = r_199.x;
                acc_147[27] = r_199.y;
                float2 _f2_370 = make_float2(acc_147[28], acc_147[29]);
                float2 _f2_371 = make_float2(_unpack_fp4_43[4], _unpack_fp4_43[5]);
                float2 r_200 = fma_f32x2_rn_noftz(_f2_371, _f2_341, _f2_370);
                acc_147[28] = r_200.x;
                acc_147[29] = r_200.y;
                float2 _f2_372 = make_float2(acc_147[30], acc_147[31]);
                float2 _f2_373 = make_float2(_unpack_fp4_43[6], _unpack_fp4_43[7]);
                float2 r_201 = fma_f32x2_rn_noftz(_f2_373, _f2_341, _f2_372);
                acc_147[30] = r_201.x;
                acc_147[31] = r_201.y;
                float2 _fp8x2_decode_11;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_11.x), "=f"(_fp8x2_decode_11.y) : "h"((uint16_t)_fp4_wire_scales_2[3]));
                float _mul_22;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(_fp8x2_decode_11.x), "f"(g_2));
                float s0_202 = _mul_22;
                float _mul_23;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_fp8x2_decode_11.y), "f"(g_2));
                float s1_203 = _mul_23;
                float2 _f2_374 = make_float2(s0_202, s0_202);
                float2 _f2_375 = make_float2(s1_203, s1_203);
                float _unpack_fp4_44[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[12 + 0]));
                    (&_unpack_fp4_44[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_44[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_44[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_44[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_44[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_44[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_44[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_44[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_44[0]), "=f"(_unpack_fp4_44[1]), "=f"(_unpack_fp4_44[2]), "=f"(_unpack_fp4_44[3]), "=f"(_unpack_fp4_44[4]), "=f"(_unpack_fp4_44[5]), "=f"(_unpack_fp4_44[6]), "=f"(_unpack_fp4_44[7]) : "r"(_fp4_wire_values_2[12 + 0]));
                    #endif
                }
                float2 _f2_376 = make_float2(acc_147[0], acc_147[1]);
                float2 _f2_377 = make_float2(_unpack_fp4_44[0], _unpack_fp4_44[1]);
                float2 r_204 = fma_f32x2_rn_noftz(_f2_377, _f2_374, _f2_376);
                acc_147[0] = r_204.x;
                acc_147[1] = r_204.y;
                float2 _f2_378 = make_float2(acc_147[2], acc_147[3]);
                float2 _f2_379 = make_float2(_unpack_fp4_44[2], _unpack_fp4_44[3]);
                float2 r_205 = fma_f32x2_rn_noftz(_f2_379, _f2_374, _f2_378);
                acc_147[2] = r_205.x;
                acc_147[3] = r_205.y;
                float2 _f2_380 = make_float2(acc_147[4], acc_147[5]);
                float2 _f2_381 = make_float2(_unpack_fp4_44[4], _unpack_fp4_44[5]);
                float2 r_206 = fma_f32x2_rn_noftz(_f2_381, _f2_374, _f2_380);
                acc_147[4] = r_206.x;
                acc_147[5] = r_206.y;
                float2 _f2_382 = make_float2(acc_147[6], acc_147[7]);
                float2 _f2_383 = make_float2(_unpack_fp4_44[6], _unpack_fp4_44[7]);
                float2 r_207 = fma_f32x2_rn_noftz(_f2_383, _f2_374, _f2_382);
                acc_147[6] = r_207.x;
                acc_147[7] = r_207.y;
                float _unpack_fp4_45[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[13 + 0]));
                    (&_unpack_fp4_45[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_45[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_45[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_45[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_45[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_45[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_45[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_45[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_45[0]), "=f"(_unpack_fp4_45[1]), "=f"(_unpack_fp4_45[2]), "=f"(_unpack_fp4_45[3]), "=f"(_unpack_fp4_45[4]), "=f"(_unpack_fp4_45[5]), "=f"(_unpack_fp4_45[6]), "=f"(_unpack_fp4_45[7]) : "r"(_fp4_wire_values_2[13 + 0]));
                    #endif
                }
                float2 _f2_384 = make_float2(acc_147[8], acc_147[9]);
                float2 _f2_385 = make_float2(_unpack_fp4_45[0], _unpack_fp4_45[1]);
                float2 r_208 = fma_f32x2_rn_noftz(_f2_385, _f2_374, _f2_384);
                acc_147[8] = r_208.x;
                acc_147[9] = r_208.y;
                float2 _f2_386 = make_float2(acc_147[10], acc_147[11]);
                float2 _f2_387 = make_float2(_unpack_fp4_45[2], _unpack_fp4_45[3]);
                float2 r_209 = fma_f32x2_rn_noftz(_f2_387, _f2_374, _f2_386);
                acc_147[10] = r_209.x;
                acc_147[11] = r_209.y;
                float2 _f2_388 = make_float2(acc_147[12], acc_147[13]);
                float2 _f2_389 = make_float2(_unpack_fp4_45[4], _unpack_fp4_45[5]);
                float2 r_210 = fma_f32x2_rn_noftz(_f2_389, _f2_374, _f2_388);
                acc_147[12] = r_210.x;
                acc_147[13] = r_210.y;
                float2 _f2_390 = make_float2(acc_147[14], acc_147[15]);
                float2 _f2_391 = make_float2(_unpack_fp4_45[6], _unpack_fp4_45[7]);
                float2 r_211 = fma_f32x2_rn_noftz(_f2_391, _f2_374, _f2_390);
                acc_147[14] = r_211.x;
                acc_147[15] = r_211.y;
                float _unpack_fp4_46[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[14 + 0]));
                    (&_unpack_fp4_46[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_46[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_46[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_46[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_46[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_46[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_46[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_46[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_46[0]), "=f"(_unpack_fp4_46[1]), "=f"(_unpack_fp4_46[2]), "=f"(_unpack_fp4_46[3]), "=f"(_unpack_fp4_46[4]), "=f"(_unpack_fp4_46[5]), "=f"(_unpack_fp4_46[6]), "=f"(_unpack_fp4_46[7]) : "r"(_fp4_wire_values_2[14 + 0]));
                    #endif
                }
                float2 _f2_392 = make_float2(acc_147[16], acc_147[17]);
                float2 _f2_393 = make_float2(_unpack_fp4_46[0], _unpack_fp4_46[1]);
                float2 r_212 = fma_f32x2_rn_noftz(_f2_393, _f2_375, _f2_392);
                acc_147[16] = r_212.x;
                acc_147[17] = r_212.y;
                float2 _f2_394 = make_float2(acc_147[18], acc_147[19]);
                float2 _f2_395 = make_float2(_unpack_fp4_46[2], _unpack_fp4_46[3]);
                float2 r_213 = fma_f32x2_rn_noftz(_f2_395, _f2_375, _f2_394);
                acc_147[18] = r_213.x;
                acc_147[19] = r_213.y;
                float2 _f2_396 = make_float2(acc_147[20], acc_147[21]);
                float2 _f2_397 = make_float2(_unpack_fp4_46[4], _unpack_fp4_46[5]);
                float2 r_214 = fma_f32x2_rn_noftz(_f2_397, _f2_375, _f2_396);
                acc_147[20] = r_214.x;
                acc_147[21] = r_214.y;
                float2 _f2_398 = make_float2(acc_147[22], acc_147[23]);
                float2 _f2_399 = make_float2(_unpack_fp4_46[6], _unpack_fp4_46[7]);
                float2 r_215 = fma_f32x2_rn_noftz(_f2_399, _f2_375, _f2_398);
                acc_147[22] = r_215.x;
                acc_147[23] = r_215.y;
                float _unpack_fp4_47[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_2[15 + 0]));
                    (&_unpack_fp4_47[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_47[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_47[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_47[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_47[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_47[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_47[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_47[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_47[0]), "=f"(_unpack_fp4_47[1]), "=f"(_unpack_fp4_47[2]), "=f"(_unpack_fp4_47[3]), "=f"(_unpack_fp4_47[4]), "=f"(_unpack_fp4_47[5]), "=f"(_unpack_fp4_47[6]), "=f"(_unpack_fp4_47[7]) : "r"(_fp4_wire_values_2[15 + 0]));
                    #endif
                }
                float2 _f2_400 = make_float2(acc_147[24], acc_147[25]);
                float2 _f2_401 = make_float2(_unpack_fp4_47[0], _unpack_fp4_47[1]);
                float2 r_216 = fma_f32x2_rn_noftz(_f2_401, _f2_375, _f2_400);
                acc_147[24] = r_216.x;
                acc_147[25] = r_216.y;
                float2 _f2_402 = make_float2(acc_147[26], acc_147[27]);
                float2 _f2_403 = make_float2(_unpack_fp4_47[2], _unpack_fp4_47[3]);
                float2 r_217 = fma_f32x2_rn_noftz(_f2_403, _f2_375, _f2_402);
                acc_147[26] = r_217.x;
                acc_147[27] = r_217.y;
                float2 _f2_404 = make_float2(acc_147[28], acc_147[29]);
                float2 _f2_405 = make_float2(_unpack_fp4_47[4], _unpack_fp4_47[5]);
                float2 r_218 = fma_f32x2_rn_noftz(_f2_405, _f2_375, _f2_404);
                acc_147[28] = r_218.x;
                acc_147[29] = r_218.y;
                float2 _f2_406 = make_float2(acc_147[30], acc_147[31]);
                float2 _f2_407 = make_float2(_unpack_fp4_47[6], _unpack_fp4_47[7]);
                float2 r_219 = fma_f32x2_rn_noftz(_f2_407, _f2_375, _f2_406);
                acc_147[30] = r_219.x;
                acc_147[31] = r_219.y;
                float am = 0.0f;
                float cur = am;
                float _fabs_0 = fabsf(acc[0]);
                float _fmax_0 = fmaxf(cur, _fabs_0);
                cur = _fmax_0;
                float _fabs_1 = fabsf(acc[1]);
                float _fmax_1 = fmaxf(cur, _fabs_1);
                cur = _fmax_1;
                float _fabs_2 = fabsf(acc[2]);
                float _fmax_2 = fmaxf(cur, _fabs_2);
                cur = _fmax_2;
                float _fabs_3 = fabsf(acc[3]);
                float _fmax_3 = fmaxf(cur, _fabs_3);
                cur = _fmax_3;
                float _fabs_4 = fabsf(acc[4]);
                float _fmax_4 = fmaxf(cur, _fabs_4);
                cur = _fmax_4;
                float _fabs_5 = fabsf(acc[5]);
                float _fmax_5 = fmaxf(cur, _fabs_5);
                cur = _fmax_5;
                float _fabs_6 = fabsf(acc[6]);
                float _fmax_6 = fmaxf(cur, _fabs_6);
                cur = _fmax_6;
                float _fabs_7 = fabsf(acc[7]);
                float _fmax_7 = fmaxf(cur, _fabs_7);
                cur = _fmax_7;
                float _fabs_8 = fabsf(acc[8]);
                float _fmax_8 = fmaxf(cur, _fabs_8);
                cur = _fmax_8;
                float _fabs_9 = fabsf(acc[9]);
                float _fmax_9 = fmaxf(cur, _fabs_9);
                cur = _fmax_9;
                float _fabs_10 = fabsf(acc[10]);
                float _fmax_10 = fmaxf(cur, _fabs_10);
                cur = _fmax_10;
                float _fabs_11 = fabsf(acc[11]);
                float _fmax_11 = fmaxf(cur, _fabs_11);
                cur = _fmax_11;
                float _fabs_12 = fabsf(acc[12]);
                float _fmax_12 = fmaxf(cur, _fabs_12);
                cur = _fmax_12;
                float _fabs_13 = fabsf(acc[13]);
                float _fmax_13 = fmaxf(cur, _fabs_13);
                cur = _fmax_13;
                float _fabs_14 = fabsf(acc[14]);
                float _fmax_14 = fmaxf(cur, _fabs_14);
                cur = _fmax_14;
                float _fabs_15 = fabsf(acc[15]);
                float _fmax_15 = fmaxf(cur, _fabs_15);
                cur = _fmax_15;
                float _fabs_16 = fabsf(acc[16]);
                float _fmax_16 = fmaxf(cur, _fabs_16);
                cur = _fmax_16;
                float _fabs_17 = fabsf(acc[17]);
                float _fmax_17 = fmaxf(cur, _fabs_17);
                cur = _fmax_17;
                float _fabs_18 = fabsf(acc[18]);
                float _fmax_18 = fmaxf(cur, _fabs_18);
                cur = _fmax_18;
                float _fabs_19 = fabsf(acc[19]);
                float _fmax_19 = fmaxf(cur, _fabs_19);
                cur = _fmax_19;
                float _fabs_20 = fabsf(acc[20]);
                float _fmax_20 = fmaxf(cur, _fabs_20);
                cur = _fmax_20;
                float _fabs_21 = fabsf(acc[21]);
                float _fmax_21 = fmaxf(cur, _fabs_21);
                cur = _fmax_21;
                float _fabs_22 = fabsf(acc[22]);
                float _fmax_22 = fmaxf(cur, _fabs_22);
                cur = _fmax_22;
                float _fabs_23 = fabsf(acc[23]);
                float _fmax_23 = fmaxf(cur, _fabs_23);
                cur = _fmax_23;
                float _fabs_24 = fabsf(acc[24]);
                float _fmax_24 = fmaxf(cur, _fabs_24);
                cur = _fmax_24;
                float _fabs_25 = fabsf(acc[25]);
                float _fmax_25 = fmaxf(cur, _fabs_25);
                cur = _fmax_25;
                float _fabs_26 = fabsf(acc[26]);
                float _fmax_26 = fmaxf(cur, _fabs_26);
                cur = _fmax_26;
                float _fabs_27 = fabsf(acc[27]);
                float _fmax_27 = fmaxf(cur, _fabs_27);
                cur = _fmax_27;
                float _fabs_28 = fabsf(acc[28]);
                float _fmax_28 = fmaxf(cur, _fabs_28);
                cur = _fmax_28;
                float _fabs_29 = fabsf(acc[29]);
                float _fmax_29 = fmaxf(cur, _fabs_29);
                cur = _fmax_29;
                float _fabs_30 = fabsf(acc[30]);
                float _fmax_30 = fmaxf(cur, _fabs_30);
                cur = _fmax_30;
                float _fabs_31 = fabsf(acc[31]);
                float _fmax_31 = fmaxf(cur, _fabs_31);
                cur = _fmax_31;
                am = cur;
                float cur_220 = am;
                float _fabs_32 = fabsf(acc_73[0]);
                float _fmax_32 = fmaxf(cur_220, _fabs_32);
                cur_220 = _fmax_32;
                float _fabs_33 = fabsf(acc_73[1]);
                float _fmax_33 = fmaxf(cur_220, _fabs_33);
                cur_220 = _fmax_33;
                float _fabs_34 = fabsf(acc_73[2]);
                float _fmax_34 = fmaxf(cur_220, _fabs_34);
                cur_220 = _fmax_34;
                float _fabs_35 = fabsf(acc_73[3]);
                float _fmax_35 = fmaxf(cur_220, _fabs_35);
                cur_220 = _fmax_35;
                float _fabs_36 = fabsf(acc_73[4]);
                float _fmax_36 = fmaxf(cur_220, _fabs_36);
                cur_220 = _fmax_36;
                float _fabs_37 = fabsf(acc_73[5]);
                float _fmax_37 = fmaxf(cur_220, _fabs_37);
                cur_220 = _fmax_37;
                float _fabs_38 = fabsf(acc_73[6]);
                float _fmax_38 = fmaxf(cur_220, _fabs_38);
                cur_220 = _fmax_38;
                float _fabs_39 = fabsf(acc_73[7]);
                float _fmax_39 = fmaxf(cur_220, _fabs_39);
                cur_220 = _fmax_39;
                float _fabs_40 = fabsf(acc_73[8]);
                float _fmax_40 = fmaxf(cur_220, _fabs_40);
                cur_220 = _fmax_40;
                float _fabs_41 = fabsf(acc_73[9]);
                float _fmax_41 = fmaxf(cur_220, _fabs_41);
                cur_220 = _fmax_41;
                float _fabs_42 = fabsf(acc_73[10]);
                float _fmax_42 = fmaxf(cur_220, _fabs_42);
                cur_220 = _fmax_42;
                float _fabs_43 = fabsf(acc_73[11]);
                float _fmax_43 = fmaxf(cur_220, _fabs_43);
                cur_220 = _fmax_43;
                float _fabs_44 = fabsf(acc_73[12]);
                float _fmax_44 = fmaxf(cur_220, _fabs_44);
                cur_220 = _fmax_44;
                float _fabs_45 = fabsf(acc_73[13]);
                float _fmax_45 = fmaxf(cur_220, _fabs_45);
                cur_220 = _fmax_45;
                float _fabs_46 = fabsf(acc_73[14]);
                float _fmax_46 = fmaxf(cur_220, _fabs_46);
                cur_220 = _fmax_46;
                float _fabs_47 = fabsf(acc_73[15]);
                float _fmax_47 = fmaxf(cur_220, _fabs_47);
                cur_220 = _fmax_47;
                float _fabs_48 = fabsf(acc_73[16]);
                float _fmax_48 = fmaxf(cur_220, _fabs_48);
                cur_220 = _fmax_48;
                float _fabs_49 = fabsf(acc_73[17]);
                float _fmax_49 = fmaxf(cur_220, _fabs_49);
                cur_220 = _fmax_49;
                float _fabs_50 = fabsf(acc_73[18]);
                float _fmax_50 = fmaxf(cur_220, _fabs_50);
                cur_220 = _fmax_50;
                float _fabs_51 = fabsf(acc_73[19]);
                float _fmax_51 = fmaxf(cur_220, _fabs_51);
                cur_220 = _fmax_51;
                float _fabs_52 = fabsf(acc_73[20]);
                float _fmax_52 = fmaxf(cur_220, _fabs_52);
                cur_220 = _fmax_52;
                float _fabs_53 = fabsf(acc_73[21]);
                float _fmax_53 = fmaxf(cur_220, _fabs_53);
                cur_220 = _fmax_53;
                float _fabs_54 = fabsf(acc_73[22]);
                float _fmax_54 = fmaxf(cur_220, _fabs_54);
                cur_220 = _fmax_54;
                float _fabs_55 = fabsf(acc_73[23]);
                float _fmax_55 = fmaxf(cur_220, _fabs_55);
                cur_220 = _fmax_55;
                float _fabs_56 = fabsf(acc_73[24]);
                float _fmax_56 = fmaxf(cur_220, _fabs_56);
                cur_220 = _fmax_56;
                float _fabs_57 = fabsf(acc_73[25]);
                float _fmax_57 = fmaxf(cur_220, _fabs_57);
                cur_220 = _fmax_57;
                float _fabs_58 = fabsf(acc_73[26]);
                float _fmax_58 = fmaxf(cur_220, _fabs_58);
                cur_220 = _fmax_58;
                float _fabs_59 = fabsf(acc_73[27]);
                float _fmax_59 = fmaxf(cur_220, _fabs_59);
                cur_220 = _fmax_59;
                float _fabs_60 = fabsf(acc_73[28]);
                float _fmax_60 = fmaxf(cur_220, _fabs_60);
                cur_220 = _fmax_60;
                float _fabs_61 = fabsf(acc_73[29]);
                float _fmax_61 = fmaxf(cur_220, _fabs_61);
                cur_220 = _fmax_61;
                float _fabs_62 = fabsf(acc_73[30]);
                float _fmax_62 = fmaxf(cur_220, _fabs_62);
                cur_220 = _fmax_62;
                float _fabs_63 = fabsf(acc_73[31]);
                float _fmax_63 = fmaxf(cur_220, _fabs_63);
                cur_220 = _fmax_63;
                am = cur_220;
                float cur_221 = am;
                float _fabs_64 = fabsf(acc_147[0]);
                float _fmax_64 = fmaxf(cur_221, _fabs_64);
                cur_221 = _fmax_64;
                float _fabs_65 = fabsf(acc_147[1]);
                float _fmax_65 = fmaxf(cur_221, _fabs_65);
                cur_221 = _fmax_65;
                float _fabs_66 = fabsf(acc_147[2]);
                float _fmax_66 = fmaxf(cur_221, _fabs_66);
                cur_221 = _fmax_66;
                float _fabs_67 = fabsf(acc_147[3]);
                float _fmax_67 = fmaxf(cur_221, _fabs_67);
                cur_221 = _fmax_67;
                float _fabs_68 = fabsf(acc_147[4]);
                float _fmax_68 = fmaxf(cur_221, _fabs_68);
                cur_221 = _fmax_68;
                float _fabs_69 = fabsf(acc_147[5]);
                float _fmax_69 = fmaxf(cur_221, _fabs_69);
                cur_221 = _fmax_69;
                float _fabs_70 = fabsf(acc_147[6]);
                float _fmax_70 = fmaxf(cur_221, _fabs_70);
                cur_221 = _fmax_70;
                float _fabs_71 = fabsf(acc_147[7]);
                float _fmax_71 = fmaxf(cur_221, _fabs_71);
                cur_221 = _fmax_71;
                float _fabs_72 = fabsf(acc_147[8]);
                float _fmax_72 = fmaxf(cur_221, _fabs_72);
                cur_221 = _fmax_72;
                float _fabs_73 = fabsf(acc_147[9]);
                float _fmax_73 = fmaxf(cur_221, _fabs_73);
                cur_221 = _fmax_73;
                float _fabs_74 = fabsf(acc_147[10]);
                float _fmax_74 = fmaxf(cur_221, _fabs_74);
                cur_221 = _fmax_74;
                float _fabs_75 = fabsf(acc_147[11]);
                float _fmax_75 = fmaxf(cur_221, _fabs_75);
                cur_221 = _fmax_75;
                float _fabs_76 = fabsf(acc_147[12]);
                float _fmax_76 = fmaxf(cur_221, _fabs_76);
                cur_221 = _fmax_76;
                float _fabs_77 = fabsf(acc_147[13]);
                float _fmax_77 = fmaxf(cur_221, _fabs_77);
                cur_221 = _fmax_77;
                float _fabs_78 = fabsf(acc_147[14]);
                float _fmax_78 = fmaxf(cur_221, _fabs_78);
                cur_221 = _fmax_78;
                float _fabs_79 = fabsf(acc_147[15]);
                float _fmax_79 = fmaxf(cur_221, _fabs_79);
                cur_221 = _fmax_79;
                float _fabs_80 = fabsf(acc_147[16]);
                float _fmax_80 = fmaxf(cur_221, _fabs_80);
                cur_221 = _fmax_80;
                float _fabs_81 = fabsf(acc_147[17]);
                float _fmax_81 = fmaxf(cur_221, _fabs_81);
                cur_221 = _fmax_81;
                float _fabs_82 = fabsf(acc_147[18]);
                float _fmax_82 = fmaxf(cur_221, _fabs_82);
                cur_221 = _fmax_82;
                float _fabs_83 = fabsf(acc_147[19]);
                float _fmax_83 = fmaxf(cur_221, _fabs_83);
                cur_221 = _fmax_83;
                float _fabs_84 = fabsf(acc_147[20]);
                float _fmax_84 = fmaxf(cur_221, _fabs_84);
                cur_221 = _fmax_84;
                float _fabs_85 = fabsf(acc_147[21]);
                float _fmax_85 = fmaxf(cur_221, _fabs_85);
                cur_221 = _fmax_85;
                float _fabs_86 = fabsf(acc_147[22]);
                float _fmax_86 = fmaxf(cur_221, _fabs_86);
                cur_221 = _fmax_86;
                float _fabs_87 = fabsf(acc_147[23]);
                float _fmax_87 = fmaxf(cur_221, _fabs_87);
                cur_221 = _fmax_87;
                float _fabs_88 = fabsf(acc_147[24]);
                float _fmax_88 = fmaxf(cur_221, _fabs_88);
                cur_221 = _fmax_88;
                float _fabs_89 = fabsf(acc_147[25]);
                float _fmax_89 = fmaxf(cur_221, _fabs_89);
                cur_221 = _fmax_89;
                float _fabs_90 = fabsf(acc_147[26]);
                float _fmax_90 = fmaxf(cur_221, _fabs_90);
                cur_221 = _fmax_90;
                float _fabs_91 = fabsf(acc_147[27]);
                float _fmax_91 = fmaxf(cur_221, _fabs_91);
                cur_221 = _fmax_91;
                float _fabs_92 = fabsf(acc_147[28]);
                float _fmax_92 = fmaxf(cur_221, _fabs_92);
                cur_221 = _fmax_92;
                float _fabs_93 = fabsf(acc_147[29]);
                float _fmax_93 = fmaxf(cur_221, _fabs_93);
                cur_221 = _fmax_93;
                float _fabs_94 = fabsf(acc_147[30]);
                float _fmax_94 = fmaxf(cur_221, _fabs_94);
                cur_221 = _fmax_94;
                float _fabs_95 = fabsf(acc_147[31]);
                float _fmax_95 = fmaxf(cur_221, _fabs_95);
                cur_221 = _fmax_95;
                am = cur_221;
                float _warp_reduce_0 = am;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 = fmaxf(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                am = _warp_reduce_0;
                int sam_base = 4 * t + 2 * kbp;
                if (lane_1 == 0) {
                    s_am[sam_base + h] = am;
                }
                asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                float _fmax_96 = fmaxf(s_am[sam_base], s_am[sam_base + 1]);
                float am2 = _fmax_96;
                float rs2 = 1.0f;
                float zero = 0.0f;
                float target = 2688.0f;
                float _fdiv_rn_0 = __fdiv_rn(target, am2);
                float inv = ((am2 > zero) ? _fdiv_rn_0 : zero);
                float _fdiv_rn_1 = __fdiv_rn(am2, target);
                float rs = _fdiv_rn_1;
                rs2 = rs;
                int pp2 = lane_1 + 32 * (3 * h);
                unsigned int q[4];
                float zero_222 = 0.0f;
                float six = 6.0f;
                float bm0 = 0.0f;
                float bm1 = 0.0f;
                float _fabs_96 = fabsf(acc[0]);
                float _fmax_97 = fmaxf(bm0, _fabs_96);
                bm0 = _fmax_97;
                float _fabs_97 = fabsf(acc[16]);
                float _fmax_98 = fmaxf(bm1, _fabs_97);
                bm1 = _fmax_98;
                float _fabs_98 = fabsf(acc[1]);
                float _fmax_99 = fmaxf(bm0, _fabs_98);
                bm0 = _fmax_99;
                float _fabs_99 = fabsf(acc[17]);
                float _fmax_100 = fmaxf(bm1, _fabs_99);
                bm1 = _fmax_100;
                float _fabs_100 = fabsf(acc[2]);
                float _fmax_101 = fmaxf(bm0, _fabs_100);
                bm0 = _fmax_101;
                float _fabs_101 = fabsf(acc[18]);
                float _fmax_102 = fmaxf(bm1, _fabs_101);
                bm1 = _fmax_102;
                float _fabs_102 = fabsf(acc[3]);
                float _fmax_103 = fmaxf(bm0, _fabs_102);
                bm0 = _fmax_103;
                float _fabs_103 = fabsf(acc[19]);
                float _fmax_104 = fmaxf(bm1, _fabs_103);
                bm1 = _fmax_104;
                float _fabs_104 = fabsf(acc[4]);
                float _fmax_105 = fmaxf(bm0, _fabs_104);
                bm0 = _fmax_105;
                float _fabs_105 = fabsf(acc[20]);
                float _fmax_106 = fmaxf(bm1, _fabs_105);
                bm1 = _fmax_106;
                float _fabs_106 = fabsf(acc[5]);
                float _fmax_107 = fmaxf(bm0, _fabs_106);
                bm0 = _fmax_107;
                float _fabs_107 = fabsf(acc[21]);
                float _fmax_108 = fmaxf(bm1, _fabs_107);
                bm1 = _fmax_108;
                float _fabs_108 = fabsf(acc[6]);
                float _fmax_109 = fmaxf(bm0, _fabs_108);
                bm0 = _fmax_109;
                float _fabs_109 = fabsf(acc[22]);
                float _fmax_110 = fmaxf(bm1, _fabs_109);
                bm1 = _fmax_110;
                float _fabs_110 = fabsf(acc[7]);
                float _fmax_111 = fmaxf(bm0, _fabs_110);
                bm0 = _fmax_111;
                float _fabs_111 = fabsf(acc[23]);
                float _fmax_112 = fmaxf(bm1, _fabs_111);
                bm1 = _fmax_112;
                float _fabs_112 = fabsf(acc[8]);
                float _fmax_113 = fmaxf(bm0, _fabs_112);
                bm0 = _fmax_113;
                float _fabs_113 = fabsf(acc[24]);
                float _fmax_114 = fmaxf(bm1, _fabs_113);
                bm1 = _fmax_114;
                float _fabs_114 = fabsf(acc[9]);
                float _fmax_115 = fmaxf(bm0, _fabs_114);
                bm0 = _fmax_115;
                float _fabs_115 = fabsf(acc[25]);
                float _fmax_116 = fmaxf(bm1, _fabs_115);
                bm1 = _fmax_116;
                float _fabs_116 = fabsf(acc[10]);
                float _fmax_117 = fmaxf(bm0, _fabs_116);
                bm0 = _fmax_117;
                float _fabs_117 = fabsf(acc[26]);
                float _fmax_118 = fmaxf(bm1, _fabs_117);
                bm1 = _fmax_118;
                float _fabs_118 = fabsf(acc[11]);
                float _fmax_119 = fmaxf(bm0, _fabs_118);
                bm0 = _fmax_119;
                float _fabs_119 = fabsf(acc[27]);
                float _fmax_120 = fmaxf(bm1, _fabs_119);
                bm1 = _fmax_120;
                float _fabs_120 = fabsf(acc[12]);
                float _fmax_121 = fmaxf(bm0, _fabs_120);
                bm0 = _fmax_121;
                float _fabs_121 = fabsf(acc[28]);
                float _fmax_122 = fmaxf(bm1, _fabs_121);
                bm1 = _fmax_122;
                float _fabs_122 = fabsf(acc[13]);
                float _fmax_123 = fmaxf(bm0, _fabs_122);
                bm0 = _fmax_123;
                float _fabs_123 = fabsf(acc[29]);
                float _fmax_124 = fmaxf(bm1, _fabs_123);
                bm1 = _fmax_124;
                float _fabs_124 = fabsf(acc[14]);
                float _fmax_125 = fmaxf(bm0, _fabs_124);
                bm0 = _fmax_125;
                float _fabs_125 = fabsf(acc[30]);
                float _fmax_126 = fmaxf(bm1, _fabs_125);
                bm1 = _fmax_126;
                float _fabs_126 = fabsf(acc[15]);
                float _fmax_127 = fmaxf(bm0, _fabs_126);
                bm0 = _fmax_127;
                float _fabs_127 = fabsf(acc[31]);
                float _fmax_128 = fmaxf(bm1, _fabs_127);
                bm1 = _fmax_128;
                float _mul_24;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(bm0), "f"(inv));
                float _fdiv_rn_2 = __fdiv_rn(_mul_24, six);
                float _mul_25;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(bm1), "f"(inv));
                float _fdiv_rn_3 = __fdiv_rn(_mul_25, six);
                uint16_t _e4m3x2_f32_0;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_fdiv_rn_3), "f"(_fdiv_rn_2));
                float2 _fp8x2_decode_12;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_12.x), "=f"(_fp8x2_decode_12.y) : "h"(_e4m3x2_f32_0));
                float _fdiv_rn_4 = __fdiv_rn(inv, _fp8x2_decode_12.x);
                float _fdiv_rn_5 = __fdiv_rn(inv, _fp8x2_decode_12.y);
                float m0 = ((zero_222 < _fp8x2_decode_12.x) ? _fdiv_rn_4 : zero_222);
                float m1 = ((zero_222 < _fp8x2_decode_12.y) ? _fdiv_rn_5 : zero_222);
                float2 _f2_408 = make_float2(m0, m0);
                float2 _f2_409 = make_float2(m1, m1);
                float2 _f2_410 = make_float2(acc[0], acc[1]);
                float2 _f32x2_mul_b_11 = ((1) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_410), "l"(*(const unsigned long long*)&_f32x2_mul_b_11));
                uint32_t _fp4_pair_0;
                uint16_t _e2m1_pair_h_12;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_12) : "f"(_mul_f32x2_0.x), "f"(_mul_f32x2_0.y));
                _fp4_pair_0 = (uint32_t)_e2m1_pair_h_12;
                float2 _f2_411 = make_float2(acc[2], acc[3]);
                float2 _f32x2_mul_b_13 = ((1) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_f2_411), "l"(*(const unsigned long long*)&_f32x2_mul_b_13));
                uint32_t _fp4_pair_1;
                uint16_t _e2m1_pair_h_14;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_14) : "f"(_mul_f32x2_1.x), "f"(_mul_f32x2_1.y));
                _fp4_pair_1 = (uint32_t)_e2m1_pair_h_14;
                float2 _f2_412 = make_float2(acc[4], acc[5]);
                float2 _f32x2_mul_b_15 = ((1) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_412), "l"(*(const unsigned long long*)&_f32x2_mul_b_15));
                uint32_t _fp4_pair_2;
                uint16_t _e2m1_pair_h_16;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_16) : "f"(_mul_f32x2_2.x), "f"(_mul_f32x2_2.y));
                _fp4_pair_2 = (uint32_t)_e2m1_pair_h_16;
                float2 _f2_413 = make_float2(acc[6], acc[7]);
                float2 _f32x2_mul_b_17 = ((1) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_f2_413), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
                uint32_t _fp4_pair_3;
                uint16_t _e2m1_pair_h_18;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_18) : "f"(_mul_f32x2_3.x), "f"(_mul_f32x2_3.y));
                _fp4_pair_3 = (uint32_t)_e2m1_pair_h_18;
                unsigned int word = _fp4_pair_0 | _fp4_pair_1 << 8 | _fp4_pair_2 << 16 | _fp4_pair_3 << 24;
                uint32_t _e2m1_canon_0;
                {
                    const uint32_t _canon_w_19 = word;
                    const uint32_t _canon_s_19 = (_canon_w_19 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_0 = _canon_w_19 & (_canon_s_19 | 0x77777777u);
                }
                q[0] = _e2m1_canon_0;
                float2 _f2_414 = make_float2(acc[8], acc[9]);
                float2 _f32x2_mul_b_20 = ((1) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_414), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
                uint32_t _fp4_pair_4;
                uint16_t _e2m1_pair_h_21;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_21) : "f"(_mul_f32x2_4.x), "f"(_mul_f32x2_4.y));
                _fp4_pair_4 = (uint32_t)_e2m1_pair_h_21;
                float2 _f2_415 = make_float2(acc[10], acc[11]);
                float2 _f32x2_mul_b_22 = ((1) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_f2_415), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
                uint32_t _fp4_pair_5;
                uint16_t _e2m1_pair_h_23;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_23) : "f"(_mul_f32x2_5.x), "f"(_mul_f32x2_5.y));
                _fp4_pair_5 = (uint32_t)_e2m1_pair_h_23;
                float2 _f2_416 = make_float2(acc[12], acc[13]);
                float2 _f32x2_mul_b_24 = ((1) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&_f2_416), "l"(*(const unsigned long long*)&_f32x2_mul_b_24));
                uint32_t _fp4_pair_6;
                uint16_t _e2m1_pair_h_25;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_25) : "f"(_mul_f32x2_6.x), "f"(_mul_f32x2_6.y));
                _fp4_pair_6 = (uint32_t)_e2m1_pair_h_25;
                float2 _f2_417 = make_float2(acc[14], acc[15]);
                float2 _f32x2_mul_b_26 = ((1) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&_f2_417), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
                uint32_t _fp4_pair_7;
                uint16_t _e2m1_pair_h_27;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_27) : "f"(_mul_f32x2_7.x), "f"(_mul_f32x2_7.y));
                _fp4_pair_7 = (uint32_t)_e2m1_pair_h_27;
                unsigned int word_223 = _fp4_pair_4 | _fp4_pair_5 << 8 | _fp4_pair_6 << 16 | _fp4_pair_7 << 24;
                uint32_t _e2m1_canon_1;
                {
                    const uint32_t _canon_w_28 = word_223;
                    const uint32_t _canon_s_28 = (_canon_w_28 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_1 = _canon_w_28 & (_canon_s_28 | 0x77777777u);
                }
                q[1] = _e2m1_canon_1;
                float2 _f2_418 = make_float2(acc[16], acc[17]);
                float2 _f32x2_mul_b_29 = ((0) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_8) : "l"(*(const unsigned long long*)&_f2_418), "l"(*(const unsigned long long*)&_f32x2_mul_b_29));
                uint32_t _fp4_pair_8;
                uint16_t _e2m1_pair_h_30;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_30) : "f"(_mul_f32x2_8.x), "f"(_mul_f32x2_8.y));
                _fp4_pair_8 = (uint32_t)_e2m1_pair_h_30;
                float2 _f2_419 = make_float2(acc[18], acc[19]);
                float2 _f32x2_mul_b_31 = ((0) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&_f2_419), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
                uint32_t _fp4_pair_9;
                uint16_t _e2m1_pair_h_32;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_32) : "f"(_mul_f32x2_9.x), "f"(_mul_f32x2_9.y));
                _fp4_pair_9 = (uint32_t)_e2m1_pair_h_32;
                float2 _f2_420 = make_float2(acc[20], acc[21]);
                float2 _f32x2_mul_b_33 = ((0) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_10) : "l"(*(const unsigned long long*)&_f2_420), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
                uint32_t _fp4_pair_10;
                uint16_t _e2m1_pair_h_34;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_34) : "f"(_mul_f32x2_10.x), "f"(_mul_f32x2_10.y));
                _fp4_pair_10 = (uint32_t)_e2m1_pair_h_34;
                float2 _f2_421 = make_float2(acc[22], acc[23]);
                float2 _f32x2_mul_b_35 = ((0) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_11) : "l"(*(const unsigned long long*)&_f2_421), "l"(*(const unsigned long long*)&_f32x2_mul_b_35));
                uint32_t _fp4_pair_11;
                uint16_t _e2m1_pair_h_36;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_36) : "f"(_mul_f32x2_11.x), "f"(_mul_f32x2_11.y));
                _fp4_pair_11 = (uint32_t)_e2m1_pair_h_36;
                unsigned int word_224 = _fp4_pair_8 | _fp4_pair_9 << 8 | _fp4_pair_10 << 16 | _fp4_pair_11 << 24;
                uint32_t _e2m1_canon_2;
                {
                    const uint32_t _canon_w_37 = word_224;
                    const uint32_t _canon_s_37 = (_canon_w_37 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_2 = _canon_w_37 & (_canon_s_37 | 0x77777777u);
                }
                q[2] = _e2m1_canon_2;
                float2 _f2_422 = make_float2(acc[24], acc[25]);
                float2 _f32x2_mul_b_38 = ((0) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_12) : "l"(*(const unsigned long long*)&_f2_422), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
                uint32_t _fp4_pair_12;
                uint16_t _e2m1_pair_h_39;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_39) : "f"(_mul_f32x2_12.x), "f"(_mul_f32x2_12.y));
                _fp4_pair_12 = (uint32_t)_e2m1_pair_h_39;
                float2 _f2_423 = make_float2(acc[26], acc[27]);
                float2 _f32x2_mul_b_40 = ((0) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_13) : "l"(*(const unsigned long long*)&_f2_423), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
                uint32_t _fp4_pair_13;
                uint16_t _e2m1_pair_h_41;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_41) : "f"(_mul_f32x2_13.x), "f"(_mul_f32x2_13.y));
                _fp4_pair_13 = (uint32_t)_e2m1_pair_h_41;
                float2 _f2_424 = make_float2(acc[28], acc[29]);
                float2 _f32x2_mul_b_42 = ((0) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&_f2_424), "l"(*(const unsigned long long*)&_f32x2_mul_b_42));
                uint32_t _fp4_pair_14;
                uint16_t _e2m1_pair_h_43;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_43) : "f"(_mul_f32x2_14.x), "f"(_mul_f32x2_14.y));
                _fp4_pair_14 = (uint32_t)_e2m1_pair_h_43;
                float2 _f2_425 = make_float2(acc[30], acc[31]);
                float2 _f32x2_mul_b_44 = ((0) ? _f2_408 : _f2_409);
                float2 _mul_f32x2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&_f2_425), "l"(*(const unsigned long long*)&_f32x2_mul_b_44));
                uint32_t _fp4_pair_15;
                uint16_t _e2m1_pair_h_45;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_45) : "f"(_mul_f32x2_15.x), "f"(_mul_f32x2_15.y));
                _fp4_pair_15 = (uint32_t)_e2m1_pair_h_45;
                unsigned int word_225 = _fp4_pair_12 | _fp4_pair_13 << 8 | _fp4_pair_14 << 16 | _fp4_pair_15 << 24;
                uint32_t _e2m1_canon_3;
                {
                    const uint32_t _canon_w_46 = word_225;
                    const uint32_t _canon_s_46 = (_canon_w_46 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_3 = _canon_w_46 & (_canon_s_46 | 0x77777777u);
                }
                q[3] = _e2m1_canon_3;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(tob_off + 16 * pp2)), "r"(q[0]), "r"(q[1]), "r"(q[2]), "r"(q[3]) : "memory");
                wsm16[(tob_off + 3072 >> 1) + pp2] = _e4m3x2_f32_0;
                float d[32];
                d[0] = 0.0f;
                d[1] = 0.0f;
                d[2] = 0.0f;
                d[3] = 0.0f;
                d[4] = 0.0f;
                d[5] = 0.0f;
                d[6] = 0.0f;
                d[7] = 0.0f;
                d[8] = 0.0f;
                d[9] = 0.0f;
                d[10] = 0.0f;
                d[11] = 0.0f;
                d[12] = 0.0f;
                d[13] = 0.0f;
                d[14] = 0.0f;
                d[15] = 0.0f;
                d[16] = 0.0f;
                d[17] = 0.0f;
                d[18] = 0.0f;
                d[19] = 0.0f;
                d[20] = 0.0f;
                d[21] = 0.0f;
                d[22] = 0.0f;
                d[23] = 0.0f;
                d[24] = 0.0f;
                d[25] = 0.0f;
                d[26] = 0.0f;
                d[27] = 0.0f;
                d[28] = 0.0f;
                d[29] = 0.0f;
                d[30] = 0.0f;
                d[31] = 0.0f;
                float2 _fp8x2_decode_13;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_13.x), "=f"(_fp8x2_decode_13.y) : "h"((uint16_t)_e4m3x2_f32_0));
                float _mul_26;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(_fp8x2_decode_13.x), "f"(rs2));
                float s0_226 = _mul_26;
                float _mul_27;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(_fp8x2_decode_13.y), "f"(rs2));
                float s1_227 = _mul_27;
                float2 _f2_426 = make_float2(s0_226, s0_226);
                float2 _f2_427 = make_float2(s1_227, s1_227);
                float _unpack_fp4_48[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q[0]));
                    (&_unpack_fp4_48[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_48[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_48[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_48[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_48[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_48[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_48[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_48[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_48[0]), "=f"(_unpack_fp4_48[1]), "=f"(_unpack_fp4_48[2]), "=f"(_unpack_fp4_48[3]), "=f"(_unpack_fp4_48[4]), "=f"(_unpack_fp4_48[5]), "=f"(_unpack_fp4_48[6]), "=f"(_unpack_fp4_48[7]) : "r"(q[0]));
                    #endif
                }
                float2 _f2_428 = make_float2(d[0], d[1]);
                float2 _f2_429 = make_float2(_unpack_fp4_48[0], _unpack_fp4_48[1]);
                float2 r_228 = fma_f32x2_rn_noftz(_f2_429, _f2_426, _f2_428);
                d[0] = r_228.x;
                d[1] = r_228.y;
                float2 _f2_430 = make_float2(d[2], d[3]);
                float2 _f2_431 = make_float2(_unpack_fp4_48[2], _unpack_fp4_48[3]);
                float2 r_229 = fma_f32x2_rn_noftz(_f2_431, _f2_426, _f2_430);
                d[2] = r_229.x;
                d[3] = r_229.y;
                float2 _f2_432 = make_float2(d[4], d[5]);
                float2 _f2_433 = make_float2(_unpack_fp4_48[4], _unpack_fp4_48[5]);
                float2 r_230 = fma_f32x2_rn_noftz(_f2_433, _f2_426, _f2_432);
                d[4] = r_230.x;
                d[5] = r_230.y;
                float2 _f2_434 = make_float2(d[6], d[7]);
                float2 _f2_435 = make_float2(_unpack_fp4_48[6], _unpack_fp4_48[7]);
                float2 r_231 = fma_f32x2_rn_noftz(_f2_435, _f2_426, _f2_434);
                d[6] = r_231.x;
                d[7] = r_231.y;
                float _unpack_fp4_49[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q[1 + 0]));
                    (&_unpack_fp4_49[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_49[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_49[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_49[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_49[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_49[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_49[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_49[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_49[0]), "=f"(_unpack_fp4_49[1]), "=f"(_unpack_fp4_49[2]), "=f"(_unpack_fp4_49[3]), "=f"(_unpack_fp4_49[4]), "=f"(_unpack_fp4_49[5]), "=f"(_unpack_fp4_49[6]), "=f"(_unpack_fp4_49[7]) : "r"(q[1 + 0]));
                    #endif
                }
                float2 _f2_436 = make_float2(d[8], d[9]);
                float2 _f2_437 = make_float2(_unpack_fp4_49[0], _unpack_fp4_49[1]);
                float2 r_232 = fma_f32x2_rn_noftz(_f2_437, _f2_426, _f2_436);
                d[8] = r_232.x;
                d[9] = r_232.y;
                float2 _f2_438 = make_float2(d[10], d[11]);
                float2 _f2_439 = make_float2(_unpack_fp4_49[2], _unpack_fp4_49[3]);
                float2 r_233 = fma_f32x2_rn_noftz(_f2_439, _f2_426, _f2_438);
                d[10] = r_233.x;
                d[11] = r_233.y;
                float2 _f2_440 = make_float2(d[12], d[13]);
                float2 _f2_441 = make_float2(_unpack_fp4_49[4], _unpack_fp4_49[5]);
                float2 r_234 = fma_f32x2_rn_noftz(_f2_441, _f2_426, _f2_440);
                d[12] = r_234.x;
                d[13] = r_234.y;
                float2 _f2_442 = make_float2(d[14], d[15]);
                float2 _f2_443 = make_float2(_unpack_fp4_49[6], _unpack_fp4_49[7]);
                float2 r_235 = fma_f32x2_rn_noftz(_f2_443, _f2_426, _f2_442);
                d[14] = r_235.x;
                d[15] = r_235.y;
                float _unpack_fp4_50[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q[2 + 0]));
                    (&_unpack_fp4_50[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_50[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_50[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_50[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_50[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_50[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_50[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_50[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_50[0]), "=f"(_unpack_fp4_50[1]), "=f"(_unpack_fp4_50[2]), "=f"(_unpack_fp4_50[3]), "=f"(_unpack_fp4_50[4]), "=f"(_unpack_fp4_50[5]), "=f"(_unpack_fp4_50[6]), "=f"(_unpack_fp4_50[7]) : "r"(q[2 + 0]));
                    #endif
                }
                float2 _f2_444 = make_float2(d[16], d[17]);
                float2 _f2_445 = make_float2(_unpack_fp4_50[0], _unpack_fp4_50[1]);
                float2 r_236 = fma_f32x2_rn_noftz(_f2_445, _f2_427, _f2_444);
                d[16] = r_236.x;
                d[17] = r_236.y;
                float2 _f2_446 = make_float2(d[18], d[19]);
                float2 _f2_447 = make_float2(_unpack_fp4_50[2], _unpack_fp4_50[3]);
                float2 r_237 = fma_f32x2_rn_noftz(_f2_447, _f2_427, _f2_446);
                d[18] = r_237.x;
                d[19] = r_237.y;
                float2 _f2_448 = make_float2(d[20], d[21]);
                float2 _f2_449 = make_float2(_unpack_fp4_50[4], _unpack_fp4_50[5]);
                float2 r_238 = fma_f32x2_rn_noftz(_f2_449, _f2_427, _f2_448);
                d[20] = r_238.x;
                d[21] = r_238.y;
                float2 _f2_450 = make_float2(d[22], d[23]);
                float2 _f2_451 = make_float2(_unpack_fp4_50[6], _unpack_fp4_50[7]);
                float2 r_239 = fma_f32x2_rn_noftz(_f2_451, _f2_427, _f2_450);
                d[22] = r_239.x;
                d[23] = r_239.y;
                float _unpack_fp4_51[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q[3 + 0]));
                    (&_unpack_fp4_51[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_51[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_51[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_51[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_51[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_51[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_51[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_51[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_51[0]), "=f"(_unpack_fp4_51[1]), "=f"(_unpack_fp4_51[2]), "=f"(_unpack_fp4_51[3]), "=f"(_unpack_fp4_51[4]), "=f"(_unpack_fp4_51[5]), "=f"(_unpack_fp4_51[6]), "=f"(_unpack_fp4_51[7]) : "r"(q[3 + 0]));
                    #endif
                }
                float2 _f2_452 = make_float2(d[24], d[25]);
                float2 _f2_453 = make_float2(_unpack_fp4_51[0], _unpack_fp4_51[1]);
                float2 r_240 = fma_f32x2_rn_noftz(_f2_453, _f2_427, _f2_452);
                d[24] = r_240.x;
                d[25] = r_240.y;
                float2 _f2_454 = make_float2(d[26], d[27]);
                float2 _f2_455 = make_float2(_unpack_fp4_51[2], _unpack_fp4_51[3]);
                float2 r_241 = fma_f32x2_rn_noftz(_f2_455, _f2_427, _f2_454);
                d[26] = r_241.x;
                d[27] = r_241.y;
                float2 _f2_456 = make_float2(d[28], d[29]);
                float2 _f2_457 = make_float2(_unpack_fp4_51[4], _unpack_fp4_51[5]);
                float2 r_242 = fma_f32x2_rn_noftz(_f2_457, _f2_427, _f2_456);
                d[28] = r_242.x;
                d[29] = r_242.y;
                float2 _f2_458 = make_float2(d[30], d[31]);
                float2 _f2_459 = make_float2(_unpack_fp4_51[6], _unpack_fp4_51[7]);
                float2 r_243 = fma_f32x2_rn_noftz(_f2_459, _f2_427, _f2_458);
                d[30] = r_243.x;
                d[31] = r_243.y;
                uint32_t d_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(d[_lp*2 + 0], d[_lp*2+1 + 0]));
                    d_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(d_bf16[0]), "r"(d_bf16[1]), "r"(d_bf16[2]), "r"(d_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(d_bf16[4]), "r"(d_bf16[5]), "r"(d_bf16[6]), "r"(d_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(d_bf16[8]), "r"(d_bf16[9]), "r"(d_bf16[10]), "r"(d_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(d_bf16[12]), "r"(d_bf16[13]), "r"(d_bf16[14]), "r"(d_bf16[15]) : "memory");
                __syncwarp();
                int c = lane_1;
                int l = c >> 2;
                unsigned int w4[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l + (c & 3 ^ l >> 1 & 3)))));
                long long idx = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c));
                {
                    int4 _iv4 = make_int4(w4[0 + 0], w4[0 + 1], w4[0 + 2], w4[0 + 3]);
                    *reinterpret_cast<int4*>(out_i32 + (idx)) = _iv4;
                }
                int c_244 = lane_1 + 32;
                int l_245 = c_244 >> 2;
                unsigned int w4_246[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_246[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_246[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_246[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_246[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_245 + (c_244 & 3 ^ l_245 >> 1 & 3)))));
                long long idx_247 = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c_244));
                {
                    int4 _iv4 = make_int4(w4_246[0 + 0], w4_246[0 + 1], w4_246[0 + 2], w4_246[0 + 3]);
                    *reinterpret_cast<int4*>(out_i32 + (idx_247)) = _iv4;
                }
                int c_248 = lane_1 + 64;
                int l_249 = c_248 >> 2;
                unsigned int w4_250[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_250[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_250[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_250[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_250[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_249 + (c_248 & 3 ^ l_249 >> 1 & 3)))));
                long long idx_251 = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c_248));
                {
                    int4 _iv4 = make_int4(w4_250[0 + 0], w4_250[0 + 1], w4_250[0 + 2], w4_250[0 + 3]);
                    *reinterpret_cast<int4*>(out_i32 + (idx_251)) = _iv4;
                }
                int c_252 = lane_1 + 96;
                int l_253 = c_252 >> 2;
                unsigned int w4_254[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_254[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_254[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_254[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_254[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_253 + (c_252 & 3 ^ l_253 >> 1 & 3)))));
                long long idx_255 = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c_252));
                {
                    int4 _iv4 = make_int4(w4_254[0 + 0], w4_254[0 + 1], w4_254[0 + 2], w4_254[0 + 3]);
                    *reinterpret_cast<int4*>(out_i32 + (idx_255)) = _iv4;
                }
                __syncwarp();
                int pp2_256 = lane_1 + 32 * (3 * h + 1);
                unsigned int q_257[4];
                float zero_258 = 0.0f;
                float six_259 = 6.0f;
                float bm0_260 = 0.0f;
                float bm1_261 = 0.0f;
                float _fabs_128 = fabsf(acc_73[0]);
                float _fmax_129 = fmaxf(bm0_260, _fabs_128);
                bm0_260 = _fmax_129;
                float _fabs_129 = fabsf(acc_73[16]);
                float _fmax_130 = fmaxf(bm1_261, _fabs_129);
                bm1_261 = _fmax_130;
                float _fabs_130 = fabsf(acc_73[1]);
                float _fmax_131 = fmaxf(bm0_260, _fabs_130);
                bm0_260 = _fmax_131;
                float _fabs_131 = fabsf(acc_73[17]);
                float _fmax_132 = fmaxf(bm1_261, _fabs_131);
                bm1_261 = _fmax_132;
                float _fabs_132 = fabsf(acc_73[2]);
                float _fmax_133 = fmaxf(bm0_260, _fabs_132);
                bm0_260 = _fmax_133;
                float _fabs_133 = fabsf(acc_73[18]);
                float _fmax_134 = fmaxf(bm1_261, _fabs_133);
                bm1_261 = _fmax_134;
                float _fabs_134 = fabsf(acc_73[3]);
                float _fmax_135 = fmaxf(bm0_260, _fabs_134);
                bm0_260 = _fmax_135;
                float _fabs_135 = fabsf(acc_73[19]);
                float _fmax_136 = fmaxf(bm1_261, _fabs_135);
                bm1_261 = _fmax_136;
                float _fabs_136 = fabsf(acc_73[4]);
                float _fmax_137 = fmaxf(bm0_260, _fabs_136);
                bm0_260 = _fmax_137;
                float _fabs_137 = fabsf(acc_73[20]);
                float _fmax_138 = fmaxf(bm1_261, _fabs_137);
                bm1_261 = _fmax_138;
                float _fabs_138 = fabsf(acc_73[5]);
                float _fmax_139 = fmaxf(bm0_260, _fabs_138);
                bm0_260 = _fmax_139;
                float _fabs_139 = fabsf(acc_73[21]);
                float _fmax_140 = fmaxf(bm1_261, _fabs_139);
                bm1_261 = _fmax_140;
                float _fabs_140 = fabsf(acc_73[6]);
                float _fmax_141 = fmaxf(bm0_260, _fabs_140);
                bm0_260 = _fmax_141;
                float _fabs_141 = fabsf(acc_73[22]);
                float _fmax_142 = fmaxf(bm1_261, _fabs_141);
                bm1_261 = _fmax_142;
                float _fabs_142 = fabsf(acc_73[7]);
                float _fmax_143 = fmaxf(bm0_260, _fabs_142);
                bm0_260 = _fmax_143;
                float _fabs_143 = fabsf(acc_73[23]);
                float _fmax_144 = fmaxf(bm1_261, _fabs_143);
                bm1_261 = _fmax_144;
                float _fabs_144 = fabsf(acc_73[8]);
                float _fmax_145 = fmaxf(bm0_260, _fabs_144);
                bm0_260 = _fmax_145;
                float _fabs_145 = fabsf(acc_73[24]);
                float _fmax_146 = fmaxf(bm1_261, _fabs_145);
                bm1_261 = _fmax_146;
                float _fabs_146 = fabsf(acc_73[9]);
                float _fmax_147 = fmaxf(bm0_260, _fabs_146);
                bm0_260 = _fmax_147;
                float _fabs_147 = fabsf(acc_73[25]);
                float _fmax_148 = fmaxf(bm1_261, _fabs_147);
                bm1_261 = _fmax_148;
                float _fabs_148 = fabsf(acc_73[10]);
                float _fmax_149 = fmaxf(bm0_260, _fabs_148);
                bm0_260 = _fmax_149;
                float _fabs_149 = fabsf(acc_73[26]);
                float _fmax_150 = fmaxf(bm1_261, _fabs_149);
                bm1_261 = _fmax_150;
                float _fabs_150 = fabsf(acc_73[11]);
                float _fmax_151 = fmaxf(bm0_260, _fabs_150);
                bm0_260 = _fmax_151;
                float _fabs_151 = fabsf(acc_73[27]);
                float _fmax_152 = fmaxf(bm1_261, _fabs_151);
                bm1_261 = _fmax_152;
                float _fabs_152 = fabsf(acc_73[12]);
                float _fmax_153 = fmaxf(bm0_260, _fabs_152);
                bm0_260 = _fmax_153;
                float _fabs_153 = fabsf(acc_73[28]);
                float _fmax_154 = fmaxf(bm1_261, _fabs_153);
                bm1_261 = _fmax_154;
                float _fabs_154 = fabsf(acc_73[13]);
                float _fmax_155 = fmaxf(bm0_260, _fabs_154);
                bm0_260 = _fmax_155;
                float _fabs_155 = fabsf(acc_73[29]);
                float _fmax_156 = fmaxf(bm1_261, _fabs_155);
                bm1_261 = _fmax_156;
                float _fabs_156 = fabsf(acc_73[14]);
                float _fmax_157 = fmaxf(bm0_260, _fabs_156);
                bm0_260 = _fmax_157;
                float _fabs_157 = fabsf(acc_73[30]);
                float _fmax_158 = fmaxf(bm1_261, _fabs_157);
                bm1_261 = _fmax_158;
                float _fabs_158 = fabsf(acc_73[15]);
                float _fmax_159 = fmaxf(bm0_260, _fabs_158);
                bm0_260 = _fmax_159;
                float _fabs_159 = fabsf(acc_73[31]);
                float _fmax_160 = fmaxf(bm1_261, _fabs_159);
                bm1_261 = _fmax_160;
                float _mul_28;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(bm0_260), "f"(inv));
                float _fdiv_rn_6 = __fdiv_rn(_mul_28, six_259);
                float _mul_29;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(bm1_261), "f"(inv));
                float _fdiv_rn_7 = __fdiv_rn(_mul_29, six_259);
                uint16_t _e4m3x2_f32_1;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_fdiv_rn_7), "f"(_fdiv_rn_6));
                float2 _fp8x2_decode_14;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_14.x), "=f"(_fp8x2_decode_14.y) : "h"(_e4m3x2_f32_1));
                float _fdiv_rn_8 = __fdiv_rn(inv, _fp8x2_decode_14.x);
                float _fdiv_rn_9 = __fdiv_rn(inv, _fp8x2_decode_14.y);
                float m0_262 = ((zero_258 < _fp8x2_decode_14.x) ? _fdiv_rn_8 : zero_258);
                float m1_263 = ((zero_258 < _fp8x2_decode_14.y) ? _fdiv_rn_9 : zero_258);
                float2 _f2_460 = make_float2(m0_262, m0_262);
                float2 _f2_461 = make_float2(m1_263, m1_263);
                float2 _f2_462 = make_float2(acc_73[0], acc_73[1]);
                float2 _f32x2_mul_b_47 = ((1) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_16;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_16) : "l"(*(const unsigned long long*)&_f2_462), "l"(*(const unsigned long long*)&_f32x2_mul_b_47));
                uint32_t _fp4_pair_16;
                uint16_t _e2m1_pair_h_48;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_48) : "f"(_mul_f32x2_16.x), "f"(_mul_f32x2_16.y));
                _fp4_pair_16 = (uint32_t)_e2m1_pair_h_48;
                float2 _f2_463 = make_float2(acc_73[2], acc_73[3]);
                float2 _f32x2_mul_b_49 = ((1) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_17;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_17) : "l"(*(const unsigned long long*)&_f2_463), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
                uint32_t _fp4_pair_17;
                uint16_t _e2m1_pair_h_50;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_50) : "f"(_mul_f32x2_17.x), "f"(_mul_f32x2_17.y));
                _fp4_pair_17 = (uint32_t)_e2m1_pair_h_50;
                float2 _f2_464 = make_float2(acc_73[4], acc_73[5]);
                float2 _f32x2_mul_b_51 = ((1) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_18;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_18) : "l"(*(const unsigned long long*)&_f2_464), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
                uint32_t _fp4_pair_18;
                uint16_t _e2m1_pair_h_52;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_52) : "f"(_mul_f32x2_18.x), "f"(_mul_f32x2_18.y));
                _fp4_pair_18 = (uint32_t)_e2m1_pair_h_52;
                float2 _f2_465 = make_float2(acc_73[6], acc_73[7]);
                float2 _f32x2_mul_b_53 = ((1) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_19;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_19) : "l"(*(const unsigned long long*)&_f2_465), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
                uint32_t _fp4_pair_19;
                uint16_t _e2m1_pair_h_54;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_54) : "f"(_mul_f32x2_19.x), "f"(_mul_f32x2_19.y));
                _fp4_pair_19 = (uint32_t)_e2m1_pair_h_54;
                unsigned int word_264 = _fp4_pair_16 | _fp4_pair_17 << 8 | _fp4_pair_18 << 16 | _fp4_pair_19 << 24;
                uint32_t _e2m1_canon_4;
                {
                    const uint32_t _canon_w_55 = word_264;
                    const uint32_t _canon_s_55 = (_canon_w_55 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_4 = _canon_w_55 & (_canon_s_55 | 0x77777777u);
                }
                q_257[0] = _e2m1_canon_4;
                float2 _f2_466 = make_float2(acc_73[8], acc_73[9]);
                float2 _f32x2_mul_b_56 = ((1) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_20) : "l"(*(const unsigned long long*)&_f2_466), "l"(*(const unsigned long long*)&_f32x2_mul_b_56));
                uint32_t _fp4_pair_20;
                uint16_t _e2m1_pair_h_57;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_57) : "f"(_mul_f32x2_20.x), "f"(_mul_f32x2_20.y));
                _fp4_pair_20 = (uint32_t)_e2m1_pair_h_57;
                float2 _f2_467 = make_float2(acc_73[10], acc_73[11]);
                float2 _f32x2_mul_b_58 = ((1) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_21) : "l"(*(const unsigned long long*)&_f2_467), "l"(*(const unsigned long long*)&_f32x2_mul_b_58));
                uint32_t _fp4_pair_21;
                uint16_t _e2m1_pair_h_59;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_59) : "f"(_mul_f32x2_21.x), "f"(_mul_f32x2_21.y));
                _fp4_pair_21 = (uint32_t)_e2m1_pair_h_59;
                float2 _f2_468 = make_float2(acc_73[12], acc_73[13]);
                float2 _f32x2_mul_b_60 = ((1) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_22) : "l"(*(const unsigned long long*)&_f2_468), "l"(*(const unsigned long long*)&_f32x2_mul_b_60));
                uint32_t _fp4_pair_22;
                uint16_t _e2m1_pair_h_61;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_61) : "f"(_mul_f32x2_22.x), "f"(_mul_f32x2_22.y));
                _fp4_pair_22 = (uint32_t)_e2m1_pair_h_61;
                float2 _f2_469 = make_float2(acc_73[14], acc_73[15]);
                float2 _f32x2_mul_b_62 = ((1) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_23) : "l"(*(const unsigned long long*)&_f2_469), "l"(*(const unsigned long long*)&_f32x2_mul_b_62));
                uint32_t _fp4_pair_23;
                uint16_t _e2m1_pair_h_63;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_63) : "f"(_mul_f32x2_23.x), "f"(_mul_f32x2_23.y));
                _fp4_pair_23 = (uint32_t)_e2m1_pair_h_63;
                unsigned int word_265 = _fp4_pair_20 | _fp4_pair_21 << 8 | _fp4_pair_22 << 16 | _fp4_pair_23 << 24;
                uint32_t _e2m1_canon_5;
                {
                    const uint32_t _canon_w_64 = word_265;
                    const uint32_t _canon_s_64 = (_canon_w_64 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_5 = _canon_w_64 & (_canon_s_64 | 0x77777777u);
                }
                q_257[1] = _e2m1_canon_5;
                float2 _f2_470 = make_float2(acc_73[16], acc_73[17]);
                float2 _f32x2_mul_b_65 = ((0) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_24;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_24) : "l"(*(const unsigned long long*)&_f2_470), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
                uint32_t _fp4_pair_24;
                uint16_t _e2m1_pair_h_66;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_66) : "f"(_mul_f32x2_24.x), "f"(_mul_f32x2_24.y));
                _fp4_pair_24 = (uint32_t)_e2m1_pair_h_66;
                float2 _f2_471 = make_float2(acc_73[18], acc_73[19]);
                float2 _f32x2_mul_b_67 = ((0) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_25;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_25) : "l"(*(const unsigned long long*)&_f2_471), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
                uint32_t _fp4_pair_25;
                uint16_t _e2m1_pair_h_68;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_68) : "f"(_mul_f32x2_25.x), "f"(_mul_f32x2_25.y));
                _fp4_pair_25 = (uint32_t)_e2m1_pair_h_68;
                float2 _f2_472 = make_float2(acc_73[20], acc_73[21]);
                float2 _f32x2_mul_b_69 = ((0) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_26;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_26) : "l"(*(const unsigned long long*)&_f2_472), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
                uint32_t _fp4_pair_26;
                uint16_t _e2m1_pair_h_70;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_70) : "f"(_mul_f32x2_26.x), "f"(_mul_f32x2_26.y));
                _fp4_pair_26 = (uint32_t)_e2m1_pair_h_70;
                float2 _f2_473 = make_float2(acc_73[22], acc_73[23]);
                float2 _f32x2_mul_b_71 = ((0) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_27;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_27) : "l"(*(const unsigned long long*)&_f2_473), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
                uint32_t _fp4_pair_27;
                uint16_t _e2m1_pair_h_72;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_72) : "f"(_mul_f32x2_27.x), "f"(_mul_f32x2_27.y));
                _fp4_pair_27 = (uint32_t)_e2m1_pair_h_72;
                unsigned int word_266 = _fp4_pair_24 | _fp4_pair_25 << 8 | _fp4_pair_26 << 16 | _fp4_pair_27 << 24;
                uint32_t _e2m1_canon_6;
                {
                    const uint32_t _canon_w_73 = word_266;
                    const uint32_t _canon_s_73 = (_canon_w_73 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_6 = _canon_w_73 & (_canon_s_73 | 0x77777777u);
                }
                q_257[2] = _e2m1_canon_6;
                float2 _f2_474 = make_float2(acc_73[24], acc_73[25]);
                float2 _f32x2_mul_b_74 = ((0) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_28;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_28) : "l"(*(const unsigned long long*)&_f2_474), "l"(*(const unsigned long long*)&_f32x2_mul_b_74));
                uint32_t _fp4_pair_28;
                uint16_t _e2m1_pair_h_75;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_75) : "f"(_mul_f32x2_28.x), "f"(_mul_f32x2_28.y));
                _fp4_pair_28 = (uint32_t)_e2m1_pair_h_75;
                float2 _f2_475 = make_float2(acc_73[26], acc_73[27]);
                float2 _f32x2_mul_b_76 = ((0) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_29;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_29) : "l"(*(const unsigned long long*)&_f2_475), "l"(*(const unsigned long long*)&_f32x2_mul_b_76));
                uint32_t _fp4_pair_29;
                uint16_t _e2m1_pair_h_77;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_77) : "f"(_mul_f32x2_29.x), "f"(_mul_f32x2_29.y));
                _fp4_pair_29 = (uint32_t)_e2m1_pair_h_77;
                float2 _f2_476 = make_float2(acc_73[28], acc_73[29]);
                float2 _f32x2_mul_b_78 = ((0) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_30;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_30) : "l"(*(const unsigned long long*)&_f2_476), "l"(*(const unsigned long long*)&_f32x2_mul_b_78));
                uint32_t _fp4_pair_30;
                uint16_t _e2m1_pair_h_79;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_79) : "f"(_mul_f32x2_30.x), "f"(_mul_f32x2_30.y));
                _fp4_pair_30 = (uint32_t)_e2m1_pair_h_79;
                float2 _f2_477 = make_float2(acc_73[30], acc_73[31]);
                float2 _f32x2_mul_b_80 = ((0) ? _f2_460 : _f2_461);
                float2 _mul_f32x2_31;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_31) : "l"(*(const unsigned long long*)&_f2_477), "l"(*(const unsigned long long*)&_f32x2_mul_b_80));
                uint32_t _fp4_pair_31;
                uint16_t _e2m1_pair_h_81;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_81) : "f"(_mul_f32x2_31.x), "f"(_mul_f32x2_31.y));
                _fp4_pair_31 = (uint32_t)_e2m1_pair_h_81;
                unsigned int word_267 = _fp4_pair_28 | _fp4_pair_29 << 8 | _fp4_pair_30 << 16 | _fp4_pair_31 << 24;
                uint32_t _e2m1_canon_7;
                {
                    const uint32_t _canon_w_82 = word_267;
                    const uint32_t _canon_s_82 = (_canon_w_82 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_7 = _canon_w_82 & (_canon_s_82 | 0x77777777u);
                }
                q_257[3] = _e2m1_canon_7;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(tob_off + 16 * pp2_256)), "r"(q_257[0]), "r"(q_257[1]), "r"(q_257[2]), "r"(q_257[3]) : "memory");
                wsm16[(tob_off + 3072 >> 1) + pp2_256] = _e4m3x2_f32_1;
                float d_268[32];
                d_268[0] = 0.0f;
                d_268[1] = 0.0f;
                d_268[2] = 0.0f;
                d_268[3] = 0.0f;
                d_268[4] = 0.0f;
                d_268[5] = 0.0f;
                d_268[6] = 0.0f;
                d_268[7] = 0.0f;
                d_268[8] = 0.0f;
                d_268[9] = 0.0f;
                d_268[10] = 0.0f;
                d_268[11] = 0.0f;
                d_268[12] = 0.0f;
                d_268[13] = 0.0f;
                d_268[14] = 0.0f;
                d_268[15] = 0.0f;
                d_268[16] = 0.0f;
                d_268[17] = 0.0f;
                d_268[18] = 0.0f;
                d_268[19] = 0.0f;
                d_268[20] = 0.0f;
                d_268[21] = 0.0f;
                d_268[22] = 0.0f;
                d_268[23] = 0.0f;
                d_268[24] = 0.0f;
                d_268[25] = 0.0f;
                d_268[26] = 0.0f;
                d_268[27] = 0.0f;
                d_268[28] = 0.0f;
                d_268[29] = 0.0f;
                d_268[30] = 0.0f;
                d_268[31] = 0.0f;
                float2 _fp8x2_decode_15;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_15.x), "=f"(_fp8x2_decode_15.y) : "h"((uint16_t)_e4m3x2_f32_1));
                float _mul_30;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(_fp8x2_decode_15.x), "f"(rs2));
                float s0_269 = _mul_30;
                float _mul_31;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(_fp8x2_decode_15.y), "f"(rs2));
                float s1_270 = _mul_31;
                float2 _f2_478 = make_float2(s0_269, s0_269);
                float2 _f2_479 = make_float2(s1_270, s1_270);
                float _unpack_fp4_52[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_257[0]));
                    (&_unpack_fp4_52[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_52[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_52[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_52[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_52[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_52[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_52[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_52[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_52[0]), "=f"(_unpack_fp4_52[1]), "=f"(_unpack_fp4_52[2]), "=f"(_unpack_fp4_52[3]), "=f"(_unpack_fp4_52[4]), "=f"(_unpack_fp4_52[5]), "=f"(_unpack_fp4_52[6]), "=f"(_unpack_fp4_52[7]) : "r"(q_257[0]));
                    #endif
                }
                float2 _f2_480 = make_float2(d_268[0], d_268[1]);
                float2 _f2_481 = make_float2(_unpack_fp4_52[0], _unpack_fp4_52[1]);
                float2 r_271 = fma_f32x2_rn_noftz(_f2_481, _f2_478, _f2_480);
                d_268[0] = r_271.x;
                d_268[1] = r_271.y;
                float2 _f2_482 = make_float2(d_268[2], d_268[3]);
                float2 _f2_483 = make_float2(_unpack_fp4_52[2], _unpack_fp4_52[3]);
                float2 r_272 = fma_f32x2_rn_noftz(_f2_483, _f2_478, _f2_482);
                d_268[2] = r_272.x;
                d_268[3] = r_272.y;
                float2 _f2_484 = make_float2(d_268[4], d_268[5]);
                float2 _f2_485 = make_float2(_unpack_fp4_52[4], _unpack_fp4_52[5]);
                float2 r_273 = fma_f32x2_rn_noftz(_f2_485, _f2_478, _f2_484);
                d_268[4] = r_273.x;
                d_268[5] = r_273.y;
                float2 _f2_486 = make_float2(d_268[6], d_268[7]);
                float2 _f2_487 = make_float2(_unpack_fp4_52[6], _unpack_fp4_52[7]);
                float2 r_274 = fma_f32x2_rn_noftz(_f2_487, _f2_478, _f2_486);
                d_268[6] = r_274.x;
                d_268[7] = r_274.y;
                float _unpack_fp4_53[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_257[1 + 0]));
                    (&_unpack_fp4_53[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_53[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_53[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_53[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_53[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_53[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_53[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_53[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_53[0]), "=f"(_unpack_fp4_53[1]), "=f"(_unpack_fp4_53[2]), "=f"(_unpack_fp4_53[3]), "=f"(_unpack_fp4_53[4]), "=f"(_unpack_fp4_53[5]), "=f"(_unpack_fp4_53[6]), "=f"(_unpack_fp4_53[7]) : "r"(q_257[1 + 0]));
                    #endif
                }
                float2 _f2_488 = make_float2(d_268[8], d_268[9]);
                float2 _f2_489 = make_float2(_unpack_fp4_53[0], _unpack_fp4_53[1]);
                float2 r_275 = fma_f32x2_rn_noftz(_f2_489, _f2_478, _f2_488);
                d_268[8] = r_275.x;
                d_268[9] = r_275.y;
                float2 _f2_490 = make_float2(d_268[10], d_268[11]);
                float2 _f2_491 = make_float2(_unpack_fp4_53[2], _unpack_fp4_53[3]);
                float2 r_276 = fma_f32x2_rn_noftz(_f2_491, _f2_478, _f2_490);
                d_268[10] = r_276.x;
                d_268[11] = r_276.y;
                float2 _f2_492 = make_float2(d_268[12], d_268[13]);
                float2 _f2_493 = make_float2(_unpack_fp4_53[4], _unpack_fp4_53[5]);
                float2 r_277 = fma_f32x2_rn_noftz(_f2_493, _f2_478, _f2_492);
                d_268[12] = r_277.x;
                d_268[13] = r_277.y;
                float2 _f2_494 = make_float2(d_268[14], d_268[15]);
                float2 _f2_495 = make_float2(_unpack_fp4_53[6], _unpack_fp4_53[7]);
                float2 r_278 = fma_f32x2_rn_noftz(_f2_495, _f2_478, _f2_494);
                d_268[14] = r_278.x;
                d_268[15] = r_278.y;
                float _unpack_fp4_54[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_257[2 + 0]));
                    (&_unpack_fp4_54[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_54[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_54[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_54[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_54[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_54[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_54[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_54[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_54[0]), "=f"(_unpack_fp4_54[1]), "=f"(_unpack_fp4_54[2]), "=f"(_unpack_fp4_54[3]), "=f"(_unpack_fp4_54[4]), "=f"(_unpack_fp4_54[5]), "=f"(_unpack_fp4_54[6]), "=f"(_unpack_fp4_54[7]) : "r"(q_257[2 + 0]));
                    #endif
                }
                float2 _f2_496 = make_float2(d_268[16], d_268[17]);
                float2 _f2_497 = make_float2(_unpack_fp4_54[0], _unpack_fp4_54[1]);
                float2 r_279 = fma_f32x2_rn_noftz(_f2_497, _f2_479, _f2_496);
                d_268[16] = r_279.x;
                d_268[17] = r_279.y;
                float2 _f2_498 = make_float2(d_268[18], d_268[19]);
                float2 _f2_499 = make_float2(_unpack_fp4_54[2], _unpack_fp4_54[3]);
                float2 r_280 = fma_f32x2_rn_noftz(_f2_499, _f2_479, _f2_498);
                d_268[18] = r_280.x;
                d_268[19] = r_280.y;
                float2 _f2_500 = make_float2(d_268[20], d_268[21]);
                float2 _f2_501 = make_float2(_unpack_fp4_54[4], _unpack_fp4_54[5]);
                float2 r_281 = fma_f32x2_rn_noftz(_f2_501, _f2_479, _f2_500);
                d_268[20] = r_281.x;
                d_268[21] = r_281.y;
                float2 _f2_502 = make_float2(d_268[22], d_268[23]);
                float2 _f2_503 = make_float2(_unpack_fp4_54[6], _unpack_fp4_54[7]);
                float2 r_282 = fma_f32x2_rn_noftz(_f2_503, _f2_479, _f2_502);
                d_268[22] = r_282.x;
                d_268[23] = r_282.y;
                float _unpack_fp4_55[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_257[3 + 0]));
                    (&_unpack_fp4_55[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_55[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_55[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_55[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_55[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_55[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_55[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_55[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_55[0]), "=f"(_unpack_fp4_55[1]), "=f"(_unpack_fp4_55[2]), "=f"(_unpack_fp4_55[3]), "=f"(_unpack_fp4_55[4]), "=f"(_unpack_fp4_55[5]), "=f"(_unpack_fp4_55[6]), "=f"(_unpack_fp4_55[7]) : "r"(q_257[3 + 0]));
                    #endif
                }
                float2 _f2_504 = make_float2(d_268[24], d_268[25]);
                float2 _f2_505 = make_float2(_unpack_fp4_55[0], _unpack_fp4_55[1]);
                float2 r_283 = fma_f32x2_rn_noftz(_f2_505, _f2_479, _f2_504);
                d_268[24] = r_283.x;
                d_268[25] = r_283.y;
                float2 _f2_506 = make_float2(d_268[26], d_268[27]);
                float2 _f2_507 = make_float2(_unpack_fp4_55[2], _unpack_fp4_55[3]);
                float2 r_284 = fma_f32x2_rn_noftz(_f2_507, _f2_479, _f2_506);
                d_268[26] = r_284.x;
                d_268[27] = r_284.y;
                float2 _f2_508 = make_float2(d_268[28], d_268[29]);
                float2 _f2_509 = make_float2(_unpack_fp4_55[4], _unpack_fp4_55[5]);
                float2 r_285 = fma_f32x2_rn_noftz(_f2_509, _f2_479, _f2_508);
                d_268[28] = r_285.x;
                d_268[29] = r_285.y;
                float2 _f2_510 = make_float2(d_268[30], d_268[31]);
                float2 _f2_511 = make_float2(_unpack_fp4_55[6], _unpack_fp4_55[7]);
                float2 r_286 = fma_f32x2_rn_noftz(_f2_511, _f2_479, _f2_510);
                d_268[30] = r_286.x;
                d_268[31] = r_286.y;
                uint32_t d_268_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(d_268[_lp*2 + 0], d_268[_lp*2+1 + 0]));
                    d_268_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(d_268_bf16[0]), "r"(d_268_bf16[1]), "r"(d_268_bf16[2]), "r"(d_268_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(d_268_bf16[4]), "r"(d_268_bf16[5]), "r"(d_268_bf16[6]), "r"(d_268_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(d_268_bf16[8]), "r"(d_268_bf16[9]), "r"(d_268_bf16[10]), "r"(d_268_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(d_268_bf16[12]), "r"(d_268_bf16[13]), "r"(d_268_bf16[14]), "r"(d_268_bf16[15]) : "memory");
                __syncwarp();
                int c_287 = lane_1;
                int l_288 = c_287 >> 2;
                unsigned int w4_289[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_289[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_289[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_289[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_289[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_288 + (c_287 & 3 ^ l_288 >> 1 & 3)))));
                long long idx_290 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_287));
                {
                    int4 _iv4 = make_int4(w4_289[0 + 0], w4_289[0 + 1], w4_289[0 + 2], w4_289[0 + 3]);
                    *reinterpret_cast<int4*>(out_i32 + (idx_290)) = _iv4;
                }
                int c_291 = lane_1 + 32;
                int l_292 = c_291 >> 2;
                unsigned int w4_293[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_293[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_293[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_293[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_293[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_292 + (c_291 & 3 ^ l_292 >> 1 & 3)))));
                long long idx_294 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_291));
                {
                    int4 _iv4 = make_int4(w4_293[0 + 0], w4_293[0 + 1], w4_293[0 + 2], w4_293[0 + 3]);
                    *reinterpret_cast<int4*>(out_i32 + (idx_294)) = _iv4;
                }
                int c_295 = lane_1 + 64;
                int l_296 = c_295 >> 2;
                unsigned int w4_297[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_297[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_297[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_297[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_297[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_296 + (c_295 & 3 ^ l_296 >> 1 & 3)))));
                long long idx_298 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_295));
                {
                    int4 _iv4 = make_int4(w4_297[0 + 0], w4_297[0 + 1], w4_297[0 + 2], w4_297[0 + 3]);
                    *reinterpret_cast<int4*>(out_i32 + (idx_298)) = _iv4;
                }
                int c_299 = lane_1 + 96;
                int l_300 = c_299 >> 2;
                unsigned int w4_301[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_301[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_301[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_301[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_301[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_300 + (c_299 & 3 ^ l_300 >> 1 & 3)))));
                long long idx_302 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_299));
                {
                    int4 _iv4 = make_int4(w4_301[0 + 0], w4_301[0 + 1], w4_301[0 + 2], w4_301[0 + 3]);
                    *reinterpret_cast<int4*>(out_i32 + (idx_302)) = _iv4;
                }
                __syncwarp();
                int pp2_303 = lane_1 + 32 * (3 * h + 2);
                unsigned int q_304[4];
                float zero_305 = 0.0f;
                float six_306 = 6.0f;
                float bm0_307 = 0.0f;
                float bm1_308 = 0.0f;
                float _fabs_160 = fabsf(acc_147[0]);
                float _fmax_161 = fmaxf(bm0_307, _fabs_160);
                bm0_307 = _fmax_161;
                float _fabs_161 = fabsf(acc_147[16]);
                float _fmax_162 = fmaxf(bm1_308, _fabs_161);
                bm1_308 = _fmax_162;
                float _fabs_162 = fabsf(acc_147[1]);
                float _fmax_163 = fmaxf(bm0_307, _fabs_162);
                bm0_307 = _fmax_163;
                float _fabs_163 = fabsf(acc_147[17]);
                float _fmax_164 = fmaxf(bm1_308, _fabs_163);
                bm1_308 = _fmax_164;
                float _fabs_164 = fabsf(acc_147[2]);
                float _fmax_165 = fmaxf(bm0_307, _fabs_164);
                bm0_307 = _fmax_165;
                float _fabs_165 = fabsf(acc_147[18]);
                float _fmax_166 = fmaxf(bm1_308, _fabs_165);
                bm1_308 = _fmax_166;
                float _fabs_166 = fabsf(acc_147[3]);
                float _fmax_167 = fmaxf(bm0_307, _fabs_166);
                bm0_307 = _fmax_167;
                float _fabs_167 = fabsf(acc_147[19]);
                float _fmax_168 = fmaxf(bm1_308, _fabs_167);
                bm1_308 = _fmax_168;
                float _fabs_168 = fabsf(acc_147[4]);
                float _fmax_169 = fmaxf(bm0_307, _fabs_168);
                bm0_307 = _fmax_169;
                float _fabs_169 = fabsf(acc_147[20]);
                float _fmax_170 = fmaxf(bm1_308, _fabs_169);
                bm1_308 = _fmax_170;
                float _fabs_170 = fabsf(acc_147[5]);
                float _fmax_171 = fmaxf(bm0_307, _fabs_170);
                bm0_307 = _fmax_171;
                float _fabs_171 = fabsf(acc_147[21]);
                float _fmax_172 = fmaxf(bm1_308, _fabs_171);
                bm1_308 = _fmax_172;
                float _fabs_172 = fabsf(acc_147[6]);
                float _fmax_173 = fmaxf(bm0_307, _fabs_172);
                bm0_307 = _fmax_173;
                float _fabs_173 = fabsf(acc_147[22]);
                float _fmax_174 = fmaxf(bm1_308, _fabs_173);
                bm1_308 = _fmax_174;
                float _fabs_174 = fabsf(acc_147[7]);
                float _fmax_175 = fmaxf(bm0_307, _fabs_174);
                bm0_307 = _fmax_175;
                float _fabs_175 = fabsf(acc_147[23]);
                float _fmax_176 = fmaxf(bm1_308, _fabs_175);
                bm1_308 = _fmax_176;
                float _fabs_176 = fabsf(acc_147[8]);
                float _fmax_177 = fmaxf(bm0_307, _fabs_176);
                bm0_307 = _fmax_177;
                float _fabs_177 = fabsf(acc_147[24]);
                float _fmax_178 = fmaxf(bm1_308, _fabs_177);
                bm1_308 = _fmax_178;
                float _fabs_178 = fabsf(acc_147[9]);
                float _fmax_179 = fmaxf(bm0_307, _fabs_178);
                bm0_307 = _fmax_179;
                float _fabs_179 = fabsf(acc_147[25]);
                float _fmax_180 = fmaxf(bm1_308, _fabs_179);
                bm1_308 = _fmax_180;
                float _fabs_180 = fabsf(acc_147[10]);
                float _fmax_181 = fmaxf(bm0_307, _fabs_180);
                bm0_307 = _fmax_181;
                float _fabs_181 = fabsf(acc_147[26]);
                float _fmax_182 = fmaxf(bm1_308, _fabs_181);
                bm1_308 = _fmax_182;
                float _fabs_182 = fabsf(acc_147[11]);
                float _fmax_183 = fmaxf(bm0_307, _fabs_182);
                bm0_307 = _fmax_183;
                float _fabs_183 = fabsf(acc_147[27]);
                float _fmax_184 = fmaxf(bm1_308, _fabs_183);
                bm1_308 = _fmax_184;
                float _fabs_184 = fabsf(acc_147[12]);
                float _fmax_185 = fmaxf(bm0_307, _fabs_184);
                bm0_307 = _fmax_185;
                float _fabs_185 = fabsf(acc_147[28]);
                float _fmax_186 = fmaxf(bm1_308, _fabs_185);
                bm1_308 = _fmax_186;
                float _fabs_186 = fabsf(acc_147[13]);
                float _fmax_187 = fmaxf(bm0_307, _fabs_186);
                bm0_307 = _fmax_187;
                float _fabs_187 = fabsf(acc_147[29]);
                float _fmax_188 = fmaxf(bm1_308, _fabs_187);
                bm1_308 = _fmax_188;
                float _fabs_188 = fabsf(acc_147[14]);
                float _fmax_189 = fmaxf(bm0_307, _fabs_188);
                bm0_307 = _fmax_189;
                float _fabs_189 = fabsf(acc_147[30]);
                float _fmax_190 = fmaxf(bm1_308, _fabs_189);
                bm1_308 = _fmax_190;
                float _fabs_190 = fabsf(acc_147[15]);
                float _fmax_191 = fmaxf(bm0_307, _fabs_190);
                bm0_307 = _fmax_191;
                float _fabs_191 = fabsf(acc_147[31]);
                float _fmax_192 = fmaxf(bm1_308, _fabs_191);
                bm1_308 = _fmax_192;
                float _mul_32;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(bm0_307), "f"(inv));
                float _fdiv_rn_10 = __fdiv_rn(_mul_32, six_306);
                float _mul_33;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(bm1_308), "f"(inv));
                float _fdiv_rn_11 = __fdiv_rn(_mul_33, six_306);
                uint16_t _e4m3x2_f32_2;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(_fdiv_rn_11), "f"(_fdiv_rn_10));
                float2 _fp8x2_decode_16;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_16.x), "=f"(_fp8x2_decode_16.y) : "h"(_e4m3x2_f32_2));
                float _fdiv_rn_12 = __fdiv_rn(inv, _fp8x2_decode_16.x);
                float _fdiv_rn_13 = __fdiv_rn(inv, _fp8x2_decode_16.y);
                float m0_309 = ((zero_305 < _fp8x2_decode_16.x) ? _fdiv_rn_12 : zero_305);
                float m1_310 = ((zero_305 < _fp8x2_decode_16.y) ? _fdiv_rn_13 : zero_305);
                float2 _f2_512 = make_float2(m0_309, m0_309);
                float2 _f2_513 = make_float2(m1_310, m1_310);
                float2 _f2_514 = make_float2(acc_147[0], acc_147[1]);
                float2 _f32x2_mul_b_83 = ((1) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_32;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_32) : "l"(*(const unsigned long long*)&_f2_514), "l"(*(const unsigned long long*)&_f32x2_mul_b_83));
                uint32_t _fp4_pair_32;
                uint16_t _e2m1_pair_h_84;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_84) : "f"(_mul_f32x2_32.x), "f"(_mul_f32x2_32.y));
                _fp4_pair_32 = (uint32_t)_e2m1_pair_h_84;
                float2 _f2_515 = make_float2(acc_147[2], acc_147[3]);
                float2 _f32x2_mul_b_85 = ((1) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_33;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_33) : "l"(*(const unsigned long long*)&_f2_515), "l"(*(const unsigned long long*)&_f32x2_mul_b_85));
                uint32_t _fp4_pair_33;
                uint16_t _e2m1_pair_h_86;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_86) : "f"(_mul_f32x2_33.x), "f"(_mul_f32x2_33.y));
                _fp4_pair_33 = (uint32_t)_e2m1_pair_h_86;
                float2 _f2_516 = make_float2(acc_147[4], acc_147[5]);
                float2 _f32x2_mul_b_87 = ((1) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_34;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_34) : "l"(*(const unsigned long long*)&_f2_516), "l"(*(const unsigned long long*)&_f32x2_mul_b_87));
                uint32_t _fp4_pair_34;
                uint16_t _e2m1_pair_h_88;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_88) : "f"(_mul_f32x2_34.x), "f"(_mul_f32x2_34.y));
                _fp4_pair_34 = (uint32_t)_e2m1_pair_h_88;
                float2 _f2_517 = make_float2(acc_147[6], acc_147[7]);
                float2 _f32x2_mul_b_89 = ((1) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_35;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_35) : "l"(*(const unsigned long long*)&_f2_517), "l"(*(const unsigned long long*)&_f32x2_mul_b_89));
                uint32_t _fp4_pair_35;
                uint16_t _e2m1_pair_h_90;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_90) : "f"(_mul_f32x2_35.x), "f"(_mul_f32x2_35.y));
                _fp4_pair_35 = (uint32_t)_e2m1_pair_h_90;
                unsigned int word_311 = _fp4_pair_32 | _fp4_pair_33 << 8 | _fp4_pair_34 << 16 | _fp4_pair_35 << 24;
                uint32_t _e2m1_canon_8;
                {
                    const uint32_t _canon_w_91 = word_311;
                    const uint32_t _canon_s_91 = (_canon_w_91 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_8 = _canon_w_91 & (_canon_s_91 | 0x77777777u);
                }
                q_304[0] = _e2m1_canon_8;
                float2 _f2_518 = make_float2(acc_147[8], acc_147[9]);
                float2 _f32x2_mul_b_92 = ((1) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_36;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_36) : "l"(*(const unsigned long long*)&_f2_518), "l"(*(const unsigned long long*)&_f32x2_mul_b_92));
                uint32_t _fp4_pair_36;
                uint16_t _e2m1_pair_h_93;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_93) : "f"(_mul_f32x2_36.x), "f"(_mul_f32x2_36.y));
                _fp4_pair_36 = (uint32_t)_e2m1_pair_h_93;
                float2 _f2_519 = make_float2(acc_147[10], acc_147[11]);
                float2 _f32x2_mul_b_94 = ((1) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_37;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_37) : "l"(*(const unsigned long long*)&_f2_519), "l"(*(const unsigned long long*)&_f32x2_mul_b_94));
                uint32_t _fp4_pair_37;
                uint16_t _e2m1_pair_h_95;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_95) : "f"(_mul_f32x2_37.x), "f"(_mul_f32x2_37.y));
                _fp4_pair_37 = (uint32_t)_e2m1_pair_h_95;
                float2 _f2_520 = make_float2(acc_147[12], acc_147[13]);
                float2 _f32x2_mul_b_96 = ((1) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_38;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_38) : "l"(*(const unsigned long long*)&_f2_520), "l"(*(const unsigned long long*)&_f32x2_mul_b_96));
                uint32_t _fp4_pair_38;
                uint16_t _e2m1_pair_h_97;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_97) : "f"(_mul_f32x2_38.x), "f"(_mul_f32x2_38.y));
                _fp4_pair_38 = (uint32_t)_e2m1_pair_h_97;
                float2 _f2_521 = make_float2(acc_147[14], acc_147[15]);
                float2 _f32x2_mul_b_98 = ((1) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_39;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_39) : "l"(*(const unsigned long long*)&_f2_521), "l"(*(const unsigned long long*)&_f32x2_mul_b_98));
                uint32_t _fp4_pair_39;
                uint16_t _e2m1_pair_h_99;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_99) : "f"(_mul_f32x2_39.x), "f"(_mul_f32x2_39.y));
                _fp4_pair_39 = (uint32_t)_e2m1_pair_h_99;
                unsigned int word_312 = _fp4_pair_36 | _fp4_pair_37 << 8 | _fp4_pair_38 << 16 | _fp4_pair_39 << 24;
                uint32_t _e2m1_canon_9;
                {
                    const uint32_t _canon_w_100 = word_312;
                    const uint32_t _canon_s_100 = (_canon_w_100 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_9 = _canon_w_100 & (_canon_s_100 | 0x77777777u);
                }
                q_304[1] = _e2m1_canon_9;
                float2 _f2_522 = make_float2(acc_147[16], acc_147[17]);
                float2 _f32x2_mul_b_101 = ((0) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_40;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_40) : "l"(*(const unsigned long long*)&_f2_522), "l"(*(const unsigned long long*)&_f32x2_mul_b_101));
                uint32_t _fp4_pair_40;
                uint16_t _e2m1_pair_h_102;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_102) : "f"(_mul_f32x2_40.x), "f"(_mul_f32x2_40.y));
                _fp4_pair_40 = (uint32_t)_e2m1_pair_h_102;
                float2 _f2_523 = make_float2(acc_147[18], acc_147[19]);
                float2 _f32x2_mul_b_103 = ((0) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_41;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_41) : "l"(*(const unsigned long long*)&_f2_523), "l"(*(const unsigned long long*)&_f32x2_mul_b_103));
                uint32_t _fp4_pair_41;
                uint16_t _e2m1_pair_h_104;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_104) : "f"(_mul_f32x2_41.x), "f"(_mul_f32x2_41.y));
                _fp4_pair_41 = (uint32_t)_e2m1_pair_h_104;
                float2 _f2_524 = make_float2(acc_147[20], acc_147[21]);
                float2 _f32x2_mul_b_105 = ((0) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_42;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_42) : "l"(*(const unsigned long long*)&_f2_524), "l"(*(const unsigned long long*)&_f32x2_mul_b_105));
                uint32_t _fp4_pair_42;
                uint16_t _e2m1_pair_h_106;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_106) : "f"(_mul_f32x2_42.x), "f"(_mul_f32x2_42.y));
                _fp4_pair_42 = (uint32_t)_e2m1_pair_h_106;
                float2 _f2_525 = make_float2(acc_147[22], acc_147[23]);
                float2 _f32x2_mul_b_107 = ((0) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_43;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_43) : "l"(*(const unsigned long long*)&_f2_525), "l"(*(const unsigned long long*)&_f32x2_mul_b_107));
                uint32_t _fp4_pair_43;
                uint16_t _e2m1_pair_h_108;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_108) : "f"(_mul_f32x2_43.x), "f"(_mul_f32x2_43.y));
                _fp4_pair_43 = (uint32_t)_e2m1_pair_h_108;
                unsigned int word_313 = _fp4_pair_40 | _fp4_pair_41 << 8 | _fp4_pair_42 << 16 | _fp4_pair_43 << 24;
                uint32_t _e2m1_canon_10;
                {
                    const uint32_t _canon_w_109 = word_313;
                    const uint32_t _canon_s_109 = (_canon_w_109 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_10 = _canon_w_109 & (_canon_s_109 | 0x77777777u);
                }
                q_304[2] = _e2m1_canon_10;
                float2 _f2_526 = make_float2(acc_147[24], acc_147[25]);
                float2 _f32x2_mul_b_110 = ((0) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_44;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_44) : "l"(*(const unsigned long long*)&_f2_526), "l"(*(const unsigned long long*)&_f32x2_mul_b_110));
                uint32_t _fp4_pair_44;
                uint16_t _e2m1_pair_h_111;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_111) : "f"(_mul_f32x2_44.x), "f"(_mul_f32x2_44.y));
                _fp4_pair_44 = (uint32_t)_e2m1_pair_h_111;
                float2 _f2_527 = make_float2(acc_147[26], acc_147[27]);
                float2 _f32x2_mul_b_112 = ((0) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_45;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_45) : "l"(*(const unsigned long long*)&_f2_527), "l"(*(const unsigned long long*)&_f32x2_mul_b_112));
                uint32_t _fp4_pair_45;
                uint16_t _e2m1_pair_h_113;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_113) : "f"(_mul_f32x2_45.x), "f"(_mul_f32x2_45.y));
                _fp4_pair_45 = (uint32_t)_e2m1_pair_h_113;
                float2 _f2_528 = make_float2(acc_147[28], acc_147[29]);
                float2 _f32x2_mul_b_114 = ((0) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_46;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_46) : "l"(*(const unsigned long long*)&_f2_528), "l"(*(const unsigned long long*)&_f32x2_mul_b_114));
                uint32_t _fp4_pair_46;
                uint16_t _e2m1_pair_h_115;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_115) : "f"(_mul_f32x2_46.x), "f"(_mul_f32x2_46.y));
                _fp4_pair_46 = (uint32_t)_e2m1_pair_h_115;
                float2 _f2_529 = make_float2(acc_147[30], acc_147[31]);
                float2 _f32x2_mul_b_116 = ((0) ? _f2_512 : _f2_513);
                float2 _mul_f32x2_47;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_47) : "l"(*(const unsigned long long*)&_f2_529), "l"(*(const unsigned long long*)&_f32x2_mul_b_116));
                uint32_t _fp4_pair_47;
                uint16_t _e2m1_pair_h_117;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_117) : "f"(_mul_f32x2_47.x), "f"(_mul_f32x2_47.y));
                _fp4_pair_47 = (uint32_t)_e2m1_pair_h_117;
                unsigned int word_314 = _fp4_pair_44 | _fp4_pair_45 << 8 | _fp4_pair_46 << 16 | _fp4_pair_47 << 24;
                uint32_t _e2m1_canon_11;
                {
                    const uint32_t _canon_w_118 = word_314;
                    const uint32_t _canon_s_118 = (_canon_w_118 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_11 = _canon_w_118 & (_canon_s_118 | 0x77777777u);
                }
                q_304[3] = _e2m1_canon_11;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(tob_off + 16 * pp2_303)), "r"(q_304[0]), "r"(q_304[1]), "r"(q_304[2]), "r"(q_304[3]) : "memory");
                wsm16[(tob_off + 3072 >> 1) + pp2_303] = _e4m3x2_f32_2;
                float d_315[32];
                d_315[0] = 0.0f;
                d_315[1] = 0.0f;
                d_315[2] = 0.0f;
                d_315[3] = 0.0f;
                d_315[4] = 0.0f;
                d_315[5] = 0.0f;
                d_315[6] = 0.0f;
                d_315[7] = 0.0f;
                d_315[8] = 0.0f;
                d_315[9] = 0.0f;
                d_315[10] = 0.0f;
                d_315[11] = 0.0f;
                d_315[12] = 0.0f;
                d_315[13] = 0.0f;
                d_315[14] = 0.0f;
                d_315[15] = 0.0f;
                d_315[16] = 0.0f;
                d_315[17] = 0.0f;
                d_315[18] = 0.0f;
                d_315[19] = 0.0f;
                d_315[20] = 0.0f;
                d_315[21] = 0.0f;
                d_315[22] = 0.0f;
                d_315[23] = 0.0f;
                d_315[24] = 0.0f;
                d_315[25] = 0.0f;
                d_315[26] = 0.0f;
                d_315[27] = 0.0f;
                d_315[28] = 0.0f;
                d_315[29] = 0.0f;
                d_315[30] = 0.0f;
                d_315[31] = 0.0f;
                float2 _fp8x2_decode_17;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_17.x), "=f"(_fp8x2_decode_17.y) : "h"((uint16_t)_e4m3x2_f32_2));
                float _mul_34;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(_fp8x2_decode_17.x), "f"(rs2));
                float s0_316 = _mul_34;
                float _mul_35;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(_fp8x2_decode_17.y), "f"(rs2));
                float s1_317 = _mul_35;
                float2 _f2_530 = make_float2(s0_316, s0_316);
                float2 _f2_531 = make_float2(s1_317, s1_317);
                float _unpack_fp4_56[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_304[0]));
                    (&_unpack_fp4_56[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_56[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_56[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_56[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_56[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_56[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_56[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_56[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_56[0]), "=f"(_unpack_fp4_56[1]), "=f"(_unpack_fp4_56[2]), "=f"(_unpack_fp4_56[3]), "=f"(_unpack_fp4_56[4]), "=f"(_unpack_fp4_56[5]), "=f"(_unpack_fp4_56[6]), "=f"(_unpack_fp4_56[7]) : "r"(q_304[0]));
                    #endif
                }
                float2 _f2_532 = make_float2(d_315[0], d_315[1]);
                float2 _f2_533 = make_float2(_unpack_fp4_56[0], _unpack_fp4_56[1]);
                float2 r_318 = fma_f32x2_rn_noftz(_f2_533, _f2_530, _f2_532);
                d_315[0] = r_318.x;
                d_315[1] = r_318.y;
                float2 _f2_534 = make_float2(d_315[2], d_315[3]);
                float2 _f2_535 = make_float2(_unpack_fp4_56[2], _unpack_fp4_56[3]);
                float2 r_319 = fma_f32x2_rn_noftz(_f2_535, _f2_530, _f2_534);
                d_315[2] = r_319.x;
                d_315[3] = r_319.y;
                float2 _f2_536 = make_float2(d_315[4], d_315[5]);
                float2 _f2_537 = make_float2(_unpack_fp4_56[4], _unpack_fp4_56[5]);
                float2 r_320 = fma_f32x2_rn_noftz(_f2_537, _f2_530, _f2_536);
                d_315[4] = r_320.x;
                d_315[5] = r_320.y;
                float2 _f2_538 = make_float2(d_315[6], d_315[7]);
                float2 _f2_539 = make_float2(_unpack_fp4_56[6], _unpack_fp4_56[7]);
                float2 r_321 = fma_f32x2_rn_noftz(_f2_539, _f2_530, _f2_538);
                d_315[6] = r_321.x;
                d_315[7] = r_321.y;
                float _unpack_fp4_57[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_304[1 + 0]));
                    (&_unpack_fp4_57[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_57[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_57[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_57[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_57[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_57[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_57[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_57[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_57[0]), "=f"(_unpack_fp4_57[1]), "=f"(_unpack_fp4_57[2]), "=f"(_unpack_fp4_57[3]), "=f"(_unpack_fp4_57[4]), "=f"(_unpack_fp4_57[5]), "=f"(_unpack_fp4_57[6]), "=f"(_unpack_fp4_57[7]) : "r"(q_304[1 + 0]));
                    #endif
                }
                float2 _f2_540 = make_float2(d_315[8], d_315[9]);
                float2 _f2_541 = make_float2(_unpack_fp4_57[0], _unpack_fp4_57[1]);
                float2 r_322 = fma_f32x2_rn_noftz(_f2_541, _f2_530, _f2_540);
                d_315[8] = r_322.x;
                d_315[9] = r_322.y;
                float2 _f2_542 = make_float2(d_315[10], d_315[11]);
                float2 _f2_543 = make_float2(_unpack_fp4_57[2], _unpack_fp4_57[3]);
                float2 r_323 = fma_f32x2_rn_noftz(_f2_543, _f2_530, _f2_542);
                d_315[10] = r_323.x;
                d_315[11] = r_323.y;
                float2 _f2_544 = make_float2(d_315[12], d_315[13]);
                float2 _f2_545 = make_float2(_unpack_fp4_57[4], _unpack_fp4_57[5]);
                float2 r_324 = fma_f32x2_rn_noftz(_f2_545, _f2_530, _f2_544);
                d_315[12] = r_324.x;
                d_315[13] = r_324.y;
                float2 _f2_546 = make_float2(d_315[14], d_315[15]);
                float2 _f2_547 = make_float2(_unpack_fp4_57[6], _unpack_fp4_57[7]);
                float2 r_325 = fma_f32x2_rn_noftz(_f2_547, _f2_530, _f2_546);
                d_315[14] = r_325.x;
                d_315[15] = r_325.y;
                float _unpack_fp4_58[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_304[2 + 0]));
                    (&_unpack_fp4_58[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_58[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_58[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_58[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_58[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_58[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_58[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_58[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_58[0]), "=f"(_unpack_fp4_58[1]), "=f"(_unpack_fp4_58[2]), "=f"(_unpack_fp4_58[3]), "=f"(_unpack_fp4_58[4]), "=f"(_unpack_fp4_58[5]), "=f"(_unpack_fp4_58[6]), "=f"(_unpack_fp4_58[7]) : "r"(q_304[2 + 0]));
                    #endif
                }
                float2 _f2_548 = make_float2(d_315[16], d_315[17]);
                float2 _f2_549 = make_float2(_unpack_fp4_58[0], _unpack_fp4_58[1]);
                float2 r_326 = fma_f32x2_rn_noftz(_f2_549, _f2_531, _f2_548);
                d_315[16] = r_326.x;
                d_315[17] = r_326.y;
                float2 _f2_550 = make_float2(d_315[18], d_315[19]);
                float2 _f2_551 = make_float2(_unpack_fp4_58[2], _unpack_fp4_58[3]);
                float2 r_327 = fma_f32x2_rn_noftz(_f2_551, _f2_531, _f2_550);
                d_315[18] = r_327.x;
                d_315[19] = r_327.y;
                float2 _f2_552 = make_float2(d_315[20], d_315[21]);
                float2 _f2_553 = make_float2(_unpack_fp4_58[4], _unpack_fp4_58[5]);
                float2 r_328 = fma_f32x2_rn_noftz(_f2_553, _f2_531, _f2_552);
                d_315[20] = r_328.x;
                d_315[21] = r_328.y;
                float2 _f2_554 = make_float2(d_315[22], d_315[23]);
                float2 _f2_555 = make_float2(_unpack_fp4_58[6], _unpack_fp4_58[7]);
                float2 r_329 = fma_f32x2_rn_noftz(_f2_555, _f2_531, _f2_554);
                d_315[22] = r_329.x;
                d_315[23] = r_329.y;
                float _unpack_fp4_59[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_304[3 + 0]));
                    (&_unpack_fp4_59[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_59[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_59[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_59[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_59[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_59[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_59[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_59[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_59[0]), "=f"(_unpack_fp4_59[1]), "=f"(_unpack_fp4_59[2]), "=f"(_unpack_fp4_59[3]), "=f"(_unpack_fp4_59[4]), "=f"(_unpack_fp4_59[5]), "=f"(_unpack_fp4_59[6]), "=f"(_unpack_fp4_59[7]) : "r"(q_304[3 + 0]));
                    #endif
                }
                float2 _f2_556 = make_float2(d_315[24], d_315[25]);
                float2 _f2_557 = make_float2(_unpack_fp4_59[0], _unpack_fp4_59[1]);
                float2 r_330 = fma_f32x2_rn_noftz(_f2_557, _f2_531, _f2_556);
                d_315[24] = r_330.x;
                d_315[25] = r_330.y;
                float2 _f2_558 = make_float2(d_315[26], d_315[27]);
                float2 _f2_559 = make_float2(_unpack_fp4_59[2], _unpack_fp4_59[3]);
                float2 r_331 = fma_f32x2_rn_noftz(_f2_559, _f2_531, _f2_558);
                d_315[26] = r_331.x;
                d_315[27] = r_331.y;
                float2 _f2_560 = make_float2(d_315[28], d_315[29]);
                float2 _f2_561 = make_float2(_unpack_fp4_59[4], _unpack_fp4_59[5]);
                float2 r_332 = fma_f32x2_rn_noftz(_f2_561, _f2_531, _f2_560);
                d_315[28] = r_332.x;
                d_315[29] = r_332.y;
                float2 _f2_562 = make_float2(d_315[30], d_315[31]);
                float2 _f2_563 = make_float2(_unpack_fp4_59[6], _unpack_fp4_59[7]);
                float2 r_333 = fma_f32x2_rn_noftz(_f2_563, _f2_531, _f2_562);
                d_315[30] = r_333.x;
                d_315[31] = r_333.y;
                uint32_t d_315_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(d_315[_lp*2 + 0], d_315[_lp*2+1 + 0]));
                    d_315_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(d_315_bf16[0]), "r"(d_315_bf16[1]), "r"(d_315_bf16[2]), "r"(d_315_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(d_315_bf16[4]), "r"(d_315_bf16[5]), "r"(d_315_bf16[6]), "r"(d_315_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(d_315_bf16[8]), "r"(d_315_bf16[9]), "r"(d_315_bf16[10]), "r"(d_315_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(d_315_bf16[12]), "r"(d_315_bf16[13]), "r"(d_315_bf16[14]), "r"(d_315_bf16[15]) : "memory");
                __syncwarp();
                int c_334 = lane_1;
                int l_335 = c_334 >> 2;
                unsigned int w4_336[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_336[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_336[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_336[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_336[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_335 + (c_334 & 3 ^ l_335 >> 1 & 3)))));
                long long idx_337 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_334));
                {
                    int4 _iv4 = make_int4(w4_336[0 + 0], w4_336[0 + 1], w4_336[0 + 2], w4_336[0 + 3]);
                    *reinterpret_cast<int4*>(out_i32 + (idx_337)) = _iv4;
                }
                int c_338 = lane_1 + 32;
                int l_339 = c_338 >> 2;
                unsigned int w4_340[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_340[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_340[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_340[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_340[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_339 + (c_338 & 3 ^ l_339 >> 1 & 3)))));
                long long idx_341 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_338));
                {
                    int4 _iv4 = make_int4(w4_340[0 + 0], w4_340[0 + 1], w4_340[0 + 2], w4_340[0 + 3]);
                    *reinterpret_cast<int4*>(out_i32 + (idx_341)) = _iv4;
                }
                int c_342 = lane_1 + 64;
                int l_343 = c_342 >> 2;
                unsigned int w4_344[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_344[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_344[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_344[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_344[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_343 + (c_342 & 3 ^ l_343 >> 1 & 3)))));
                long long idx_345 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_342));
                {
                    int4 _iv4 = make_int4(w4_344[0 + 0], w4_344[0 + 1], w4_344[0 + 2], w4_344[0 + 3]);
                    *reinterpret_cast<int4*>(out_i32 + (idx_345)) = _iv4;
                }
                int c_346 = lane_1 + 96;
                int l_347 = c_346 >> 2;
                unsigned int w4_348[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_348[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_348[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_348[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_348[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_347 + (c_346 & 3 ^ l_347 >> 1 & 3)))));
                long long idx_349 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_346));
                {
                    int4 _iv4 = make_int4(w4_348[0 + 0], w4_348[0 + 1], w4_348[0 + 2], w4_348[0 + 3]);
                    *reinterpret_cast<int4*>(out_i32 + (idx_349)) = _iv4;
                }
                __syncwarp();
                if (h == 0 && lane_1 < 4) {
                    wsm32[(tob_off + 3072 + 384 >> 2) + lane_1] = rs2;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                if (h == 0 && lane_1 == 0) {
                    {
                        void* _cpbulk_dst_119 = reinterpret_cast<void*>(peer_base_2 + ((int)ag_base + (int)tix64 * 3072));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_119), "r"(wsm_addr + (unsigned int)tob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_120 = reinterpret_cast<void*>(peer_base_2 + ((int)ag_base + 50331648 + (int)tix64 * 384));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_120), "r"(wsm_addr + (unsigned int)tob_off + 3072), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_121 = reinterpret_cast<void*>(peer_base_2 + ((int)ag_base + 56623104 + (int)tix64 * 16));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_121), "r"(wsm_addr + (unsigned int)tob_off + 3072 + 384), "r"((uint32_t)(16))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_122 = reinterpret_cast<void*>(peer_base_3 + ((int)ag_base + (int)tix64 * 3072));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_122), "r"(wsm_addr + (unsigned int)tob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_123 = reinterpret_cast<void*>(peer_base_3 + ((int)ag_base + 50331648 + (int)tix64 * 384));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_123), "r"(wsm_addr + (unsigned int)tob_off + 3072), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_124 = reinterpret_cast<void*>(peer_base_3 + ((int)ag_base + 56623104 + (int)tix64 * 16));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_124), "r"(wsm_addr + (unsigned int)tob_off + 3072 + 384), "r"((uint32_t)(16))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_125 = reinterpret_cast<void*>(peer_base_4 + ((int)ag_base + (int)tix64 * 3072));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_125), "r"(wsm_addr + (unsigned int)tob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_126 = reinterpret_cast<void*>(peer_base_4 + ((int)ag_base + 50331648 + (int)tix64 * 384));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_126), "r"(wsm_addr + (unsigned int)tob_off + 3072), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_127 = reinterpret_cast<void*>(peer_base_4 + ((int)ag_base + 56623104 + (int)tix64 * 16));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_127), "r"(wsm_addr + (unsigned int)tob_off + 3072 + 384), "r"((uint32_t)(16))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                kb = kb + 1;
            }
            int fme = rb0 - r0;
            #pragma unroll 1
            for (int c_1 = wC; c_1 < nC; c_1 += 10) {
                int blk = c_1 / 3;
                int w = c_1 % 3;
                int bump = ((w >= fme) ? 1 : 0);
                int crow = r0 + blk * 4 + w + bump;
                long long co64 = (long long)(crow % 4);
                long long cix64 = (long long)(crow / 4);
                long long crow64 = (long long)crow;
                long long csrc64 = co64;
                float rs_1 = 0.0f;
                uint32_t _sysv_poll32_group_0[1];
                do {
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[0]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)ag_base + 56623104 + ((int)csrc64 * 4096 + (int)cix64) * 16))) : "memory");
                } while (
                    (_sysv_poll32_group_0[0] == 0x80000000u)
                );
                rs_1 = __uint_as_float(_sysv_poll32_group_0[0]);
                #pragma unroll 1
                for (int jj = 0; jj < 2; jj++) {
                    int pp0 = lane_1 + 96 * jj;
                    int pp1 = pp0 + 32;
                    int pp2_1 = pp0 + 64;
                    uint32_t _fp4_wire_values_3[12];
                    uint32_t _fp4_wire_scales_3[3];
                    uint16_t _wire_h16_129_0;
                    uint16_t _wire_h16_129_1;
                    uint16_t _wire_h16_129_2;
                    uint32_t _wire_vpend_129 = 7u;
                    uint32_t _wire_spend_129 = 7u;
                    do {
                        if (_wire_vpend_129 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_3[0]), "=r"(_fp4_wire_values_3[1]), "=r"(_fp4_wire_values_3[2]), "=r"(_fp4_wire_values_3[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)ag_base + ((int)csrc64 * 4096 + (int)cix64) * 3072 + 16 * pp0))) : "memory");
                        }
                        if (_wire_vpend_129 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_3[4]), "=r"(_fp4_wire_values_3[5]), "=r"(_fp4_wire_values_3[6]), "=r"(_fp4_wire_values_3[7]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)ag_base + ((int)csrc64 * 4096 + (int)cix64) * 3072 + 16 * pp1))) : "memory");
                        }
                        if (_wire_vpend_129 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_3[8]), "=r"(_fp4_wire_values_3[9]), "=r"(_fp4_wire_values_3[10]), "=r"(_fp4_wire_values_3[11]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + ((int)ag_base + ((int)csrc64 * 4096 + (int)cix64) * 3072 + 16 * pp2_1))) : "memory");
                        }
                        if (_wire_spend_129 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_129_0) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)ag_base + 50331648 + ((int)csrc64 * 4096 + (int)cix64) * 384 + 2 * pp0))) : "memory");
                            _fp4_wire_scales_3[0] = (uint32_t)_wire_h16_129_0;
                        }
                        if (_wire_spend_129 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_129_1) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)ag_base + 50331648 + ((int)csrc64 * 4096 + (int)cix64) * 384 + 2 * pp1))) : "memory");
                            _fp4_wire_scales_3[1] = (uint32_t)_wire_h16_129_1;
                        }
                        if (_wire_spend_129 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_129_2) : "l"(reinterpret_cast<uint16_t*>(self_u8 + ((int)ag_base + 50331648 + ((int)csrc64 * 4096 + (int)cix64) * 384 + 2 * pp2_1))) : "memory");
                            _fp4_wire_scales_3[2] = (uint32_t)_wire_h16_129_2;
                        }
                        if (_wire_vpend_129 & 1u) {
                            if (!((((_fp4_wire_values_3[0] & ~((_fp4_wire_values_3[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_3[1] & ~((_fp4_wire_values_3[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_3[2] & ~((_fp4_wire_values_3[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_3[3] & ~((_fp4_wire_values_3[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_129 &= ~1u;
                        }
                        if (_wire_vpend_129 & 2u) {
                            if (!((((_fp4_wire_values_3[4] & ~((_fp4_wire_values_3[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_3[5] & ~((_fp4_wire_values_3[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_3[6] & ~((_fp4_wire_values_3[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_3[7] & ~((_fp4_wire_values_3[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_129 &= ~2u;
                        }
                        if (_wire_vpend_129 & 4u) {
                            if (!((((_fp4_wire_values_3[8] & ~((_fp4_wire_values_3[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_3[9] & ~((_fp4_wire_values_3[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_3[10] & ~((_fp4_wire_values_3[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_3[11] & ~((_fp4_wire_values_3[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_129 &= ~4u;
                        }
                        if (_wire_spend_129 & 1u) {
                            if (!(((_fp4_wire_scales_3[0] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_3[0] >> 8) == 0xFFu))) _wire_spend_129 &= ~1u;
                        }
                        if (_wire_spend_129 & 2u) {
                            if (!(((_fp4_wire_scales_3[1] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_3[1] >> 8) == 0xFFu))) _wire_spend_129 &= ~2u;
                        }
                        if (_wire_spend_129 & 4u) {
                            if (!(((_fp4_wire_scales_3[2] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_3[2] >> 8) == 0xFFu))) _wire_spend_129 &= ~4u;
                        }
                    } while ((_wire_vpend_129 | _wire_spend_129) != 0u);
                    float acc_1[32];
                    acc_1[0] = 0.0f;
                    acc_1[1] = 0.0f;
                    acc_1[2] = 0.0f;
                    acc_1[3] = 0.0f;
                    acc_1[4] = 0.0f;
                    acc_1[5] = 0.0f;
                    acc_1[6] = 0.0f;
                    acc_1[7] = 0.0f;
                    acc_1[8] = 0.0f;
                    acc_1[9] = 0.0f;
                    acc_1[10] = 0.0f;
                    acc_1[11] = 0.0f;
                    acc_1[12] = 0.0f;
                    acc_1[13] = 0.0f;
                    acc_1[14] = 0.0f;
                    acc_1[15] = 0.0f;
                    acc_1[16] = 0.0f;
                    acc_1[17] = 0.0f;
                    acc_1[18] = 0.0f;
                    acc_1[19] = 0.0f;
                    acc_1[20] = 0.0f;
                    acc_1[21] = 0.0f;
                    acc_1[22] = 0.0f;
                    acc_1[23] = 0.0f;
                    acc_1[24] = 0.0f;
                    acc_1[25] = 0.0f;
                    acc_1[26] = 0.0f;
                    acc_1[27] = 0.0f;
                    acc_1[28] = 0.0f;
                    acc_1[29] = 0.0f;
                    acc_1[30] = 0.0f;
                    acc_1[31] = 0.0f;
                    float2 _fp8x2_decode_18;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_18.x), "=f"(_fp8x2_decode_18.y) : "h"((uint16_t)_fp4_wire_scales_3[0]));
                    float _mul_36;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(_fp8x2_decode_18.x), "f"(rs_1));
                    float s0_1 = _mul_36;
                    float _mul_37;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(_fp8x2_decode_18.y), "f"(rs_1));
                    float s1_1 = _mul_37;
                    float2 _f2_564 = make_float2(s0_1, s0_1);
                    float2 _f2_565 = make_float2(s1_1, s1_1);
                    float _unpack_fp4_60[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_3[0]));
                        (&_unpack_fp4_60[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_60[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_60[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_60[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_60[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_60[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_60[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_60[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_60[0]), "=f"(_unpack_fp4_60[1]), "=f"(_unpack_fp4_60[2]), "=f"(_unpack_fp4_60[3]), "=f"(_unpack_fp4_60[4]), "=f"(_unpack_fp4_60[5]), "=f"(_unpack_fp4_60[6]), "=f"(_unpack_fp4_60[7]) : "r"(_fp4_wire_values_3[0]));
                        #endif
                    }
                    float2 _f2_566 = make_float2(acc_1[0], acc_1[1]);
                    float2 _f2_567 = make_float2(_unpack_fp4_60[0], _unpack_fp4_60[1]);
                    float2 r_1 = fma_f32x2_rn_noftz(_f2_567, _f2_564, _f2_566);
                    acc_1[0] = r_1.x;
                    acc_1[1] = r_1.y;
                    float2 _f2_568 = make_float2(acc_1[2], acc_1[3]);
                    float2 _f2_569 = make_float2(_unpack_fp4_60[2], _unpack_fp4_60[3]);
                    float2 r_0 = fma_f32x2_rn_noftz(_f2_569, _f2_564, _f2_568);
                    acc_1[2] = r_0.x;
                    acc_1[3] = r_0.y;
                    float2 _f2_570 = make_float2(acc_1[4], acc_1[5]);
                    float2 _f2_571 = make_float2(_unpack_fp4_60[4], _unpack_fp4_60[5]);
                    float2 r_1_1 = fma_f32x2_rn_noftz(_f2_571, _f2_564, _f2_570);
                    acc_1[4] = r_1_1.x;
                    acc_1[5] = r_1_1.y;
                    float2 _f2_572 = make_float2(acc_1[6], acc_1[7]);
                    float2 _f2_573 = make_float2(_unpack_fp4_60[6], _unpack_fp4_60[7]);
                    float2 r_2 = fma_f32x2_rn_noftz(_f2_573, _f2_564, _f2_572);
                    acc_1[6] = r_2.x;
                    acc_1[7] = r_2.y;
                    float _unpack_fp4_61[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_3[1 + 0]));
                        (&_unpack_fp4_61[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_61[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_61[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_61[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_61[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_61[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_61[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_61[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_61[0]), "=f"(_unpack_fp4_61[1]), "=f"(_unpack_fp4_61[2]), "=f"(_unpack_fp4_61[3]), "=f"(_unpack_fp4_61[4]), "=f"(_unpack_fp4_61[5]), "=f"(_unpack_fp4_61[6]), "=f"(_unpack_fp4_61[7]) : "r"(_fp4_wire_values_3[1 + 0]));
                        #endif
                    }
                    float2 _f2_574 = make_float2(acc_1[8], acc_1[9]);
                    float2 _f2_575 = make_float2(_unpack_fp4_61[0], _unpack_fp4_61[1]);
                    float2 r_3_1 = fma_f32x2_rn_noftz(_f2_575, _f2_564, _f2_574);
                    acc_1[8] = r_3_1.x;
                    acc_1[9] = r_3_1.y;
                    float2 _f2_576 = make_float2(acc_1[10], acc_1[11]);
                    float2 _f2_577 = make_float2(_unpack_fp4_61[2], _unpack_fp4_61[3]);
                    float2 r_4_1 = fma_f32x2_rn_noftz(_f2_577, _f2_564, _f2_576);
                    acc_1[10] = r_4_1.x;
                    acc_1[11] = r_4_1.y;
                    float2 _f2_578 = make_float2(acc_1[12], acc_1[13]);
                    float2 _f2_579 = make_float2(_unpack_fp4_61[4], _unpack_fp4_61[5]);
                    float2 r_5_1 = fma_f32x2_rn_noftz(_f2_579, _f2_564, _f2_578);
                    acc_1[12] = r_5_1.x;
                    acc_1[13] = r_5_1.y;
                    float2 _f2_580 = make_float2(acc_1[14], acc_1[15]);
                    float2 _f2_581 = make_float2(_unpack_fp4_61[6], _unpack_fp4_61[7]);
                    float2 r_6_1 = fma_f32x2_rn_noftz(_f2_581, _f2_564, _f2_580);
                    acc_1[14] = r_6_1.x;
                    acc_1[15] = r_6_1.y;
                    float _unpack_fp4_62[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_3[2 + 0]));
                        (&_unpack_fp4_62[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_62[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_62[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_62[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_62[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_62[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_62[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_62[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_62[0]), "=f"(_unpack_fp4_62[1]), "=f"(_unpack_fp4_62[2]), "=f"(_unpack_fp4_62[3]), "=f"(_unpack_fp4_62[4]), "=f"(_unpack_fp4_62[5]), "=f"(_unpack_fp4_62[6]), "=f"(_unpack_fp4_62[7]) : "r"(_fp4_wire_values_3[2 + 0]));
                        #endif
                    }
                    float2 _f2_582 = make_float2(acc_1[16], acc_1[17]);
                    float2 _f2_583 = make_float2(_unpack_fp4_62[0], _unpack_fp4_62[1]);
                    float2 r_7_1 = fma_f32x2_rn_noftz(_f2_583, _f2_565, _f2_582);
                    acc_1[16] = r_7_1.x;
                    acc_1[17] = r_7_1.y;
                    float2 _f2_584 = make_float2(acc_1[18], acc_1[19]);
                    float2 _f2_585 = make_float2(_unpack_fp4_62[2], _unpack_fp4_62[3]);
                    float2 r_8_1 = fma_f32x2_rn_noftz(_f2_585, _f2_565, _f2_584);
                    acc_1[18] = r_8_1.x;
                    acc_1[19] = r_8_1.y;
                    float2 _f2_586 = make_float2(acc_1[20], acc_1[21]);
                    float2 _f2_587 = make_float2(_unpack_fp4_62[4], _unpack_fp4_62[5]);
                    float2 r_9_1 = fma_f32x2_rn_noftz(_f2_587, _f2_565, _f2_586);
                    acc_1[20] = r_9_1.x;
                    acc_1[21] = r_9_1.y;
                    float2 _f2_588 = make_float2(acc_1[22], acc_1[23]);
                    float2 _f2_589 = make_float2(_unpack_fp4_62[6], _unpack_fp4_62[7]);
                    float2 r_10_1 = fma_f32x2_rn_noftz(_f2_589, _f2_565, _f2_588);
                    acc_1[22] = r_10_1.x;
                    acc_1[23] = r_10_1.y;
                    float _unpack_fp4_63[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_3[3 + 0]));
                        (&_unpack_fp4_63[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_63[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_63[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_63[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_63[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_63[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_63[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_63[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_63[0]), "=f"(_unpack_fp4_63[1]), "=f"(_unpack_fp4_63[2]), "=f"(_unpack_fp4_63[3]), "=f"(_unpack_fp4_63[4]), "=f"(_unpack_fp4_63[5]), "=f"(_unpack_fp4_63[6]), "=f"(_unpack_fp4_63[7]) : "r"(_fp4_wire_values_3[3 + 0]));
                        #endif
                    }
                    float2 _f2_590 = make_float2(acc_1[24], acc_1[25]);
                    float2 _f2_591 = make_float2(_unpack_fp4_63[0], _unpack_fp4_63[1]);
                    float2 r_11_1 = fma_f32x2_rn_noftz(_f2_591, _f2_565, _f2_590);
                    acc_1[24] = r_11_1.x;
                    acc_1[25] = r_11_1.y;
                    float2 _f2_592 = make_float2(acc_1[26], acc_1[27]);
                    float2 _f2_593 = make_float2(_unpack_fp4_63[2], _unpack_fp4_63[3]);
                    float2 r_12_1 = fma_f32x2_rn_noftz(_f2_593, _f2_565, _f2_592);
                    acc_1[26] = r_12_1.x;
                    acc_1[27] = r_12_1.y;
                    float2 _f2_594 = make_float2(acc_1[28], acc_1[29]);
                    float2 _f2_595 = make_float2(_unpack_fp4_63[4], _unpack_fp4_63[5]);
                    float2 r_13_1 = fma_f32x2_rn_noftz(_f2_595, _f2_565, _f2_594);
                    acc_1[28] = r_13_1.x;
                    acc_1[29] = r_13_1.y;
                    float2 _f2_596 = make_float2(acc_1[30], acc_1[31]);
                    float2 _f2_597 = make_float2(_unpack_fp4_63[6], _unpack_fp4_63[7]);
                    float2 r_14_1 = fma_f32x2_rn_noftz(_f2_597, _f2_565, _f2_596);
                    acc_1[30] = r_14_1.x;
                    acc_1[31] = r_14_1.y;
                    uint32_t acc_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_1[_lp*2 + 0], acc_1[_lp*2+1 + 0]));
                        acc_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_bf16[0]), "r"(acc_bf16[1]), "r"(acc_bf16[2]), "r"(acc_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_bf16[4]), "r"(acc_bf16[5]), "r"(acc_bf16[6]), "r"(acc_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_bf16[8]), "r"(acc_bf16[9]), "r"(acc_bf16[10]), "r"(acc_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_bf16[12]), "r"(acc_bf16[13]), "r"(acc_bf16[14]), "r"(acc_bf16[15]) : "memory");
                    __syncwarp();
                    int c_15 = lane_1;
                    int l_1 = c_15 >> 2;
                    unsigned int w4_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_1 + (c_15 & 3 ^ l_1 >> 1 & 3)))));
                    long long idx_1 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_15));
                    {
                        int4 _iv4 = make_int4(w4_1[0 + 0], w4_1[0 + 1], w4_1[0 + 2], w4_1[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_1)) = _iv4;
                    }
                    int c_16 = lane_1 + 32;
                    int l_17 = c_16 >> 2;
                    unsigned int w4_18[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_17 + (c_16 & 3 ^ l_17 >> 1 & 3)))));
                    long long idx_19 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_16));
                    {
                        int4 _iv4 = make_int4(w4_18[0 + 0], w4_18[0 + 1], w4_18[0 + 2], w4_18[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_19)) = _iv4;
                    }
                    int c_20 = lane_1 + 64;
                    int l_21 = c_20 >> 2;
                    unsigned int w4_22[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_21 + (c_20 & 3 ^ l_21 >> 1 & 3)))));
                    long long idx_23 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_20));
                    {
                        int4 _iv4 = make_int4(w4_22[0 + 0], w4_22[0 + 1], w4_22[0 + 2], w4_22[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_23)) = _iv4;
                    }
                    int c_24 = lane_1 + 96;
                    int l_25 = c_24 >> 2;
                    unsigned int w4_26[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_26[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_25 + (c_24 & 3 ^ l_25 >> 1 & 3)))));
                    long long idx_27 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_24));
                    {
                        int4 _iv4 = make_int4(w4_26[0 + 0], w4_26[0 + 1], w4_26[0 + 2], w4_26[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_27)) = _iv4;
                    }
                    __syncwarp();
                    float acc_28[32];
                    acc_28[0] = 0.0f;
                    acc_28[1] = 0.0f;
                    acc_28[2] = 0.0f;
                    acc_28[3] = 0.0f;
                    acc_28[4] = 0.0f;
                    acc_28[5] = 0.0f;
                    acc_28[6] = 0.0f;
                    acc_28[7] = 0.0f;
                    acc_28[8] = 0.0f;
                    acc_28[9] = 0.0f;
                    acc_28[10] = 0.0f;
                    acc_28[11] = 0.0f;
                    acc_28[12] = 0.0f;
                    acc_28[13] = 0.0f;
                    acc_28[14] = 0.0f;
                    acc_28[15] = 0.0f;
                    acc_28[16] = 0.0f;
                    acc_28[17] = 0.0f;
                    acc_28[18] = 0.0f;
                    acc_28[19] = 0.0f;
                    acc_28[20] = 0.0f;
                    acc_28[21] = 0.0f;
                    acc_28[22] = 0.0f;
                    acc_28[23] = 0.0f;
                    acc_28[24] = 0.0f;
                    acc_28[25] = 0.0f;
                    acc_28[26] = 0.0f;
                    acc_28[27] = 0.0f;
                    acc_28[28] = 0.0f;
                    acc_28[29] = 0.0f;
                    acc_28[30] = 0.0f;
                    acc_28[31] = 0.0f;
                    float2 _fp8x2_decode_19;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_19.x), "=f"(_fp8x2_decode_19.y) : "h"((uint16_t)_fp4_wire_scales_3[1]));
                    float _mul_38;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(_fp8x2_decode_19.x), "f"(rs_1));
                    float s0_29 = _mul_38;
                    float _mul_39;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(_fp8x2_decode_19.y), "f"(rs_1));
                    float s1_30 = _mul_39;
                    float2 _f2_598 = make_float2(s0_29, s0_29);
                    float2 _f2_599 = make_float2(s1_30, s1_30);
                    float _unpack_fp4_64[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_3[4 + 0]));
                        (&_unpack_fp4_64[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_64[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_64[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_64[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_64[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_64[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_64[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_64[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_64[0]), "=f"(_unpack_fp4_64[1]), "=f"(_unpack_fp4_64[2]), "=f"(_unpack_fp4_64[3]), "=f"(_unpack_fp4_64[4]), "=f"(_unpack_fp4_64[5]), "=f"(_unpack_fp4_64[6]), "=f"(_unpack_fp4_64[7]) : "r"(_fp4_wire_values_3[4 + 0]));
                        #endif
                    }
                    float2 _f2_600 = make_float2(acc_28[0], acc_28[1]);
                    float2 _f2_601 = make_float2(_unpack_fp4_64[0], _unpack_fp4_64[1]);
                    float2 r_31_1 = fma_f32x2_rn_noftz(_f2_601, _f2_598, _f2_600);
                    acc_28[0] = r_31_1.x;
                    acc_28[1] = r_31_1.y;
                    float2 _f2_602 = make_float2(acc_28[2], acc_28[3]);
                    float2 _f2_603 = make_float2(_unpack_fp4_64[2], _unpack_fp4_64[3]);
                    float2 r_32_1 = fma_f32x2_rn_noftz(_f2_603, _f2_598, _f2_602);
                    acc_28[2] = r_32_1.x;
                    acc_28[3] = r_32_1.y;
                    float2 _f2_604 = make_float2(acc_28[4], acc_28[5]);
                    float2 _f2_605 = make_float2(_unpack_fp4_64[4], _unpack_fp4_64[5]);
                    float2 r_33_1 = fma_f32x2_rn_noftz(_f2_605, _f2_598, _f2_604);
                    acc_28[4] = r_33_1.x;
                    acc_28[5] = r_33_1.y;
                    float2 _f2_606 = make_float2(acc_28[6], acc_28[7]);
                    float2 _f2_607 = make_float2(_unpack_fp4_64[6], _unpack_fp4_64[7]);
                    float2 r_34_1 = fma_f32x2_rn_noftz(_f2_607, _f2_598, _f2_606);
                    acc_28[6] = r_34_1.x;
                    acc_28[7] = r_34_1.y;
                    float _unpack_fp4_65[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_3[5 + 0]));
                        (&_unpack_fp4_65[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_65[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_65[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_65[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_65[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_65[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_65[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_65[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_65[0]), "=f"(_unpack_fp4_65[1]), "=f"(_unpack_fp4_65[2]), "=f"(_unpack_fp4_65[3]), "=f"(_unpack_fp4_65[4]), "=f"(_unpack_fp4_65[5]), "=f"(_unpack_fp4_65[6]), "=f"(_unpack_fp4_65[7]) : "r"(_fp4_wire_values_3[5 + 0]));
                        #endif
                    }
                    float2 _f2_608 = make_float2(acc_28[8], acc_28[9]);
                    float2 _f2_609 = make_float2(_unpack_fp4_65[0], _unpack_fp4_65[1]);
                    float2 r_35_1 = fma_f32x2_rn_noftz(_f2_609, _f2_598, _f2_608);
                    acc_28[8] = r_35_1.x;
                    acc_28[9] = r_35_1.y;
                    float2 _f2_610 = make_float2(acc_28[10], acc_28[11]);
                    float2 _f2_611 = make_float2(_unpack_fp4_65[2], _unpack_fp4_65[3]);
                    float2 r_36 = fma_f32x2_rn_noftz(_f2_611, _f2_598, _f2_610);
                    acc_28[10] = r_36.x;
                    acc_28[11] = r_36.y;
                    float2 _f2_612 = make_float2(acc_28[12], acc_28[13]);
                    float2 _f2_613 = make_float2(_unpack_fp4_65[4], _unpack_fp4_65[5]);
                    float2 r_37 = fma_f32x2_rn_noftz(_f2_613, _f2_598, _f2_612);
                    acc_28[12] = r_37.x;
                    acc_28[13] = r_37.y;
                    float2 _f2_614 = make_float2(acc_28[14], acc_28[15]);
                    float2 _f2_615 = make_float2(_unpack_fp4_65[6], _unpack_fp4_65[7]);
                    float2 r_38_1 = fma_f32x2_rn_noftz(_f2_615, _f2_598, _f2_614);
                    acc_28[14] = r_38_1.x;
                    acc_28[15] = r_38_1.y;
                    float _unpack_fp4_66[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_3[6 + 0]));
                        (&_unpack_fp4_66[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_66[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_66[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_66[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_66[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_66[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_66[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_66[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_66[0]), "=f"(_unpack_fp4_66[1]), "=f"(_unpack_fp4_66[2]), "=f"(_unpack_fp4_66[3]), "=f"(_unpack_fp4_66[4]), "=f"(_unpack_fp4_66[5]), "=f"(_unpack_fp4_66[6]), "=f"(_unpack_fp4_66[7]) : "r"(_fp4_wire_values_3[6 + 0]));
                        #endif
                    }
                    float2 _f2_616 = make_float2(acc_28[16], acc_28[17]);
                    float2 _f2_617 = make_float2(_unpack_fp4_66[0], _unpack_fp4_66[1]);
                    float2 r_39_1 = fma_f32x2_rn_noftz(_f2_617, _f2_599, _f2_616);
                    acc_28[16] = r_39_1.x;
                    acc_28[17] = r_39_1.y;
                    float2 _f2_618 = make_float2(acc_28[18], acc_28[19]);
                    float2 _f2_619 = make_float2(_unpack_fp4_66[2], _unpack_fp4_66[3]);
                    float2 r_40_1 = fma_f32x2_rn_noftz(_f2_619, _f2_599, _f2_618);
                    acc_28[18] = r_40_1.x;
                    acc_28[19] = r_40_1.y;
                    float2 _f2_620 = make_float2(acc_28[20], acc_28[21]);
                    float2 _f2_621 = make_float2(_unpack_fp4_66[4], _unpack_fp4_66[5]);
                    float2 r_41_1 = fma_f32x2_rn_noftz(_f2_621, _f2_599, _f2_620);
                    acc_28[20] = r_41_1.x;
                    acc_28[21] = r_41_1.y;
                    float2 _f2_622 = make_float2(acc_28[22], acc_28[23]);
                    float2 _f2_623 = make_float2(_unpack_fp4_66[6], _unpack_fp4_66[7]);
                    float2 r_42_1 = fma_f32x2_rn_noftz(_f2_623, _f2_599, _f2_622);
                    acc_28[22] = r_42_1.x;
                    acc_28[23] = r_42_1.y;
                    float _unpack_fp4_67[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_3[7 + 0]));
                        (&_unpack_fp4_67[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_67[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_67[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_67[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_67[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_67[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_67[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_67[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_67[0]), "=f"(_unpack_fp4_67[1]), "=f"(_unpack_fp4_67[2]), "=f"(_unpack_fp4_67[3]), "=f"(_unpack_fp4_67[4]), "=f"(_unpack_fp4_67[5]), "=f"(_unpack_fp4_67[6]), "=f"(_unpack_fp4_67[7]) : "r"(_fp4_wire_values_3[7 + 0]));
                        #endif
                    }
                    float2 _f2_624 = make_float2(acc_28[24], acc_28[25]);
                    float2 _f2_625 = make_float2(_unpack_fp4_67[0], _unpack_fp4_67[1]);
                    float2 r_43_1 = fma_f32x2_rn_noftz(_f2_625, _f2_599, _f2_624);
                    acc_28[24] = r_43_1.x;
                    acc_28[25] = r_43_1.y;
                    float2 _f2_626 = make_float2(acc_28[26], acc_28[27]);
                    float2 _f2_627 = make_float2(_unpack_fp4_67[2], _unpack_fp4_67[3]);
                    float2 r_44_1 = fma_f32x2_rn_noftz(_f2_627, _f2_599, _f2_626);
                    acc_28[26] = r_44_1.x;
                    acc_28[27] = r_44_1.y;
                    float2 _f2_628 = make_float2(acc_28[28], acc_28[29]);
                    float2 _f2_629 = make_float2(_unpack_fp4_67[4], _unpack_fp4_67[5]);
                    float2 r_45_1 = fma_f32x2_rn_noftz(_f2_629, _f2_599, _f2_628);
                    acc_28[28] = r_45_1.x;
                    acc_28[29] = r_45_1.y;
                    float2 _f2_630 = make_float2(acc_28[30], acc_28[31]);
                    float2 _f2_631 = make_float2(_unpack_fp4_67[6], _unpack_fp4_67[7]);
                    float2 r_46_1 = fma_f32x2_rn_noftz(_f2_631, _f2_599, _f2_630);
                    acc_28[30] = r_46_1.x;
                    acc_28[31] = r_46_1.y;
                    uint32_t acc_28_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_28[_lp*2 + 0], acc_28[_lp*2+1 + 0]));
                        acc_28_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_28_bf16[0]), "r"(acc_28_bf16[1]), "r"(acc_28_bf16[2]), "r"(acc_28_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_28_bf16[4]), "r"(acc_28_bf16[5]), "r"(acc_28_bf16[6]), "r"(acc_28_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_28_bf16[8]), "r"(acc_28_bf16[9]), "r"(acc_28_bf16[10]), "r"(acc_28_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_28_bf16[12]), "r"(acc_28_bf16[13]), "r"(acc_28_bf16[14]), "r"(acc_28_bf16[15]) : "memory");
                    __syncwarp();
                    int c_47 = lane_1;
                    int l_48 = c_47 >> 2;
                    unsigned int w4_49[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_49[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_49[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_49[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_49[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_48 + (c_47 & 3 ^ l_48 >> 1 & 3)))));
                    long long idx_50 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_47));
                    {
                        int4 _iv4 = make_int4(w4_49[0 + 0], w4_49[0 + 1], w4_49[0 + 2], w4_49[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_50)) = _iv4;
                    }
                    int c_51 = lane_1 + 32;
                    int l_52 = c_51 >> 2;
                    unsigned int w4_53[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_53[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_53[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_53[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_53[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_52 + (c_51 & 3 ^ l_52 >> 1 & 3)))));
                    long long idx_54 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_51));
                    {
                        int4 _iv4 = make_int4(w4_53[0 + 0], w4_53[0 + 1], w4_53[0 + 2], w4_53[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_54)) = _iv4;
                    }
                    int c_55 = lane_1 + 64;
                    int l_56 = c_55 >> 2;
                    unsigned int w4_57[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_57[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_57[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_57[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_57[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_56 + (c_55 & 3 ^ l_56 >> 1 & 3)))));
                    long long idx_58 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_55));
                    {
                        int4 _iv4 = make_int4(w4_57[0 + 0], w4_57[0 + 1], w4_57[0 + 2], w4_57[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_58)) = _iv4;
                    }
                    int c_59 = lane_1 + 96;
                    int l_60 = c_59 >> 2;
                    unsigned int w4_61[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_61[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_61[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_61[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_61[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_60 + (c_59 & 3 ^ l_60 >> 1 & 3)))));
                    long long idx_62 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_59));
                    {
                        int4 _iv4 = make_int4(w4_61[0 + 0], w4_61[0 + 1], w4_61[0 + 2], w4_61[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_62)) = _iv4;
                    }
                    __syncwarp();
                    float acc_63[32];
                    acc_63[0] = 0.0f;
                    acc_63[1] = 0.0f;
                    acc_63[2] = 0.0f;
                    acc_63[3] = 0.0f;
                    acc_63[4] = 0.0f;
                    acc_63[5] = 0.0f;
                    acc_63[6] = 0.0f;
                    acc_63[7] = 0.0f;
                    acc_63[8] = 0.0f;
                    acc_63[9] = 0.0f;
                    acc_63[10] = 0.0f;
                    acc_63[11] = 0.0f;
                    acc_63[12] = 0.0f;
                    acc_63[13] = 0.0f;
                    acc_63[14] = 0.0f;
                    acc_63[15] = 0.0f;
                    acc_63[16] = 0.0f;
                    acc_63[17] = 0.0f;
                    acc_63[18] = 0.0f;
                    acc_63[19] = 0.0f;
                    acc_63[20] = 0.0f;
                    acc_63[21] = 0.0f;
                    acc_63[22] = 0.0f;
                    acc_63[23] = 0.0f;
                    acc_63[24] = 0.0f;
                    acc_63[25] = 0.0f;
                    acc_63[26] = 0.0f;
                    acc_63[27] = 0.0f;
                    acc_63[28] = 0.0f;
                    acc_63[29] = 0.0f;
                    acc_63[30] = 0.0f;
                    acc_63[31] = 0.0f;
                    float2 _fp8x2_decode_20;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_20.x), "=f"(_fp8x2_decode_20.y) : "h"((uint16_t)_fp4_wire_scales_3[2]));
                    float _mul_40;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_40) : "f"(_fp8x2_decode_20.x), "f"(rs_1));
                    float s0_64 = _mul_40;
                    float _mul_41;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_41) : "f"(_fp8x2_decode_20.y), "f"(rs_1));
                    float s1_65 = _mul_41;
                    float2 _f2_632 = make_float2(s0_64, s0_64);
                    float2 _f2_633 = make_float2(s1_65, s1_65);
                    float _unpack_fp4_68[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_3[8 + 0]));
                        (&_unpack_fp4_68[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_68[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_68[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_68[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_68[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_68[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_68[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_68[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_68[0]), "=f"(_unpack_fp4_68[1]), "=f"(_unpack_fp4_68[2]), "=f"(_unpack_fp4_68[3]), "=f"(_unpack_fp4_68[4]), "=f"(_unpack_fp4_68[5]), "=f"(_unpack_fp4_68[6]), "=f"(_unpack_fp4_68[7]) : "r"(_fp4_wire_values_3[8 + 0]));
                        #endif
                    }
                    float2 _f2_634 = make_float2(acc_63[0], acc_63[1]);
                    float2 _f2_635 = make_float2(_unpack_fp4_68[0], _unpack_fp4_68[1]);
                    float2 r_66_1 = fma_f32x2_rn_noftz(_f2_635, _f2_632, _f2_634);
                    acc_63[0] = r_66_1.x;
                    acc_63[1] = r_66_1.y;
                    float2 _f2_636 = make_float2(acc_63[2], acc_63[3]);
                    float2 _f2_637 = make_float2(_unpack_fp4_68[2], _unpack_fp4_68[3]);
                    float2 r_67_1 = fma_f32x2_rn_noftz(_f2_637, _f2_632, _f2_636);
                    acc_63[2] = r_67_1.x;
                    acc_63[3] = r_67_1.y;
                    float2 _f2_638 = make_float2(acc_63[4], acc_63[5]);
                    float2 _f2_639 = make_float2(_unpack_fp4_68[4], _unpack_fp4_68[5]);
                    float2 r_68_1 = fma_f32x2_rn_noftz(_f2_639, _f2_632, _f2_638);
                    acc_63[4] = r_68_1.x;
                    acc_63[5] = r_68_1.y;
                    float2 _f2_640 = make_float2(acc_63[6], acc_63[7]);
                    float2 _f2_641 = make_float2(_unpack_fp4_68[6], _unpack_fp4_68[7]);
                    float2 r_69_1 = fma_f32x2_rn_noftz(_f2_641, _f2_632, _f2_640);
                    acc_63[6] = r_69_1.x;
                    acc_63[7] = r_69_1.y;
                    float _unpack_fp4_69[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_3[9 + 0]));
                        (&_unpack_fp4_69[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_69[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_69[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_69[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_69[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_69[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_69[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_69[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_69[0]), "=f"(_unpack_fp4_69[1]), "=f"(_unpack_fp4_69[2]), "=f"(_unpack_fp4_69[3]), "=f"(_unpack_fp4_69[4]), "=f"(_unpack_fp4_69[5]), "=f"(_unpack_fp4_69[6]), "=f"(_unpack_fp4_69[7]) : "r"(_fp4_wire_values_3[9 + 0]));
                        #endif
                    }
                    float2 _f2_642 = make_float2(acc_63[8], acc_63[9]);
                    float2 _f2_643 = make_float2(_unpack_fp4_69[0], _unpack_fp4_69[1]);
                    float2 r_70_1 = fma_f32x2_rn_noftz(_f2_643, _f2_632, _f2_642);
                    acc_63[8] = r_70_1.x;
                    acc_63[9] = r_70_1.y;
                    float2 _f2_644 = make_float2(acc_63[10], acc_63[11]);
                    float2 _f2_645 = make_float2(_unpack_fp4_69[2], _unpack_fp4_69[3]);
                    float2 r_71_1 = fma_f32x2_rn_noftz(_f2_645, _f2_632, _f2_644);
                    acc_63[10] = r_71_1.x;
                    acc_63[11] = r_71_1.y;
                    float2 _f2_646 = make_float2(acc_63[12], acc_63[13]);
                    float2 _f2_647 = make_float2(_unpack_fp4_69[4], _unpack_fp4_69[5]);
                    float2 r_72 = fma_f32x2_rn_noftz(_f2_647, _f2_632, _f2_646);
                    acc_63[12] = r_72.x;
                    acc_63[13] = r_72.y;
                    float2 _f2_648 = make_float2(acc_63[14], acc_63[15]);
                    float2 _f2_649 = make_float2(_unpack_fp4_69[6], _unpack_fp4_69[7]);
                    float2 r_73 = fma_f32x2_rn_noftz(_f2_649, _f2_632, _f2_648);
                    acc_63[14] = r_73.x;
                    acc_63[15] = r_73.y;
                    float _unpack_fp4_70[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_3[10 + 0]));
                        (&_unpack_fp4_70[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_70[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_70[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_70[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_70[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_70[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_70[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_70[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_70[0]), "=f"(_unpack_fp4_70[1]), "=f"(_unpack_fp4_70[2]), "=f"(_unpack_fp4_70[3]), "=f"(_unpack_fp4_70[4]), "=f"(_unpack_fp4_70[5]), "=f"(_unpack_fp4_70[6]), "=f"(_unpack_fp4_70[7]) : "r"(_fp4_wire_values_3[10 + 0]));
                        #endif
                    }
                    float2 _f2_650 = make_float2(acc_63[16], acc_63[17]);
                    float2 _f2_651 = make_float2(_unpack_fp4_70[0], _unpack_fp4_70[1]);
                    float2 r_74 = fma_f32x2_rn_noftz(_f2_651, _f2_633, _f2_650);
                    acc_63[16] = r_74.x;
                    acc_63[17] = r_74.y;
                    float2 _f2_652 = make_float2(acc_63[18], acc_63[19]);
                    float2 _f2_653 = make_float2(_unpack_fp4_70[2], _unpack_fp4_70[3]);
                    float2 r_75 = fma_f32x2_rn_noftz(_f2_653, _f2_633, _f2_652);
                    acc_63[18] = r_75.x;
                    acc_63[19] = r_75.y;
                    float2 _f2_654 = make_float2(acc_63[20], acc_63[21]);
                    float2 _f2_655 = make_float2(_unpack_fp4_70[4], _unpack_fp4_70[5]);
                    float2 r_76_1 = fma_f32x2_rn_noftz(_f2_655, _f2_633, _f2_654);
                    acc_63[20] = r_76_1.x;
                    acc_63[21] = r_76_1.y;
                    float2 _f2_656 = make_float2(acc_63[22], acc_63[23]);
                    float2 _f2_657 = make_float2(_unpack_fp4_70[6], _unpack_fp4_70[7]);
                    float2 r_77_1 = fma_f32x2_rn_noftz(_f2_657, _f2_633, _f2_656);
                    acc_63[22] = r_77_1.x;
                    acc_63[23] = r_77_1.y;
                    float _unpack_fp4_71[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_3[11 + 0]));
                        (&_unpack_fp4_71[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_71[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_71[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_71[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_71[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_71[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_71[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_71[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_71[0]), "=f"(_unpack_fp4_71[1]), "=f"(_unpack_fp4_71[2]), "=f"(_unpack_fp4_71[3]), "=f"(_unpack_fp4_71[4]), "=f"(_unpack_fp4_71[5]), "=f"(_unpack_fp4_71[6]), "=f"(_unpack_fp4_71[7]) : "r"(_fp4_wire_values_3[11 + 0]));
                        #endif
                    }
                    float2 _f2_658 = make_float2(acc_63[24], acc_63[25]);
                    float2 _f2_659 = make_float2(_unpack_fp4_71[0], _unpack_fp4_71[1]);
                    float2 r_78_1 = fma_f32x2_rn_noftz(_f2_659, _f2_633, _f2_658);
                    acc_63[24] = r_78_1.x;
                    acc_63[25] = r_78_1.y;
                    float2 _f2_660 = make_float2(acc_63[26], acc_63[27]);
                    float2 _f2_661 = make_float2(_unpack_fp4_71[2], _unpack_fp4_71[3]);
                    float2 r_79_1 = fma_f32x2_rn_noftz(_f2_661, _f2_633, _f2_660);
                    acc_63[26] = r_79_1.x;
                    acc_63[27] = r_79_1.y;
                    float2 _f2_662 = make_float2(acc_63[28], acc_63[29]);
                    float2 _f2_663 = make_float2(_unpack_fp4_71[4], _unpack_fp4_71[5]);
                    float2 r_80_1 = fma_f32x2_rn_noftz(_f2_663, _f2_633, _f2_662);
                    acc_63[28] = r_80_1.x;
                    acc_63[29] = r_80_1.y;
                    float2 _f2_664 = make_float2(acc_63[30], acc_63[31]);
                    float2 _f2_665 = make_float2(_unpack_fp4_71[6], _unpack_fp4_71[7]);
                    float2 r_81_1 = fma_f32x2_rn_noftz(_f2_665, _f2_633, _f2_664);
                    acc_63[30] = r_81_1.x;
                    acc_63[31] = r_81_1.y;
                    uint32_t acc_63_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_63[_lp*2 + 0], acc_63[_lp*2+1 + 0]));
                        acc_63_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_63_bf16[0]), "r"(acc_63_bf16[1]), "r"(acc_63_bf16[2]), "r"(acc_63_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_63_bf16[4]), "r"(acc_63_bf16[5]), "r"(acc_63_bf16[6]), "r"(acc_63_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_63_bf16[8]), "r"(acc_63_bf16[9]), "r"(acc_63_bf16[10]), "r"(acc_63_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_63_bf16[12]), "r"(acc_63_bf16[13]), "r"(acc_63_bf16[14]), "r"(acc_63_bf16[15]) : "memory");
                    __syncwarp();
                    int c_82 = lane_1;
                    int l_83 = c_82 >> 2;
                    unsigned int w4_84[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_84[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_84[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_84[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_84[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_83 + (c_82 & 3 ^ l_83 >> 1 & 3)))));
                    long long idx_85 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_82));
                    {
                        int4 _iv4 = make_int4(w4_84[0 + 0], w4_84[0 + 1], w4_84[0 + 2], w4_84[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_85)) = _iv4;
                    }
                    int c_86 = lane_1 + 32;
                    int l_87 = c_86 >> 2;
                    unsigned int w4_88[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_88[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_88[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_88[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_88[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_87 + (c_86 & 3 ^ l_87 >> 1 & 3)))));
                    long long idx_89 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_86));
                    {
                        int4 _iv4 = make_int4(w4_88[0 + 0], w4_88[0 + 1], w4_88[0 + 2], w4_88[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_89)) = _iv4;
                    }
                    int c_90 = lane_1 + 64;
                    int l_91 = c_90 >> 2;
                    unsigned int w4_92[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_92[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_92[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_92[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_92[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_91 + (c_90 & 3 ^ l_91 >> 1 & 3)))));
                    long long idx_93 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_90));
                    {
                        int4 _iv4 = make_int4(w4_92[0 + 0], w4_92[0 + 1], w4_92[0 + 2], w4_92[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_93)) = _iv4;
                    }
                    int c_94 = lane_1 + 96;
                    int l_95 = c_94 >> 2;
                    unsigned int w4_96[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_96[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_96[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_96[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_96[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_95 + (c_94 & 3 ^ l_95 >> 1 & 3)))));
                    long long idx_97 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_94));
                    {
                        int4 _iv4 = make_int4(w4_96[0 + 0], w4_96[0 + 1], w4_96[0 + 2], w4_96[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_97)) = _iv4;
                    }
                    __syncwarp();
                }
            }
            if (lane_1 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            asm volatile("barrier.sync 6, 352;" ::: "memory");
            if (tid == 0) {
                __threadfence();
                unsigned int one = 1;
                unsigned int _atomic_old_0 = atomicAdd(&ctrl[1], one);
                unsigned int last = (unsigned int)num_bids - 1;
                if (_atomic_old_0 == last) {
                    unsigned int zero_1 = 0;
                    unsigned int rows_u = (unsigned int)rows;
                    unsigned int algo_u = 2;
                    ctrl[1] = zero_1;
                    ctrl[2] = rows_u;
                    ctrl[3] = algo_u;
                    __threadfence();
                    ctrl[0] = call + 1;
                    __threadfence();
                }
            }
        }
    }
    // ---- Role: reset_warp ----
    if (warp == 10) {
        { // reset_warp_main
            int lane_0 = (int)lane;
            unsigned int sv = 143165576;
            sv = sv | sv << 4;
            unsigned int ss = 268435455;
            ss = ss | ss << 4;
            unsigned int sr = 1;
            sr = sr << 31;
            #pragma unroll 1
            for (int i_1 = lane_0; i_1 < 1024; i_1 += 32) {
                pat[i_1] = sv;
                pat[1024 + i_1] = ss;
                pat[2048 + i_1] = sr;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("barrier.sync 0, 352;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int call_1 = ctl[0];
            unsigned int prev_rows = ctl[1];
            unsigned int prev_algo = ctl[2];
            unsigned int pbuf = (call_1 + 2) % 3;
#if __CUDA_ARCH__ == 1000
            int* self_i32 = reinterpret_cast<int*>(workspace[0]);
#else
            uint8_t* self_u8_1 = reinterpret_cast<uint8_t*>(workspace[0]);
#endif
            int g32_1 = (int)num_bids;
            if (rows / g32_1 <= 8) {
                asm volatile("barrier.sync 7, 352;" ::: "memory");
            }
            if (prev_algo == 2) {
                long long bid64_1 = (long long)bid;
                long long g64_1 = (long long)num_bids;
#if __CUDA_ARCH__ == 1000
                long long lane64 = (long long)lane_0;
#endif
                long long pr64 = (long long)prev_rows;
                long long pro64 = (pr64 + 3) / 4;
                long long pb64 = (long long)pbuf;
                long long rs_base_1 = pb64 * 113770496;
                long long ag_base_1 = rs_base_1 + 56885248;
#if __CUDA_ARCH__ == 1000
                unsigned int sv4[4];
                unsigned int ss4[4];
                unsigned int sr4[4];
                sv4[0] = sv;
                ss4[0] = ss;
                sr4[0] = sr;
                sv4[1] = sv;
                ss4[1] = ss;
                sr4[1] = sr;
                sv4[2] = sv;
                ss4[2] = ss;
                sr4[2] = sr;
                sv4[3] = sv;
                ss4[3] = ss;
                sr4[3] = sr;
#else
                int jbase = 0;
#endif
                long long u = pro64 * 3072 / 16;
#if __CUDA_ARCH__ == 1000
                long long lo = u * bid64_1 / g64_1;
                long long hi = u * (bid64_1 + 1) / g64_1;
                long long b64 = (long long)(int)rs_base_1;
                long long g0 = lo + lane64;
#else
                long long lo = u * bid64_1 / g64_1 * 16;
                long long hi = u * (bid64_1 + 1) / g64_1 * 16;
                int npages = (int)((hi - lo + 4095) / 4096);
                int i0_1 = lane_0 - jbase & 31;
                long long pg = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_3 = g0; g_3 < hi; g_3 += 32) {
                    long long widx = b64 + 16 * g_3 >> 2;
#else
                for (int i_2 = i0_1; i_2 < npages; i_2 += 32) {
                    long long o = lo + (long long)i_2 * 4096;
                    long long rem = hi - o;
                    unsigned int nb = (unsigned int)(((rem >= pg) ? pg : rem));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx)) = _iv4;
#else
                        void* _cpbulk_dst_0 = reinterpret_cast<void*>(self_u8_1 + ((long long)(int)rs_base_1 + o));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_0), "r"(pat_addr), "r"((uint32_t)(nb))
                            : "memory");
#endif
                    }
                }
#if !(__CUDA_ARCH__ == 1000)
                jbase = jbase + npages;
#endif
                long long u_0 = pro64 * 384 / 16;
#if __CUDA_ARCH__ == 1000
                long long lo_1 = u_0 * bid64_1 / g64_1;
                long long hi_2 = u_0 * (bid64_1 + 1) / g64_1;
                long long b64_3 = (long long)((int)rs_base_1 + 50331648);
                long long g0_4 = lo_1 + lane64;
#else
                long long lo_1 = u_0 * bid64_1 / g64_1 * 16;
                long long hi_2 = u_0 * (bid64_1 + 1) / g64_1 * 16;
                int npages_3 = (int)((hi_2 - lo_1 + 4095) / 4096);
                int i0_4 = lane_0 - jbase & 31;
                long long pg_5 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_4 = g0_4; g_4 < hi_2; g_4 += 32) {
                    long long widx_1 = b64_3 + 16 * g_4 >> 2;
#else
                for (int i_3 = i0_4; i_3 < npages_3; i_3 += 32) {
                    long long o_1 = lo_1 + (long long)i_3 * 4096;
                    long long rem_1 = hi_2 - o_1;
                    unsigned int nb_1 = (unsigned int)(((rem_1 >= pg_5) ? pg_5 : rem_1));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_1)) = _iv4;
#else
                        void* _cpbulk_dst_1 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)rs_base_1 + 50331648) + o_1));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_1), "r"(pat_addr + 4096), "r"((uint32_t)(nb_1))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_5 = pro64 * 16 / 16;
                long long lo_6 = u_5 * bid64_1 / g64_1;
                long long hi_7 = u_5 * (bid64_1 + 1) / g64_1;
                long long b64_8 = (long long)((int)rs_base_1 + 56623104);
                long long g0_9 = lo_6 + lane64;
#else
                jbase = jbase + npages_3;
                long long u_6 = pro64 * 16 / 16;
                long long lo_7 = u_6 * bid64_1 / g64_1 * 16;
                long long hi_8 = u_6 * (bid64_1 + 1) / g64_1 * 16;
                int npages_9 = (int)((hi_8 - lo_7 + 4095) / 4096);
                int i0_10 = lane_0 - jbase & 31;
                long long pg_11 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_5 = g0_9; g_5 < hi_7; g_5 += 32) {
                    long long widx_2 = b64_8 + 16 * g_5 >> 2;
#else
                for (int i_4 = i0_10; i_4 < npages_9; i_4 += 32) {
                    long long o_2 = lo_7 + (long long)i_4 * 4096;
                    long long rem_2 = hi_8 - o_2;
                    unsigned int nb_2 = (unsigned int)(((rem_2 >= pg_11) ? pg_11 : rem_2));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sr4[0 + 0], sr4[0 + 1], sr4[0 + 2], sr4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_2)) = _iv4;
#else
                        void* _cpbulk_dst_2 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)rs_base_1 + 56623104) + o_2));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_2), "r"(pat_addr + 8192), "r"((uint32_t)(nb_2))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_10 = pro64 * 3072 / 16;
                long long lo_11 = u_10 * bid64_1 / g64_1;
                long long hi_12 = u_10 * (bid64_1 + 1) / g64_1;
                long long b64_13 = (long long)((int)rs_base_1 + 12582912);
                long long g0_14 = lo_11 + lane64;
#else
                jbase = jbase + npages_9;
                long long u_12 = pro64 * 3072 / 16;
                long long lo_13 = u_12 * bid64_1 / g64_1 * 16;
                long long hi_14 = u_12 * (bid64_1 + 1) / g64_1 * 16;
                int npages_15 = (int)((hi_14 - lo_13 + 4095) / 4096);
                int i0_16 = lane_0 - jbase & 31;
                long long pg_17 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_6 = g0_14; g_6 < hi_12; g_6 += 32) {
                    long long widx_3 = b64_13 + 16 * g_6 >> 2;
#else
                for (int i_5 = i0_16; i_5 < npages_15; i_5 += 32) {
                    long long o_3 = lo_13 + (long long)i_5 * 4096;
                    long long rem_3 = hi_14 - o_3;
                    unsigned int nb_3 = (unsigned int)(((rem_3 >= pg_17) ? pg_17 : rem_3));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_3)) = _iv4;
#else
                        void* _cpbulk_dst_3 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)rs_base_1 + 12582912) + o_3));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_3), "r"(pat_addr), "r"((uint32_t)(nb_3))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_15 = pro64 * 384 / 16;
                long long lo_16 = u_15 * bid64_1 / g64_1;
                long long hi_17 = u_15 * (bid64_1 + 1) / g64_1;
                long long b64_18 = (long long)((int)rs_base_1 + 50331648 + 1572864);
                long long g0_19 = lo_16 + lane64;
#else
                jbase = jbase + npages_15;
                long long u_18 = pro64 * 384 / 16;
                long long lo_19 = u_18 * bid64_1 / g64_1 * 16;
                long long hi_20 = u_18 * (bid64_1 + 1) / g64_1 * 16;
                int npages_21 = (int)((hi_20 - lo_19 + 4095) / 4096);
                int i0_22 = lane_0 - jbase & 31;
                long long pg_23 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_7 = g0_19; g_7 < hi_17; g_7 += 32) {
                    long long widx_4 = b64_18 + 16 * g_7 >> 2;
#else
                for (int i_6 = i0_22; i_6 < npages_21; i_6 += 32) {
                    long long o_4 = lo_19 + (long long)i_6 * 4096;
                    long long rem_4 = hi_20 - o_4;
                    unsigned int nb_4 = (unsigned int)(((rem_4 >= pg_23) ? pg_23 : rem_4));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_4)) = _iv4;
#else
                        void* _cpbulk_dst_4 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)rs_base_1 + 50331648 + 1572864) + o_4));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_4), "r"(pat_addr + 4096), "r"((uint32_t)(nb_4))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_20 = pro64 * 16 / 16;
                long long lo_21 = u_20 * bid64_1 / g64_1;
                long long hi_22 = u_20 * (bid64_1 + 1) / g64_1;
                long long b64_23 = (long long)((int)rs_base_1 + 56623104 + 65536);
                long long g0_24 = lo_21 + lane64;
#else
                jbase = jbase + npages_21;
                long long u_24 = pro64 * 16 / 16;
                long long lo_25 = u_24 * bid64_1 / g64_1 * 16;
                long long hi_26 = u_24 * (bid64_1 + 1) / g64_1 * 16;
                int npages_27 = (int)((hi_26 - lo_25 + 4095) / 4096);
                int i0_28 = lane_0 - jbase & 31;
                long long pg_29 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_8 = g0_24; g_8 < hi_22; g_8 += 32) {
                    long long widx_5 = b64_23 + 16 * g_8 >> 2;
#else
                for (int i_7 = i0_28; i_7 < npages_27; i_7 += 32) {
                    long long o_5 = lo_25 + (long long)i_7 * 4096;
                    long long rem_5 = hi_26 - o_5;
                    unsigned int nb_5 = (unsigned int)(((rem_5 >= pg_29) ? pg_29 : rem_5));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sr4[0 + 0], sr4[0 + 1], sr4[0 + 2], sr4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_5)) = _iv4;
#else
                        void* _cpbulk_dst_5 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)rs_base_1 + 56623104 + 65536) + o_5));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_5), "r"(pat_addr + 8192), "r"((uint32_t)(nb_5))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_25 = pro64 * 3072 / 16;
                long long lo_26 = u_25 * bid64_1 / g64_1;
                long long hi_27 = u_25 * (bid64_1 + 1) / g64_1;
                long long b64_28 = (long long)((int)rs_base_1 + 25165824);
                long long g0_29 = lo_26 + lane64;
#else
                jbase = jbase + npages_27;
                long long u_30 = pro64 * 3072 / 16;
                long long lo_31 = u_30 * bid64_1 / g64_1 * 16;
                long long hi_32 = u_30 * (bid64_1 + 1) / g64_1 * 16;
                int npages_33 = (int)((hi_32 - lo_31 + 4095) / 4096);
                int i0_34 = lane_0 - jbase & 31;
                long long pg_35 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_9 = g0_29; g_9 < hi_27; g_9 += 32) {
                    long long widx_6 = b64_28 + 16 * g_9 >> 2;
#else
                for (int i_8 = i0_34; i_8 < npages_33; i_8 += 32) {
                    long long o_6 = lo_31 + (long long)i_8 * 4096;
                    long long rem_6 = hi_32 - o_6;
                    unsigned int nb_6 = (unsigned int)(((rem_6 >= pg_35) ? pg_35 : rem_6));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_6)) = _iv4;
#else
                        void* _cpbulk_dst_6 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)rs_base_1 + 25165824) + o_6));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_6), "r"(pat_addr), "r"((uint32_t)(nb_6))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_30 = pro64 * 384 / 16;
                long long lo_31 = u_30 * bid64_1 / g64_1;
                long long hi_32 = u_30 * (bid64_1 + 1) / g64_1;
                long long b64_33 = (long long)((int)rs_base_1 + 50331648 + 3145728);
                long long g0_34 = lo_31 + lane64;
#else
                jbase = jbase + npages_33;
                long long u_36 = pro64 * 384 / 16;
                long long lo_37 = u_36 * bid64_1 / g64_1 * 16;
                long long hi_38 = u_36 * (bid64_1 + 1) / g64_1 * 16;
                int npages_39 = (int)((hi_38 - lo_37 + 4095) / 4096);
                int i0_40 = lane_0 - jbase & 31;
                long long pg_41 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_10 = g0_34; g_10 < hi_32; g_10 += 32) {
                    long long widx_7 = b64_33 + 16 * g_10 >> 2;
#else
                for (int i_9 = i0_40; i_9 < npages_39; i_9 += 32) {
                    long long o_7 = lo_37 + (long long)i_9 * 4096;
                    long long rem_7 = hi_38 - o_7;
                    unsigned int nb_7 = (unsigned int)(((rem_7 >= pg_41) ? pg_41 : rem_7));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_7)) = _iv4;
#else
                        void* _cpbulk_dst_7 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)rs_base_1 + 50331648 + 3145728) + o_7));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_7), "r"(pat_addr + 4096), "r"((uint32_t)(nb_7))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_35 = pro64 * 16 / 16;
                long long lo_36 = u_35 * bid64_1 / g64_1;
                long long hi_37 = u_35 * (bid64_1 + 1) / g64_1;
                long long b64_38 = (long long)((int)rs_base_1 + 56623104 + 131072);
                long long g0_39 = lo_36 + lane64;
#else
                jbase = jbase + npages_39;
                long long u_42 = pro64 * 16 / 16;
                long long lo_43 = u_42 * bid64_1 / g64_1 * 16;
                long long hi_44 = u_42 * (bid64_1 + 1) / g64_1 * 16;
                int npages_45 = (int)((hi_44 - lo_43 + 4095) / 4096);
                int i0_46 = lane_0 - jbase & 31;
                long long pg_47 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_11 = g0_39; g_11 < hi_37; g_11 += 32) {
                    long long widx_8 = b64_38 + 16 * g_11 >> 2;
#else
                for (int i_10 = i0_46; i_10 < npages_45; i_10 += 32) {
                    long long o_8 = lo_43 + (long long)i_10 * 4096;
                    long long rem_8 = hi_44 - o_8;
                    unsigned int nb_8 = (unsigned int)(((rem_8 >= pg_47) ? pg_47 : rem_8));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sr4[0 + 0], sr4[0 + 1], sr4[0 + 2], sr4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_8)) = _iv4;
#else
                        void* _cpbulk_dst_8 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)rs_base_1 + 56623104 + 131072) + o_8));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_8), "r"(pat_addr + 8192), "r"((uint32_t)(nb_8))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_40 = pro64 * 3072 / 16;
                long long lo_41 = u_40 * bid64_1 / g64_1;
                long long hi_42 = u_40 * (bid64_1 + 1) / g64_1;
                long long b64_43 = (long long)((int)rs_base_1 + 37748736);
                long long g0_44 = lo_41 + lane64;
#else
                jbase = jbase + npages_45;
                long long u_48 = pro64 * 3072 / 16;
                long long lo_49 = u_48 * bid64_1 / g64_1 * 16;
                long long hi_50 = u_48 * (bid64_1 + 1) / g64_1 * 16;
                int npages_51 = (int)((hi_50 - lo_49 + 4095) / 4096);
                int i0_52 = lane_0 - jbase & 31;
                long long pg_53 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_12 = g0_44; g_12 < hi_42; g_12 += 32) {
                    long long widx_9 = b64_43 + 16 * g_12 >> 2;
#else
                for (int i_11 = i0_52; i_11 < npages_51; i_11 += 32) {
                    long long o_9 = lo_49 + (long long)i_11 * 4096;
                    long long rem_9 = hi_50 - o_9;
                    unsigned int nb_9 = (unsigned int)(((rem_9 >= pg_53) ? pg_53 : rem_9));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_9)) = _iv4;
#else
                        void* _cpbulk_dst_9 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)rs_base_1 + 37748736) + o_9));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_9), "r"(pat_addr), "r"((uint32_t)(nb_9))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_45 = pro64 * 384 / 16;
                long long lo_46 = u_45 * bid64_1 / g64_1;
                long long hi_47 = u_45 * (bid64_1 + 1) / g64_1;
                long long b64_48 = (long long)((int)rs_base_1 + 50331648 + 4718592);
                long long g0_49 = lo_46 + lane64;
#else
                jbase = jbase + npages_51;
                long long u_54 = pro64 * 384 / 16;
                long long lo_55 = u_54 * bid64_1 / g64_1 * 16;
                long long hi_56 = u_54 * (bid64_1 + 1) / g64_1 * 16;
                int npages_57 = (int)((hi_56 - lo_55 + 4095) / 4096);
                int i0_58 = lane_0 - jbase & 31;
                long long pg_59 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_13 = g0_49; g_13 < hi_47; g_13 += 32) {
                    long long widx_10 = b64_48 + 16 * g_13 >> 2;
#else
                for (int i_12 = i0_58; i_12 < npages_57; i_12 += 32) {
                    long long o_10 = lo_55 + (long long)i_12 * 4096;
                    long long rem_10 = hi_56 - o_10;
                    unsigned int nb_10 = (unsigned int)(((rem_10 >= pg_59) ? pg_59 : rem_10));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_10)) = _iv4;
#else
                        void* _cpbulk_dst_10 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)rs_base_1 + 50331648 + 4718592) + o_10));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_10), "r"(pat_addr + 4096), "r"((uint32_t)(nb_10))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_50 = pro64 * 16 / 16;
                long long lo_51 = u_50 * bid64_1 / g64_1;
                long long hi_52 = u_50 * (bid64_1 + 1) / g64_1;
                long long b64_53 = (long long)((int)rs_base_1 + 56623104 + 196608);
                long long g0_54 = lo_51 + lane64;
#else
                jbase = jbase + npages_57;
                long long u_60 = pro64 * 16 / 16;
                long long lo_61 = u_60 * bid64_1 / g64_1 * 16;
                long long hi_62 = u_60 * (bid64_1 + 1) / g64_1 * 16;
                int npages_63 = (int)((hi_62 - lo_61 + 4095) / 4096);
                int i0_64 = lane_0 - jbase & 31;
                long long pg_65 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_14 = g0_54; g_14 < hi_52; g_14 += 32) {
                    long long widx_11 = b64_53 + 16 * g_14 >> 2;
#else
                for (int i_13 = i0_64; i_13 < npages_63; i_13 += 32) {
                    long long o_11 = lo_61 + (long long)i_13 * 4096;
                    long long rem_11 = hi_62 - o_11;
                    unsigned int nb_11 = (unsigned int)(((rem_11 >= pg_65) ? pg_65 : rem_11));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sr4[0 + 0], sr4[0 + 1], sr4[0 + 2], sr4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_11)) = _iv4;
#else
                        void* _cpbulk_dst_11 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)rs_base_1 + 56623104 + 196608) + o_11));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_11), "r"(pat_addr + 8192), "r"((uint32_t)(nb_11))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_55 = pro64 * 3072 / 16;
                long long lo_56 = u_55 * bid64_1 / g64_1;
                long long hi_57 = u_55 * (bid64_1 + 1) / g64_1;
                long long b64_58 = (long long)((int)ag_base_1 + 12582912);
                long long g0_59 = lo_56 + lane64;
#else
                jbase = jbase + npages_63;
                long long u_66 = pro64 * 3072 / 16;
                long long lo_67 = u_66 * bid64_1 / g64_1 * 16;
                long long hi_68 = u_66 * (bid64_1 + 1) / g64_1 * 16;
                int npages_69 = (int)((hi_68 - lo_67 + 4095) / 4096);
                int i0_70 = lane_0 - jbase & 31;
                long long pg_71 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_15 = g0_59; g_15 < hi_57; g_15 += 32) {
                    long long widx_12 = b64_58 + 16 * g_15 >> 2;
#else
                for (int i_14 = i0_70; i_14 < npages_69; i_14 += 32) {
                    long long o_12 = lo_67 + (long long)i_14 * 4096;
                    long long rem_12 = hi_68 - o_12;
                    unsigned int nb_12 = (unsigned int)(((rem_12 >= pg_71) ? pg_71 : rem_12));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_12)) = _iv4;
#else
                        void* _cpbulk_dst_12 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)ag_base_1 + 12582912) + o_12));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_12), "r"(pat_addr), "r"((uint32_t)(nb_12))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_60 = pro64 * 384 / 16;
                long long lo_61 = u_60 * bid64_1 / g64_1;
                long long hi_62 = u_60 * (bid64_1 + 1) / g64_1;
                long long b64_63 = (long long)((int)ag_base_1 + 50331648 + 1572864);
                long long g0_64 = lo_61 + lane64;
#else
                jbase = jbase + npages_69;
                long long u_72 = pro64 * 384 / 16;
                long long lo_73 = u_72 * bid64_1 / g64_1 * 16;
                long long hi_74 = u_72 * (bid64_1 + 1) / g64_1 * 16;
                int npages_75 = (int)((hi_74 - lo_73 + 4095) / 4096);
                int i0_76 = lane_0 - jbase & 31;
                long long pg_77 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_16 = g0_64; g_16 < hi_62; g_16 += 32) {
                    long long widx_13 = b64_63 + 16 * g_16 >> 2;
#else
                for (int i_15 = i0_76; i_15 < npages_75; i_15 += 32) {
                    long long o_13 = lo_73 + (long long)i_15 * 4096;
                    long long rem_13 = hi_74 - o_13;
                    unsigned int nb_13 = (unsigned int)(((rem_13 >= pg_77) ? pg_77 : rem_13));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_13)) = _iv4;
#else
                        void* _cpbulk_dst_13 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)ag_base_1 + 50331648 + 1572864) + o_13));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_13), "r"(pat_addr + 4096), "r"((uint32_t)(nb_13))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_65 = pro64 * 16 / 16;
                long long lo_66 = u_65 * bid64_1 / g64_1;
                long long hi_67 = u_65 * (bid64_1 + 1) / g64_1;
                long long b64_68 = (long long)((int)ag_base_1 + 56623104 + 65536);
                long long g0_69 = lo_66 + lane64;
#else
                jbase = jbase + npages_75;
                long long u_78 = pro64 * 16 / 16;
                long long lo_79 = u_78 * bid64_1 / g64_1 * 16;
                long long hi_80 = u_78 * (bid64_1 + 1) / g64_1 * 16;
                int npages_81 = (int)((hi_80 - lo_79 + 4095) / 4096);
                int i0_82 = lane_0 - jbase & 31;
                long long pg_83 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_17 = g0_69; g_17 < hi_67; g_17 += 32) {
                    long long widx_14 = b64_68 + 16 * g_17 >> 2;
#else
                for (int i_16 = i0_82; i_16 < npages_81; i_16 += 32) {
                    long long o_14 = lo_79 + (long long)i_16 * 4096;
                    long long rem_14 = hi_80 - o_14;
                    unsigned int nb_14 = (unsigned int)(((rem_14 >= pg_83) ? pg_83 : rem_14));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sr4[0 + 0], sr4[0 + 1], sr4[0 + 2], sr4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_14)) = _iv4;
#else
                        void* _cpbulk_dst_14 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)ag_base_1 + 56623104 + 65536) + o_14));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_14), "r"(pat_addr + 8192), "r"((uint32_t)(nb_14))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_70 = pro64 * 3072 / 16;
                long long lo_71 = u_70 * bid64_1 / g64_1;
                long long hi_72 = u_70 * (bid64_1 + 1) / g64_1;
                long long b64_73 = (long long)((int)ag_base_1 + 25165824);
                long long g0_74 = lo_71 + lane64;
#else
                jbase = jbase + npages_81;
                long long u_84 = pro64 * 3072 / 16;
                long long lo_85 = u_84 * bid64_1 / g64_1 * 16;
                long long hi_86 = u_84 * (bid64_1 + 1) / g64_1 * 16;
                int npages_87 = (int)((hi_86 - lo_85 + 4095) / 4096);
                int i0_88 = lane_0 - jbase & 31;
                long long pg_89 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_18 = g0_74; g_18 < hi_72; g_18 += 32) {
                    long long widx_15 = b64_73 + 16 * g_18 >> 2;
#else
                for (int i_17 = i0_88; i_17 < npages_87; i_17 += 32) {
                    long long o_15 = lo_85 + (long long)i_17 * 4096;
                    long long rem_15 = hi_86 - o_15;
                    unsigned int nb_15 = (unsigned int)(((rem_15 >= pg_89) ? pg_89 : rem_15));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_15)) = _iv4;
#else
                        void* _cpbulk_dst_15 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)ag_base_1 + 25165824) + o_15));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_15), "r"(pat_addr), "r"((uint32_t)(nb_15))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_75 = pro64 * 384 / 16;
                long long lo_76 = u_75 * bid64_1 / g64_1;
                long long hi_77 = u_75 * (bid64_1 + 1) / g64_1;
                long long b64_78 = (long long)((int)ag_base_1 + 50331648 + 3145728);
                long long g0_79 = lo_76 + lane64;
#else
                jbase = jbase + npages_87;
                long long u_90 = pro64 * 384 / 16;
                long long lo_91 = u_90 * bid64_1 / g64_1 * 16;
                long long hi_92 = u_90 * (bid64_1 + 1) / g64_1 * 16;
                int npages_93 = (int)((hi_92 - lo_91 + 4095) / 4096);
                int i0_94 = lane_0 - jbase & 31;
                long long pg_95 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_19 = g0_79; g_19 < hi_77; g_19 += 32) {
                    long long widx_16 = b64_78 + 16 * g_19 >> 2;
#else
                for (int i_18 = i0_94; i_18 < npages_93; i_18 += 32) {
                    long long o_16 = lo_91 + (long long)i_18 * 4096;
                    long long rem_16 = hi_92 - o_16;
                    unsigned int nb_16 = (unsigned int)(((rem_16 >= pg_95) ? pg_95 : rem_16));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_16)) = _iv4;
#else
                        void* _cpbulk_dst_16 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)ag_base_1 + 50331648 + 3145728) + o_16));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_16), "r"(pat_addr + 4096), "r"((uint32_t)(nb_16))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_80 = pro64 * 16 / 16;
                long long lo_81 = u_80 * bid64_1 / g64_1;
                long long hi_82 = u_80 * (bid64_1 + 1) / g64_1;
                long long b64_83 = (long long)((int)ag_base_1 + 56623104 + 131072);
                long long g0_84 = lo_81 + lane64;
#else
                jbase = jbase + npages_93;
                long long u_96 = pro64 * 16 / 16;
                long long lo_97 = u_96 * bid64_1 / g64_1 * 16;
                long long hi_98 = u_96 * (bid64_1 + 1) / g64_1 * 16;
                int npages_99 = (int)((hi_98 - lo_97 + 4095) / 4096);
                int i0_100 = lane_0 - jbase & 31;
                long long pg_101 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_20 = g0_84; g_20 < hi_82; g_20 += 32) {
                    long long widx_17 = b64_83 + 16 * g_20 >> 2;
#else
                for (int i_19 = i0_100; i_19 < npages_99; i_19 += 32) {
                    long long o_17 = lo_97 + (long long)i_19 * 4096;
                    long long rem_17 = hi_98 - o_17;
                    unsigned int nb_17 = (unsigned int)(((rem_17 >= pg_101) ? pg_101 : rem_17));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sr4[0 + 0], sr4[0 + 1], sr4[0 + 2], sr4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_17)) = _iv4;
#else
                        void* _cpbulk_dst_17 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)ag_base_1 + 56623104 + 131072) + o_17));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_17), "r"(pat_addr + 8192), "r"((uint32_t)(nb_17))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_85 = pro64 * 3072 / 16;
                long long lo_86 = u_85 * bid64_1 / g64_1;
                long long hi_87 = u_85 * (bid64_1 + 1) / g64_1;
                long long b64_88 = (long long)((int)ag_base_1 + 37748736);
                long long g0_89 = lo_86 + lane64;
#else
                jbase = jbase + npages_99;
                long long u_102 = pro64 * 3072 / 16;
                long long lo_103 = u_102 * bid64_1 / g64_1 * 16;
                long long hi_104 = u_102 * (bid64_1 + 1) / g64_1 * 16;
                int npages_105 = (int)((hi_104 - lo_103 + 4095) / 4096);
                int i0_106 = lane_0 - jbase & 31;
                long long pg_107 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_21 = g0_89; g_21 < hi_87; g_21 += 32) {
                    long long widx_18 = b64_88 + 16 * g_21 >> 2;
#else
                for (int i_20 = i0_106; i_20 < npages_105; i_20 += 32) {
                    long long o_18 = lo_103 + (long long)i_20 * 4096;
                    long long rem_18 = hi_104 - o_18;
                    unsigned int nb_18 = (unsigned int)(((rem_18 >= pg_107) ? pg_107 : rem_18));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_18)) = _iv4;
#else
                        void* _cpbulk_dst_18 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)ag_base_1 + 37748736) + o_18));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_18), "r"(pat_addr), "r"((uint32_t)(nb_18))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_90 = pro64 * 384 / 16;
                long long lo_91 = u_90 * bid64_1 / g64_1;
                long long hi_92 = u_90 * (bid64_1 + 1) / g64_1;
                long long b64_93 = (long long)((int)ag_base_1 + 50331648 + 4718592);
                long long g0_94 = lo_91 + lane64;
#else
                jbase = jbase + npages_105;
                long long u_108 = pro64 * 384 / 16;
                long long lo_109 = u_108 * bid64_1 / g64_1 * 16;
                long long hi_110 = u_108 * (bid64_1 + 1) / g64_1 * 16;
                int npages_111 = (int)((hi_110 - lo_109 + 4095) / 4096);
                int i0_112 = lane_0 - jbase & 31;
                long long pg_113 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_22 = g0_94; g_22 < hi_92; g_22 += 32) {
                    long long widx_19 = b64_93 + 16 * g_22 >> 2;
#else
                for (int i_21 = i0_112; i_21 < npages_111; i_21 += 32) {
                    long long o_19 = lo_109 + (long long)i_21 * 4096;
                    long long rem_19 = hi_110 - o_19;
                    unsigned int nb_19 = (unsigned int)(((rem_19 >= pg_113) ? pg_113 : rem_19));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_19)) = _iv4;
#else
                        void* _cpbulk_dst_19 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)ag_base_1 + 50331648 + 4718592) + o_19));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_19), "r"(pat_addr + 4096), "r"((uint32_t)(nb_19))
                            : "memory");
#endif
                    }
                }
#if __CUDA_ARCH__ == 1000
                long long u_95 = pro64 * 16 / 16;
                long long lo_96 = u_95 * bid64_1 / g64_1;
                long long hi_97 = u_95 * (bid64_1 + 1) / g64_1;
                long long b64_98 = (long long)((int)ag_base_1 + 56623104 + 196608);
                long long g0_99 = lo_96 + lane64;
#else
                jbase = jbase + npages_111;
                long long u_114 = pro64 * 16 / 16;
                long long lo_115 = u_114 * bid64_1 / g64_1 * 16;
                long long hi_116 = u_114 * (bid64_1 + 1) / g64_1 * 16;
                int npages_117 = (int)((hi_116 - lo_115 + 4095) / 4096);
                int i0_118 = lane_0 - jbase & 31;
                long long pg_119 = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g_23 = g0_99; g_23 < hi_97; g_23 += 32) {
                    long long widx_20 = b64_98 + 16 * g_23 >> 2;
#else
                for (int i_22 = i0_118; i_22 < npages_117; i_22 += 32) {
                    long long o_20 = lo_115 + (long long)i_22 * 4096;
                    long long rem_20 = hi_116 - o_20;
                    unsigned int nb_20 = (unsigned int)(((rem_20 >= pg_119) ? pg_119 : rem_20));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sr4[0 + 0], sr4[0 + 1], sr4[0 + 2], sr4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_20)) = _iv4;
#else
                        void* _cpbulk_dst_20 = reinterpret_cast<void*>(self_u8_1 + ((long long)((int)ag_base_1 + 56623104 + 196608) + o_20));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_20), "r"(pat_addr + 8192), "r"((uint32_t)(nb_20))
                            : "memory");
#endif
                    }
                }
#if !(__CUDA_ARCH__ == 1000)
                jbase = jbase + npages_117;
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group 0;");
#endif
            }
            asm volatile("barrier.sync 6, 352;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
