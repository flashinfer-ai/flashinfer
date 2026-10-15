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
#define SMEM_WSM_OFF 13376
#define SMEM_WSM_STAGE_BYTES 212800
#define SMEM_WSM_STRIDE 212800
#define SMEM_WSM16_OFF 13376
#define SMEM_WSM16_STAGE_BYTES 212800
#define SMEM_WSM16_STRIDE 212800
#define SMEM_TOTAL 226176
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
kernel_fp4_allreduce_tp2_oneshot_nvfp4_fused_r1_cap16384(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ out, unsigned int* __restrict__ ctrl, int rows, const unsigned long long* __restrict__ workspace)
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
    uint8_t* wsm = reinterpret_cast<uint8_t*>(smem_raw + SMEM_WSM_OFF);
    const int wsm_addr = smem + SMEM_WSM_OFF;
    uint16_t* wsm16 = reinterpret_cast<uint16_t*>(smem_raw + SMEM_WSM16_OFF);
    const int wsm16_addr = smem + SMEM_WSM16_OFF;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // xb_full: 10 barriers, init_count=1
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
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: workers ----
    if (warp <= 9) {
        { // workers_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            float g = gs[0];
            float g_0 = gs[1];
            long long bid64 = (long long)bid;
            long long g64 = (long long)num_bids;
            long long rows64 = (long long)rows;
            int r0 = (int)(bid64 * rows64 / g64);
            int r1 = (int)((bid64 + 1) * rows64 / g64);
            int nrows = r1 - r0;
            int warp_1 = (int)warp;
            int lane_2 = (int)lane;
            int t = warp_1 >> 1;
            int h = warp_1 & 1;
            int leader = t * 2;
            if (lane_2 == 0 && h == 0 && nrows > t) {
                long long first_row = (long long)(r0 + t);
                mbarrier_arrive_expect_tx(xb_full_addr + (leader) * 8, 3456);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(leader * 21280), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(xq) + ((unsigned long long)(first_row * 3456) * (unsigned long long)1)), 3456, xb_full_addr + (leader) * 8);
            }
            if (tid == 0) {
                int base = bid * 4;
                ctl[0] = ctrl[base];
                ctl[1] = ctrl[base + 2];
                ctl[2] = ctrl[base + 3];
            }
            asm volatile("barrier.sync 0, 352;" ::: "memory");
            unsigned int call = ctl[0];
            unsigned int buf = call % 3;
            int* out_i32 = reinterpret_cast<int*>(out);
            int n = 0;
            if (nrows > t) {
                n = (nrows - t + 5 - 1) / 5;
            }
            int f = lane_2 >> 1 & 3;
            int ring_off = leader * 21280;
            int stg_off = warp_1 * 21280 + 12288 + 6944;
            uint8_t* qself_u8 = reinterpret_cast<uint8_t*>(workspace[3]);
            uint8_t* peer_base = reinterpret_cast<uint8_t*>(workspace[2]);
            long long q_base = (long long)buf * 113246208;
            unsigned int pf_v[12];
            unsigned int pf_s[3];
            pf_v[0] = 0;
            pf_v[1] = 0;
            pf_v[2] = 0;
            pf_v[3] = 0;
            pf_v[4] = 0;
            pf_v[5] = 0;
            pf_v[6] = 0;
            pf_v[7] = 0;
            pf_v[8] = 0;
            pf_v[9] = 0;
            pf_v[10] = 0;
            pf_v[11] = 0;
            pf_s[0] = 0;
            pf_s[1] = 0;
            pf_s[2] = 0;
            #pragma unroll 1
            for (int k = 0; k < n + 1; k++) {
                if (n > k) {
                    int slot = k % 3;
                    int ob_off = ring_off + slot * 3472;
                    mbarrier_wait(xb_full_addr + (leader) * 8, k & 1);
                    if (h == 0 && lane_2 == 0) {
                        {
                            long long row64 = (long long)(r0 + t + 5 * k);
                            long long rec_off = q_base + 56623104 + row64 * 3456;
                            {
                                void* _cpbulk_dst_0 = reinterpret_cast<void*>(peer_base + rec_off);
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_0), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3456))
                                    : "memory");
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                    if (h == 0 && lane_2 == 0 && n > k + 1) {
                        asm volatile("cp.async.bulk.wait_group.read 2;");
                        long long next_row = (long long)(r0 + t + 5 * (k + 1));
                        int next_slot = (k + 1) % 3;
                        mbarrier_arrive_expect_tx(xb_full_addr + (leader) * 8, 3456);
                        cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(ring_off + next_slot * 3472), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(xq) + ((unsigned long long)(next_row * 3456) * (unsigned long long)1)), 3456, xb_full_addr + (leader) * 8);
                    }
                }
                int kd = k - 1;
                if (kd >= 0 && kd < n) {
                    long long drow = (long long)(r0 + t + 5 * kd);
                    int dslot = kd % 3;
                    int dob_off = ring_off + dslot * 3472;
                    long long rec = q_base + drow * 3456;
                    unsigned int own[12];
                    unsigned int own_sc[3];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[0])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * (lane_2 + 32 * (3 * h)))));
                    own_sc[0] = (unsigned int)wsm16[(dob_off + 3072 >> 1) + (lane_2 + 32 * (3 * h))];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[4])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * (lane_2 + 32 * (3 * h + 1)))));
                    own_sc[1] = (unsigned int)wsm16[(dob_off + 3072 >> 1) + (lane_2 + 32 * (3 * h + 1))];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[8])), "=r"(*reinterpret_cast<uint32_t*>(&own[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * (lane_2 + 32 * (3 * h + 2)))));
                    own_sc[2] = (unsigned int)wsm16[(dob_off + 3072 >> 1) + (lane_2 + 32 * (3 * h + 2))];
                    unsigned int d = 0;
                    unsigned int w = pf_v[0];
                    d = d | w & ~((w & 2004318071) + 2004318071) & 2290649224u;
                    unsigned int w_0 = pf_v[1];
                    d = d | w_0 & ~((w_0 & 2004318071) + 2004318071) & 2290649224u;
                    unsigned int w_1 = pf_v[2];
                    d = d | w_1 & ~((w_1 & 2004318071) + 2004318071) & 2290649224u;
                    unsigned int w_2 = pf_v[3];
                    d = d | w_2 & ~((w_2 & 2004318071) + 2004318071) & 2290649224u;
                    unsigned int w_3 = pf_v[4];
                    d = d | w_3 & ~((w_3 & 2004318071) + 2004318071) & 2290649224u;
                    unsigned int w_4 = pf_v[5];
                    d = d | w_4 & ~((w_4 & 2004318071) + 2004318071) & 2290649224u;
                    unsigned int w_5 = pf_v[6];
                    d = d | w_5 & ~((w_5 & 2004318071) + 2004318071) & 2290649224u;
                    unsigned int w_6 = pf_v[7];
                    d = d | w_6 & ~((w_6 & 2004318071) + 2004318071) & 2290649224u;
                    unsigned int w_7 = pf_v[8];
                    d = d | w_7 & ~((w_7 & 2004318071) + 2004318071) & 2290649224u;
                    unsigned int w_8 = pf_v[9];
                    d = d | w_8 & ~((w_8 & 2004318071) + 2004318071) & 2290649224u;
                    unsigned int w_9 = pf_v[10];
                    d = d | w_9 & ~((w_9 & 2004318071) + 2004318071) & 2290649224u;
                    unsigned int w_10 = pf_v[11];
                    d = d | w_10 & ~((w_10 & 2004318071) + 2004318071) & 2290649224u;
                    unsigned int sc = pf_s[0];
                    d = d | ((sc & 255) + 1 | (sc >> 8) + 1) & 256;
                    unsigned int sc_11 = pf_s[1];
                    d = d | ((sc_11 & 255) + 1 | (sc_11 >> 8) + 1) & 256;
                    unsigned int sc_12 = pf_s[2];
                    d = d | ((sc_12 & 255) + 1 | (sc_12 >> 8) + 1) & 256;
                    if (d != 0) {
                        uint32_t _fp4_wire_values_0[12];
                        uint32_t _fp4_wire_scales_0[3];
                        uint16_t _wire_h16_1_0;
                        uint16_t _wire_h16_1_1;
                        uint16_t _wire_h16_1_2;
                        uint32_t _wire_vpend_1 = 7u;
                        uint32_t _wire_spend_1 = 7u;
                        do {
                            if (_wire_vpend_1 & 1u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(qself_u8 + (rec + (long long)(16 * (lane_2 + 32 * (3 * h)))))) : "memory");
                            }
                            if (_wire_vpend_1 & 2u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(qself_u8 + (rec + (long long)(16 * (lane_2 + 32 * (3 * h + 1)))))) : "memory");
                            }
                            if (_wire_vpend_1 & 4u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[8]), "=r"(_fp4_wire_values_0[9]), "=r"(_fp4_wire_values_0[10]), "=r"(_fp4_wire_values_0[11]) : "l"(reinterpret_cast<unsigned int*>(qself_u8 + (rec + (long long)(16 * (lane_2 + 32 * (3 * h + 2)))))) : "memory");
                            }
                            if (_wire_spend_1 & 1u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_1_0) : "l"(reinterpret_cast<uint16_t*>(qself_u8 + (rec + 3072 + (long long)(2 * (lane_2 + 32 * (3 * h)))))) : "memory");
                                _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_1_0;
                            }
                            if (_wire_spend_1 & 2u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_1_1) : "l"(reinterpret_cast<uint16_t*>(qself_u8 + (rec + 3072 + (long long)(2 * (lane_2 + 32 * (3 * h + 1)))))) : "memory");
                                _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_1_1;
                            }
                            if (_wire_spend_1 & 4u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_1_2) : "l"(reinterpret_cast<uint16_t*>(qself_u8 + (rec + 3072 + (long long)(2 * (lane_2 + 32 * (3 * h + 2)))))) : "memory");
                                _fp4_wire_scales_0[2] = (uint32_t)_wire_h16_1_2;
                            }
                            if (_wire_vpend_1 & 1u) {
                                if (!((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_1 &= ~1u;
                            }
                            if (_wire_vpend_1 & 2u) {
                                if (!((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_1 &= ~2u;
                            }
                            if (_wire_vpend_1 & 4u) {
                                if (!((((_fp4_wire_values_0[8] & ~((_fp4_wire_values_0[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[9] & ~((_fp4_wire_values_0[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[10] & ~((_fp4_wire_values_0[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[11] & ~((_fp4_wire_values_0[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_1 &= ~4u;
                            }
                            if (_wire_spend_1 & 1u) {
                                if (!(((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[0] >> 8) == 0xFFu))) _wire_spend_1 &= ~1u;
                            }
                            if (_wire_spend_1 & 2u) {
                                if (!(((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[1] >> 8) == 0xFFu))) _wire_spend_1 &= ~2u;
                            }
                            if (_wire_spend_1 & 4u) {
                                if (!(((_fp4_wire_scales_0[2] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[2] >> 8) == 0xFFu))) _wire_spend_1 &= ~4u;
                            }
                        } while ((_wire_vpend_1 | _wire_spend_1) != 0u);
                        pf_v[0] = _fp4_wire_values_0[0];
                        pf_v[1] = _fp4_wire_values_0[1];
                        pf_v[2] = _fp4_wire_values_0[2];
                        pf_v[3] = _fp4_wire_values_0[3];
                        pf_v[4] = _fp4_wire_values_0[4];
                        pf_v[5] = _fp4_wire_values_0[5];
                        pf_v[6] = _fp4_wire_values_0[6];
                        pf_v[7] = _fp4_wire_values_0[7];
                        pf_v[8] = _fp4_wire_values_0[8];
                        pf_v[9] = _fp4_wire_values_0[9];
                        pf_v[10] = _fp4_wire_values_0[10];
                        pf_v[11] = _fp4_wire_values_0[11];
                        pf_s[0] = _fp4_wire_scales_0[0];
                        pf_s[1] = _fp4_wire_scales_0[1];
                        pf_s[2] = _fp4_wire_scales_0[2];
                    }
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
                        : "=f"(_fp8x2_decode_0.x), "=f"(_fp8x2_decode_0.y) : "h"((uint16_t)pf_s[0]));
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
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[0]));
                        (&_unpack_fp4_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_0[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_0[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_0[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_0[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_0[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_0[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_0[0]), "=f"(_unpack_fp4_0[1]), "=f"(_unpack_fp4_0[2]), "=f"(_unpack_fp4_0[3]), "=f"(_unpack_fp4_0[4]), "=f"(_unpack_fp4_0[5]), "=f"(_unpack_fp4_0[6]), "=f"(_unpack_fp4_0[7]) : "r"(pf_v[0]));
                        #endif
                    }
                    float2 _f2_2 = make_float2(acc[0], acc[1]);
                    float2 _f2_3 = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                    float2 r = fma_f32x2_rn_noftz(_f2_3, _f2_0, _f2_2);
                    acc[0] = r.x;
                    acc[1] = r.y;
                    float2 _f2_4 = make_float2(acc[2], acc[3]);
                    float2 _f2_5 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                    float2 r_13 = fma_f32x2_rn_noftz(_f2_5, _f2_0, _f2_4);
                    acc[2] = r_13.x;
                    acc[3] = r_13.y;
                    float2 _f2_6 = make_float2(acc[4], acc[5]);
                    float2 _f2_7 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                    float2 r_14 = fma_f32x2_rn_noftz(_f2_7, _f2_0, _f2_6);
                    acc[4] = r_14.x;
                    acc[5] = r_14.y;
                    float2 _f2_8 = make_float2(acc[6], acc[7]);
                    float2 _f2_9 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                    float2 r_15 = fma_f32x2_rn_noftz(_f2_9, _f2_0, _f2_8);
                    acc[6] = r_15.x;
                    acc[7] = r_15.y;
                    float _unpack_fp4_1[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[1 + 0]));
                        (&_unpack_fp4_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_1[0]), "=f"(_unpack_fp4_1[1]), "=f"(_unpack_fp4_1[2]), "=f"(_unpack_fp4_1[3]), "=f"(_unpack_fp4_1[4]), "=f"(_unpack_fp4_1[5]), "=f"(_unpack_fp4_1[6]), "=f"(_unpack_fp4_1[7]) : "r"(pf_v[1 + 0]));
                        #endif
                    }
                    float2 _f2_10 = make_float2(acc[8], acc[9]);
                    float2 _f2_11 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                    float2 r_16 = fma_f32x2_rn_noftz(_f2_11, _f2_0, _f2_10);
                    acc[8] = r_16.x;
                    acc[9] = r_16.y;
                    float2 _f2_12 = make_float2(acc[10], acc[11]);
                    float2 _f2_13 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                    float2 r_17 = fma_f32x2_rn_noftz(_f2_13, _f2_0, _f2_12);
                    acc[10] = r_17.x;
                    acc[11] = r_17.y;
                    float2 _f2_14 = make_float2(acc[12], acc[13]);
                    float2 _f2_15 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                    float2 r_18 = fma_f32x2_rn_noftz(_f2_15, _f2_0, _f2_14);
                    acc[12] = r_18.x;
                    acc[13] = r_18.y;
                    float2 _f2_16 = make_float2(acc[14], acc[15]);
                    float2 _f2_17 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                    float2 r_19 = fma_f32x2_rn_noftz(_f2_17, _f2_0, _f2_16);
                    acc[14] = r_19.x;
                    acc[15] = r_19.y;
                    float _unpack_fp4_2[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[2 + 0]));
                        (&_unpack_fp4_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_2[0]), "=f"(_unpack_fp4_2[1]), "=f"(_unpack_fp4_2[2]), "=f"(_unpack_fp4_2[3]), "=f"(_unpack_fp4_2[4]), "=f"(_unpack_fp4_2[5]), "=f"(_unpack_fp4_2[6]), "=f"(_unpack_fp4_2[7]) : "r"(pf_v[2 + 0]));
                        #endif
                    }
                    float2 _f2_18 = make_float2(acc[16], acc[17]);
                    float2 _f2_19 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                    float2 r_20 = fma_f32x2_rn_noftz(_f2_19, _f2_1, _f2_18);
                    acc[16] = r_20.x;
                    acc[17] = r_20.y;
                    float2 _f2_20 = make_float2(acc[18], acc[19]);
                    float2 _f2_21 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                    float2 r_21 = fma_f32x2_rn_noftz(_f2_21, _f2_1, _f2_20);
                    acc[18] = r_21.x;
                    acc[19] = r_21.y;
                    float2 _f2_22 = make_float2(acc[20], acc[21]);
                    float2 _f2_23 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                    float2 r_22 = fma_f32x2_rn_noftz(_f2_23, _f2_1, _f2_22);
                    acc[20] = r_22.x;
                    acc[21] = r_22.y;
                    float2 _f2_24 = make_float2(acc[22], acc[23]);
                    float2 _f2_25 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                    float2 r_23 = fma_f32x2_rn_noftz(_f2_25, _f2_1, _f2_24);
                    acc[22] = r_23.x;
                    acc[23] = r_23.y;
                    float _unpack_fp4_3[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[3 + 0]));
                        (&_unpack_fp4_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_3[0]), "=f"(_unpack_fp4_3[1]), "=f"(_unpack_fp4_3[2]), "=f"(_unpack_fp4_3[3]), "=f"(_unpack_fp4_3[4]), "=f"(_unpack_fp4_3[5]), "=f"(_unpack_fp4_3[6]), "=f"(_unpack_fp4_3[7]) : "r"(pf_v[3 + 0]));
                        #endif
                    }
                    float2 _f2_26 = make_float2(acc[24], acc[25]);
                    float2 _f2_27 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                    float2 r_24 = fma_f32x2_rn_noftz(_f2_27, _f2_1, _f2_26);
                    acc[24] = r_24.x;
                    acc[25] = r_24.y;
                    float2 _f2_28 = make_float2(acc[26], acc[27]);
                    float2 _f2_29 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                    float2 r_25 = fma_f32x2_rn_noftz(_f2_29, _f2_1, _f2_28);
                    acc[26] = r_25.x;
                    acc[27] = r_25.y;
                    float2 _f2_30 = make_float2(acc[28], acc[29]);
                    float2 _f2_31 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                    float2 r_26 = fma_f32x2_rn_noftz(_f2_31, _f2_1, _f2_30);
                    acc[28] = r_26.x;
                    acc[29] = r_26.y;
                    float2 _f2_32 = make_float2(acc[30], acc[31]);
                    float2 _f2_33 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                    float2 r_27 = fma_f32x2_rn_noftz(_f2_33, _f2_1, _f2_32);
                    acc[30] = r_27.x;
                    acc[31] = r_27.y;
                    float2 _fp8x2_decode_1;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_1.x), "=f"(_fp8x2_decode_1.y) : "h"((uint16_t)own_sc[0]));
                    float _mul_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_fp8x2_decode_1.x), "f"(g_0));
                    float s0_28 = _mul_2;
                    float _mul_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_fp8x2_decode_1.y), "f"(g_0));
                    float s1_29 = _mul_3;
                    float2 _f2_34 = make_float2(s0_28, s0_28);
                    float2 _f2_35 = make_float2(s1_29, s1_29);
                    float _unpack_fp4_4[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[0]));
                        (&_unpack_fp4_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_4[0]), "=f"(_unpack_fp4_4[1]), "=f"(_unpack_fp4_4[2]), "=f"(_unpack_fp4_4[3]), "=f"(_unpack_fp4_4[4]), "=f"(_unpack_fp4_4[5]), "=f"(_unpack_fp4_4[6]), "=f"(_unpack_fp4_4[7]) : "r"(own[0]));
                        #endif
                    }
                    float2 _f2_36 = make_float2(acc[0], acc[1]);
                    float2 _f2_37 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                    float2 r_30 = fma_f32x2_rn_noftz(_f2_37, _f2_34, _f2_36);
                    acc[0] = r_30.x;
                    acc[1] = r_30.y;
                    float2 _f2_38 = make_float2(acc[2], acc[3]);
                    float2 _f2_39 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                    float2 r_31 = fma_f32x2_rn_noftz(_f2_39, _f2_34, _f2_38);
                    acc[2] = r_31.x;
                    acc[3] = r_31.y;
                    float2 _f2_40 = make_float2(acc[4], acc[5]);
                    float2 _f2_41 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                    float2 r_32 = fma_f32x2_rn_noftz(_f2_41, _f2_34, _f2_40);
                    acc[4] = r_32.x;
                    acc[5] = r_32.y;
                    float2 _f2_42 = make_float2(acc[6], acc[7]);
                    float2 _f2_43 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                    float2 r_33 = fma_f32x2_rn_noftz(_f2_43, _f2_34, _f2_42);
                    acc[6] = r_33.x;
                    acc[7] = r_33.y;
                    float _unpack_fp4_5[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[1 + 0]));
                        (&_unpack_fp4_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_5[0]), "=f"(_unpack_fp4_5[1]), "=f"(_unpack_fp4_5[2]), "=f"(_unpack_fp4_5[3]), "=f"(_unpack_fp4_5[4]), "=f"(_unpack_fp4_5[5]), "=f"(_unpack_fp4_5[6]), "=f"(_unpack_fp4_5[7]) : "r"(own[1 + 0]));
                        #endif
                    }
                    float2 _f2_44 = make_float2(acc[8], acc[9]);
                    float2 _f2_45 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                    float2 r_34 = fma_f32x2_rn_noftz(_f2_45, _f2_34, _f2_44);
                    acc[8] = r_34.x;
                    acc[9] = r_34.y;
                    float2 _f2_46 = make_float2(acc[10], acc[11]);
                    float2 _f2_47 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                    float2 r_35 = fma_f32x2_rn_noftz(_f2_47, _f2_34, _f2_46);
                    acc[10] = r_35.x;
                    acc[11] = r_35.y;
                    float2 _f2_48 = make_float2(acc[12], acc[13]);
                    float2 _f2_49 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                    float2 r_36 = fma_f32x2_rn_noftz(_f2_49, _f2_34, _f2_48);
                    acc[12] = r_36.x;
                    acc[13] = r_36.y;
                    float2 _f2_50 = make_float2(acc[14], acc[15]);
                    float2 _f2_51 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                    float2 r_37 = fma_f32x2_rn_noftz(_f2_51, _f2_34, _f2_50);
                    acc[14] = r_37.x;
                    acc[15] = r_37.y;
                    float _unpack_fp4_6[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[2 + 0]));
                        (&_unpack_fp4_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_6[0]), "=f"(_unpack_fp4_6[1]), "=f"(_unpack_fp4_6[2]), "=f"(_unpack_fp4_6[3]), "=f"(_unpack_fp4_6[4]), "=f"(_unpack_fp4_6[5]), "=f"(_unpack_fp4_6[6]), "=f"(_unpack_fp4_6[7]) : "r"(own[2 + 0]));
                        #endif
                    }
                    float2 _f2_52 = make_float2(acc[16], acc[17]);
                    float2 _f2_53 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                    float2 r_38 = fma_f32x2_rn_noftz(_f2_53, _f2_35, _f2_52);
                    acc[16] = r_38.x;
                    acc[17] = r_38.y;
                    float2 _f2_54 = make_float2(acc[18], acc[19]);
                    float2 _f2_55 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                    float2 r_39 = fma_f32x2_rn_noftz(_f2_55, _f2_35, _f2_54);
                    acc[18] = r_39.x;
                    acc[19] = r_39.y;
                    float2 _f2_56 = make_float2(acc[20], acc[21]);
                    float2 _f2_57 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                    float2 r_40 = fma_f32x2_rn_noftz(_f2_57, _f2_35, _f2_56);
                    acc[20] = r_40.x;
                    acc[21] = r_40.y;
                    float2 _f2_58 = make_float2(acc[22], acc[23]);
                    float2 _f2_59 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                    float2 r_41 = fma_f32x2_rn_noftz(_f2_59, _f2_35, _f2_58);
                    acc[22] = r_41.x;
                    acc[23] = r_41.y;
                    float _unpack_fp4_7[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[3 + 0]));
                        (&_unpack_fp4_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_7[0]), "=f"(_unpack_fp4_7[1]), "=f"(_unpack_fp4_7[2]), "=f"(_unpack_fp4_7[3]), "=f"(_unpack_fp4_7[4]), "=f"(_unpack_fp4_7[5]), "=f"(_unpack_fp4_7[6]), "=f"(_unpack_fp4_7[7]) : "r"(own[3 + 0]));
                        #endif
                    }
                    float2 _f2_60 = make_float2(acc[24], acc[25]);
                    float2 _f2_61 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                    float2 r_42 = fma_f32x2_rn_noftz(_f2_61, _f2_35, _f2_60);
                    acc[24] = r_42.x;
                    acc[25] = r_42.y;
                    float2 _f2_62 = make_float2(acc[26], acc[27]);
                    float2 _f2_63 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                    float2 r_43 = fma_f32x2_rn_noftz(_f2_63, _f2_35, _f2_62);
                    acc[26] = r_43.x;
                    acc[27] = r_43.y;
                    float2 _f2_64 = make_float2(acc[28], acc[29]);
                    float2 _f2_65 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                    float2 r_44 = fma_f32x2_rn_noftz(_f2_65, _f2_35, _f2_64);
                    acc[28] = r_44.x;
                    acc[29] = r_44.y;
                    float2 _f2_66 = make_float2(acc[30], acc[31]);
                    float2 _f2_67 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                    float2 r_45 = fma_f32x2_rn_noftz(_f2_67, _f2_35, _f2_66);
                    acc[30] = r_45.x;
                    acc[31] = r_45.y;
                    uint32_t acc_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                        acc_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_2 + (f ^ 0)))), "r"(acc_bf16[0]), "r"(acc_bf16[1]), "r"(acc_bf16[2]), "r"(acc_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_2 + (f ^ 1)))), "r"(acc_bf16[4]), "r"(acc_bf16[5]), "r"(acc_bf16[6]), "r"(acc_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_2 + (f ^ 2)))), "r"(acc_bf16[8]), "r"(acc_bf16[9]), "r"(acc_bf16[10]), "r"(acc_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_2 + (f ^ 3)))), "r"(acc_bf16[12]), "r"(acc_bf16[13]), "r"(acc_bf16[14]), "r"(acc_bf16[15]) : "memory");
                    __syncwarp();
                    int c = lane_2;
                    int l = c >> 2;
                    unsigned int w4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l + (c & 3 ^ l >> 1 & 3)))));
                    long long idx = drow * 3072 + (long long)(4 * (128 * (3 * h) + c));
                    {
                        int4 _iv4 = make_int4(w4[0 + 0], w4[0 + 1], w4[0 + 2], w4[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx)) = _iv4;
                    }
                    int c_46 = lane_2 + 32;
                    int l_47 = c_46 >> 2;
                    unsigned int w4_48[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_48[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_48[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_48[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_48[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_47 + (c_46 & 3 ^ l_47 >> 1 & 3)))));
                    long long idx_49 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_46));
                    {
                        int4 _iv4 = make_int4(w4_48[0 + 0], w4_48[0 + 1], w4_48[0 + 2], w4_48[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_49)) = _iv4;
                    }
                    int c_50 = lane_2 + 64;
                    int l_51 = c_50 >> 2;
                    unsigned int w4_52[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_52[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_52[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_52[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_52[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_51 + (c_50 & 3 ^ l_51 >> 1 & 3)))));
                    long long idx_53 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_50));
                    {
                        int4 _iv4 = make_int4(w4_52[0 + 0], w4_52[0 + 1], w4_52[0 + 2], w4_52[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_53)) = _iv4;
                    }
                    int c_54 = lane_2 + 96;
                    int l_55 = c_54 >> 2;
                    unsigned int w4_56[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_56[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_56[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_56[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_56[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_55 + (c_54 & 3 ^ l_55 >> 1 & 3)))));
                    long long idx_57 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_54));
                    {
                        int4 _iv4 = make_int4(w4_56[0 + 0], w4_56[0 + 1], w4_56[0 + 2], w4_56[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_57)) = _iv4;
                    }
                    __syncwarp();
                    float acc_58[32];
                    acc_58[0] = 0.0f;
                    acc_58[1] = 0.0f;
                    acc_58[2] = 0.0f;
                    acc_58[3] = 0.0f;
                    acc_58[4] = 0.0f;
                    acc_58[5] = 0.0f;
                    acc_58[6] = 0.0f;
                    acc_58[7] = 0.0f;
                    acc_58[8] = 0.0f;
                    acc_58[9] = 0.0f;
                    acc_58[10] = 0.0f;
                    acc_58[11] = 0.0f;
                    acc_58[12] = 0.0f;
                    acc_58[13] = 0.0f;
                    acc_58[14] = 0.0f;
                    acc_58[15] = 0.0f;
                    acc_58[16] = 0.0f;
                    acc_58[17] = 0.0f;
                    acc_58[18] = 0.0f;
                    acc_58[19] = 0.0f;
                    acc_58[20] = 0.0f;
                    acc_58[21] = 0.0f;
                    acc_58[22] = 0.0f;
                    acc_58[23] = 0.0f;
                    acc_58[24] = 0.0f;
                    acc_58[25] = 0.0f;
                    acc_58[26] = 0.0f;
                    acc_58[27] = 0.0f;
                    acc_58[28] = 0.0f;
                    acc_58[29] = 0.0f;
                    acc_58[30] = 0.0f;
                    acc_58[31] = 0.0f;
                    float2 _fp8x2_decode_2;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_2.x), "=f"(_fp8x2_decode_2.y) : "h"((uint16_t)pf_s[1]));
                    float _mul_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_fp8x2_decode_2.x), "f"(g));
                    float s0_59 = _mul_4;
                    float _mul_5;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_fp8x2_decode_2.y), "f"(g));
                    float s1_60 = _mul_5;
                    float2 _f2_68 = make_float2(s0_59, s0_59);
                    float2 _f2_69 = make_float2(s1_60, s1_60);
                    float _unpack_fp4_8[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[4 + 0]));
                        (&_unpack_fp4_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_8[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_8[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_8[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_8[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_8[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_8[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_8[0]), "=f"(_unpack_fp4_8[1]), "=f"(_unpack_fp4_8[2]), "=f"(_unpack_fp4_8[3]), "=f"(_unpack_fp4_8[4]), "=f"(_unpack_fp4_8[5]), "=f"(_unpack_fp4_8[6]), "=f"(_unpack_fp4_8[7]) : "r"(pf_v[4 + 0]));
                        #endif
                    }
                    float2 _f2_70 = make_float2(acc_58[0], acc_58[1]);
                    float2 _f2_71 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                    float2 r_61 = fma_f32x2_rn_noftz(_f2_71, _f2_68, _f2_70);
                    acc_58[0] = r_61.x;
                    acc_58[1] = r_61.y;
                    float2 _f2_72 = make_float2(acc_58[2], acc_58[3]);
                    float2 _f2_73 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                    float2 r_62 = fma_f32x2_rn_noftz(_f2_73, _f2_68, _f2_72);
                    acc_58[2] = r_62.x;
                    acc_58[3] = r_62.y;
                    float2 _f2_74 = make_float2(acc_58[4], acc_58[5]);
                    float2 _f2_75 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                    float2 r_63 = fma_f32x2_rn_noftz(_f2_75, _f2_68, _f2_74);
                    acc_58[4] = r_63.x;
                    acc_58[5] = r_63.y;
                    float2 _f2_76 = make_float2(acc_58[6], acc_58[7]);
                    float2 _f2_77 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                    float2 r_64 = fma_f32x2_rn_noftz(_f2_77, _f2_68, _f2_76);
                    acc_58[6] = r_64.x;
                    acc_58[7] = r_64.y;
                    float _unpack_fp4_9[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[5 + 0]));
                        (&_unpack_fp4_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_9[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_9[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_9[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_9[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_9[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_9[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_9[0]), "=f"(_unpack_fp4_9[1]), "=f"(_unpack_fp4_9[2]), "=f"(_unpack_fp4_9[3]), "=f"(_unpack_fp4_9[4]), "=f"(_unpack_fp4_9[5]), "=f"(_unpack_fp4_9[6]), "=f"(_unpack_fp4_9[7]) : "r"(pf_v[5 + 0]));
                        #endif
                    }
                    float2 _f2_78 = make_float2(acc_58[8], acc_58[9]);
                    float2 _f2_79 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                    float2 r_65 = fma_f32x2_rn_noftz(_f2_79, _f2_68, _f2_78);
                    acc_58[8] = r_65.x;
                    acc_58[9] = r_65.y;
                    float2 _f2_80 = make_float2(acc_58[10], acc_58[11]);
                    float2 _f2_81 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                    float2 r_66 = fma_f32x2_rn_noftz(_f2_81, _f2_68, _f2_80);
                    acc_58[10] = r_66.x;
                    acc_58[11] = r_66.y;
                    float2 _f2_82 = make_float2(acc_58[12], acc_58[13]);
                    float2 _f2_83 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                    float2 r_67 = fma_f32x2_rn_noftz(_f2_83, _f2_68, _f2_82);
                    acc_58[12] = r_67.x;
                    acc_58[13] = r_67.y;
                    float2 _f2_84 = make_float2(acc_58[14], acc_58[15]);
                    float2 _f2_85 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                    float2 r_68 = fma_f32x2_rn_noftz(_f2_85, _f2_68, _f2_84);
                    acc_58[14] = r_68.x;
                    acc_58[15] = r_68.y;
                    float _unpack_fp4_10[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[6 + 0]));
                        (&_unpack_fp4_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_10[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_10[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_10[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_10[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_10[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_10[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_10[0]), "=f"(_unpack_fp4_10[1]), "=f"(_unpack_fp4_10[2]), "=f"(_unpack_fp4_10[3]), "=f"(_unpack_fp4_10[4]), "=f"(_unpack_fp4_10[5]), "=f"(_unpack_fp4_10[6]), "=f"(_unpack_fp4_10[7]) : "r"(pf_v[6 + 0]));
                        #endif
                    }
                    float2 _f2_86 = make_float2(acc_58[16], acc_58[17]);
                    float2 _f2_87 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                    float2 r_69 = fma_f32x2_rn_noftz(_f2_87, _f2_69, _f2_86);
                    acc_58[16] = r_69.x;
                    acc_58[17] = r_69.y;
                    float2 _f2_88 = make_float2(acc_58[18], acc_58[19]);
                    float2 _f2_89 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                    float2 r_70 = fma_f32x2_rn_noftz(_f2_89, _f2_69, _f2_88);
                    acc_58[18] = r_70.x;
                    acc_58[19] = r_70.y;
                    float2 _f2_90 = make_float2(acc_58[20], acc_58[21]);
                    float2 _f2_91 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                    float2 r_71 = fma_f32x2_rn_noftz(_f2_91, _f2_69, _f2_90);
                    acc_58[20] = r_71.x;
                    acc_58[21] = r_71.y;
                    float2 _f2_92 = make_float2(acc_58[22], acc_58[23]);
                    float2 _f2_93 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                    float2 r_72 = fma_f32x2_rn_noftz(_f2_93, _f2_69, _f2_92);
                    acc_58[22] = r_72.x;
                    acc_58[23] = r_72.y;
                    float _unpack_fp4_11[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[7 + 0]));
                        (&_unpack_fp4_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_11[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_11[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_11[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_11[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_11[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_11[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_11[0]), "=f"(_unpack_fp4_11[1]), "=f"(_unpack_fp4_11[2]), "=f"(_unpack_fp4_11[3]), "=f"(_unpack_fp4_11[4]), "=f"(_unpack_fp4_11[5]), "=f"(_unpack_fp4_11[6]), "=f"(_unpack_fp4_11[7]) : "r"(pf_v[7 + 0]));
                        #endif
                    }
                    float2 _f2_94 = make_float2(acc_58[24], acc_58[25]);
                    float2 _f2_95 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                    float2 r_73 = fma_f32x2_rn_noftz(_f2_95, _f2_69, _f2_94);
                    acc_58[24] = r_73.x;
                    acc_58[25] = r_73.y;
                    float2 _f2_96 = make_float2(acc_58[26], acc_58[27]);
                    float2 _f2_97 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                    float2 r_74 = fma_f32x2_rn_noftz(_f2_97, _f2_69, _f2_96);
                    acc_58[26] = r_74.x;
                    acc_58[27] = r_74.y;
                    float2 _f2_98 = make_float2(acc_58[28], acc_58[29]);
                    float2 _f2_99 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                    float2 r_75 = fma_f32x2_rn_noftz(_f2_99, _f2_69, _f2_98);
                    acc_58[28] = r_75.x;
                    acc_58[29] = r_75.y;
                    float2 _f2_100 = make_float2(acc_58[30], acc_58[31]);
                    float2 _f2_101 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                    float2 r_76 = fma_f32x2_rn_noftz(_f2_101, _f2_69, _f2_100);
                    acc_58[30] = r_76.x;
                    acc_58[31] = r_76.y;
                    float2 _fp8x2_decode_3;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_3.x), "=f"(_fp8x2_decode_3.y) : "h"((uint16_t)own_sc[1]));
                    float _mul_6;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_fp8x2_decode_3.x), "f"(g_0));
                    float s0_77 = _mul_6;
                    float _mul_7;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_fp8x2_decode_3.y), "f"(g_0));
                    float s1_78 = _mul_7;
                    float2 _f2_102 = make_float2(s0_77, s0_77);
                    float2 _f2_103 = make_float2(s1_78, s1_78);
                    float _unpack_fp4_12[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[4 + 0]));
                        (&_unpack_fp4_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_12[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_12[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_12[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_12[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_12[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_12[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_12[0]), "=f"(_unpack_fp4_12[1]), "=f"(_unpack_fp4_12[2]), "=f"(_unpack_fp4_12[3]), "=f"(_unpack_fp4_12[4]), "=f"(_unpack_fp4_12[5]), "=f"(_unpack_fp4_12[6]), "=f"(_unpack_fp4_12[7]) : "r"(own[4 + 0]));
                        #endif
                    }
                    float2 _f2_104 = make_float2(acc_58[0], acc_58[1]);
                    float2 _f2_105 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                    float2 r_79 = fma_f32x2_rn_noftz(_f2_105, _f2_102, _f2_104);
                    acc_58[0] = r_79.x;
                    acc_58[1] = r_79.y;
                    float2 _f2_106 = make_float2(acc_58[2], acc_58[3]);
                    float2 _f2_107 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                    float2 r_80 = fma_f32x2_rn_noftz(_f2_107, _f2_102, _f2_106);
                    acc_58[2] = r_80.x;
                    acc_58[3] = r_80.y;
                    float2 _f2_108 = make_float2(acc_58[4], acc_58[5]);
                    float2 _f2_109 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                    float2 r_81 = fma_f32x2_rn_noftz(_f2_109, _f2_102, _f2_108);
                    acc_58[4] = r_81.x;
                    acc_58[5] = r_81.y;
                    float2 _f2_110 = make_float2(acc_58[6], acc_58[7]);
                    float2 _f2_111 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                    float2 r_82 = fma_f32x2_rn_noftz(_f2_111, _f2_102, _f2_110);
                    acc_58[6] = r_82.x;
                    acc_58[7] = r_82.y;
                    float _unpack_fp4_13[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[5 + 0]));
                        (&_unpack_fp4_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_13[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_13[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_13[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_13[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_13[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_13[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_13[0]), "=f"(_unpack_fp4_13[1]), "=f"(_unpack_fp4_13[2]), "=f"(_unpack_fp4_13[3]), "=f"(_unpack_fp4_13[4]), "=f"(_unpack_fp4_13[5]), "=f"(_unpack_fp4_13[6]), "=f"(_unpack_fp4_13[7]) : "r"(own[5 + 0]));
                        #endif
                    }
                    float2 _f2_112 = make_float2(acc_58[8], acc_58[9]);
                    float2 _f2_113 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                    float2 r_83 = fma_f32x2_rn_noftz(_f2_113, _f2_102, _f2_112);
                    acc_58[8] = r_83.x;
                    acc_58[9] = r_83.y;
                    float2 _f2_114 = make_float2(acc_58[10], acc_58[11]);
                    float2 _f2_115 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                    float2 r_84 = fma_f32x2_rn_noftz(_f2_115, _f2_102, _f2_114);
                    acc_58[10] = r_84.x;
                    acc_58[11] = r_84.y;
                    float2 _f2_116 = make_float2(acc_58[12], acc_58[13]);
                    float2 _f2_117 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                    float2 r_85 = fma_f32x2_rn_noftz(_f2_117, _f2_102, _f2_116);
                    acc_58[12] = r_85.x;
                    acc_58[13] = r_85.y;
                    float2 _f2_118 = make_float2(acc_58[14], acc_58[15]);
                    float2 _f2_119 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                    float2 r_86 = fma_f32x2_rn_noftz(_f2_119, _f2_102, _f2_118);
                    acc_58[14] = r_86.x;
                    acc_58[15] = r_86.y;
                    float _unpack_fp4_14[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[6 + 0]));
                        (&_unpack_fp4_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_14[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_14[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_14[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_14[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_14[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_14[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_14[0]), "=f"(_unpack_fp4_14[1]), "=f"(_unpack_fp4_14[2]), "=f"(_unpack_fp4_14[3]), "=f"(_unpack_fp4_14[4]), "=f"(_unpack_fp4_14[5]), "=f"(_unpack_fp4_14[6]), "=f"(_unpack_fp4_14[7]) : "r"(own[6 + 0]));
                        #endif
                    }
                    float2 _f2_120 = make_float2(acc_58[16], acc_58[17]);
                    float2 _f2_121 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                    float2 r_87 = fma_f32x2_rn_noftz(_f2_121, _f2_103, _f2_120);
                    acc_58[16] = r_87.x;
                    acc_58[17] = r_87.y;
                    float2 _f2_122 = make_float2(acc_58[18], acc_58[19]);
                    float2 _f2_123 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                    float2 r_88 = fma_f32x2_rn_noftz(_f2_123, _f2_103, _f2_122);
                    acc_58[18] = r_88.x;
                    acc_58[19] = r_88.y;
                    float2 _f2_124 = make_float2(acc_58[20], acc_58[21]);
                    float2 _f2_125 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                    float2 r_89 = fma_f32x2_rn_noftz(_f2_125, _f2_103, _f2_124);
                    acc_58[20] = r_89.x;
                    acc_58[21] = r_89.y;
                    float2 _f2_126 = make_float2(acc_58[22], acc_58[23]);
                    float2 _f2_127 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                    float2 r_90 = fma_f32x2_rn_noftz(_f2_127, _f2_103, _f2_126);
                    acc_58[22] = r_90.x;
                    acc_58[23] = r_90.y;
                    float _unpack_fp4_15[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[7 + 0]));
                        (&_unpack_fp4_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_15[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_15[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_15[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_15[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_15[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_15[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_15[0]), "=f"(_unpack_fp4_15[1]), "=f"(_unpack_fp4_15[2]), "=f"(_unpack_fp4_15[3]), "=f"(_unpack_fp4_15[4]), "=f"(_unpack_fp4_15[5]), "=f"(_unpack_fp4_15[6]), "=f"(_unpack_fp4_15[7]) : "r"(own[7 + 0]));
                        #endif
                    }
                    float2 _f2_128 = make_float2(acc_58[24], acc_58[25]);
                    float2 _f2_129 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                    float2 r_91 = fma_f32x2_rn_noftz(_f2_129, _f2_103, _f2_128);
                    acc_58[24] = r_91.x;
                    acc_58[25] = r_91.y;
                    float2 _f2_130 = make_float2(acc_58[26], acc_58[27]);
                    float2 _f2_131 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                    float2 r_92 = fma_f32x2_rn_noftz(_f2_131, _f2_103, _f2_130);
                    acc_58[26] = r_92.x;
                    acc_58[27] = r_92.y;
                    float2 _f2_132 = make_float2(acc_58[28], acc_58[29]);
                    float2 _f2_133 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                    float2 r_93 = fma_f32x2_rn_noftz(_f2_133, _f2_103, _f2_132);
                    acc_58[28] = r_93.x;
                    acc_58[29] = r_93.y;
                    float2 _f2_134 = make_float2(acc_58[30], acc_58[31]);
                    float2 _f2_135 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                    float2 r_94 = fma_f32x2_rn_noftz(_f2_135, _f2_103, _f2_134);
                    acc_58[30] = r_94.x;
                    acc_58[31] = r_94.y;
                    uint32_t acc_58_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_58[_lp*2 + 0], acc_58[_lp*2+1 + 0]));
                        acc_58_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_2 + (f ^ 0)))), "r"(acc_58_bf16[0]), "r"(acc_58_bf16[1]), "r"(acc_58_bf16[2]), "r"(acc_58_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_2 + (f ^ 1)))), "r"(acc_58_bf16[4]), "r"(acc_58_bf16[5]), "r"(acc_58_bf16[6]), "r"(acc_58_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_2 + (f ^ 2)))), "r"(acc_58_bf16[8]), "r"(acc_58_bf16[9]), "r"(acc_58_bf16[10]), "r"(acc_58_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_2 + (f ^ 3)))), "r"(acc_58_bf16[12]), "r"(acc_58_bf16[13]), "r"(acc_58_bf16[14]), "r"(acc_58_bf16[15]) : "memory");
                    __syncwarp();
                    int c_95 = lane_2;
                    int l_96 = c_95 >> 2;
                    unsigned int w4_97[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_97[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_97[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_97[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_97[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_96 + (c_95 & 3 ^ l_96 >> 1 & 3)))));
                    long long idx_98 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_95));
                    {
                        int4 _iv4 = make_int4(w4_97[0 + 0], w4_97[0 + 1], w4_97[0 + 2], w4_97[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_98)) = _iv4;
                    }
                    int c_99 = lane_2 + 32;
                    int l_100 = c_99 >> 2;
                    unsigned int w4_101[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_101[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_101[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_101[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_101[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_100 + (c_99 & 3 ^ l_100 >> 1 & 3)))));
                    long long idx_102 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_99));
                    {
                        int4 _iv4 = make_int4(w4_101[0 + 0], w4_101[0 + 1], w4_101[0 + 2], w4_101[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_102)) = _iv4;
                    }
                    int c_103 = lane_2 + 64;
                    int l_104 = c_103 >> 2;
                    unsigned int w4_105[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_105[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_105[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_105[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_105[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_104 + (c_103 & 3 ^ l_104 >> 1 & 3)))));
                    long long idx_106 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_103));
                    {
                        int4 _iv4 = make_int4(w4_105[0 + 0], w4_105[0 + 1], w4_105[0 + 2], w4_105[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_106)) = _iv4;
                    }
                    int c_107 = lane_2 + 96;
                    int l_108 = c_107 >> 2;
                    unsigned int w4_109[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_109[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_109[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_109[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_109[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_108 + (c_107 & 3 ^ l_108 >> 1 & 3)))));
                    long long idx_110 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_107));
                    {
                        int4 _iv4 = make_int4(w4_109[0 + 0], w4_109[0 + 1], w4_109[0 + 2], w4_109[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_110)) = _iv4;
                    }
                    __syncwarp();
                    float acc_111[32];
                    acc_111[0] = 0.0f;
                    acc_111[1] = 0.0f;
                    acc_111[2] = 0.0f;
                    acc_111[3] = 0.0f;
                    acc_111[4] = 0.0f;
                    acc_111[5] = 0.0f;
                    acc_111[6] = 0.0f;
                    acc_111[7] = 0.0f;
                    acc_111[8] = 0.0f;
                    acc_111[9] = 0.0f;
                    acc_111[10] = 0.0f;
                    acc_111[11] = 0.0f;
                    acc_111[12] = 0.0f;
                    acc_111[13] = 0.0f;
                    acc_111[14] = 0.0f;
                    acc_111[15] = 0.0f;
                    acc_111[16] = 0.0f;
                    acc_111[17] = 0.0f;
                    acc_111[18] = 0.0f;
                    acc_111[19] = 0.0f;
                    acc_111[20] = 0.0f;
                    acc_111[21] = 0.0f;
                    acc_111[22] = 0.0f;
                    acc_111[23] = 0.0f;
                    acc_111[24] = 0.0f;
                    acc_111[25] = 0.0f;
                    acc_111[26] = 0.0f;
                    acc_111[27] = 0.0f;
                    acc_111[28] = 0.0f;
                    acc_111[29] = 0.0f;
                    acc_111[30] = 0.0f;
                    acc_111[31] = 0.0f;
                    float2 _fp8x2_decode_4;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_4.x), "=f"(_fp8x2_decode_4.y) : "h"((uint16_t)pf_s[2]));
                    float _mul_8;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_fp8x2_decode_4.x), "f"(g));
                    float s0_112 = _mul_8;
                    float _mul_9;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_fp8x2_decode_4.y), "f"(g));
                    float s1_113 = _mul_9;
                    float2 _f2_136 = make_float2(s0_112, s0_112);
                    float2 _f2_137 = make_float2(s1_113, s1_113);
                    float _unpack_fp4_16[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[8 + 0]));
                        (&_unpack_fp4_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_16[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_16[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_16[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_16[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_16[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_16[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_16[0]), "=f"(_unpack_fp4_16[1]), "=f"(_unpack_fp4_16[2]), "=f"(_unpack_fp4_16[3]), "=f"(_unpack_fp4_16[4]), "=f"(_unpack_fp4_16[5]), "=f"(_unpack_fp4_16[6]), "=f"(_unpack_fp4_16[7]) : "r"(pf_v[8 + 0]));
                        #endif
                    }
                    float2 _f2_138 = make_float2(acc_111[0], acc_111[1]);
                    float2 _f2_139 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                    float2 r_114 = fma_f32x2_rn_noftz(_f2_139, _f2_136, _f2_138);
                    acc_111[0] = r_114.x;
                    acc_111[1] = r_114.y;
                    float2 _f2_140 = make_float2(acc_111[2], acc_111[3]);
                    float2 _f2_141 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                    float2 r_115 = fma_f32x2_rn_noftz(_f2_141, _f2_136, _f2_140);
                    acc_111[2] = r_115.x;
                    acc_111[3] = r_115.y;
                    float2 _f2_142 = make_float2(acc_111[4], acc_111[5]);
                    float2 _f2_143 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                    float2 r_116 = fma_f32x2_rn_noftz(_f2_143, _f2_136, _f2_142);
                    acc_111[4] = r_116.x;
                    acc_111[5] = r_116.y;
                    float2 _f2_144 = make_float2(acc_111[6], acc_111[7]);
                    float2 _f2_145 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                    float2 r_117 = fma_f32x2_rn_noftz(_f2_145, _f2_136, _f2_144);
                    acc_111[6] = r_117.x;
                    acc_111[7] = r_117.y;
                    float _unpack_fp4_17[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[9 + 0]));
                        (&_unpack_fp4_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_17[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_17[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_17[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_17[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_17[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_17[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_17[0]), "=f"(_unpack_fp4_17[1]), "=f"(_unpack_fp4_17[2]), "=f"(_unpack_fp4_17[3]), "=f"(_unpack_fp4_17[4]), "=f"(_unpack_fp4_17[5]), "=f"(_unpack_fp4_17[6]), "=f"(_unpack_fp4_17[7]) : "r"(pf_v[9 + 0]));
                        #endif
                    }
                    float2 _f2_146 = make_float2(acc_111[8], acc_111[9]);
                    float2 _f2_147 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                    float2 r_118 = fma_f32x2_rn_noftz(_f2_147, _f2_136, _f2_146);
                    acc_111[8] = r_118.x;
                    acc_111[9] = r_118.y;
                    float2 _f2_148 = make_float2(acc_111[10], acc_111[11]);
                    float2 _f2_149 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                    float2 r_119 = fma_f32x2_rn_noftz(_f2_149, _f2_136, _f2_148);
                    acc_111[10] = r_119.x;
                    acc_111[11] = r_119.y;
                    float2 _f2_150 = make_float2(acc_111[12], acc_111[13]);
                    float2 _f2_151 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                    float2 r_120 = fma_f32x2_rn_noftz(_f2_151, _f2_136, _f2_150);
                    acc_111[12] = r_120.x;
                    acc_111[13] = r_120.y;
                    float2 _f2_152 = make_float2(acc_111[14], acc_111[15]);
                    float2 _f2_153 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                    float2 r_121 = fma_f32x2_rn_noftz(_f2_153, _f2_136, _f2_152);
                    acc_111[14] = r_121.x;
                    acc_111[15] = r_121.y;
                    float _unpack_fp4_18[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[10 + 0]));
                        (&_unpack_fp4_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_18[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_18[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_18[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_18[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_18[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_18[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_18[0]), "=f"(_unpack_fp4_18[1]), "=f"(_unpack_fp4_18[2]), "=f"(_unpack_fp4_18[3]), "=f"(_unpack_fp4_18[4]), "=f"(_unpack_fp4_18[5]), "=f"(_unpack_fp4_18[6]), "=f"(_unpack_fp4_18[7]) : "r"(pf_v[10 + 0]));
                        #endif
                    }
                    float2 _f2_154 = make_float2(acc_111[16], acc_111[17]);
                    float2 _f2_155 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                    float2 r_122 = fma_f32x2_rn_noftz(_f2_155, _f2_137, _f2_154);
                    acc_111[16] = r_122.x;
                    acc_111[17] = r_122.y;
                    float2 _f2_156 = make_float2(acc_111[18], acc_111[19]);
                    float2 _f2_157 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                    float2 r_123 = fma_f32x2_rn_noftz(_f2_157, _f2_137, _f2_156);
                    acc_111[18] = r_123.x;
                    acc_111[19] = r_123.y;
                    float2 _f2_158 = make_float2(acc_111[20], acc_111[21]);
                    float2 _f2_159 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                    float2 r_124 = fma_f32x2_rn_noftz(_f2_159, _f2_137, _f2_158);
                    acc_111[20] = r_124.x;
                    acc_111[21] = r_124.y;
                    float2 _f2_160 = make_float2(acc_111[22], acc_111[23]);
                    float2 _f2_161 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                    float2 r_125 = fma_f32x2_rn_noftz(_f2_161, _f2_137, _f2_160);
                    acc_111[22] = r_125.x;
                    acc_111[23] = r_125.y;
                    float _unpack_fp4_19[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[11 + 0]));
                        (&_unpack_fp4_19[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_19[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_19[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_19[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_19[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_19[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_19[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_19[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_19[0]), "=f"(_unpack_fp4_19[1]), "=f"(_unpack_fp4_19[2]), "=f"(_unpack_fp4_19[3]), "=f"(_unpack_fp4_19[4]), "=f"(_unpack_fp4_19[5]), "=f"(_unpack_fp4_19[6]), "=f"(_unpack_fp4_19[7]) : "r"(pf_v[11 + 0]));
                        #endif
                    }
                    float2 _f2_162 = make_float2(acc_111[24], acc_111[25]);
                    float2 _f2_163 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                    float2 r_126 = fma_f32x2_rn_noftz(_f2_163, _f2_137, _f2_162);
                    acc_111[24] = r_126.x;
                    acc_111[25] = r_126.y;
                    float2 _f2_164 = make_float2(acc_111[26], acc_111[27]);
                    float2 _f2_165 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                    float2 r_127 = fma_f32x2_rn_noftz(_f2_165, _f2_137, _f2_164);
                    acc_111[26] = r_127.x;
                    acc_111[27] = r_127.y;
                    float2 _f2_166 = make_float2(acc_111[28], acc_111[29]);
                    float2 _f2_167 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                    float2 r_128 = fma_f32x2_rn_noftz(_f2_167, _f2_137, _f2_166);
                    acc_111[28] = r_128.x;
                    acc_111[29] = r_128.y;
                    float2 _f2_168 = make_float2(acc_111[30], acc_111[31]);
                    float2 _f2_169 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                    float2 r_129 = fma_f32x2_rn_noftz(_f2_169, _f2_137, _f2_168);
                    acc_111[30] = r_129.x;
                    acc_111[31] = r_129.y;
                    float2 _fp8x2_decode_5;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_5.x), "=f"(_fp8x2_decode_5.y) : "h"((uint16_t)own_sc[2]));
                    float _mul_10;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(_fp8x2_decode_5.x), "f"(g_0));
                    float s0_130 = _mul_10;
                    float _mul_11;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_fp8x2_decode_5.y), "f"(g_0));
                    float s1_131 = _mul_11;
                    float2 _f2_170 = make_float2(s0_130, s0_130);
                    float2 _f2_171 = make_float2(s1_131, s1_131);
                    float _unpack_fp4_20[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[8 + 0]));
                        (&_unpack_fp4_20[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_20[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_20[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_20[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_20[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_20[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_20[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_20[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_20[0]), "=f"(_unpack_fp4_20[1]), "=f"(_unpack_fp4_20[2]), "=f"(_unpack_fp4_20[3]), "=f"(_unpack_fp4_20[4]), "=f"(_unpack_fp4_20[5]), "=f"(_unpack_fp4_20[6]), "=f"(_unpack_fp4_20[7]) : "r"(own[8 + 0]));
                        #endif
                    }
                    float2 _f2_172 = make_float2(acc_111[0], acc_111[1]);
                    float2 _f2_173 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                    float2 r_132 = fma_f32x2_rn_noftz(_f2_173, _f2_170, _f2_172);
                    acc_111[0] = r_132.x;
                    acc_111[1] = r_132.y;
                    float2 _f2_174 = make_float2(acc_111[2], acc_111[3]);
                    float2 _f2_175 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                    float2 r_133 = fma_f32x2_rn_noftz(_f2_175, _f2_170, _f2_174);
                    acc_111[2] = r_133.x;
                    acc_111[3] = r_133.y;
                    float2 _f2_176 = make_float2(acc_111[4], acc_111[5]);
                    float2 _f2_177 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                    float2 r_134 = fma_f32x2_rn_noftz(_f2_177, _f2_170, _f2_176);
                    acc_111[4] = r_134.x;
                    acc_111[5] = r_134.y;
                    float2 _f2_178 = make_float2(acc_111[6], acc_111[7]);
                    float2 _f2_179 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                    float2 r_135 = fma_f32x2_rn_noftz(_f2_179, _f2_170, _f2_178);
                    acc_111[6] = r_135.x;
                    acc_111[7] = r_135.y;
                    float _unpack_fp4_21[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[9 + 0]));
                        (&_unpack_fp4_21[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_21[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_21[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_21[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_21[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_21[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_21[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_21[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_21[0]), "=f"(_unpack_fp4_21[1]), "=f"(_unpack_fp4_21[2]), "=f"(_unpack_fp4_21[3]), "=f"(_unpack_fp4_21[4]), "=f"(_unpack_fp4_21[5]), "=f"(_unpack_fp4_21[6]), "=f"(_unpack_fp4_21[7]) : "r"(own[9 + 0]));
                        #endif
                    }
                    float2 _f2_180 = make_float2(acc_111[8], acc_111[9]);
                    float2 _f2_181 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                    float2 r_136 = fma_f32x2_rn_noftz(_f2_181, _f2_170, _f2_180);
                    acc_111[8] = r_136.x;
                    acc_111[9] = r_136.y;
                    float2 _f2_182 = make_float2(acc_111[10], acc_111[11]);
                    float2 _f2_183 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                    float2 r_137 = fma_f32x2_rn_noftz(_f2_183, _f2_170, _f2_182);
                    acc_111[10] = r_137.x;
                    acc_111[11] = r_137.y;
                    float2 _f2_184 = make_float2(acc_111[12], acc_111[13]);
                    float2 _f2_185 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                    float2 r_138 = fma_f32x2_rn_noftz(_f2_185, _f2_170, _f2_184);
                    acc_111[12] = r_138.x;
                    acc_111[13] = r_138.y;
                    float2 _f2_186 = make_float2(acc_111[14], acc_111[15]);
                    float2 _f2_187 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                    float2 r_139 = fma_f32x2_rn_noftz(_f2_187, _f2_170, _f2_186);
                    acc_111[14] = r_139.x;
                    acc_111[15] = r_139.y;
                    float _unpack_fp4_22[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[10 + 0]));
                        (&_unpack_fp4_22[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_22[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_22[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_22[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_22[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_22[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_22[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_22[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_22[0]), "=f"(_unpack_fp4_22[1]), "=f"(_unpack_fp4_22[2]), "=f"(_unpack_fp4_22[3]), "=f"(_unpack_fp4_22[4]), "=f"(_unpack_fp4_22[5]), "=f"(_unpack_fp4_22[6]), "=f"(_unpack_fp4_22[7]) : "r"(own[10 + 0]));
                        #endif
                    }
                    float2 _f2_188 = make_float2(acc_111[16], acc_111[17]);
                    float2 _f2_189 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                    float2 r_140 = fma_f32x2_rn_noftz(_f2_189, _f2_171, _f2_188);
                    acc_111[16] = r_140.x;
                    acc_111[17] = r_140.y;
                    float2 _f2_190 = make_float2(acc_111[18], acc_111[19]);
                    float2 _f2_191 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                    float2 r_141 = fma_f32x2_rn_noftz(_f2_191, _f2_171, _f2_190);
                    acc_111[18] = r_141.x;
                    acc_111[19] = r_141.y;
                    float2 _f2_192 = make_float2(acc_111[20], acc_111[21]);
                    float2 _f2_193 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                    float2 r_142 = fma_f32x2_rn_noftz(_f2_193, _f2_171, _f2_192);
                    acc_111[20] = r_142.x;
                    acc_111[21] = r_142.y;
                    float2 _f2_194 = make_float2(acc_111[22], acc_111[23]);
                    float2 _f2_195 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                    float2 r_143 = fma_f32x2_rn_noftz(_f2_195, _f2_171, _f2_194);
                    acc_111[22] = r_143.x;
                    acc_111[23] = r_143.y;
                    float _unpack_fp4_23[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[11 + 0]));
                        (&_unpack_fp4_23[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_23[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_23[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_23[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_23[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_23[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_23[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_23[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_23[0]), "=f"(_unpack_fp4_23[1]), "=f"(_unpack_fp4_23[2]), "=f"(_unpack_fp4_23[3]), "=f"(_unpack_fp4_23[4]), "=f"(_unpack_fp4_23[5]), "=f"(_unpack_fp4_23[6]), "=f"(_unpack_fp4_23[7]) : "r"(own[11 + 0]));
                        #endif
                    }
                    float2 _f2_196 = make_float2(acc_111[24], acc_111[25]);
                    float2 _f2_197 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                    float2 r_144 = fma_f32x2_rn_noftz(_f2_197, _f2_171, _f2_196);
                    acc_111[24] = r_144.x;
                    acc_111[25] = r_144.y;
                    float2 _f2_198 = make_float2(acc_111[26], acc_111[27]);
                    float2 _f2_199 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                    float2 r_145 = fma_f32x2_rn_noftz(_f2_199, _f2_171, _f2_198);
                    acc_111[26] = r_145.x;
                    acc_111[27] = r_145.y;
                    float2 _f2_200 = make_float2(acc_111[28], acc_111[29]);
                    float2 _f2_201 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                    float2 r_146 = fma_f32x2_rn_noftz(_f2_201, _f2_171, _f2_200);
                    acc_111[28] = r_146.x;
                    acc_111[29] = r_146.y;
                    float2 _f2_202 = make_float2(acc_111[30], acc_111[31]);
                    float2 _f2_203 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                    float2 r_147 = fma_f32x2_rn_noftz(_f2_203, _f2_171, _f2_202);
                    acc_111[30] = r_147.x;
                    acc_111[31] = r_147.y;
                    uint32_t acc_111_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_111[_lp*2 + 0], acc_111[_lp*2+1 + 0]));
                        acc_111_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_2 + (f ^ 0)))), "r"(acc_111_bf16[0]), "r"(acc_111_bf16[1]), "r"(acc_111_bf16[2]), "r"(acc_111_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_2 + (f ^ 1)))), "r"(acc_111_bf16[4]), "r"(acc_111_bf16[5]), "r"(acc_111_bf16[6]), "r"(acc_111_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_2 + (f ^ 2)))), "r"(acc_111_bf16[8]), "r"(acc_111_bf16[9]), "r"(acc_111_bf16[10]), "r"(acc_111_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_2 + (f ^ 3)))), "r"(acc_111_bf16[12]), "r"(acc_111_bf16[13]), "r"(acc_111_bf16[14]), "r"(acc_111_bf16[15]) : "memory");
                    __syncwarp();
                    int c_148 = lane_2;
                    int l_149 = c_148 >> 2;
                    unsigned int w4_150[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_150[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_150[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_150[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_150[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_149 + (c_148 & 3 ^ l_149 >> 1 & 3)))));
                    long long idx_151 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_148));
                    {
                        int4 _iv4 = make_int4(w4_150[0 + 0], w4_150[0 + 1], w4_150[0 + 2], w4_150[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_151)) = _iv4;
                    }
                    int c_152 = lane_2 + 32;
                    int l_153 = c_152 >> 2;
                    unsigned int w4_154[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_154[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_154[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_154[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_154[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_153 + (c_152 & 3 ^ l_153 >> 1 & 3)))));
                    long long idx_155 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_152));
                    {
                        int4 _iv4 = make_int4(w4_154[0 + 0], w4_154[0 + 1], w4_154[0 + 2], w4_154[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_155)) = _iv4;
                    }
                    int c_156 = lane_2 + 64;
                    int l_157 = c_156 >> 2;
                    unsigned int w4_158[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_158[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_158[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_158[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_158[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_157 + (c_156 & 3 ^ l_157 >> 1 & 3)))));
                    long long idx_159 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_156));
                    {
                        int4 _iv4 = make_int4(w4_158[0 + 0], w4_158[0 + 1], w4_158[0 + 2], w4_158[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_159)) = _iv4;
                    }
                    int c_160 = lane_2 + 96;
                    int l_161 = c_160 >> 2;
                    unsigned int w4_162[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_162[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_162[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_162[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_162[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_161 + (c_160 & 3 ^ l_161 >> 1 & 3)))));
                    long long idx_163 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_160));
                    {
                        int4 _iv4 = make_int4(w4_162[0 + 0], w4_162[0 + 1], w4_162[0 + 2], w4_162[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_163)) = _iv4;
                    }
                    __syncwarp();
                }
                if (n > k) {
                    long long prow = (long long)(r0 + t + 5 * k);
                    long long rec_1 = q_base + prow * 3456;
                    uint32_t _sysv_ld_0[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_0[0]), "=r"(_sysv_ld_0[1]), "=r"(_sysv_ld_0[2]), "=r"(_sysv_ld_0[3]) : "l"(reinterpret_cast<unsigned int*>(qself_u8 + (rec_1 + (long long)(16 * (lane_2 + 32 * (3 * h)))))) : "memory");
                    pf_v[0] = _sysv_ld_0[0];
                    pf_v[1] = _sysv_ld_0[1];
                    pf_v[2] = _sysv_ld_0[2];
                    pf_v[3] = _sysv_ld_0[3];
                    unsigned int _vec_load_0[1];
                    {
                        uint16_t _scalar_bits_2;
                        asm volatile("ld.volatile.global.u16 %0, [%1];"
                            : "=h"(_scalar_bits_2) : "l"((const void*)(reinterpret_cast<uint16_t*>(qself_u8 + (rec_1 + 3072 + (long long)(2 * (lane_2 + 32 * (3 * h))))) + (0))) : "memory");
                        _vec_load_0[0] = (uint16_t)_scalar_bits_2;
                    }
                    pf_s[0] = _vec_load_0[0];
                    uint32_t _sysv_ld_1[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_1[0]), "=r"(_sysv_ld_1[1]), "=r"(_sysv_ld_1[2]), "=r"(_sysv_ld_1[3]) : "l"(reinterpret_cast<unsigned int*>(qself_u8 + (rec_1 + (long long)(16 * (lane_2 + 32 * (3 * h + 1)))))) : "memory");
                    pf_v[4] = _sysv_ld_1[0];
                    pf_v[5] = _sysv_ld_1[1];
                    pf_v[6] = _sysv_ld_1[2];
                    pf_v[7] = _sysv_ld_1[3];
                    unsigned int _vec_load_1[1];
                    {
                        uint16_t _scalar_bits_3;
                        asm volatile("ld.volatile.global.u16 %0, [%1];"
                            : "=h"(_scalar_bits_3) : "l"((const void*)(reinterpret_cast<uint16_t*>(qself_u8 + (rec_1 + 3072 + (long long)(2 * (lane_2 + 32 * (3 * h + 1))))) + (0))) : "memory");
                        _vec_load_1[0] = (uint16_t)_scalar_bits_3;
                    }
                    pf_s[1] = _vec_load_1[0];
                    uint32_t _sysv_ld_2[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_2[0]), "=r"(_sysv_ld_2[1]), "=r"(_sysv_ld_2[2]), "=r"(_sysv_ld_2[3]) : "l"(reinterpret_cast<unsigned int*>(qself_u8 + (rec_1 + (long long)(16 * (lane_2 + 32 * (3 * h + 2)))))) : "memory");
                    pf_v[8] = _sysv_ld_2[0];
                    pf_v[9] = _sysv_ld_2[1];
                    pf_v[10] = _sysv_ld_2[2];
                    pf_v[11] = _sysv_ld_2[3];
                    unsigned int _vec_load_2[1];
                    {
                        uint16_t _scalar_bits_4;
                        asm volatile("ld.volatile.global.u16 %0, [%1];"
                            : "=h"(_scalar_bits_4) : "l"((const void*)(reinterpret_cast<uint16_t*>(qself_u8 + (rec_1 + 3072 + (long long)(2 * (lane_2 + 32 * (3 * h + 2))))) + (0))) : "memory");
                        _vec_load_2[0] = (uint16_t)_scalar_bits_4;
                    }
                    pf_s[2] = _vec_load_2[0];
                }
            }
            if (lane_2 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            if (tid == 0) {
                int base_1 = bid * 4;
                unsigned int rows_u = (unsigned int)rows;
                unsigned int algo_u = 3;
                ctrl[base_1 + 2] = rows_u;
                ctrl[base_1 + 3] = algo_u;
                ctrl[base_1] = call + 1;
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
            for (int i = lane_0; i < 1024; i += 32) {
                pat[i] = sv;
                pat[1024 + i] = ss;
                pat[2048 + i] = sr;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            asm volatile("barrier.sync 0, 352;" ::: "memory");
            unsigned int call_1 = ctl[0];
            unsigned int prev_rows = ctl[1];
            unsigned int prev_algo = ctl[2];
            unsigned int pbuf = (call_1 + 2) % 3;
            if (prev_algo != 0) {
                uint8_t* self_u8 = reinterpret_cast<uint8_t*>(workspace[1]);
                int* qself_i32 = reinterpret_cast<int*>(workspace[3]);
                long long bid64_1 = (long long)bid;
                long long g64_1 = (long long)num_bids;
                long long pr64 = (long long)prev_rows;
                long long pb64 = (long long)pbuf;
                long long lane64 = (long long)lane_0;
                if (prev_algo == 3) {
                    unsigned int sv_0 = 143165576;
                    sv_0 = sv_0 | sv_0 << 4;
                    unsigned int ss_1 = 268435455;
                    ss_1 = ss_1 | ss_1 << 4;
                    unsigned int sv4[4];
                    unsigned int ss4[4];
                    sv4[0] = sv_0;
                    ss4[0] = ss_1;
                    sv4[1] = sv_0;
                    ss4[1] = ss_1;
                    sv4[2] = sv_0;
                    ss4[2] = ss_1;
                    sv4[3] = sv_0;
                    ss4[3] = ss_1;
                    long long base_2 = pb64 * 113246208;
                    long long u = pr64 * 3456 / 16;
                    long long lo = u * bid64_1 / g64_1;
                    long long hi = u * (bid64_1 + 1) / g64_1;
                    long long g0 = lo + lane64;
                    int gi = (int)(g0 % 216);
                    #pragma unroll 1
                    for (long long g_1 = g0; g_1 < hi; g_1 += 32) {
                        unsigned int w4_1[4];
                        w4_1[0] = ((gi < 192) ? sv4[0] : ss4[0]);
                        w4_1[1] = ((gi < 192) ? sv4[1] : ss4[1]);
                        w4_1[2] = ((gi < 192) ? sv4[2] : ss4[2]);
                        w4_1[3] = ((gi < 192) ? sv4[3] : ss4[3]);
                        long long widx = base_2 + 16 * g_1 >> 2;
                        {
                            int4 _iv4 = make_int4(w4_1[0 + 0], w4_1[0 + 1], w4_1[0 + 2], w4_1[0 + 3]);
                            *reinterpret_cast<int4*>(qself_i32 + (widx)) = _iv4;
                        }
                        int gn = gi + 32;
                        gi = ((gn >= 216) ? gn - 216 : gn);
                    }
                } else if (prev_algo == 1) {
                    int jbase = 0;
                    long long v_base = pb64 * 113770496;
                    long long sc_base = pb64 * 113770496 + 100663296;
                    long long m_base = pb64 * 113770496 + 113246208;
                    long long u_1 = pr64 * 3072 / 16;
                    long long lo_1 = u_1 * bid64_1 / g64_1 * 16;
                    long long hi_1 = u_1 * (bid64_1 + 1) / g64_1 * 16;
                    int npages = (int)((hi_1 - lo_1 + 4095) / 4096);
                    long long pg = 4096;
                    #pragma unroll 1
                    for (int i_1 = 0; i_1 < npages; i_1++) {
                        if ((jbase + i_1 & 31) == lane_0) {
                            long long o = lo_1 + (long long)i_1 * 4096;
                            long long rem = hi_1 - o;
                            unsigned int nb = (unsigned int)(((rem >= pg) ? pg : rem));
                            {
                                void* _cpbulk_dst_0 = reinterpret_cast<void*>(self_u8 + (v_base + o));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_0), "r"(pat_addr), "r"((uint32_t)(nb))
                                    : "memory");
                            }
                        }
                    }
                    jbase = jbase + npages;
                    long long u_0 = pr64 * 384 / 16;
                    long long lo_1_1 = u_0 * bid64_1 / g64_1 * 16;
                    long long hi_2 = u_0 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_3 = (int)((hi_2 - lo_1_1 + 4095) / 4096);
                    long long pg_4 = 4096;
                    #pragma unroll 1
                    for (int i_2 = 0; i_2 < npages_3; i_2++) {
                        if ((jbase + i_2 & 31) == lane_0) {
                            long long o_1 = lo_1_1 + (long long)i_2 * 4096;
                            long long rem_1 = hi_2 - o_1;
                            unsigned int nb_1 = (unsigned int)(((rem_1 >= pg_4) ? pg_4 : rem_1));
                            {
                                void* _cpbulk_dst_1 = reinterpret_cast<void*>(self_u8 + (sc_base + o_1));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_1), "r"(pat_addr + 4096), "r"((uint32_t)(nb_1))
                                    : "memory");
                            }
                        }
                    }
                    jbase = jbase + npages_3;
                    long long u_5 = pr64 * 16 / 16;
                    long long lo_6 = u_5 * bid64_1 / g64_1 * 16;
                    long long hi_7 = u_5 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_8 = (int)((hi_7 - lo_6 + 4095) / 4096);
                    long long pg_9 = 4096;
                    #pragma unroll 1
                    for (int i_3 = 0; i_3 < npages_8; i_3++) {
                        if ((jbase + i_3 & 31) == lane_0) {
                            long long o_2 = lo_6 + (long long)i_3 * 4096;
                            long long rem_2 = hi_7 - o_2;
                            unsigned int nb_2 = (unsigned int)(((rem_2 >= pg_9) ? pg_9 : rem_2));
                            {
                                void* _cpbulk_dst_2 = reinterpret_cast<void*>(self_u8 + (m_base + o_2));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_2), "r"(pat_addr + 8192), "r"((uint32_t)(nb_2))
                                    : "memory");
                            }
                        }
                    }
                    jbase = jbase + npages_8;
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
