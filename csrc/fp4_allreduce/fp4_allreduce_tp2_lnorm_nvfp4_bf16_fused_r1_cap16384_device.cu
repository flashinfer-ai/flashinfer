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
#define SMEM_WROW_OFF 1024
#define SMEM_WROW_STAGE_BYTES 12288
#define SMEM_WROW_STRIDE 12288
#define SMEM_CTL_OFF 13312
#define SMEM_CTL_STAGE_BYTES 16
#define SMEM_CTL_STRIDE 16
#define SMEM_WSM_OFF 13376
#define SMEM_WSM_STAGE_BYTES 185200
#define SMEM_WSM_STRIDE 185200
#define SMEM_WSM16_OFF 13376
#define SMEM_WSM16_STAGE_BYTES 185200
#define SMEM_WSM16_STRIDE 185200
#define SMEM_WSM32_OFF 13376
#define SMEM_WSM32_STAGE_BYTES 185200
#define SMEM_WSM32_STRIDE 185200
#define SMEM_TOTAL 198656
#define THREADS 192
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


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_fp4_allreduce_tp2_lnorm_nvfp4_bf16_fused_r1_cap16384(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ residual_in, __nv_bfloat16* __restrict__ weight, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ residual_out, unsigned int* __restrict__ ctrl, int rows, const unsigned long long* __restrict__ workspace)
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
    #define rb_full_addr (mbar_base + 40)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* wrow = reinterpret_cast<uint8_t*>(smem_raw + SMEM_WROW_OFF);
    const int wrow_addr = smem + SMEM_WROW_OFF;
    unsigned int* ctl = reinterpret_cast<unsigned int*>(smem_raw + SMEM_CTL_OFF);
    const int ctl_addr = smem + SMEM_CTL_OFF;
    uint8_t* wsm = reinterpret_cast<uint8_t*>(smem_raw + SMEM_WSM_OFF);
    const int wsm_addr = smem + SMEM_WSM_OFF;
    uint16_t* wsm16 = reinterpret_cast<uint16_t*>(smem_raw + SMEM_WSM16_OFF);
    const int wsm16_addr = smem + SMEM_WSM16_OFF;
    float* wsm32 = reinterpret_cast<float*>(smem_raw + SMEM_WSM32_OFF);
    const int wsm32_addr = smem + SMEM_WSM32_OFF;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // xb_full: 5 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // rb_full: 5 barriers, init_count=1
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
    if (warp <= 4) {
        { // workers_main
            uint8_t* w_u8 = reinterpret_cast<uint8_t*>(weight);
            int tid_0 = (int)tid;
            #pragma unroll 1
            for (int i = tid_0; i < 768; i += 160) {
                unsigned int _vec_load_0[4];
                {
                    uint32_t _uv4_0_0;
                    uint32_t _uv4_0_1;
                    uint32_t _uv4_0_2;
                    uint32_t _uv4_0_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(w_u8 + (16 * i)) + (0))) : "memory");
                    _vec_load_0[0 + 0] = _uv4_0_0;
                    _vec_load_0[0 + 1] = _uv4_0_1;
                    _vec_load_0[0 + 2] = _uv4_0_2;
                    _vec_load_0[0 + 3] = _uv4_0_3;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow_addr + (unsigned int)(16 * i)), "r"(_vec_load_0[0]), "r"(_vec_load_0[1]), "r"(_vec_load_0[2]), "r"(_vec_load_0[3]) : "memory");
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            float g = gs[0];
            float g_1 = gs[1];
            long long bid64 = (long long)bid;
            long long g64 = (long long)num_bids;
            long long rows64 = (long long)rows;
            int r0 = (int)(bid64 * rows64 / g64);
            int r1 = (int)((bid64 + 1) * rows64 / g64);
            int nrows = r1 - r0;
            int warp_2 = (int)warp;
            int lane_3 = (int)lane;
            int n = 0;
            if (nrows > warp_2) {
                n = (nrows - warp_2 + 5 - 1) / 5;
            }
            int wbase0 = warp_2 * 37040;
            if (lane_3 == 0 && n > 0) {
                long long first_row = (long long)(r0 + warp_2);
                mbarrier_arrive_expect_tx(rb_full_addr + (warp_2) * 8, 12288);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(wbase0 + 12464), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(residual_in) + ((unsigned long long)(first_row * 6144) * (unsigned long long)2)), 12288, rb_full_addr + (warp_2) * 8);
                mbarrier_arrive_expect_tx(xb_full_addr + (warp_2) * 8, 3456);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)wbase0, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(xq) + ((unsigned long long)(first_row * 3456) * (unsigned long long)1)), 3456, xb_full_addr + (warp_2) * 8);
            }
            if (tid == 0) {
                ctl[0] = ctrl[0];
                ctl[1] = ctrl[2];
                ctl[2] = ctrl[3];
            }
            asm volatile("barrier.sync 0, 192;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int call = ctl[0];
            unsigned int buf = call % 3;
            int* resout_i32 = reinterpret_cast<int*>(residual_out);
            int* out_i32 = reinterpret_cast<int*>(out);
            int f = lane_3 >> 1 & 3;
            int wbase = warp_2 * 37040;
            int qr_off = wbase;
            int stg_off = wbase + 10416;
            int rb_off = wbase + 12464;
            int zb_off = wbase + 24752;
            uint8_t* self_u8 = reinterpret_cast<uint8_t*>(workspace[1]);
            uint8_t* peer_base = reinterpret_cast<uint8_t*>(workspace[0]);
            long long os_base = (long long)buf * 113770496;
            float gs_me = g_1;
            #pragma unroll 1
            for (int k = 0; k < n + 1; k++) {
                if (n > k) {
                    long long row64 = (long long)(r0 + warp_2 + 5 * k);
                    int slot = k % 3;
                    int ob_off = qr_off + slot * 3472;
                    if (lane_3 == 0) {
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(residual_in) + ((unsigned long long)(row64 * 6144) * (unsigned long long)2))), "r"((uint32_t)(12288)) : "memory");
                    }
                    float rs = 1.0f;
                    {
                        mbarrier_wait(xb_full_addr + (warp_2) * 8, k & 1);
                        rs = gs_me;
                        if (n > k + 1) {
                            if (lane_3 == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                                long long next_row = (long long)(r0 + warp_2 + 5 * (k + 1));
                                int next_slot = (k + 1) % 3;
                                mbarrier_arrive_expect_tx(xb_full_addr + (warp_2) * 8, 3456);
                                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(qr_off + next_slot * 3472), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(xq) + ((unsigned long long)(next_row * 3456) * (unsigned long long)1)), 3456, xb_full_addr + (warp_2) * 8);
                            }
                        }
                    }
                    if (lane_3 < 4) {
                        wsm32[(ob_off + 3072 + 384 >> 2) + lane_3] = rs;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane_3 == 0) {
                        {
                            long long v_off = os_base + (16384 + row64) * 3072;
                            long long sc_off = os_base + 100663296 + (16384 + row64) * 384;
                            long long m_off = os_base + 113246208 + (16384 + row64) * 16;
                            {
                                void* _cpbulk_dst_1 = reinterpret_cast<void*>(peer_base + v_off);
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_1), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_2 = reinterpret_cast<void*>(peer_base + sc_off);
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_2), "r"(wsm_addr + (unsigned int)ob_off + 3072), "r"((uint32_t)(384))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_3 = reinterpret_cast<void*>(peer_base + m_off);
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_3), "r"(wsm_addr + (unsigned int)ob_off + 3072 + 384), "r"((uint32_t)(16))
                                    : "memory");
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                int kd = k - 1;
                if (kd >= 0 && kd < n) {
                    long long drow = (long long)(r0 + warp_2 + 5 * kd);
                    int dslot = kd % 3;
                    int dob_off = qr_off + dslot * 3472;
                    long long v_base = os_base + drow * 3072;
                    long long sc_base = os_base + 100663296 + drow * 384;
                    long long m_base = os_base + 113246208 + drow * 16;
                    uint32_t _sysv_poll32_group_0[1];
                    do {
                        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[0]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + m_base)) : "memory");
                    } while (
                        (_sysv_poll32_group_0[0] == 0x80000000u)
                    );
                    float ssa[2];
                    ssa[0] = 0.0f;
                    ssa[1] = 0.0f;
                    #pragma unroll 1
                    for (int g_2 = 0; g_2 < 3; g_2++) {
                        int j0 = 2 * g_2;
                        unsigned int own[8];
                        unsigned int own_sc[2];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&own[0])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * (lane_3 + 32 * j0))));
                        own_sc[0] = (unsigned int)wsm16[(dob_off + 3072 >> 1) + (lane_3 + 32 * j0)];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&own[4])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * (lane_3 + 32 * (j0 + 1)))));
                        own_sc[1] = (unsigned int)wsm16[(dob_off + 3072 >> 1) + (lane_3 + 32 * (j0 + 1))];
                        uint32_t _fp4_wire_values_0[8];
                        uint32_t _fp4_wire_scales_0[2];
                        uint16_t _wire_h16_5_0;
                        uint16_t _wire_h16_5_1;
                        uint32_t _wire_vpend_5 = 3u;
                        uint32_t _wire_spend_5 = 3u;
                        do {
                            if (_wire_vpend_5 & 1u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_3 + 32 * j0))))) : "memory");
                            }
                            if (_wire_vpend_5 & 2u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_3 + 32 * (j0 + 1)))))) : "memory");
                            }
                            if (_wire_spend_5 & 1u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_5_0) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_3 + 32 * j0))))) : "memory");
                                _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_5_0;
                            }
                            if (_wire_spend_5 & 2u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_5_1) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_3 + 32 * (j0 + 1)))))) : "memory");
                                _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_5_1;
                            }
                            if (_wire_vpend_5 & 1u) {
                                if (!((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_5 &= ~1u;
                            }
                            if (_wire_vpend_5 & 2u) {
                                if (!((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_5 &= ~2u;
                            }
                            if (_wire_spend_5 & 1u) {
                                if (!(((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[0] >> 8) == 0xFFu))) _wire_spend_5 &= ~1u;
                            }
                            if (_wire_spend_5 & 2u) {
                                if (!(((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[1] >> 8) == 0xFFu))) _wire_spend_5 &= ~2u;
                            }
                        } while ((_wire_vpend_5 | _wire_spend_5) != 0u);
                        {
                            if (g_2 == 0) {
                                mbarrier_wait(rb_full_addr + (warp_2) * 8, kd & 1);
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
                                : "=f"(_fp8x2_decode_0.x), "=f"(_fp8x2_decode_0.y) : "h"((uint16_t)_fp4_wire_scales_0[0]));
                            float _mul_0;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(_fp8x2_decode_0.x), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                            float s0 = _mul_0;
                            float _mul_1;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(_fp8x2_decode_0.y), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
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
                            float2 r_0 = fma_f32x2_rn_noftz(_f2_5, _f2_0, _f2_4);
                            acc[2] = r_0.x;
                            acc[3] = r_0.y;
                            float2 _f2_6 = make_float2(acc[4], acc[5]);
                            float2 _f2_7 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                            float2 r_1 = fma_f32x2_rn_noftz(_f2_7, _f2_0, _f2_6);
                            acc[4] = r_1.x;
                            acc[5] = r_1.y;
                            float2 _f2_8 = make_float2(acc[6], acc[7]);
                            float2 _f2_9 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                            float2 r_2 = fma_f32x2_rn_noftz(_f2_9, _f2_0, _f2_8);
                            acc[6] = r_2.x;
                            acc[7] = r_2.y;
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
                            float2 r_3 = fma_f32x2_rn_noftz(_f2_11, _f2_0, _f2_10);
                            acc[8] = r_3.x;
                            acc[9] = r_3.y;
                            float2 _f2_12 = make_float2(acc[10], acc[11]);
                            float2 _f2_13 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                            float2 r_4 = fma_f32x2_rn_noftz(_f2_13, _f2_0, _f2_12);
                            acc[10] = r_4.x;
                            acc[11] = r_4.y;
                            float2 _f2_14 = make_float2(acc[12], acc[13]);
                            float2 _f2_15 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                            float2 r_5 = fma_f32x2_rn_noftz(_f2_15, _f2_0, _f2_14);
                            acc[12] = r_5.x;
                            acc[13] = r_5.y;
                            float2 _f2_16 = make_float2(acc[14], acc[15]);
                            float2 _f2_17 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                            float2 r_6 = fma_f32x2_rn_noftz(_f2_17, _f2_0, _f2_16);
                            acc[14] = r_6.x;
                            acc[15] = r_6.y;
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
                            float2 r_7 = fma_f32x2_rn_noftz(_f2_19, _f2_1, _f2_18);
                            acc[16] = r_7.x;
                            acc[17] = r_7.y;
                            float2 _f2_20 = make_float2(acc[18], acc[19]);
                            float2 _f2_21 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                            float2 r_8 = fma_f32x2_rn_noftz(_f2_21, _f2_1, _f2_20);
                            acc[18] = r_8.x;
                            acc[19] = r_8.y;
                            float2 _f2_22 = make_float2(acc[20], acc[21]);
                            float2 _f2_23 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                            float2 r_9 = fma_f32x2_rn_noftz(_f2_23, _f2_1, _f2_22);
                            acc[20] = r_9.x;
                            acc[21] = r_9.y;
                            float2 _f2_24 = make_float2(acc[22], acc[23]);
                            float2 _f2_25 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                            float2 r_10 = fma_f32x2_rn_noftz(_f2_25, _f2_1, _f2_24);
                            acc[22] = r_10.x;
                            acc[23] = r_10.y;
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
                            float2 r_11 = fma_f32x2_rn_noftz(_f2_27, _f2_1, _f2_26);
                            acc[24] = r_11.x;
                            acc[25] = r_11.y;
                            float2 _f2_28 = make_float2(acc[26], acc[27]);
                            float2 _f2_29 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                            float2 r_12 = fma_f32x2_rn_noftz(_f2_29, _f2_1, _f2_28);
                            acc[26] = r_12.x;
                            acc[27] = r_12.y;
                            float2 _f2_30 = make_float2(acc[28], acc[29]);
                            float2 _f2_31 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                            float2 r_13 = fma_f32x2_rn_noftz(_f2_31, _f2_1, _f2_30);
                            acc[28] = r_13.x;
                            acc[29] = r_13.y;
                            float2 _f2_32 = make_float2(acc[30], acc[31]);
                            float2 _f2_33 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                            float2 r_14 = fma_f32x2_rn_noftz(_f2_33, _f2_1, _f2_32);
                            acc[30] = r_14.x;
                            acc[31] = r_14.y;
                            float2 _fp8x2_decode_1;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(_fp8x2_decode_1.x), "=f"(_fp8x2_decode_1.y) : "h"((uint16_t)own_sc[0]));
                            float _mul_2;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_fp8x2_decode_1.x), "f"(gs_me));
                            float s0_15 = _mul_2;
                            float _mul_3;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_fp8x2_decode_1.y), "f"(gs_me));
                            float s1_16 = _mul_3;
                            float2 _f2_34 = make_float2(s0_15, s0_15);
                            float2 _f2_35 = make_float2(s1_16, s1_16);
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
                            float2 r_17 = fma_f32x2_rn_noftz(_f2_37, _f2_34, _f2_36);
                            acc[0] = r_17.x;
                            acc[1] = r_17.y;
                            float2 _f2_38 = make_float2(acc[2], acc[3]);
                            float2 _f2_39 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                            float2 r_18 = fma_f32x2_rn_noftz(_f2_39, _f2_34, _f2_38);
                            acc[2] = r_18.x;
                            acc[3] = r_18.y;
                            float2 _f2_40 = make_float2(acc[4], acc[5]);
                            float2 _f2_41 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                            float2 r_19 = fma_f32x2_rn_noftz(_f2_41, _f2_34, _f2_40);
                            acc[4] = r_19.x;
                            acc[5] = r_19.y;
                            float2 _f2_42 = make_float2(acc[6], acc[7]);
                            float2 _f2_43 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                            float2 r_20 = fma_f32x2_rn_noftz(_f2_43, _f2_34, _f2_42);
                            acc[6] = r_20.x;
                            acc[7] = r_20.y;
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
                            float2 r_21 = fma_f32x2_rn_noftz(_f2_45, _f2_34, _f2_44);
                            acc[8] = r_21.x;
                            acc[9] = r_21.y;
                            float2 _f2_46 = make_float2(acc[10], acc[11]);
                            float2 _f2_47 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                            float2 r_22 = fma_f32x2_rn_noftz(_f2_47, _f2_34, _f2_46);
                            acc[10] = r_22.x;
                            acc[11] = r_22.y;
                            float2 _f2_48 = make_float2(acc[12], acc[13]);
                            float2 _f2_49 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                            float2 r_23 = fma_f32x2_rn_noftz(_f2_49, _f2_34, _f2_48);
                            acc[12] = r_23.x;
                            acc[13] = r_23.y;
                            float2 _f2_50 = make_float2(acc[14], acc[15]);
                            float2 _f2_51 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                            float2 r_24 = fma_f32x2_rn_noftz(_f2_51, _f2_34, _f2_50);
                            acc[14] = r_24.x;
                            acc[15] = r_24.y;
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
                            float2 r_25 = fma_f32x2_rn_noftz(_f2_53, _f2_35, _f2_52);
                            acc[16] = r_25.x;
                            acc[17] = r_25.y;
                            float2 _f2_54 = make_float2(acc[18], acc[19]);
                            float2 _f2_55 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                            float2 r_26 = fma_f32x2_rn_noftz(_f2_55, _f2_35, _f2_54);
                            acc[18] = r_26.x;
                            acc[19] = r_26.y;
                            float2 _f2_56 = make_float2(acc[20], acc[21]);
                            float2 _f2_57 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                            float2 r_27 = fma_f32x2_rn_noftz(_f2_57, _f2_35, _f2_56);
                            acc[20] = r_27.x;
                            acc[21] = r_27.y;
                            float2 _f2_58 = make_float2(acc[22], acc[23]);
                            float2 _f2_59 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                            float2 r_28 = fma_f32x2_rn_noftz(_f2_59, _f2_35, _f2_58);
                            acc[22] = r_28.x;
                            acc[23] = r_28.y;
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
                            float2 r_29 = fma_f32x2_rn_noftz(_f2_61, _f2_35, _f2_60);
                            acc[24] = r_29.x;
                            acc[25] = r_29.y;
                            float2 _f2_62 = make_float2(acc[26], acc[27]);
                            float2 _f2_63 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                            float2 r_30 = fma_f32x2_rn_noftz(_f2_63, _f2_35, _f2_62);
                            acc[26] = r_30.x;
                            acc[27] = r_30.y;
                            float2 _f2_64 = make_float2(acc[28], acc[29]);
                            float2 _f2_65 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                            float2 r_31 = fma_f32x2_rn_noftz(_f2_65, _f2_35, _f2_64);
                            acc[28] = r_31.x;
                            acc[29] = r_31.y;
                            float2 _f2_66 = make_float2(acc[30], acc[31]);
                            float2 _f2_67 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                            float2 r_32 = fma_f32x2_rn_noftz(_f2_67, _f2_35, _f2_66);
                            acc[30] = r_32.x;
                            acc[31] = r_32.y;
                            int j = j0;
                            unsigned int t[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&t[0])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 32 * j0)) + (unsigned int)(16 * (f ^ 0))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&t[4])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 32 * j0)) + (unsigned int)(16 * (f ^ 1))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&t[8])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 3]))
                                : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 32 * j0)) + (unsigned int)(16 * (f ^ 2))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&t[12])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 3]))
                                : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 32 * j0)) + (unsigned int)(16 * (f ^ 3))));
                            unsigned int s[16];
                            unsigned int r_33[16];
                            s[0] = (((f & 1) != 0) ? t[4] : t[0]);
                            s[4] = (((f & 1) != 0) ? t[0] : t[4]);
                            s[8] = (((f & 1) != 0) ? t[12] : t[8]);
                            s[12] = (((f & 1) != 0) ? t[8] : t[12]);
                            s[1] = (((f & 1) != 0) ? t[5] : t[1]);
                            s[5] = (((f & 1) != 0) ? t[1] : t[5]);
                            s[9] = (((f & 1) != 0) ? t[13] : t[9]);
                            s[13] = (((f & 1) != 0) ? t[9] : t[13]);
                            s[2] = (((f & 1) != 0) ? t[6] : t[2]);
                            s[6] = (((f & 1) != 0) ? t[2] : t[6]);
                            s[10] = (((f & 1) != 0) ? t[14] : t[10]);
                            s[14] = (((f & 1) != 0) ? t[10] : t[14]);
                            s[3] = (((f & 1) != 0) ? t[7] : t[3]);
                            s[7] = (((f & 1) != 0) ? t[3] : t[7]);
                            s[11] = (((f & 1) != 0) ? t[15] : t[11]);
                            s[15] = (((f & 1) != 0) ? t[11] : t[15]);
                            r_33[0] = (((f & 2) != 0) ? s[8] : s[0]);
                            r_33[4] = (((f & 2) != 0) ? s[12] : s[4]);
                            r_33[8] = (((f & 2) != 0) ? s[0] : s[8]);
                            r_33[12] = (((f & 2) != 0) ? s[4] : s[12]);
                            r_33[1] = (((f & 2) != 0) ? s[9] : s[1]);
                            r_33[5] = (((f & 2) != 0) ? s[13] : s[5]);
                            r_33[9] = (((f & 2) != 0) ? s[1] : s[9]);
                            r_33[13] = (((f & 2) != 0) ? s[5] : s[13]);
                            r_33[2] = (((f & 2) != 0) ? s[10] : s[2]);
                            r_33[6] = (((f & 2) != 0) ? s[14] : s[6]);
                            r_33[10] = (((f & 2) != 0) ? s[2] : s[10]);
                            r_33[14] = (((f & 2) != 0) ? s[6] : s[14]);
                            r_33[3] = (((f & 2) != 0) ? s[11] : s[3]);
                            r_33[7] = (((f & 2) != 0) ? s[15] : s[7]);
                            r_33[11] = (((f & 2) != 0) ? s[3] : s[11]);
                            r_33[15] = (((f & 2) != 0) ? s[7] : s[15]);
                            float2 _f2_68 = make_float2(acc[0], acc[1]);
                            float2 _f2_69 = make_float2(__uint_as_float(r_33[0] << 16), __uint_as_float(r_33[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_0;
                            {
                                float2 _packed_f32x2_6_0 = _f2_68;
                                float2 _packed_f32x2_6_1 = _f2_69;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_0)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_6_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_6_1)));
                            }
                            acc[0] = _packed_add_f32x2_0.x;
                            acc[1] = _packed_add_f32x2_0.y;
                            float2 _f2_70 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_0, _packed_add_f32x2_0, _f2_70).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_0, _packed_add_f32x2_0, _f2_70).y;
                            float2 _f2_71 = make_float2(acc[2], acc[3]);
                            float2 _f2_72 = make_float2(__uint_as_float(r_33[1] << 16), __uint_as_float(r_33[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_1;
                            {
                                float2 _packed_f32x2_7_0 = _f2_71;
                                float2 _packed_f32x2_7_1 = _f2_72;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_1)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_7_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_7_1)));
                            }
                            acc[2] = _packed_add_f32x2_1.x;
                            acc[3] = _packed_add_f32x2_1.y;
                            float2 _f2_73 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_1, _packed_add_f32x2_1, _f2_73).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_1, _packed_add_f32x2_1, _f2_73).y;
                            float2 _f2_74 = make_float2(acc[4], acc[5]);
                            float2 _f2_75 = make_float2(__uint_as_float(r_33[2] << 16), __uint_as_float(r_33[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_2;
                            {
                                float2 _packed_f32x2_8_0 = _f2_74;
                                float2 _packed_f32x2_8_1 = _f2_75;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_2)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_8_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_8_1)));
                            }
                            acc[4] = _packed_add_f32x2_2.x;
                            acc[5] = _packed_add_f32x2_2.y;
                            float2 _f2_76 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_2, _packed_add_f32x2_2, _f2_76).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_2, _packed_add_f32x2_2, _f2_76).y;
                            float2 _f2_77 = make_float2(acc[6], acc[7]);
                            float2 _f2_78 = make_float2(__uint_as_float(r_33[3] << 16), __uint_as_float(r_33[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_3;
                            {
                                float2 _packed_f32x2_9_0 = _f2_77;
                                float2 _packed_f32x2_9_1 = _f2_78;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_3)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_9_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_9_1)));
                            }
                            acc[6] = _packed_add_f32x2_3.x;
                            acc[7] = _packed_add_f32x2_3.y;
                            float2 _f2_79 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_3, _packed_add_f32x2_3, _f2_79).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_3, _packed_add_f32x2_3, _f2_79).y;
                            float2 _f2_80 = make_float2(acc[8], acc[9]);
                            float2 _f2_81 = make_float2(__uint_as_float(r_33[4] << 16), __uint_as_float(r_33[4] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_4;
                            {
                                float2 _packed_f32x2_10_0 = _f2_80;
                                float2 _packed_f32x2_10_1 = _f2_81;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_4)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_10_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_10_1)));
                            }
                            acc[8] = _packed_add_f32x2_4.x;
                            acc[9] = _packed_add_f32x2_4.y;
                            float2 _f2_82 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_4, _packed_add_f32x2_4, _f2_82).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_4, _packed_add_f32x2_4, _f2_82).y;
                            float2 _f2_83 = make_float2(acc[10], acc[11]);
                            float2 _f2_84 = make_float2(__uint_as_float(r_33[5] << 16), __uint_as_float(r_33[5] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_5;
                            {
                                float2 _packed_f32x2_11_0 = _f2_83;
                                float2 _packed_f32x2_11_1 = _f2_84;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_5)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_11_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_11_1)));
                            }
                            acc[10] = _packed_add_f32x2_5.x;
                            acc[11] = _packed_add_f32x2_5.y;
                            float2 _f2_85 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_5, _packed_add_f32x2_5, _f2_85).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_5, _packed_add_f32x2_5, _f2_85).y;
                            float2 _f2_86 = make_float2(acc[12], acc[13]);
                            float2 _f2_87 = make_float2(__uint_as_float(r_33[6] << 16), __uint_as_float(r_33[6] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_6;
                            {
                                float2 _packed_f32x2_12_0 = _f2_86;
                                float2 _packed_f32x2_12_1 = _f2_87;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_6)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_12_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_12_1)));
                            }
                            acc[12] = _packed_add_f32x2_6.x;
                            acc[13] = _packed_add_f32x2_6.y;
                            float2 _f2_88 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_6, _packed_add_f32x2_6, _f2_88).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_6, _packed_add_f32x2_6, _f2_88).y;
                            float2 _f2_89 = make_float2(acc[14], acc[15]);
                            float2 _f2_90 = make_float2(__uint_as_float(r_33[7] << 16), __uint_as_float(r_33[7] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_7;
                            {
                                float2 _packed_f32x2_13_0 = _f2_89;
                                float2 _packed_f32x2_13_1 = _f2_90;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_7)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_13_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_13_1)));
                            }
                            acc[14] = _packed_add_f32x2_7.x;
                            acc[15] = _packed_add_f32x2_7.y;
                            float2 _f2_91 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_7, _packed_add_f32x2_7, _f2_91).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_7, _packed_add_f32x2_7, _f2_91).y;
                            float2 _f2_92 = make_float2(acc[16], acc[17]);
                            float2 _f2_93 = make_float2(__uint_as_float(r_33[8] << 16), __uint_as_float(r_33[8] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_8;
                            {
                                float2 _packed_f32x2_14_0 = _f2_92;
                                float2 _packed_f32x2_14_1 = _f2_93;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_8)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_14_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_14_1)));
                            }
                            acc[16] = _packed_add_f32x2_8.x;
                            acc[17] = _packed_add_f32x2_8.y;
                            float2 _f2_94 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_8, _packed_add_f32x2_8, _f2_94).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_8, _packed_add_f32x2_8, _f2_94).y;
                            float2 _f2_95 = make_float2(acc[18], acc[19]);
                            float2 _f2_96 = make_float2(__uint_as_float(r_33[9] << 16), __uint_as_float(r_33[9] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_9;
                            {
                                float2 _packed_f32x2_15_0 = _f2_95;
                                float2 _packed_f32x2_15_1 = _f2_96;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_9)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_15_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_15_1)));
                            }
                            acc[18] = _packed_add_f32x2_9.x;
                            acc[19] = _packed_add_f32x2_9.y;
                            float2 _f2_97 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_9, _packed_add_f32x2_9, _f2_97).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_9, _packed_add_f32x2_9, _f2_97).y;
                            float2 _f2_98 = make_float2(acc[20], acc[21]);
                            float2 _f2_99 = make_float2(__uint_as_float(r_33[10] << 16), __uint_as_float(r_33[10] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_10;
                            {
                                float2 _packed_f32x2_16_0 = _f2_98;
                                float2 _packed_f32x2_16_1 = _f2_99;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_10)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_16_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_16_1)));
                            }
                            acc[20] = _packed_add_f32x2_10.x;
                            acc[21] = _packed_add_f32x2_10.y;
                            float2 _f2_100 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_10, _packed_add_f32x2_10, _f2_100).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_10, _packed_add_f32x2_10, _f2_100).y;
                            float2 _f2_101 = make_float2(acc[22], acc[23]);
                            float2 _f2_102 = make_float2(__uint_as_float(r_33[11] << 16), __uint_as_float(r_33[11] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_11;
                            {
                                float2 _packed_f32x2_17_0 = _f2_101;
                                float2 _packed_f32x2_17_1 = _f2_102;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_11)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_17_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_17_1)));
                            }
                            acc[22] = _packed_add_f32x2_11.x;
                            acc[23] = _packed_add_f32x2_11.y;
                            float2 _f2_103 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_11, _packed_add_f32x2_11, _f2_103).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_11, _packed_add_f32x2_11, _f2_103).y;
                            float2 _f2_104 = make_float2(acc[24], acc[25]);
                            float2 _f2_105 = make_float2(__uint_as_float(r_33[12] << 16), __uint_as_float(r_33[12] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_12;
                            {
                                float2 _packed_f32x2_18_0 = _f2_104;
                                float2 _packed_f32x2_18_1 = _f2_105;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_12)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_18_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_18_1)));
                            }
                            acc[24] = _packed_add_f32x2_12.x;
                            acc[25] = _packed_add_f32x2_12.y;
                            float2 _f2_106 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_12, _packed_add_f32x2_12, _f2_106).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_12, _packed_add_f32x2_12, _f2_106).y;
                            float2 _f2_107 = make_float2(acc[26], acc[27]);
                            float2 _f2_108 = make_float2(__uint_as_float(r_33[13] << 16), __uint_as_float(r_33[13] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_13;
                            {
                                float2 _packed_f32x2_19_0 = _f2_107;
                                float2 _packed_f32x2_19_1 = _f2_108;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_13)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_19_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_19_1)));
                            }
                            acc[26] = _packed_add_f32x2_13.x;
                            acc[27] = _packed_add_f32x2_13.y;
                            float2 _f2_109 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_13, _packed_add_f32x2_13, _f2_109).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_13, _packed_add_f32x2_13, _f2_109).y;
                            float2 _f2_110 = make_float2(acc[28], acc[29]);
                            float2 _f2_111 = make_float2(__uint_as_float(r_33[14] << 16), __uint_as_float(r_33[14] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_14;
                            {
                                float2 _packed_f32x2_20_0 = _f2_110;
                                float2 _packed_f32x2_20_1 = _f2_111;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_14)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_20_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_20_1)));
                            }
                            acc[28] = _packed_add_f32x2_14.x;
                            acc[29] = _packed_add_f32x2_14.y;
                            float2 _f2_112 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_14, _packed_add_f32x2_14, _f2_112).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_14, _packed_add_f32x2_14, _f2_112).y;
                            float2 _f2_113 = make_float2(acc[30], acc[31]);
                            float2 _f2_114 = make_float2(__uint_as_float(r_33[15] << 16), __uint_as_float(r_33[15] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_15;
                            {
                                float2 _packed_f32x2_21_0 = _f2_113;
                                float2 _packed_f32x2_21_1 = _f2_114;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_15)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_21_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_21_1)));
                            }
                            acc[30] = _packed_add_f32x2_15.x;
                            acc[31] = _packed_add_f32x2_15.y;
                            float2 _f2_115 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_15, _packed_add_f32x2_15, _f2_115).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_15, _packed_add_f32x2_15, _f2_115).y;
                            uint32_t acc_bf16[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                                acc_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 0)))), "r"(acc_bf16[0]), "r"(acc_bf16[1]), "r"(acc_bf16[2]), "r"(acc_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 1)))), "r"(acc_bf16[4]), "r"(acc_bf16[5]), "r"(acc_bf16[6]), "r"(acc_bf16[7]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 2)))), "r"(acc_bf16[8]), "r"(acc_bf16[9]), "r"(acc_bf16[10]), "r"(acc_bf16[11]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 3)))), "r"(acc_bf16[12]), "r"(acc_bf16[13]), "r"(acc_bf16[14]), "r"(acc_bf16[15]) : "memory");
                            __syncwarp();
                            int c = lane_3;
                            int l = c >> 2;
                            unsigned int w4[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l + (c & 3 ^ l >> 1 & 3)))));
                            long long idx = drow * 3072 + (long long)(4 * (128 * j + c));
                            {
                                int4 _iv4 = make_int4(w4[0 + 0], w4[0 + 1], w4[0 + 2], w4[0 + 3]);
                                *reinterpret_cast<int4*>(resout_i32 + (idx)) = _iv4;
                            }
                            int c_34 = lane_3 + 32;
                            int l_35 = c_34 >> 2;
                            unsigned int w4_36[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_35 + (c_34 & 3 ^ l_35 >> 1 & 3)))));
                            long long idx_37 = drow * 3072 + (long long)(4 * (128 * j + c_34));
                            {
                                int4 _iv4 = make_int4(w4_36[0 + 0], w4_36[0 + 1], w4_36[0 + 2], w4_36[0 + 3]);
                                *reinterpret_cast<int4*>(resout_i32 + (idx_37)) = _iv4;
                            }
                            int c_38 = lane_3 + 64;
                            int l_39 = c_38 >> 2;
                            unsigned int w4_40[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_40[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_39 + (c_38 & 3 ^ l_39 >> 1 & 3)))));
                            long long idx_41 = drow * 3072 + (long long)(4 * (128 * j + c_38));
                            {
                                int4 _iv4 = make_int4(w4_40[0 + 0], w4_40[0 + 1], w4_40[0 + 2], w4_40[0 + 3]);
                                *reinterpret_cast<int4*>(resout_i32 + (idx_41)) = _iv4;
                            }
                            int c_42 = lane_3 + 96;
                            int l_43 = c_42 >> 2;
                            unsigned int w4_44[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_44[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_43 + (c_42 & 3 ^ l_43 >> 1 & 3)))));
                            long long idx_45 = drow * 3072 + (long long)(4 * (128 * j + c_42));
                            {
                                int4 _iv4 = make_int4(w4_44[0 + 0], w4_44[0 + 1], w4_44[0 + 2], w4_44[0 + 3]);
                                *reinterpret_cast<int4*>(resout_i32 + (idx_45)) = _iv4;
                            }
                            __syncwarp();
                            int zlo = zb_off + 4096 * j;
                            int zhi = rb_off + 4096 * (j - 3);
                            int zbase = ((j < 3) ? zlo : zhi);
                            int zlane = zbase + 128 * lane_3;
                            int l7 = lane_3 & 7;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 0))), "r"(__as_u32(acc[0])), "r"(__as_u32(acc[1])), "r"(__as_u32(acc[2])), "r"(__as_u32(acc[3])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 1))), "r"(__as_u32(acc[4])), "r"(__as_u32(acc[5])), "r"(__as_u32(acc[6])), "r"(__as_u32(acc[7])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 2))), "r"(__as_u32(acc[8])), "r"(__as_u32(acc[9])), "r"(__as_u32(acc[10])), "r"(__as_u32(acc[11])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 3))), "r"(__as_u32(acc[12])), "r"(__as_u32(acc[13])), "r"(__as_u32(acc[14])), "r"(__as_u32(acc[15])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 4))), "r"(__as_u32(acc[16])), "r"(__as_u32(acc[17])), "r"(__as_u32(acc[18])), "r"(__as_u32(acc[19])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 5))), "r"(__as_u32(acc[20])), "r"(__as_u32(acc[21])), "r"(__as_u32(acc[22])), "r"(__as_u32(acc[23])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 6))), "r"(__as_u32(acc[24])), "r"(__as_u32(acc[25])), "r"(__as_u32(acc[26])), "r"(__as_u32(acc[27])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 7))), "r"(__as_u32(acc[28])), "r"(__as_u32(acc[29])), "r"(__as_u32(acc[30])), "r"(__as_u32(acc[31])) : "memory");
                            float acc_46[32];
                            acc_46[0] = 0.0f;
                            acc_46[1] = 0.0f;
                            acc_46[2] = 0.0f;
                            acc_46[3] = 0.0f;
                            acc_46[4] = 0.0f;
                            acc_46[5] = 0.0f;
                            acc_46[6] = 0.0f;
                            acc_46[7] = 0.0f;
                            acc_46[8] = 0.0f;
                            acc_46[9] = 0.0f;
                            acc_46[10] = 0.0f;
                            acc_46[11] = 0.0f;
                            acc_46[12] = 0.0f;
                            acc_46[13] = 0.0f;
                            acc_46[14] = 0.0f;
                            acc_46[15] = 0.0f;
                            acc_46[16] = 0.0f;
                            acc_46[17] = 0.0f;
                            acc_46[18] = 0.0f;
                            acc_46[19] = 0.0f;
                            acc_46[20] = 0.0f;
                            acc_46[21] = 0.0f;
                            acc_46[22] = 0.0f;
                            acc_46[23] = 0.0f;
                            acc_46[24] = 0.0f;
                            acc_46[25] = 0.0f;
                            acc_46[26] = 0.0f;
                            acc_46[27] = 0.0f;
                            acc_46[28] = 0.0f;
                            acc_46[29] = 0.0f;
                            acc_46[30] = 0.0f;
                            acc_46[31] = 0.0f;
                            float2 _fp8x2_decode_2;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(_fp8x2_decode_2.x), "=f"(_fp8x2_decode_2.y) : "h"((uint16_t)_fp4_wire_scales_0[1]));
                            float _mul_4;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_fp8x2_decode_2.x), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                            float s0_47 = _mul_4;
                            float _mul_5;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_fp8x2_decode_2.y), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                            float s1_48 = _mul_5;
                            float2 _f2_116 = make_float2(s0_47, s0_47);
                            float2 _f2_117 = make_float2(s1_48, s1_48);
                            float _unpack_fp4_8[8];
                            {
                                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                                uint32_t _fp4_decode_pairs[4];
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[4 + 0]));
                                (&_unpack_fp4_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                                (&_unpack_fp4_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                                (&_unpack_fp4_8[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                                (&_unpack_fp4_8[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                                (&_unpack_fp4_8[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                                (&_unpack_fp4_8[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                                (&_unpack_fp4_8[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                                (&_unpack_fp4_8[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                                #else
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_8[0]), "=f"(_unpack_fp4_8[1]), "=f"(_unpack_fp4_8[2]), "=f"(_unpack_fp4_8[3]), "=f"(_unpack_fp4_8[4]), "=f"(_unpack_fp4_8[5]), "=f"(_unpack_fp4_8[6]), "=f"(_unpack_fp4_8[7]) : "r"(_fp4_wire_values_0[4 + 0]));
                                #endif
                            }
                            float2 _f2_118 = make_float2(acc_46[0], acc_46[1]);
                            float2 _f2_119 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                            float2 r_49 = fma_f32x2_rn_noftz(_f2_119, _f2_116, _f2_118);
                            acc_46[0] = r_49.x;
                            acc_46[1] = r_49.y;
                            float2 _f2_120 = make_float2(acc_46[2], acc_46[3]);
                            float2 _f2_121 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                            float2 r_50 = fma_f32x2_rn_noftz(_f2_121, _f2_116, _f2_120);
                            acc_46[2] = r_50.x;
                            acc_46[3] = r_50.y;
                            float2 _f2_122 = make_float2(acc_46[4], acc_46[5]);
                            float2 _f2_123 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                            float2 r_51 = fma_f32x2_rn_noftz(_f2_123, _f2_116, _f2_122);
                            acc_46[4] = r_51.x;
                            acc_46[5] = r_51.y;
                            float2 _f2_124 = make_float2(acc_46[6], acc_46[7]);
                            float2 _f2_125 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                            float2 r_52 = fma_f32x2_rn_noftz(_f2_125, _f2_116, _f2_124);
                            acc_46[6] = r_52.x;
                            acc_46[7] = r_52.y;
                            float _unpack_fp4_9[8];
                            {
                                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                                uint32_t _fp4_decode_pairs[4];
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[5 + 0]));
                                (&_unpack_fp4_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                                (&_unpack_fp4_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                                (&_unpack_fp4_9[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                                (&_unpack_fp4_9[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                                (&_unpack_fp4_9[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                                (&_unpack_fp4_9[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                                (&_unpack_fp4_9[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                                (&_unpack_fp4_9[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                                #else
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_9[0]), "=f"(_unpack_fp4_9[1]), "=f"(_unpack_fp4_9[2]), "=f"(_unpack_fp4_9[3]), "=f"(_unpack_fp4_9[4]), "=f"(_unpack_fp4_9[5]), "=f"(_unpack_fp4_9[6]), "=f"(_unpack_fp4_9[7]) : "r"(_fp4_wire_values_0[5 + 0]));
                                #endif
                            }
                            float2 _f2_126 = make_float2(acc_46[8], acc_46[9]);
                            float2 _f2_127 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                            float2 r_53 = fma_f32x2_rn_noftz(_f2_127, _f2_116, _f2_126);
                            acc_46[8] = r_53.x;
                            acc_46[9] = r_53.y;
                            float2 _f2_128 = make_float2(acc_46[10], acc_46[11]);
                            float2 _f2_129 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                            float2 r_54 = fma_f32x2_rn_noftz(_f2_129, _f2_116, _f2_128);
                            acc_46[10] = r_54.x;
                            acc_46[11] = r_54.y;
                            float2 _f2_130 = make_float2(acc_46[12], acc_46[13]);
                            float2 _f2_131 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                            float2 r_55 = fma_f32x2_rn_noftz(_f2_131, _f2_116, _f2_130);
                            acc_46[12] = r_55.x;
                            acc_46[13] = r_55.y;
                            float2 _f2_132 = make_float2(acc_46[14], acc_46[15]);
                            float2 _f2_133 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                            float2 r_56 = fma_f32x2_rn_noftz(_f2_133, _f2_116, _f2_132);
                            acc_46[14] = r_56.x;
                            acc_46[15] = r_56.y;
                            float _unpack_fp4_10[8];
                            {
                                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                                uint32_t _fp4_decode_pairs[4];
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[6 + 0]));
                                (&_unpack_fp4_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                                (&_unpack_fp4_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                                (&_unpack_fp4_10[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                                (&_unpack_fp4_10[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                                (&_unpack_fp4_10[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                                (&_unpack_fp4_10[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                                (&_unpack_fp4_10[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                                (&_unpack_fp4_10[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                                #else
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_10[0]), "=f"(_unpack_fp4_10[1]), "=f"(_unpack_fp4_10[2]), "=f"(_unpack_fp4_10[3]), "=f"(_unpack_fp4_10[4]), "=f"(_unpack_fp4_10[5]), "=f"(_unpack_fp4_10[6]), "=f"(_unpack_fp4_10[7]) : "r"(_fp4_wire_values_0[6 + 0]));
                                #endif
                            }
                            float2 _f2_134 = make_float2(acc_46[16], acc_46[17]);
                            float2 _f2_135 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                            float2 r_57 = fma_f32x2_rn_noftz(_f2_135, _f2_117, _f2_134);
                            acc_46[16] = r_57.x;
                            acc_46[17] = r_57.y;
                            float2 _f2_136 = make_float2(acc_46[18], acc_46[19]);
                            float2 _f2_137 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                            float2 r_58 = fma_f32x2_rn_noftz(_f2_137, _f2_117, _f2_136);
                            acc_46[18] = r_58.x;
                            acc_46[19] = r_58.y;
                            float2 _f2_138 = make_float2(acc_46[20], acc_46[21]);
                            float2 _f2_139 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                            float2 r_59 = fma_f32x2_rn_noftz(_f2_139, _f2_117, _f2_138);
                            acc_46[20] = r_59.x;
                            acc_46[21] = r_59.y;
                            float2 _f2_140 = make_float2(acc_46[22], acc_46[23]);
                            float2 _f2_141 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                            float2 r_60 = fma_f32x2_rn_noftz(_f2_141, _f2_117, _f2_140);
                            acc_46[22] = r_60.x;
                            acc_46[23] = r_60.y;
                            float _unpack_fp4_11[8];
                            {
                                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                                uint32_t _fp4_decode_pairs[4];
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[7 + 0]));
                                (&_unpack_fp4_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                                (&_unpack_fp4_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                                (&_unpack_fp4_11[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                                (&_unpack_fp4_11[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                                (&_unpack_fp4_11[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                                (&_unpack_fp4_11[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                                (&_unpack_fp4_11[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                                (&_unpack_fp4_11[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                                #else
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_11[0]), "=f"(_unpack_fp4_11[1]), "=f"(_unpack_fp4_11[2]), "=f"(_unpack_fp4_11[3]), "=f"(_unpack_fp4_11[4]), "=f"(_unpack_fp4_11[5]), "=f"(_unpack_fp4_11[6]), "=f"(_unpack_fp4_11[7]) : "r"(_fp4_wire_values_0[7 + 0]));
                                #endif
                            }
                            float2 _f2_142 = make_float2(acc_46[24], acc_46[25]);
                            float2 _f2_143 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                            float2 r_61 = fma_f32x2_rn_noftz(_f2_143, _f2_117, _f2_142);
                            acc_46[24] = r_61.x;
                            acc_46[25] = r_61.y;
                            float2 _f2_144 = make_float2(acc_46[26], acc_46[27]);
                            float2 _f2_145 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                            float2 r_62 = fma_f32x2_rn_noftz(_f2_145, _f2_117, _f2_144);
                            acc_46[26] = r_62.x;
                            acc_46[27] = r_62.y;
                            float2 _f2_146 = make_float2(acc_46[28], acc_46[29]);
                            float2 _f2_147 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                            float2 r_63 = fma_f32x2_rn_noftz(_f2_147, _f2_117, _f2_146);
                            acc_46[28] = r_63.x;
                            acc_46[29] = r_63.y;
                            float2 _f2_148 = make_float2(acc_46[30], acc_46[31]);
                            float2 _f2_149 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                            float2 r_64 = fma_f32x2_rn_noftz(_f2_149, _f2_117, _f2_148);
                            acc_46[30] = r_64.x;
                            acc_46[31] = r_64.y;
                            float2 _fp8x2_decode_3;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(_fp8x2_decode_3.x), "=f"(_fp8x2_decode_3.y) : "h"((uint16_t)own_sc[1]));
                            float _mul_6;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_fp8x2_decode_3.x), "f"(gs_me));
                            float s0_65 = _mul_6;
                            float _mul_7;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_fp8x2_decode_3.y), "f"(gs_me));
                            float s1_66 = _mul_7;
                            float2 _f2_150 = make_float2(s0_65, s0_65);
                            float2 _f2_151 = make_float2(s1_66, s1_66);
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
                            float2 _f2_152 = make_float2(acc_46[0], acc_46[1]);
                            float2 _f2_153 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                            float2 r_67 = fma_f32x2_rn_noftz(_f2_153, _f2_150, _f2_152);
                            acc_46[0] = r_67.x;
                            acc_46[1] = r_67.y;
                            float2 _f2_154 = make_float2(acc_46[2], acc_46[3]);
                            float2 _f2_155 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                            float2 r_68 = fma_f32x2_rn_noftz(_f2_155, _f2_150, _f2_154);
                            acc_46[2] = r_68.x;
                            acc_46[3] = r_68.y;
                            float2 _f2_156 = make_float2(acc_46[4], acc_46[5]);
                            float2 _f2_157 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                            float2 r_69 = fma_f32x2_rn_noftz(_f2_157, _f2_150, _f2_156);
                            acc_46[4] = r_69.x;
                            acc_46[5] = r_69.y;
                            float2 _f2_158 = make_float2(acc_46[6], acc_46[7]);
                            float2 _f2_159 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                            float2 r_70 = fma_f32x2_rn_noftz(_f2_159, _f2_150, _f2_158);
                            acc_46[6] = r_70.x;
                            acc_46[7] = r_70.y;
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
                            float2 _f2_160 = make_float2(acc_46[8], acc_46[9]);
                            float2 _f2_161 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                            float2 r_71 = fma_f32x2_rn_noftz(_f2_161, _f2_150, _f2_160);
                            acc_46[8] = r_71.x;
                            acc_46[9] = r_71.y;
                            float2 _f2_162 = make_float2(acc_46[10], acc_46[11]);
                            float2 _f2_163 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                            float2 r_72 = fma_f32x2_rn_noftz(_f2_163, _f2_150, _f2_162);
                            acc_46[10] = r_72.x;
                            acc_46[11] = r_72.y;
                            float2 _f2_164 = make_float2(acc_46[12], acc_46[13]);
                            float2 _f2_165 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                            float2 r_73 = fma_f32x2_rn_noftz(_f2_165, _f2_150, _f2_164);
                            acc_46[12] = r_73.x;
                            acc_46[13] = r_73.y;
                            float2 _f2_166 = make_float2(acc_46[14], acc_46[15]);
                            float2 _f2_167 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                            float2 r_74 = fma_f32x2_rn_noftz(_f2_167, _f2_150, _f2_166);
                            acc_46[14] = r_74.x;
                            acc_46[15] = r_74.y;
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
                            float2 _f2_168 = make_float2(acc_46[16], acc_46[17]);
                            float2 _f2_169 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                            float2 r_75 = fma_f32x2_rn_noftz(_f2_169, _f2_151, _f2_168);
                            acc_46[16] = r_75.x;
                            acc_46[17] = r_75.y;
                            float2 _f2_170 = make_float2(acc_46[18], acc_46[19]);
                            float2 _f2_171 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                            float2 r_76 = fma_f32x2_rn_noftz(_f2_171, _f2_151, _f2_170);
                            acc_46[18] = r_76.x;
                            acc_46[19] = r_76.y;
                            float2 _f2_172 = make_float2(acc_46[20], acc_46[21]);
                            float2 _f2_173 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                            float2 r_77 = fma_f32x2_rn_noftz(_f2_173, _f2_151, _f2_172);
                            acc_46[20] = r_77.x;
                            acc_46[21] = r_77.y;
                            float2 _f2_174 = make_float2(acc_46[22], acc_46[23]);
                            float2 _f2_175 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                            float2 r_78 = fma_f32x2_rn_noftz(_f2_175, _f2_151, _f2_174);
                            acc_46[22] = r_78.x;
                            acc_46[23] = r_78.y;
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
                            float2 _f2_176 = make_float2(acc_46[24], acc_46[25]);
                            float2 _f2_177 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                            float2 r_79 = fma_f32x2_rn_noftz(_f2_177, _f2_151, _f2_176);
                            acc_46[24] = r_79.x;
                            acc_46[25] = r_79.y;
                            float2 _f2_178 = make_float2(acc_46[26], acc_46[27]);
                            float2 _f2_179 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                            float2 r_80 = fma_f32x2_rn_noftz(_f2_179, _f2_151, _f2_178);
                            acc_46[26] = r_80.x;
                            acc_46[27] = r_80.y;
                            float2 _f2_180 = make_float2(acc_46[28], acc_46[29]);
                            float2 _f2_181 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                            float2 r_81 = fma_f32x2_rn_noftz(_f2_181, _f2_151, _f2_180);
                            acc_46[28] = r_81.x;
                            acc_46[29] = r_81.y;
                            float2 _f2_182 = make_float2(acc_46[30], acc_46[31]);
                            float2 _f2_183 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                            float2 r_82 = fma_f32x2_rn_noftz(_f2_183, _f2_151, _f2_182);
                            acc_46[30] = r_82.x;
                            acc_46[31] = r_82.y;
                            int j_83 = j0 + 1;
                            unsigned int t_84[16];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&t_84[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_84[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_84[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_84[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 32 * (j0 + 1))) + (unsigned int)(16 * (f ^ 0))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&t_84[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_84[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_84[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_84[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 32 * (j0 + 1))) + (unsigned int)(16 * (f ^ 1))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&t_84[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_84[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_84[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_84[(8) + 3]))
                                : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 32 * (j0 + 1))) + (unsigned int)(16 * (f ^ 2))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&t_84[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_84[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_84[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_84[(12) + 3]))
                                : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 32 * (j0 + 1))) + (unsigned int)(16 * (f ^ 3))));
                            unsigned int s_85[16];
                            unsigned int r_86[16];
                            s_85[0] = (((f & 1) != 0) ? t_84[4] : t_84[0]);
                            s_85[4] = (((f & 1) != 0) ? t_84[0] : t_84[4]);
                            s_85[8] = (((f & 1) != 0) ? t_84[12] : t_84[8]);
                            s_85[12] = (((f & 1) != 0) ? t_84[8] : t_84[12]);
                            s_85[1] = (((f & 1) != 0) ? t_84[5] : t_84[1]);
                            s_85[5] = (((f & 1) != 0) ? t_84[1] : t_84[5]);
                            s_85[9] = (((f & 1) != 0) ? t_84[13] : t_84[9]);
                            s_85[13] = (((f & 1) != 0) ? t_84[9] : t_84[13]);
                            s_85[2] = (((f & 1) != 0) ? t_84[6] : t_84[2]);
                            s_85[6] = (((f & 1) != 0) ? t_84[2] : t_84[6]);
                            s_85[10] = (((f & 1) != 0) ? t_84[14] : t_84[10]);
                            s_85[14] = (((f & 1) != 0) ? t_84[10] : t_84[14]);
                            s_85[3] = (((f & 1) != 0) ? t_84[7] : t_84[3]);
                            s_85[7] = (((f & 1) != 0) ? t_84[3] : t_84[7]);
                            s_85[11] = (((f & 1) != 0) ? t_84[15] : t_84[11]);
                            s_85[15] = (((f & 1) != 0) ? t_84[11] : t_84[15]);
                            r_86[0] = (((f & 2) != 0) ? s_85[8] : s_85[0]);
                            r_86[4] = (((f & 2) != 0) ? s_85[12] : s_85[4]);
                            r_86[8] = (((f & 2) != 0) ? s_85[0] : s_85[8]);
                            r_86[12] = (((f & 2) != 0) ? s_85[4] : s_85[12]);
                            r_86[1] = (((f & 2) != 0) ? s_85[9] : s_85[1]);
                            r_86[5] = (((f & 2) != 0) ? s_85[13] : s_85[5]);
                            r_86[9] = (((f & 2) != 0) ? s_85[1] : s_85[9]);
                            r_86[13] = (((f & 2) != 0) ? s_85[5] : s_85[13]);
                            r_86[2] = (((f & 2) != 0) ? s_85[10] : s_85[2]);
                            r_86[6] = (((f & 2) != 0) ? s_85[14] : s_85[6]);
                            r_86[10] = (((f & 2) != 0) ? s_85[2] : s_85[10]);
                            r_86[14] = (((f & 2) != 0) ? s_85[6] : s_85[14]);
                            r_86[3] = (((f & 2) != 0) ? s_85[11] : s_85[3]);
                            r_86[7] = (((f & 2) != 0) ? s_85[15] : s_85[7]);
                            r_86[11] = (((f & 2) != 0) ? s_85[3] : s_85[11]);
                            r_86[15] = (((f & 2) != 0) ? s_85[7] : s_85[15]);
                            float2 _f2_184 = make_float2(acc_46[0], acc_46[1]);
                            float2 _f2_185 = make_float2(__uint_as_float(r_86[0] << 16), __uint_as_float(r_86[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_16;
                            {
                                float2 _packed_f32x2_22_0 = _f2_184;
                                float2 _packed_f32x2_22_1 = _f2_185;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_16)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_22_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_22_1)));
                            }
                            acc_46[0] = _packed_add_f32x2_16.x;
                            acc_46[1] = _packed_add_f32x2_16.y;
                            float2 _f2_186 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_16, _packed_add_f32x2_16, _f2_186).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_16, _packed_add_f32x2_16, _f2_186).y;
                            float2 _f2_187 = make_float2(acc_46[2], acc_46[3]);
                            float2 _f2_188 = make_float2(__uint_as_float(r_86[1] << 16), __uint_as_float(r_86[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_17;
                            {
                                float2 _packed_f32x2_23_0 = _f2_187;
                                float2 _packed_f32x2_23_1 = _f2_188;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_17)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_23_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_23_1)));
                            }
                            acc_46[2] = _packed_add_f32x2_17.x;
                            acc_46[3] = _packed_add_f32x2_17.y;
                            float2 _f2_189 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_17, _packed_add_f32x2_17, _f2_189).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_17, _packed_add_f32x2_17, _f2_189).y;
                            float2 _f2_190 = make_float2(acc_46[4], acc_46[5]);
                            float2 _f2_191 = make_float2(__uint_as_float(r_86[2] << 16), __uint_as_float(r_86[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_18;
                            {
                                float2 _packed_f32x2_24_0 = _f2_190;
                                float2 _packed_f32x2_24_1 = _f2_191;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_18)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_24_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_24_1)));
                            }
                            acc_46[4] = _packed_add_f32x2_18.x;
                            acc_46[5] = _packed_add_f32x2_18.y;
                            float2 _f2_192 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_18, _packed_add_f32x2_18, _f2_192).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_18, _packed_add_f32x2_18, _f2_192).y;
                            float2 _f2_193 = make_float2(acc_46[6], acc_46[7]);
                            float2 _f2_194 = make_float2(__uint_as_float(r_86[3] << 16), __uint_as_float(r_86[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_19;
                            {
                                float2 _packed_f32x2_25_0 = _f2_193;
                                float2 _packed_f32x2_25_1 = _f2_194;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_19)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_25_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_25_1)));
                            }
                            acc_46[6] = _packed_add_f32x2_19.x;
                            acc_46[7] = _packed_add_f32x2_19.y;
                            float2 _f2_195 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_19, _packed_add_f32x2_19, _f2_195).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_19, _packed_add_f32x2_19, _f2_195).y;
                            float2 _f2_196 = make_float2(acc_46[8], acc_46[9]);
                            float2 _f2_197 = make_float2(__uint_as_float(r_86[4] << 16), __uint_as_float(r_86[4] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_20;
                            {
                                float2 _packed_f32x2_26_0 = _f2_196;
                                float2 _packed_f32x2_26_1 = _f2_197;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_20)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_26_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_26_1)));
                            }
                            acc_46[8] = _packed_add_f32x2_20.x;
                            acc_46[9] = _packed_add_f32x2_20.y;
                            float2 _f2_198 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_20, _packed_add_f32x2_20, _f2_198).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_20, _packed_add_f32x2_20, _f2_198).y;
                            float2 _f2_199 = make_float2(acc_46[10], acc_46[11]);
                            float2 _f2_200 = make_float2(__uint_as_float(r_86[5] << 16), __uint_as_float(r_86[5] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_21;
                            {
                                float2 _packed_f32x2_27_0 = _f2_199;
                                float2 _packed_f32x2_27_1 = _f2_200;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_21)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_27_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_27_1)));
                            }
                            acc_46[10] = _packed_add_f32x2_21.x;
                            acc_46[11] = _packed_add_f32x2_21.y;
                            float2 _f2_201 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_21, _packed_add_f32x2_21, _f2_201).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_21, _packed_add_f32x2_21, _f2_201).y;
                            float2 _f2_202 = make_float2(acc_46[12], acc_46[13]);
                            float2 _f2_203 = make_float2(__uint_as_float(r_86[6] << 16), __uint_as_float(r_86[6] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_22;
                            {
                                float2 _packed_f32x2_28_0 = _f2_202;
                                float2 _packed_f32x2_28_1 = _f2_203;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_22)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_28_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_28_1)));
                            }
                            acc_46[12] = _packed_add_f32x2_22.x;
                            acc_46[13] = _packed_add_f32x2_22.y;
                            float2 _f2_204 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_22, _packed_add_f32x2_22, _f2_204).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_22, _packed_add_f32x2_22, _f2_204).y;
                            float2 _f2_205 = make_float2(acc_46[14], acc_46[15]);
                            float2 _f2_206 = make_float2(__uint_as_float(r_86[7] << 16), __uint_as_float(r_86[7] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_23;
                            {
                                float2 _packed_f32x2_29_0 = _f2_205;
                                float2 _packed_f32x2_29_1 = _f2_206;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_23)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_29_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_29_1)));
                            }
                            acc_46[14] = _packed_add_f32x2_23.x;
                            acc_46[15] = _packed_add_f32x2_23.y;
                            float2 _f2_207 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_23, _packed_add_f32x2_23, _f2_207).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_23, _packed_add_f32x2_23, _f2_207).y;
                            float2 _f2_208 = make_float2(acc_46[16], acc_46[17]);
                            float2 _f2_209 = make_float2(__uint_as_float(r_86[8] << 16), __uint_as_float(r_86[8] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_24;
                            {
                                float2 _packed_f32x2_30_0 = _f2_208;
                                float2 _packed_f32x2_30_1 = _f2_209;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_24)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_30_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_30_1)));
                            }
                            acc_46[16] = _packed_add_f32x2_24.x;
                            acc_46[17] = _packed_add_f32x2_24.y;
                            float2 _f2_210 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_24, _packed_add_f32x2_24, _f2_210).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_24, _packed_add_f32x2_24, _f2_210).y;
                            float2 _f2_211 = make_float2(acc_46[18], acc_46[19]);
                            float2 _f2_212 = make_float2(__uint_as_float(r_86[9] << 16), __uint_as_float(r_86[9] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_25;
                            {
                                float2 _packed_f32x2_31_0 = _f2_211;
                                float2 _packed_f32x2_31_1 = _f2_212;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_25)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_31_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_31_1)));
                            }
                            acc_46[18] = _packed_add_f32x2_25.x;
                            acc_46[19] = _packed_add_f32x2_25.y;
                            float2 _f2_213 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_25, _packed_add_f32x2_25, _f2_213).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_25, _packed_add_f32x2_25, _f2_213).y;
                            float2 _f2_214 = make_float2(acc_46[20], acc_46[21]);
                            float2 _f2_215 = make_float2(__uint_as_float(r_86[10] << 16), __uint_as_float(r_86[10] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_26;
                            {
                                float2 _packed_f32x2_32_0 = _f2_214;
                                float2 _packed_f32x2_32_1 = _f2_215;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_26)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_32_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_32_1)));
                            }
                            acc_46[20] = _packed_add_f32x2_26.x;
                            acc_46[21] = _packed_add_f32x2_26.y;
                            float2 _f2_216 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_26, _packed_add_f32x2_26, _f2_216).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_26, _packed_add_f32x2_26, _f2_216).y;
                            float2 _f2_217 = make_float2(acc_46[22], acc_46[23]);
                            float2 _f2_218 = make_float2(__uint_as_float(r_86[11] << 16), __uint_as_float(r_86[11] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_27;
                            {
                                float2 _packed_f32x2_33_0 = _f2_217;
                                float2 _packed_f32x2_33_1 = _f2_218;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_27)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_33_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_33_1)));
                            }
                            acc_46[22] = _packed_add_f32x2_27.x;
                            acc_46[23] = _packed_add_f32x2_27.y;
                            float2 _f2_219 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_27, _packed_add_f32x2_27, _f2_219).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_27, _packed_add_f32x2_27, _f2_219).y;
                            float2 _f2_220 = make_float2(acc_46[24], acc_46[25]);
                            float2 _f2_221 = make_float2(__uint_as_float(r_86[12] << 16), __uint_as_float(r_86[12] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_28;
                            {
                                float2 _packed_f32x2_34_0 = _f2_220;
                                float2 _packed_f32x2_34_1 = _f2_221;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_28)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_34_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_34_1)));
                            }
                            acc_46[24] = _packed_add_f32x2_28.x;
                            acc_46[25] = _packed_add_f32x2_28.y;
                            float2 _f2_222 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_28, _packed_add_f32x2_28, _f2_222).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_28, _packed_add_f32x2_28, _f2_222).y;
                            float2 _f2_223 = make_float2(acc_46[26], acc_46[27]);
                            float2 _f2_224 = make_float2(__uint_as_float(r_86[13] << 16), __uint_as_float(r_86[13] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_29;
                            {
                                float2 _packed_f32x2_35_0 = _f2_223;
                                float2 _packed_f32x2_35_1 = _f2_224;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_29)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_35_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_35_1)));
                            }
                            acc_46[26] = _packed_add_f32x2_29.x;
                            acc_46[27] = _packed_add_f32x2_29.y;
                            float2 _f2_225 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_29, _packed_add_f32x2_29, _f2_225).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_29, _packed_add_f32x2_29, _f2_225).y;
                            float2 _f2_226 = make_float2(acc_46[28], acc_46[29]);
                            float2 _f2_227 = make_float2(__uint_as_float(r_86[14] << 16), __uint_as_float(r_86[14] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_30;
                            {
                                float2 _packed_f32x2_36_0 = _f2_226;
                                float2 _packed_f32x2_36_1 = _f2_227;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_30)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_36_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_36_1)));
                            }
                            acc_46[28] = _packed_add_f32x2_30.x;
                            acc_46[29] = _packed_add_f32x2_30.y;
                            float2 _f2_228 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_30, _packed_add_f32x2_30, _f2_228).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_30, _packed_add_f32x2_30, _f2_228).y;
                            float2 _f2_229 = make_float2(acc_46[30], acc_46[31]);
                            float2 _f2_230 = make_float2(__uint_as_float(r_86[15] << 16), __uint_as_float(r_86[15] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_31;
                            {
                                float2 _packed_f32x2_37_0 = _f2_229;
                                float2 _packed_f32x2_37_1 = _f2_230;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_31)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_37_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_37_1)));
                            }
                            acc_46[30] = _packed_add_f32x2_31.x;
                            acc_46[31] = _packed_add_f32x2_31.y;
                            float2 _f2_231 = make_float2(ssa[0], ssa[1]);
                            ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_31, _packed_add_f32x2_31, _f2_231).x;
                            ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_31, _packed_add_f32x2_31, _f2_231).y;
                            uint32_t acc_46_bf16[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_46[_lp*2 + 0], acc_46[_lp*2+1 + 0]));
                                acc_46_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 0)))), "r"(acc_46_bf16[0]), "r"(acc_46_bf16[1]), "r"(acc_46_bf16[2]), "r"(acc_46_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 1)))), "r"(acc_46_bf16[4]), "r"(acc_46_bf16[5]), "r"(acc_46_bf16[6]), "r"(acc_46_bf16[7]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 2)))), "r"(acc_46_bf16[8]), "r"(acc_46_bf16[9]), "r"(acc_46_bf16[10]), "r"(acc_46_bf16[11]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 3)))), "r"(acc_46_bf16[12]), "r"(acc_46_bf16[13]), "r"(acc_46_bf16[14]), "r"(acc_46_bf16[15]) : "memory");
                            __syncwarp();
                            int c_87 = lane_3;
                            int l_88 = c_87 >> 2;
                            unsigned int w4_89[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_89[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_89[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_89[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_89[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_88 + (c_87 & 3 ^ l_88 >> 1 & 3)))));
                            long long idx_90 = drow * 3072 + (long long)(4 * (128 * j_83 + c_87));
                            {
                                int4 _iv4 = make_int4(w4_89[0 + 0], w4_89[0 + 1], w4_89[0 + 2], w4_89[0 + 3]);
                                *reinterpret_cast<int4*>(resout_i32 + (idx_90)) = _iv4;
                            }
                            int c_91 = lane_3 + 32;
                            int l_92 = c_91 >> 2;
                            unsigned int w4_93[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_93[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_93[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_93[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_93[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_92 + (c_91 & 3 ^ l_92 >> 1 & 3)))));
                            long long idx_94 = drow * 3072 + (long long)(4 * (128 * j_83 + c_91));
                            {
                                int4 _iv4 = make_int4(w4_93[0 + 0], w4_93[0 + 1], w4_93[0 + 2], w4_93[0 + 3]);
                                *reinterpret_cast<int4*>(resout_i32 + (idx_94)) = _iv4;
                            }
                            int c_95 = lane_3 + 64;
                            int l_96 = c_95 >> 2;
                            unsigned int w4_97[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_97[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_97[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_97[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_97[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_96 + (c_95 & 3 ^ l_96 >> 1 & 3)))));
                            long long idx_98 = drow * 3072 + (long long)(4 * (128 * j_83 + c_95));
                            {
                                int4 _iv4 = make_int4(w4_97[0 + 0], w4_97[0 + 1], w4_97[0 + 2], w4_97[0 + 3]);
                                *reinterpret_cast<int4*>(resout_i32 + (idx_98)) = _iv4;
                            }
                            int c_99 = lane_3 + 96;
                            int l_100 = c_99 >> 2;
                            unsigned int w4_101[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_101[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_101[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_101[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_101[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_100 + (c_99 & 3 ^ l_100 >> 1 & 3)))));
                            long long idx_102 = drow * 3072 + (long long)(4 * (128 * j_83 + c_99));
                            {
                                int4 _iv4 = make_int4(w4_101[0 + 0], w4_101[0 + 1], w4_101[0 + 2], w4_101[0 + 3]);
                                *reinterpret_cast<int4*>(resout_i32 + (idx_102)) = _iv4;
                            }
                            __syncwarp();
                            int zlo_103 = zb_off + 4096 * j_83;
                            int zhi_104 = rb_off + 4096 * (j_83 - 3);
                            int zbase_105 = ((j_83 < 3) ? zlo_103 : zhi_104);
                            int zlane_106 = zbase_105 + 128 * lane_3;
                            int l7_107 = lane_3 & 7;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_106 + 16 * (l7_107 ^ 0))), "r"(__as_u32(acc_46[0])), "r"(__as_u32(acc_46[1])), "r"(__as_u32(acc_46[2])), "r"(__as_u32(acc_46[3])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_106 + 16 * (l7_107 ^ 1))), "r"(__as_u32(acc_46[4])), "r"(__as_u32(acc_46[5])), "r"(__as_u32(acc_46[6])), "r"(__as_u32(acc_46[7])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_106 + 16 * (l7_107 ^ 2))), "r"(__as_u32(acc_46[8])), "r"(__as_u32(acc_46[9])), "r"(__as_u32(acc_46[10])), "r"(__as_u32(acc_46[11])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_106 + 16 * (l7_107 ^ 3))), "r"(__as_u32(acc_46[12])), "r"(__as_u32(acc_46[13])), "r"(__as_u32(acc_46[14])), "r"(__as_u32(acc_46[15])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_106 + 16 * (l7_107 ^ 4))), "r"(__as_u32(acc_46[16])), "r"(__as_u32(acc_46[17])), "r"(__as_u32(acc_46[18])), "r"(__as_u32(acc_46[19])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_106 + 16 * (l7_107 ^ 5))), "r"(__as_u32(acc_46[20])), "r"(__as_u32(acc_46[21])), "r"(__as_u32(acc_46[22])), "r"(__as_u32(acc_46[23])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_106 + 16 * (l7_107 ^ 6))), "r"(__as_u32(acc_46[24])), "r"(__as_u32(acc_46[25])), "r"(__as_u32(acc_46[26])), "r"(__as_u32(acc_46[27])) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_106 + 16 * (l7_107 ^ 7))), "r"(__as_u32(acc_46[28])), "r"(__as_u32(acc_46[29])), "r"(__as_u32(acc_46[30])), "r"(__as_u32(acc_46[31])) : "memory");
                        }
                    }
                    {
                        float one = 1.0f;
                        float _fma_0;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(ssa[0]), "f"(one), "f"(ssa[1]));
                        float ss = _fma_0;
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, ss, 16);
                        float _fma_1;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(_shfl_xor_0), "f"(one), "f"(ss));
                        ss = _fma_1;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, ss, 8);
                        float _fma_2;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(_shfl_xor_1), "f"(one), "f"(ss));
                        ss = _fma_2;
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, ss, 4);
                        float _fma_3;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(_shfl_xor_2), "f"(one), "f"(ss));
                        ss = _fma_3;
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, ss, 2);
                        float _fma_4;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(_shfl_xor_3), "f"(one), "f"(ss));
                        ss = _fma_4;
                        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, ss, 1);
                        float _fma_5;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(_shfl_xor_4), "f"(one), "f"(ss));
                        ss = _fma_5;
                        float hdim = 6144.0f;
                        float eps = 1e-06f;
                        float _fdiv_rn_0 = __fdiv_rn(ss, hdim);
                        float _fma_6;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(_fdiv_rn_0), "f"(one), "f"(eps));
                        float _rsqrt_0;
                        asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(_rsqrt_0) : "f"(_fma_6));
                        float2 _f2_232 = make_float2(_rsqrt_0, _rsqrt_0);
                        float one_0 = 1.0f;
                        float2 _f2_233 = make_float2(one_0, one_0);
                        int l7_1 = lane_3 & 7;
                        #pragma unroll 2
                        for (int j_1 = 0; j_1 < 6; j_1++) {
                            int pp = lane_3 + 32 * j_1;
                            int zlo_1 = zb_off + 4096 * j_1;
                            int zhi_1 = rb_off + 4096 * (j_1 - 3);
                            int zbase_1 = ((j_1 < 3) ? zlo_1 : zhi_1);
                            int zlane_1 = zbase_1 + 128 * lane_3;
                            int kw = f ^ 0;
                            unsigned int wc[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&wc[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc[(0) + 3]))
                                : "r"(wrow_addr + (unsigned int)(64 * pp) + (unsigned int)(16 * kw)));
                            float zc[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc[0])), "=r"(*reinterpret_cast<uint32_t*>(&zc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_1 + (unsigned int)(16 * (l7_1 ^ 2 * kw))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc[4])), "=r"(*reinterpret_cast<uint32_t*>(&zc[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_1 + (unsigned int)(16 * (l7_1 ^ 2 * kw + 1))));
                            float yv[8];
                            float2 _f2_234 = make_float2(zc[0], zc[1]);
                            float2 _mul_f32x2_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_234), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_235 = make_float2(__uint_as_float(wc[0] << 16), __uint_as_float(wc[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_32;
                            {
                                float2 _packed_f32x2_38_0 = _f2_233;
                                float2 _packed_f32x2_38_1 = _f2_235;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_32)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_38_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_38_1)));
                            }
                            float2 _mul_f32x2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_mul_f32x2_0), "l"(*(const unsigned long long*)&_packed_add_f32x2_32));
                            yv[0] = _mul_f32x2_1.x;
                            yv[1] = _mul_f32x2_1.y;
                            float2 _f2_236 = make_float2(zc[2], zc[3]);
                            float2 _mul_f32x2_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_236), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_237 = make_float2(__uint_as_float(wc[1] << 16), __uint_as_float(wc[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_33;
                            {
                                float2 _packed_f32x2_39_0 = _f2_233;
                                float2 _packed_f32x2_39_1 = _f2_237;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_33)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_39_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_39_1)));
                            }
                            float2 _mul_f32x2_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_mul_f32x2_2), "l"(*(const unsigned long long*)&_packed_add_f32x2_33));
                            yv[2] = _mul_f32x2_3.x;
                            yv[3] = _mul_f32x2_3.y;
                            float2 _f2_238 = make_float2(zc[4], zc[5]);
                            float2 _mul_f32x2_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_238), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_239 = make_float2(__uint_as_float(wc[2] << 16), __uint_as_float(wc[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_34;
                            {
                                float2 _packed_f32x2_40_0 = _f2_233;
                                float2 _packed_f32x2_40_1 = _f2_239;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_34)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_40_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_40_1)));
                            }
                            float2 _mul_f32x2_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_mul_f32x2_4), "l"(*(const unsigned long long*)&_packed_add_f32x2_34));
                            yv[4] = _mul_f32x2_5.x;
                            yv[5] = _mul_f32x2_5.y;
                            float2 _f2_240 = make_float2(zc[6], zc[7]);
                            float2 _mul_f32x2_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&_f2_240), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_241 = make_float2(__uint_as_float(wc[3] << 16), __uint_as_float(wc[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_35;
                            {
                                float2 _packed_f32x2_41_0 = _f2_233;
                                float2 _packed_f32x2_41_1 = _f2_241;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_35)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_41_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_41_1)));
                            }
                            float2 _mul_f32x2_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&_mul_f32x2_6), "l"(*(const unsigned long long*)&_packed_add_f32x2_35));
                            yv[6] = _mul_f32x2_7.x;
                            yv[7] = _mul_f32x2_7.y;
                            uint32_t yv_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv[_lp*2 + 0], yv[_lp*2+1 + 0]));
                                yv_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + kw))), "r"(yv_bf16[0]), "r"(yv_bf16[1]), "r"(yv_bf16[2]), "r"(yv_bf16[3]) : "memory");
                            int kw_0 = f ^ 1;
                            unsigned int wc_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&wc_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_1[(0) + 3]))
                                : "r"(wrow_addr + (unsigned int)(64 * pp) + (unsigned int)(16 * kw_0)));
                            float zc_2[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&zc_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_2[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_1 + (unsigned int)(16 * (l7_1 ^ 2 * kw_0))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&zc_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_2[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_1 + (unsigned int)(16 * (l7_1 ^ 2 * kw_0 + 1))));
                            float yv_3[8];
                            float2 _f2_242 = make_float2(zc_2[0], zc_2[1]);
                            float2 _mul_f32x2_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_8) : "l"(*(const unsigned long long*)&_f2_242), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_243 = make_float2(__uint_as_float(wc_1[0] << 16), __uint_as_float(wc_1[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_36;
                            {
                                float2 _packed_f32x2_42_0 = _f2_233;
                                float2 _packed_f32x2_42_1 = _f2_243;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_36)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_42_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_42_1)));
                            }
                            float2 _mul_f32x2_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&_mul_f32x2_8), "l"(*(const unsigned long long*)&_packed_add_f32x2_36));
                            yv_3[0] = _mul_f32x2_9.x;
                            yv_3[1] = _mul_f32x2_9.y;
                            float2 _f2_244 = make_float2(zc_2[2], zc_2[3]);
                            float2 _mul_f32x2_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_10) : "l"(*(const unsigned long long*)&_f2_244), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_245 = make_float2(__uint_as_float(wc_1[1] << 16), __uint_as_float(wc_1[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_37;
                            {
                                float2 _packed_f32x2_43_0 = _f2_233;
                                float2 _packed_f32x2_43_1 = _f2_245;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_37)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_43_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_43_1)));
                            }
                            float2 _mul_f32x2_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_11) : "l"(*(const unsigned long long*)&_mul_f32x2_10), "l"(*(const unsigned long long*)&_packed_add_f32x2_37));
                            yv_3[2] = _mul_f32x2_11.x;
                            yv_3[3] = _mul_f32x2_11.y;
                            float2 _f2_246 = make_float2(zc_2[4], zc_2[5]);
                            float2 _mul_f32x2_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_12) : "l"(*(const unsigned long long*)&_f2_246), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_247 = make_float2(__uint_as_float(wc_1[2] << 16), __uint_as_float(wc_1[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_38;
                            {
                                float2 _packed_f32x2_44_0 = _f2_233;
                                float2 _packed_f32x2_44_1 = _f2_247;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_38)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_44_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_44_1)));
                            }
                            float2 _mul_f32x2_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_13) : "l"(*(const unsigned long long*)&_mul_f32x2_12), "l"(*(const unsigned long long*)&_packed_add_f32x2_38));
                            yv_3[4] = _mul_f32x2_13.x;
                            yv_3[5] = _mul_f32x2_13.y;
                            float2 _f2_248 = make_float2(zc_2[6], zc_2[7]);
                            float2 _mul_f32x2_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&_f2_248), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_249 = make_float2(__uint_as_float(wc_1[3] << 16), __uint_as_float(wc_1[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_39;
                            {
                                float2 _packed_f32x2_45_0 = _f2_233;
                                float2 _packed_f32x2_45_1 = _f2_249;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_39)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_45_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_45_1)));
                            }
                            float2 _mul_f32x2_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&_mul_f32x2_14), "l"(*(const unsigned long long*)&_packed_add_f32x2_39));
                            yv_3[6] = _mul_f32x2_15.x;
                            yv_3[7] = _mul_f32x2_15.y;
                            uint32_t yv_3_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_3[_lp*2 + 0], yv_3[_lp*2+1 + 0]));
                                yv_3_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + kw_0))), "r"(yv_3_bf16[0]), "r"(yv_3_bf16[1]), "r"(yv_3_bf16[2]), "r"(yv_3_bf16[3]) : "memory");
                            int kw_4 = f ^ 2;
                            unsigned int wc_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&wc_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_5[(0) + 3]))
                                : "r"(wrow_addr + (unsigned int)(64 * pp) + (unsigned int)(16 * kw_4)));
                            float zc_6[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&zc_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_6[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_1 + (unsigned int)(16 * (l7_1 ^ 2 * kw_4))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&zc_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_6[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_1 + (unsigned int)(16 * (l7_1 ^ 2 * kw_4 + 1))));
                            float yv_7[8];
                            float2 _f2_250 = make_float2(zc_6[0], zc_6[1]);
                            float2 _mul_f32x2_16;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_16) : "l"(*(const unsigned long long*)&_f2_250), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_251 = make_float2(__uint_as_float(wc_5[0] << 16), __uint_as_float(wc_5[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_40;
                            {
                                float2 _packed_f32x2_46_0 = _f2_233;
                                float2 _packed_f32x2_46_1 = _f2_251;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_40)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_46_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_46_1)));
                            }
                            float2 _mul_f32x2_17;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_17) : "l"(*(const unsigned long long*)&_mul_f32x2_16), "l"(*(const unsigned long long*)&_packed_add_f32x2_40));
                            yv_7[0] = _mul_f32x2_17.x;
                            yv_7[1] = _mul_f32x2_17.y;
                            float2 _f2_252 = make_float2(zc_6[2], zc_6[3]);
                            float2 _mul_f32x2_18;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_18) : "l"(*(const unsigned long long*)&_f2_252), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_253 = make_float2(__uint_as_float(wc_5[1] << 16), __uint_as_float(wc_5[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_41;
                            {
                                float2 _packed_f32x2_47_0 = _f2_233;
                                float2 _packed_f32x2_47_1 = _f2_253;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_41)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_47_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_47_1)));
                            }
                            float2 _mul_f32x2_19;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_19) : "l"(*(const unsigned long long*)&_mul_f32x2_18), "l"(*(const unsigned long long*)&_packed_add_f32x2_41));
                            yv_7[2] = _mul_f32x2_19.x;
                            yv_7[3] = _mul_f32x2_19.y;
                            float2 _f2_254 = make_float2(zc_6[4], zc_6[5]);
                            float2 _mul_f32x2_20;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_20) : "l"(*(const unsigned long long*)&_f2_254), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_255 = make_float2(__uint_as_float(wc_5[2] << 16), __uint_as_float(wc_5[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_42;
                            {
                                float2 _packed_f32x2_48_0 = _f2_233;
                                float2 _packed_f32x2_48_1 = _f2_255;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_42)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_48_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_48_1)));
                            }
                            float2 _mul_f32x2_21;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_21) : "l"(*(const unsigned long long*)&_mul_f32x2_20), "l"(*(const unsigned long long*)&_packed_add_f32x2_42));
                            yv_7[4] = _mul_f32x2_21.x;
                            yv_7[5] = _mul_f32x2_21.y;
                            float2 _f2_256 = make_float2(zc_6[6], zc_6[7]);
                            float2 _mul_f32x2_22;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_22) : "l"(*(const unsigned long long*)&_f2_256), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_257 = make_float2(__uint_as_float(wc_5[3] << 16), __uint_as_float(wc_5[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_43;
                            {
                                float2 _packed_f32x2_49_0 = _f2_233;
                                float2 _packed_f32x2_49_1 = _f2_257;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_43)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_49_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_49_1)));
                            }
                            float2 _mul_f32x2_23;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_23) : "l"(*(const unsigned long long*)&_mul_f32x2_22), "l"(*(const unsigned long long*)&_packed_add_f32x2_43));
                            yv_7[6] = _mul_f32x2_23.x;
                            yv_7[7] = _mul_f32x2_23.y;
                            uint32_t yv_7_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_7[_lp*2 + 0], yv_7[_lp*2+1 + 0]));
                                yv_7_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + kw_4))), "r"(yv_7_bf16[0]), "r"(yv_7_bf16[1]), "r"(yv_7_bf16[2]), "r"(yv_7_bf16[3]) : "memory");
                            int kw_8 = f ^ 3;
                            unsigned int wc_9[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&wc_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_9[(0) + 3]))
                                : "r"(wrow_addr + (unsigned int)(64 * pp) + (unsigned int)(16 * kw_8)));
                            float zc_10[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&zc_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_10[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_1 + (unsigned int)(16 * (l7_1 ^ 2 * kw_8))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&zc_10[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_10[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_10[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_1 + (unsigned int)(16 * (l7_1 ^ 2 * kw_8 + 1))));
                            float yv_11[8];
                            float2 _f2_258 = make_float2(zc_10[0], zc_10[1]);
                            float2 _mul_f32x2_24;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_24) : "l"(*(const unsigned long long*)&_f2_258), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_259 = make_float2(__uint_as_float(wc_9[0] << 16), __uint_as_float(wc_9[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_44;
                            {
                                float2 _packed_f32x2_50_0 = _f2_233;
                                float2 _packed_f32x2_50_1 = _f2_259;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_44)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_50_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_50_1)));
                            }
                            float2 _mul_f32x2_25;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_25) : "l"(*(const unsigned long long*)&_mul_f32x2_24), "l"(*(const unsigned long long*)&_packed_add_f32x2_44));
                            yv_11[0] = _mul_f32x2_25.x;
                            yv_11[1] = _mul_f32x2_25.y;
                            float2 _f2_260 = make_float2(zc_10[2], zc_10[3]);
                            float2 _mul_f32x2_26;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_26) : "l"(*(const unsigned long long*)&_f2_260), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_261 = make_float2(__uint_as_float(wc_9[1] << 16), __uint_as_float(wc_9[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_45;
                            {
                                float2 _packed_f32x2_51_0 = _f2_233;
                                float2 _packed_f32x2_51_1 = _f2_261;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_45)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_51_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_51_1)));
                            }
                            float2 _mul_f32x2_27;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_27) : "l"(*(const unsigned long long*)&_mul_f32x2_26), "l"(*(const unsigned long long*)&_packed_add_f32x2_45));
                            yv_11[2] = _mul_f32x2_27.x;
                            yv_11[3] = _mul_f32x2_27.y;
                            float2 _f2_262 = make_float2(zc_10[4], zc_10[5]);
                            float2 _mul_f32x2_28;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_28) : "l"(*(const unsigned long long*)&_f2_262), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_263 = make_float2(__uint_as_float(wc_9[2] << 16), __uint_as_float(wc_9[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_46;
                            {
                                float2 _packed_f32x2_52_0 = _f2_233;
                                float2 _packed_f32x2_52_1 = _f2_263;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_46)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_52_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_52_1)));
                            }
                            float2 _mul_f32x2_29;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_29) : "l"(*(const unsigned long long*)&_mul_f32x2_28), "l"(*(const unsigned long long*)&_packed_add_f32x2_46));
                            yv_11[4] = _mul_f32x2_29.x;
                            yv_11[5] = _mul_f32x2_29.y;
                            float2 _f2_264 = make_float2(zc_10[6], zc_10[7]);
                            float2 _mul_f32x2_30;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_30) : "l"(*(const unsigned long long*)&_f2_264), "l"(*(const unsigned long long*)&_f2_232));
                            float2 _f2_265 = make_float2(__uint_as_float(wc_9[3] << 16), __uint_as_float(wc_9[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_47;
                            {
                                float2 _packed_f32x2_53_0 = _f2_233;
                                float2 _packed_f32x2_53_1 = _f2_265;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_47)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_53_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_53_1)));
                            }
                            float2 _mul_f32x2_31;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_31) : "l"(*(const unsigned long long*)&_mul_f32x2_30), "l"(*(const unsigned long long*)&_packed_add_f32x2_47));
                            yv_11[6] = _mul_f32x2_31.x;
                            yv_11[7] = _mul_f32x2_31.y;
                            uint32_t yv_11_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_11[_lp*2 + 0], yv_11[_lp*2+1 + 0]));
                                yv_11_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + kw_8))), "r"(yv_11_bf16[0]), "r"(yv_11_bf16[1]), "r"(yv_11_bf16[2]), "r"(yv_11_bf16[3]) : "memory");
                            __syncwarp();
                            unsigned int w4_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * lane_3)));
                            long long idx_1 = drow * 3072 + (long long)(4 * (128 * j_1 + lane_3));
                            {
                                int4 _iv4 = make_int4(w4_1[0 + 0], w4_1[0 + 1], w4_1[0 + 2], w4_1[0 + 3]);
                                *reinterpret_cast<int4*>(out_i32 + (idx_1)) = _iv4;
                            }
                            unsigned int w4_12[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 32))));
                            long long idx_13 = drow * 3072 + (long long)(4 * (128 * j_1 + lane_3 + 32));
                            {
                                int4 _iv4 = make_int4(w4_12[0 + 0], w4_12[0 + 1], w4_12[0 + 2], w4_12[0 + 3]);
                                *reinterpret_cast<int4*>(out_i32 + (idx_13)) = _iv4;
                            }
                            unsigned int w4_14[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_14[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 64))));
                            long long idx_15 = drow * 3072 + (long long)(4 * (128 * j_1 + lane_3 + 64));
                            {
                                int4 _iv4 = make_int4(w4_14[0 + 0], w4_14[0 + 1], w4_14[0 + 2], w4_14[0 + 3]);
                                *reinterpret_cast<int4*>(out_i32 + (idx_15)) = _iv4;
                            }
                            unsigned int w4_16[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 96))));
                            long long idx_17 = drow * 3072 + (long long)(4 * (128 * j_1 + lane_3 + 96));
                            {
                                int4 _iv4 = make_int4(w4_16[0 + 0], w4_16[0 + 1], w4_16[0 + 2], w4_16[0 + 3]);
                                *reinterpret_cast<int4*>(out_i32 + (idx_17)) = _iv4;
                            }
                            __syncwarp();
                        }
                    }
                    if (n > kd + 1) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (lane_3 == 0) {
                            long long nrow = (long long)(r0 + warp_2 + 5 * (kd + 1));
                            mbarrier_arrive_expect_tx(rb_full_addr + (warp_2) * 8, 12288);
                            cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)rb_off, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(residual_in) + ((unsigned long long)(nrow * 6144) * (unsigned long long)2)), 12288, rb_full_addr + (warp_2) * 8);
                        }
                    }
                }
            }
            if (lane_3 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            asm volatile("barrier.sync 6, 192;" ::: "memory");
            if (tid == 0) {
                __threadfence();
                unsigned int one_1 = 1;
                unsigned int _atomic_old_0 = atomicAdd(&ctrl[1], one_1);
                unsigned int last = (unsigned int)num_bids - 1;
                if (_atomic_old_0 == last) {
                    unsigned int zero = 0;
                    unsigned int rows_u = (unsigned int)rows;
                    unsigned int algo_u = 1;
                    ctrl[1] = zero;
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
    if (warp == 5) {
        { // reset_warp_main
            int lane_0 = (int)lane;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("barrier.sync 0, 192;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int call_1 = ctl[0];
            unsigned int prev_rows = ctl[1];
            unsigned int prev_algo = ctl[2];
            unsigned int pbuf = (call_1 + 2) % 3;
            if (prev_algo != 0) {
                int* self_i32 = reinterpret_cast<int*>(workspace[1]);
                long long bid64_1 = (long long)bid;
                long long g64_1 = (long long)num_bids;
                long long pr64 = (long long)prev_rows;
                long long pb64 = (long long)pbuf;
                long long lane64 = (long long)lane_0;
                unsigned int sv = 143165576;
                sv = sv | sv << 4;
                unsigned int ss_1 = 268435455;
                ss_1 = ss_1 | ss_1 << 4;
                unsigned int sr = 1;
                sr = sr << 31;
                unsigned int w4_2[4];
                w4_2[0] = sv;
                w4_2[1] = sv;
                w4_2[2] = sv;
                w4_2[3] = sv;
                long long plane = pb64 * 113770496;
                long long u = pr64 * 3072 / 16;
                long long lo = u * bid64_1 / g64_1 * 16;
                long long hi = u * (bid64_1 + 1) / g64_1 * 16;
                #pragma unroll 1
                for (long long o = lo + 16 * lane64; o < hi; o += 512) {
                    long long widx = plane + o >> 2;
                    {
                        int4 _iv4 = make_int4(w4_2[0 + 0], w4_2[0 + 1], w4_2[0 + 2], w4_2[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx)) = _iv4;
                    }
                }
                unsigned int w4_0[4];
                w4_0[0] = ss_1;
                w4_0[1] = ss_1;
                w4_0[2] = ss_1;
                w4_0[3] = ss_1;
                long long plane_1 = pb64 * 113770496 + 100663296;
                long long u_2 = pr64 * 384 / 16;
                long long lo_3 = u_2 * bid64_1 / g64_1 * 16;
                long long hi_4 = u_2 * (bid64_1 + 1) / g64_1 * 16;
                #pragma unroll 1
                for (long long o_1 = lo_3 + 16 * lane64; o_1 < hi_4; o_1 += 512) {
                    long long widx_1 = plane_1 + o_1 >> 2;
                    {
                        int4 _iv4 = make_int4(w4_0[0 + 0], w4_0[0 + 1], w4_0[0 + 2], w4_0[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_1)) = _iv4;
                    }
                }
                unsigned int w4_5[4];
                w4_5[0] = sr;
                w4_5[1] = sr;
                w4_5[2] = sr;
                w4_5[3] = sr;
                long long plane_6 = pb64 * 113770496 + 113246208;
                long long u_7 = pr64 * 16 / 16;
                long long lo_8 = u_7 * bid64_1 / g64_1 * 16;
                long long hi_9 = u_7 * (bid64_1 + 1) / g64_1 * 16;
                #pragma unroll 1
                for (long long o_2 = lo_8 + 16 * lane64; o_2 < hi_9; o_2 += 512) {
                    long long widx_2 = plane_6 + o_2 >> 2;
                    {
                        int4 _iv4 = make_int4(w4_5[0 + 0], w4_5[0 + 1], w4_5[0 + 2], w4_5[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_2)) = _iv4;
                    }
                }
            }
            asm volatile("barrier.sync 6, 192;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
