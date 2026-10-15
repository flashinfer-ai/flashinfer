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
#define SMEM_WSM_STAGE_BYTES 201936
#define SMEM_WSM_STRIDE 201936
#define SMEM_WSM16_OFF 13376
#define SMEM_WSM16_STAGE_BYTES 201936
#define SMEM_WSM16_STRIDE 201936
#define SMEM_WSM32_OFF 13376
#define SMEM_WSM32_STAGE_BYTES 201936
#define SMEM_WSM32_STRIDE 201936
#define SMEM_TOTAL 215424
#define THREADS 256
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
    #define rb_full_addr (mbar_base + 56)

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

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 14 barriers)
    // Mbarriers at smem_raw[0..112)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // xb_full: 7 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // rb_full: 7 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: workers ----
    if (warp <= 6) {
        { // workers_main
            uint8_t* w_u8 = reinterpret_cast<uint8_t*>(weight);
            int tid_0 = (int)tid;
            #pragma unroll 1
            for (int i = tid_0; i < 768; i += 224) {
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
                n = (nrows - warp_2 + 7 - 1) / 7;
            }
            int wbase0 = warp_2 * 28848;
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
            asm volatile("barrier.sync 0, 256;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int call = ctl[0];
            unsigned int buf = call % 3;
            int* resout_i32 = reinterpret_cast<int*>(residual_out);
            int* out_i32 = reinterpret_cast<int*>(out);
            int f = lane_3 >> 1 & 3;
            int wbase = warp_2 * 28848;
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
                    long long row64 = (long long)(r0 + warp_2 + 7 * k);
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
                                long long next_row = (long long)(r0 + warp_2 + 7 * (k + 1));
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
                    long long drow = (long long)(r0 + warp_2 + 7 * kd);
                    int dslot = kd % 3;
                    int dob_off = qr_off + dslot * 3472;
                    long long v_base = os_base + drow * 3072;
                    long long sc_base = os_base + 100663296 + drow * 384;
                    long long m_base = os_base + 113246208 + drow * 16;
                    uint32_t _sysv_u32_0;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_0) : "l"(reinterpret_cast<unsigned int*>(self_u8 + m_base)) : "memory");
                    float ssa[2];
                    ssa[0] = 0.0f;
                    ssa[1] = 0.0f;
                    float zreg[32];
                    float zreg_b[32];
                    zreg[0] = 0.0f;
                    zreg_b[0] = 0.0f;
                    zreg[1] = 0.0f;
                    zreg_b[1] = 0.0f;
                    zreg[2] = 0.0f;
                    zreg_b[2] = 0.0f;
                    zreg[3] = 0.0f;
                    zreg_b[3] = 0.0f;
                    zreg[4] = 0.0f;
                    zreg_b[4] = 0.0f;
                    zreg[5] = 0.0f;
                    zreg_b[5] = 0.0f;
                    zreg[6] = 0.0f;
                    zreg_b[6] = 0.0f;
                    zreg[7] = 0.0f;
                    zreg_b[7] = 0.0f;
                    zreg[8] = 0.0f;
                    zreg_b[8] = 0.0f;
                    zreg[9] = 0.0f;
                    zreg_b[9] = 0.0f;
                    zreg[10] = 0.0f;
                    zreg_b[10] = 0.0f;
                    zreg[11] = 0.0f;
                    zreg_b[11] = 0.0f;
                    zreg[12] = 0.0f;
                    zreg_b[12] = 0.0f;
                    zreg[13] = 0.0f;
                    zreg_b[13] = 0.0f;
                    zreg[14] = 0.0f;
                    zreg_b[14] = 0.0f;
                    zreg[15] = 0.0f;
                    zreg_b[15] = 0.0f;
                    zreg[16] = 0.0f;
                    zreg_b[16] = 0.0f;
                    zreg[17] = 0.0f;
                    zreg_b[17] = 0.0f;
                    zreg[18] = 0.0f;
                    zreg_b[18] = 0.0f;
                    zreg[19] = 0.0f;
                    zreg_b[19] = 0.0f;
                    zreg[20] = 0.0f;
                    zreg_b[20] = 0.0f;
                    zreg[21] = 0.0f;
                    zreg_b[21] = 0.0f;
                    zreg[22] = 0.0f;
                    zreg_b[22] = 0.0f;
                    zreg[23] = 0.0f;
                    zreg_b[23] = 0.0f;
                    zreg[24] = 0.0f;
                    zreg_b[24] = 0.0f;
                    zreg[25] = 0.0f;
                    zreg_b[25] = 0.0f;
                    zreg[26] = 0.0f;
                    zreg_b[26] = 0.0f;
                    zreg[27] = 0.0f;
                    zreg_b[27] = 0.0f;
                    zreg[28] = 0.0f;
                    zreg_b[28] = 0.0f;
                    zreg[29] = 0.0f;
                    zreg_b[29] = 0.0f;
                    zreg[30] = 0.0f;
                    zreg_b[30] = 0.0f;
                    zreg[31] = 0.0f;
                    zreg_b[31] = 0.0f;
                    unsigned int pf_v[24];
                    unsigned int pf_s[6];
                    uint32_t _sysv_ld_0[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_0[0]), "=r"(_sysv_ld_0[1]), "=r"(_sysv_ld_0[2]), "=r"(_sysv_ld_0[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * lane_3)))) : "memory");
                    pf_v[0] = _sysv_ld_0[0];
                    pf_v[1] = _sysv_ld_0[1];
                    pf_v[2] = _sysv_ld_0[2];
                    pf_v[3] = _sysv_ld_0[3];
                    unsigned int _vec_load_1[1];
                    {
                        uint16_t _scalar_bits_4;
                        asm volatile("ld.volatile.global.u16 %0, [%1];"
                            : "=h"(_scalar_bits_4) : "l"((const void*)(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * lane_3))) + (0))) : "memory");
                        _vec_load_1[0] = (uint16_t)_scalar_bits_4;
                    }
                    pf_s[0] = _vec_load_1[0];
                    uint32_t _sysv_ld_1[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_1[0]), "=r"(_sysv_ld_1[1]), "=r"(_sysv_ld_1[2]), "=r"(_sysv_ld_1[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_3 + 32))))) : "memory");
                    pf_v[4] = _sysv_ld_1[0];
                    pf_v[5] = _sysv_ld_1[1];
                    pf_v[6] = _sysv_ld_1[2];
                    pf_v[7] = _sysv_ld_1[3];
                    unsigned int _vec_load_2[1];
                    {
                        uint16_t _scalar_bits_5;
                        asm volatile("ld.volatile.global.u16 %0, [%1];"
                            : "=h"(_scalar_bits_5) : "l"((const void*)(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_3 + 32)))) + (0))) : "memory");
                        _vec_load_2[0] = (uint16_t)_scalar_bits_5;
                    }
                    pf_s[1] = _vec_load_2[0];
                    uint32_t _sysv_ld_2[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_2[0]), "=r"(_sysv_ld_2[1]), "=r"(_sysv_ld_2[2]), "=r"(_sysv_ld_2[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_3 + 64))))) : "memory");
                    pf_v[8] = _sysv_ld_2[0];
                    pf_v[9] = _sysv_ld_2[1];
                    pf_v[10] = _sysv_ld_2[2];
                    pf_v[11] = _sysv_ld_2[3];
                    unsigned int _vec_load_3[1];
                    {
                        uint16_t _scalar_bits_6;
                        asm volatile("ld.volatile.global.u16 %0, [%1];"
                            : "=h"(_scalar_bits_6) : "l"((const void*)(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_3 + 64)))) + (0))) : "memory");
                        _vec_load_3[0] = (uint16_t)_scalar_bits_6;
                    }
                    pf_s[2] = _vec_load_3[0];
                    uint32_t _sysv_ld_3[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_3[0]), "=r"(_sysv_ld_3[1]), "=r"(_sysv_ld_3[2]), "=r"(_sysv_ld_3[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_3 + 96))))) : "memory");
                    pf_v[12] = _sysv_ld_3[0];
                    pf_v[13] = _sysv_ld_3[1];
                    pf_v[14] = _sysv_ld_3[2];
                    pf_v[15] = _sysv_ld_3[3];
                    unsigned int _vec_load_4[1];
                    {
                        uint16_t _scalar_bits_7;
                        asm volatile("ld.volatile.global.u16 %0, [%1];"
                            : "=h"(_scalar_bits_7) : "l"((const void*)(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_3 + 96)))) + (0))) : "memory");
                        _vec_load_4[0] = (uint16_t)_scalar_bits_7;
                    }
                    pf_s[3] = _vec_load_4[0];
                    uint32_t _sysv_ld_4[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_4[0]), "=r"(_sysv_ld_4[1]), "=r"(_sysv_ld_4[2]), "=r"(_sysv_ld_4[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_3 + 128))))) : "memory");
                    pf_v[16] = _sysv_ld_4[0];
                    pf_v[17] = _sysv_ld_4[1];
                    pf_v[18] = _sysv_ld_4[2];
                    pf_v[19] = _sysv_ld_4[3];
                    unsigned int _vec_load_5[1];
                    {
                        uint16_t _scalar_bits_8;
                        asm volatile("ld.volatile.global.u16 %0, [%1];"
                            : "=h"(_scalar_bits_8) : "l"((const void*)(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_3 + 128)))) + (0))) : "memory");
                        _vec_load_5[0] = (uint16_t)_scalar_bits_8;
                    }
                    pf_s[4] = _vec_load_5[0];
                    uint32_t _sysv_ld_5[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_5[0]), "=r"(_sysv_ld_5[1]), "=r"(_sysv_ld_5[2]), "=r"(_sysv_ld_5[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_3 + 160))))) : "memory");
                    pf_v[20] = _sysv_ld_5[0];
                    pf_v[21] = _sysv_ld_5[1];
                    pf_v[22] = _sysv_ld_5[2];
                    pf_v[23] = _sysv_ld_5[3];
                    unsigned int _vec_load_6[1];
                    {
                        uint16_t _scalar_bits_9;
                        asm volatile("ld.volatile.global.u16 %0, [%1];"
                            : "=h"(_scalar_bits_9) : "l"((const void*)(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_3 + 160)))) + (0))) : "memory");
                        _vec_load_6[0] = (uint16_t)_scalar_bits_9;
                    }
                    pf_s[5] = _vec_load_6[0];
                    {
                        unsigned int d = 0;
                        unsigned int w = pf_v[0];
                        d = d | w & ~((w & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_0 = pf_v[1];
                        d = d | w_0 & ~((w_0 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_1 = pf_v[2];
                        d = d | w_1 & ~((w_1 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_2 = pf_v[3];
                        d = d | w_2 & ~((w_2 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int sc = pf_s[0];
                        d = d | ((sc & 255) + 1 | (sc >> 8) + 1) & 256;
                        unsigned int w_3 = pf_v[4];
                        d = d | w_3 & ~((w_3 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_4 = pf_v[5];
                        d = d | w_4 & ~((w_4 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_5 = pf_v[6];
                        d = d | w_5 & ~((w_5 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_6 = pf_v[7];
                        d = d | w_6 & ~((w_6 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int sc_7 = pf_s[1];
                        d = d | ((sc_7 & 255) + 1 | (sc_7 >> 8) + 1) & 256;
                        if (d != 0) {
                            uint32_t _fp4_wire_values_0[8];
                            uint32_t _fp4_wire_scales_0[2];
                            uint16_t _wire_h16_10_0;
                            uint16_t _wire_h16_10_1;
                            uint32_t _wire_vpend_10 = 3u;
                            uint32_t _wire_spend_10 = 3u;
                            do {
                                if (_wire_vpend_10 & 1u) {
                                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * lane_3)))) : "memory");
                                }
                                if (_wire_vpend_10 & 2u) {
                                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_3 + 32))))) : "memory");
                                }
                                if (_wire_spend_10 & 1u) {
                                    asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_10_0) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * lane_3)))) : "memory");
                                    _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_10_0;
                                }
                                if (_wire_spend_10 & 2u) {
                                    asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_10_1) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_3 + 32))))) : "memory");
                                    _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_10_1;
                                }
                                if (_wire_vpend_10 & 1u) {
                                    if (!((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_10 &= ~1u;
                                }
                                if (_wire_vpend_10 & 2u) {
                                    if (!((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_10 &= ~2u;
                                }
                                if (_wire_spend_10 & 1u) {
                                    if (!(((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[0] >> 8) == 0xFFu))) _wire_spend_10 &= ~1u;
                                }
                                if (_wire_spend_10 & 2u) {
                                    if (!(((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[1] >> 8) == 0xFFu))) _wire_spend_10 &= ~2u;
                                }
                            } while ((_wire_vpend_10 | _wire_spend_10) != 0u);
                            pf_v[0] = _fp4_wire_values_0[0];
                            pf_v[1] = _fp4_wire_values_0[1];
                            pf_v[2] = _fp4_wire_values_0[2];
                            pf_v[3] = _fp4_wire_values_0[3];
                            pf_s[0] = _fp4_wire_scales_0[0];
                            pf_v[4] = _fp4_wire_values_0[4];
                            pf_v[5] = _fp4_wire_values_0[5];
                            pf_v[6] = _fp4_wire_values_0[6];
                            pf_v[7] = _fp4_wire_values_0[7];
                            pf_s[1] = _fp4_wire_scales_0[1];
                        }
                    }
                    unsigned int sent = 1;
                    sent = sent << 31;
                    unsigned int w_7 = _sysv_u32_0;
                    if (w_7 == sent) {
                        uint32_t _sysv_poll32_group_0[1];
                        do {
                            asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[0]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + m_base)) : "memory");
                        } while (
                            (_sysv_poll32_group_0[0] == 0x80000000u)
                        );
                        w_7 = _sysv_poll32_group_0[0];
                    }
                    {
                        mbarrier_wait(rb_full_addr + (warp_2) * 8, kd & 1);
                    }
                    {
                        unsigned int own[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&own[0])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * lane_3)));
                        unsigned int own_sc = (unsigned int)wsm16[(dob_off + 3072 >> 1) + lane_3];
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
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(_fp8x2_decode_0.x), "f"(__uint_as_float(w_7)));
                        float s0 = _mul_0;
                        float _mul_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(_fp8x2_decode_0.y), "f"(__uint_as_float(w_7)));
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
                            : "=f"(_fp8x2_decode_1.x), "=f"(_fp8x2_decode_1.y) : "h"((uint16_t)own_sc));
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
                        unsigned int t[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t[0])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * lane_3) + (unsigned int)(16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t[4])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * lane_3) + (unsigned int)(16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t[8])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * lane_3) + (unsigned int)(16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t[12])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * lane_3) + (unsigned int)(16 * (f ^ 3))));
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
                            float2 _packed_f32x2_12_0 = _f2_68;
                            float2 _packed_f32x2_12_1 = _f2_69;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_0)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_12_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_12_1)));
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
                            float2 _packed_f32x2_13_0 = _f2_71;
                            float2 _packed_f32x2_13_1 = _f2_72;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_1)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_13_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_13_1)));
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
                            float2 _packed_f32x2_14_0 = _f2_74;
                            float2 _packed_f32x2_14_1 = _f2_75;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_2)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_14_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_14_1)));
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
                            float2 _packed_f32x2_15_0 = _f2_77;
                            float2 _packed_f32x2_15_1 = _f2_78;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_3)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_15_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_15_1)));
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
                            float2 _packed_f32x2_16_0 = _f2_80;
                            float2 _packed_f32x2_16_1 = _f2_81;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_4)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_16_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_16_1)));
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
                            float2 _packed_f32x2_17_0 = _f2_83;
                            float2 _packed_f32x2_17_1 = _f2_84;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_5)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_17_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_17_1)));
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
                            float2 _packed_f32x2_18_0 = _f2_86;
                            float2 _packed_f32x2_18_1 = _f2_87;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_6)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_18_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_18_1)));
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
                            float2 _packed_f32x2_19_0 = _f2_89;
                            float2 _packed_f32x2_19_1 = _f2_90;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_7)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_19_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_19_1)));
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
                            float2 _packed_f32x2_20_0 = _f2_92;
                            float2 _packed_f32x2_20_1 = _f2_93;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_8)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_20_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_20_1)));
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
                            float2 _packed_f32x2_21_0 = _f2_95;
                            float2 _packed_f32x2_21_1 = _f2_96;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_9)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_21_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_21_1)));
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
                            float2 _packed_f32x2_22_0 = _f2_98;
                            float2 _packed_f32x2_22_1 = _f2_99;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_10)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_22_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_22_1)));
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
                            float2 _packed_f32x2_23_0 = _f2_101;
                            float2 _packed_f32x2_23_1 = _f2_102;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_11)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_23_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_23_1)));
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
                            float2 _packed_f32x2_24_0 = _f2_104;
                            float2 _packed_f32x2_24_1 = _f2_105;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_12)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_24_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_24_1)));
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
                            float2 _packed_f32x2_25_0 = _f2_107;
                            float2 _packed_f32x2_25_1 = _f2_108;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_13)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_25_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_25_1)));
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
                            float2 _packed_f32x2_26_0 = _f2_110;
                            float2 _packed_f32x2_26_1 = _f2_111;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_14)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_26_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_26_1)));
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
                            float2 _packed_f32x2_27_0 = _f2_113;
                            float2 _packed_f32x2_27_1 = _f2_114;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_15)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_27_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_27_1)));
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
                        long long idx = drow * 3072 + (long long)(4 * c);
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
                        long long idx_37 = drow * 3072 + (long long)(4 * c_34);
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
                        long long idx_41 = drow * 3072 + (long long)(4 * c_38);
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
                        long long idx_45 = drow * 3072 + (long long)(4 * c_42);
                        {
                            int4 _iv4 = make_int4(w4_44[0 + 0], w4_44[0 + 1], w4_44[0 + 2], w4_44[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_45)) = _iv4;
                        }
                        __syncwarp();
                        int zlane = zb_off + 128 * lane_3;
                        int l7 = lane_3 & 7;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 0))), "r"(__as_u32(acc[0])), "r"(__as_u32(acc[1])), "r"(__as_u32(acc[2])), "r"(__as_u32(acc[3])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 1))), "r"(__as_u32(acc[4])), "r"(__as_u32(acc[5])), "r"(__as_u32(acc[6])), "r"(__as_u32(acc[7])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 2))), "r"(__as_u32(acc[8])), "r"(__as_u32(acc[9])), "r"(__as_u32(acc[10])), "r"(__as_u32(acc[11])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 3))), "r"(__as_u32(acc[12])), "r"(__as_u32(acc[13])), "r"(__as_u32(acc[14])), "r"(__as_u32(acc[15])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 4))), "r"(__as_u32(acc[16])), "r"(__as_u32(acc[17])), "r"(__as_u32(acc[18])), "r"(__as_u32(acc[19])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 5))), "r"(__as_u32(acc[20])), "r"(__as_u32(acc[21])), "r"(__as_u32(acc[22])), "r"(__as_u32(acc[23])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 6))), "r"(__as_u32(acc[24])), "r"(__as_u32(acc[25])), "r"(__as_u32(acc[26])), "r"(__as_u32(acc[27])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane + 16 * (l7 ^ 7))), "r"(__as_u32(acc[28])), "r"(__as_u32(acc[29])), "r"(__as_u32(acc[30])), "r"(__as_u32(acc[31])) : "memory");
                    }
                    {
                        unsigned int own_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&own_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_1[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * (lane_3 + 32))));
                        unsigned int own_sc_1 = (unsigned int)wsm16[(dob_off + 3072 >> 1) + (lane_3 + 32)];
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
                        float2 _fp8x2_decode_2;
                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                            "mov.b32 {lo, hi}, pair;\n"
                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                            : "=f"(_fp8x2_decode_2.x), "=f"(_fp8x2_decode_2.y) : "h"((uint16_t)pf_s[1]));
                        float _mul_4;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_fp8x2_decode_2.x), "f"(__uint_as_float(w_7)));
                        float s0_1 = _mul_4;
                        float _mul_5;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_fp8x2_decode_2.y), "f"(__uint_as_float(w_7)));
                        float s1_1 = _mul_5;
                        float2 _f2_116 = make_float2(s0_1, s0_1);
                        float2 _f2_117 = make_float2(s1_1, s1_1);
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
                        float2 _f2_118 = make_float2(acc_1[0], acc_1[1]);
                        float2 _f2_119 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                        float2 r_15 = fma_f32x2_rn_noftz(_f2_119, _f2_116, _f2_118);
                        acc_1[0] = r_15.x;
                        acc_1[1] = r_15.y;
                        float2 _f2_120 = make_float2(acc_1[2], acc_1[3]);
                        float2 _f2_121 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                        float2 r_0_1 = fma_f32x2_rn_noftz(_f2_121, _f2_116, _f2_120);
                        acc_1[2] = r_0_1.x;
                        acc_1[3] = r_0_1.y;
                        float2 _f2_122 = make_float2(acc_1[4], acc_1[5]);
                        float2 _f2_123 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                        float2 r_1_1 = fma_f32x2_rn_noftz(_f2_123, _f2_116, _f2_122);
                        acc_1[4] = r_1_1.x;
                        acc_1[5] = r_1_1.y;
                        float2 _f2_124 = make_float2(acc_1[6], acc_1[7]);
                        float2 _f2_125 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                        float2 r_2_1 = fma_f32x2_rn_noftz(_f2_125, _f2_116, _f2_124);
                        acc_1[6] = r_2_1.x;
                        acc_1[7] = r_2_1.y;
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
                        float2 _f2_126 = make_float2(acc_1[8], acc_1[9]);
                        float2 _f2_127 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                        float2 r_3_1 = fma_f32x2_rn_noftz(_f2_127, _f2_116, _f2_126);
                        acc_1[8] = r_3_1.x;
                        acc_1[9] = r_3_1.y;
                        float2 _f2_128 = make_float2(acc_1[10], acc_1[11]);
                        float2 _f2_129 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                        float2 r_4_1 = fma_f32x2_rn_noftz(_f2_129, _f2_116, _f2_128);
                        acc_1[10] = r_4_1.x;
                        acc_1[11] = r_4_1.y;
                        float2 _f2_130 = make_float2(acc_1[12], acc_1[13]);
                        float2 _f2_131 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                        float2 r_5_1 = fma_f32x2_rn_noftz(_f2_131, _f2_116, _f2_130);
                        acc_1[12] = r_5_1.x;
                        acc_1[13] = r_5_1.y;
                        float2 _f2_132 = make_float2(acc_1[14], acc_1[15]);
                        float2 _f2_133 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                        float2 r_6_1 = fma_f32x2_rn_noftz(_f2_133, _f2_116, _f2_132);
                        acc_1[14] = r_6_1.x;
                        acc_1[15] = r_6_1.y;
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
                        float2 _f2_134 = make_float2(acc_1[16], acc_1[17]);
                        float2 _f2_135 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                        float2 r_7_1 = fma_f32x2_rn_noftz(_f2_135, _f2_117, _f2_134);
                        acc_1[16] = r_7_1.x;
                        acc_1[17] = r_7_1.y;
                        float2 _f2_136 = make_float2(acc_1[18], acc_1[19]);
                        float2 _f2_137 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                        float2 r_8_1 = fma_f32x2_rn_noftz(_f2_137, _f2_117, _f2_136);
                        acc_1[18] = r_8_1.x;
                        acc_1[19] = r_8_1.y;
                        float2 _f2_138 = make_float2(acc_1[20], acc_1[21]);
                        float2 _f2_139 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                        float2 r_9_1 = fma_f32x2_rn_noftz(_f2_139, _f2_117, _f2_138);
                        acc_1[20] = r_9_1.x;
                        acc_1[21] = r_9_1.y;
                        float2 _f2_140 = make_float2(acc_1[22], acc_1[23]);
                        float2 _f2_141 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                        float2 r_10_1 = fma_f32x2_rn_noftz(_f2_141, _f2_117, _f2_140);
                        acc_1[22] = r_10_1.x;
                        acc_1[23] = r_10_1.y;
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
                        float2 _f2_142 = make_float2(acc_1[24], acc_1[25]);
                        float2 _f2_143 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                        float2 r_11_1 = fma_f32x2_rn_noftz(_f2_143, _f2_117, _f2_142);
                        acc_1[24] = r_11_1.x;
                        acc_1[25] = r_11_1.y;
                        float2 _f2_144 = make_float2(acc_1[26], acc_1[27]);
                        float2 _f2_145 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                        float2 r_12_1 = fma_f32x2_rn_noftz(_f2_145, _f2_117, _f2_144);
                        acc_1[26] = r_12_1.x;
                        acc_1[27] = r_12_1.y;
                        float2 _f2_146 = make_float2(acc_1[28], acc_1[29]);
                        float2 _f2_147 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                        float2 r_13_1 = fma_f32x2_rn_noftz(_f2_147, _f2_117, _f2_146);
                        acc_1[28] = r_13_1.x;
                        acc_1[29] = r_13_1.y;
                        float2 _f2_148 = make_float2(acc_1[30], acc_1[31]);
                        float2 _f2_149 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                        float2 r_14_1 = fma_f32x2_rn_noftz(_f2_149, _f2_117, _f2_148);
                        acc_1[30] = r_14_1.x;
                        acc_1[31] = r_14_1.y;
                        float2 _fp8x2_decode_3;
                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                            "mov.b32 {lo, hi}, pair;\n"
                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                            : "=f"(_fp8x2_decode_3.x), "=f"(_fp8x2_decode_3.y) : "h"((uint16_t)own_sc_1));
                        float _mul_6;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_fp8x2_decode_3.x), "f"(gs_me));
                        float s0_15_1 = _mul_6;
                        float _mul_7;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_fp8x2_decode_3.y), "f"(gs_me));
                        float s1_16_1 = _mul_7;
                        float2 _f2_150 = make_float2(s0_15_1, s0_15_1);
                        float2 _f2_151 = make_float2(s1_16_1, s1_16_1);
                        float _unpack_fp4_12[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_1[0]));
                            (&_unpack_fp4_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_12[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_12[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_12[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_12[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_12[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_12[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_12[0]), "=f"(_unpack_fp4_12[1]), "=f"(_unpack_fp4_12[2]), "=f"(_unpack_fp4_12[3]), "=f"(_unpack_fp4_12[4]), "=f"(_unpack_fp4_12[5]), "=f"(_unpack_fp4_12[6]), "=f"(_unpack_fp4_12[7]) : "r"(own_1[0]));
                            #endif
                        }
                        float2 _f2_152 = make_float2(acc_1[0], acc_1[1]);
                        float2 _f2_153 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                        float2 r_17_1 = fma_f32x2_rn_noftz(_f2_153, _f2_150, _f2_152);
                        acc_1[0] = r_17_1.x;
                        acc_1[1] = r_17_1.y;
                        float2 _f2_154 = make_float2(acc_1[2], acc_1[3]);
                        float2 _f2_155 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                        float2 r_18_1 = fma_f32x2_rn_noftz(_f2_155, _f2_150, _f2_154);
                        acc_1[2] = r_18_1.x;
                        acc_1[3] = r_18_1.y;
                        float2 _f2_156 = make_float2(acc_1[4], acc_1[5]);
                        float2 _f2_157 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                        float2 r_19_1 = fma_f32x2_rn_noftz(_f2_157, _f2_150, _f2_156);
                        acc_1[4] = r_19_1.x;
                        acc_1[5] = r_19_1.y;
                        float2 _f2_158 = make_float2(acc_1[6], acc_1[7]);
                        float2 _f2_159 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                        float2 r_20_1 = fma_f32x2_rn_noftz(_f2_159, _f2_150, _f2_158);
                        acc_1[6] = r_20_1.x;
                        acc_1[7] = r_20_1.y;
                        float _unpack_fp4_13[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_1[1 + 0]));
                            (&_unpack_fp4_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_13[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_13[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_13[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_13[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_13[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_13[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_13[0]), "=f"(_unpack_fp4_13[1]), "=f"(_unpack_fp4_13[2]), "=f"(_unpack_fp4_13[3]), "=f"(_unpack_fp4_13[4]), "=f"(_unpack_fp4_13[5]), "=f"(_unpack_fp4_13[6]), "=f"(_unpack_fp4_13[7]) : "r"(own_1[1 + 0]));
                            #endif
                        }
                        float2 _f2_160 = make_float2(acc_1[8], acc_1[9]);
                        float2 _f2_161 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                        float2 r_21_1 = fma_f32x2_rn_noftz(_f2_161, _f2_150, _f2_160);
                        acc_1[8] = r_21_1.x;
                        acc_1[9] = r_21_1.y;
                        float2 _f2_162 = make_float2(acc_1[10], acc_1[11]);
                        float2 _f2_163 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                        float2 r_22_1 = fma_f32x2_rn_noftz(_f2_163, _f2_150, _f2_162);
                        acc_1[10] = r_22_1.x;
                        acc_1[11] = r_22_1.y;
                        float2 _f2_164 = make_float2(acc_1[12], acc_1[13]);
                        float2 _f2_165 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                        float2 r_23_1 = fma_f32x2_rn_noftz(_f2_165, _f2_150, _f2_164);
                        acc_1[12] = r_23_1.x;
                        acc_1[13] = r_23_1.y;
                        float2 _f2_166 = make_float2(acc_1[14], acc_1[15]);
                        float2 _f2_167 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                        float2 r_24_1 = fma_f32x2_rn_noftz(_f2_167, _f2_150, _f2_166);
                        acc_1[14] = r_24_1.x;
                        acc_1[15] = r_24_1.y;
                        float _unpack_fp4_14[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_1[2 + 0]));
                            (&_unpack_fp4_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_14[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_14[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_14[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_14[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_14[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_14[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_14[0]), "=f"(_unpack_fp4_14[1]), "=f"(_unpack_fp4_14[2]), "=f"(_unpack_fp4_14[3]), "=f"(_unpack_fp4_14[4]), "=f"(_unpack_fp4_14[5]), "=f"(_unpack_fp4_14[6]), "=f"(_unpack_fp4_14[7]) : "r"(own_1[2 + 0]));
                            #endif
                        }
                        float2 _f2_168 = make_float2(acc_1[16], acc_1[17]);
                        float2 _f2_169 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                        float2 r_25_1 = fma_f32x2_rn_noftz(_f2_169, _f2_151, _f2_168);
                        acc_1[16] = r_25_1.x;
                        acc_1[17] = r_25_1.y;
                        float2 _f2_170 = make_float2(acc_1[18], acc_1[19]);
                        float2 _f2_171 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                        float2 r_26_1 = fma_f32x2_rn_noftz(_f2_171, _f2_151, _f2_170);
                        acc_1[18] = r_26_1.x;
                        acc_1[19] = r_26_1.y;
                        float2 _f2_172 = make_float2(acc_1[20], acc_1[21]);
                        float2 _f2_173 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                        float2 r_27_1 = fma_f32x2_rn_noftz(_f2_173, _f2_151, _f2_172);
                        acc_1[20] = r_27_1.x;
                        acc_1[21] = r_27_1.y;
                        float2 _f2_174 = make_float2(acc_1[22], acc_1[23]);
                        float2 _f2_175 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                        float2 r_28_1 = fma_f32x2_rn_noftz(_f2_175, _f2_151, _f2_174);
                        acc_1[22] = r_28_1.x;
                        acc_1[23] = r_28_1.y;
                        float _unpack_fp4_15[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_1[3 + 0]));
                            (&_unpack_fp4_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_15[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_15[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_15[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_15[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_15[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_15[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_15[0]), "=f"(_unpack_fp4_15[1]), "=f"(_unpack_fp4_15[2]), "=f"(_unpack_fp4_15[3]), "=f"(_unpack_fp4_15[4]), "=f"(_unpack_fp4_15[5]), "=f"(_unpack_fp4_15[6]), "=f"(_unpack_fp4_15[7]) : "r"(own_1[3 + 0]));
                            #endif
                        }
                        float2 _f2_176 = make_float2(acc_1[24], acc_1[25]);
                        float2 _f2_177 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                        float2 r_29_1 = fma_f32x2_rn_noftz(_f2_177, _f2_151, _f2_176);
                        acc_1[24] = r_29_1.x;
                        acc_1[25] = r_29_1.y;
                        float2 _f2_178 = make_float2(acc_1[26], acc_1[27]);
                        float2 _f2_179 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                        float2 r_30_1 = fma_f32x2_rn_noftz(_f2_179, _f2_151, _f2_178);
                        acc_1[26] = r_30_1.x;
                        acc_1[27] = r_30_1.y;
                        float2 _f2_180 = make_float2(acc_1[28], acc_1[29]);
                        float2 _f2_181 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                        float2 r_31_1 = fma_f32x2_rn_noftz(_f2_181, _f2_151, _f2_180);
                        acc_1[28] = r_31_1.x;
                        acc_1[29] = r_31_1.y;
                        float2 _f2_182 = make_float2(acc_1[30], acc_1[31]);
                        float2 _f2_183 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                        float2 r_32_1 = fma_f32x2_rn_noftz(_f2_183, _f2_151, _f2_182);
                        acc_1[30] = r_32_1.x;
                        acc_1[31] = r_32_1.y;
                        unsigned int t_1[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 32)) + (unsigned int)(16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 32)) + (unsigned int)(16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 32)) + (unsigned int)(16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 32)) + (unsigned int)(16 * (f ^ 3))));
                        unsigned int s_1[16];
                        unsigned int r_33_1[16];
                        s_1[0] = (((f & 1) != 0) ? t_1[4] : t_1[0]);
                        s_1[4] = (((f & 1) != 0) ? t_1[0] : t_1[4]);
                        s_1[8] = (((f & 1) != 0) ? t_1[12] : t_1[8]);
                        s_1[12] = (((f & 1) != 0) ? t_1[8] : t_1[12]);
                        s_1[1] = (((f & 1) != 0) ? t_1[5] : t_1[1]);
                        s_1[5] = (((f & 1) != 0) ? t_1[1] : t_1[5]);
                        s_1[9] = (((f & 1) != 0) ? t_1[13] : t_1[9]);
                        s_1[13] = (((f & 1) != 0) ? t_1[9] : t_1[13]);
                        s_1[2] = (((f & 1) != 0) ? t_1[6] : t_1[2]);
                        s_1[6] = (((f & 1) != 0) ? t_1[2] : t_1[6]);
                        s_1[10] = (((f & 1) != 0) ? t_1[14] : t_1[10]);
                        s_1[14] = (((f & 1) != 0) ? t_1[10] : t_1[14]);
                        s_1[3] = (((f & 1) != 0) ? t_1[7] : t_1[3]);
                        s_1[7] = (((f & 1) != 0) ? t_1[3] : t_1[7]);
                        s_1[11] = (((f & 1) != 0) ? t_1[15] : t_1[11]);
                        s_1[15] = (((f & 1) != 0) ? t_1[11] : t_1[15]);
                        r_33_1[0] = (((f & 2) != 0) ? s_1[8] : s_1[0]);
                        r_33_1[4] = (((f & 2) != 0) ? s_1[12] : s_1[4]);
                        r_33_1[8] = (((f & 2) != 0) ? s_1[0] : s_1[8]);
                        r_33_1[12] = (((f & 2) != 0) ? s_1[4] : s_1[12]);
                        r_33_1[1] = (((f & 2) != 0) ? s_1[9] : s_1[1]);
                        r_33_1[5] = (((f & 2) != 0) ? s_1[13] : s_1[5]);
                        r_33_1[9] = (((f & 2) != 0) ? s_1[1] : s_1[9]);
                        r_33_1[13] = (((f & 2) != 0) ? s_1[5] : s_1[13]);
                        r_33_1[2] = (((f & 2) != 0) ? s_1[10] : s_1[2]);
                        r_33_1[6] = (((f & 2) != 0) ? s_1[14] : s_1[6]);
                        r_33_1[10] = (((f & 2) != 0) ? s_1[2] : s_1[10]);
                        r_33_1[14] = (((f & 2) != 0) ? s_1[6] : s_1[14]);
                        r_33_1[3] = (((f & 2) != 0) ? s_1[11] : s_1[3]);
                        r_33_1[7] = (((f & 2) != 0) ? s_1[15] : s_1[7]);
                        r_33_1[11] = (((f & 2) != 0) ? s_1[3] : s_1[11]);
                        r_33_1[15] = (((f & 2) != 0) ? s_1[7] : s_1[15]);
                        float2 _f2_184 = make_float2(acc_1[0], acc_1[1]);
                        float2 _f2_185 = make_float2(__uint_as_float(r_33_1[0] << 16), __uint_as_float(r_33_1[0] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_16;
                        {
                            float2 _packed_f32x2_28_0 = _f2_184;
                            float2 _packed_f32x2_28_1 = _f2_185;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_16)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_28_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_28_1)));
                        }
                        acc_1[0] = _packed_add_f32x2_16.x;
                        acc_1[1] = _packed_add_f32x2_16.y;
                        float2 _f2_186 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_16, _packed_add_f32x2_16, _f2_186).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_16, _packed_add_f32x2_16, _f2_186).y;
                        float2 _f2_187 = make_float2(acc_1[2], acc_1[3]);
                        float2 _f2_188 = make_float2(__uint_as_float(r_33_1[1] << 16), __uint_as_float(r_33_1[1] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_17;
                        {
                            float2 _packed_f32x2_29_0 = _f2_187;
                            float2 _packed_f32x2_29_1 = _f2_188;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_17)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_29_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_29_1)));
                        }
                        acc_1[2] = _packed_add_f32x2_17.x;
                        acc_1[3] = _packed_add_f32x2_17.y;
                        float2 _f2_189 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_17, _packed_add_f32x2_17, _f2_189).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_17, _packed_add_f32x2_17, _f2_189).y;
                        float2 _f2_190 = make_float2(acc_1[4], acc_1[5]);
                        float2 _f2_191 = make_float2(__uint_as_float(r_33_1[2] << 16), __uint_as_float(r_33_1[2] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_18;
                        {
                            float2 _packed_f32x2_30_0 = _f2_190;
                            float2 _packed_f32x2_30_1 = _f2_191;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_18)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_30_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_30_1)));
                        }
                        acc_1[4] = _packed_add_f32x2_18.x;
                        acc_1[5] = _packed_add_f32x2_18.y;
                        float2 _f2_192 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_18, _packed_add_f32x2_18, _f2_192).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_18, _packed_add_f32x2_18, _f2_192).y;
                        float2 _f2_193 = make_float2(acc_1[6], acc_1[7]);
                        float2 _f2_194 = make_float2(__uint_as_float(r_33_1[3] << 16), __uint_as_float(r_33_1[3] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_19;
                        {
                            float2 _packed_f32x2_31_0 = _f2_193;
                            float2 _packed_f32x2_31_1 = _f2_194;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_19)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_31_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_31_1)));
                        }
                        acc_1[6] = _packed_add_f32x2_19.x;
                        acc_1[7] = _packed_add_f32x2_19.y;
                        float2 _f2_195 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_19, _packed_add_f32x2_19, _f2_195).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_19, _packed_add_f32x2_19, _f2_195).y;
                        float2 _f2_196 = make_float2(acc_1[8], acc_1[9]);
                        float2 _f2_197 = make_float2(__uint_as_float(r_33_1[4] << 16), __uint_as_float(r_33_1[4] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_20;
                        {
                            float2 _packed_f32x2_32_0 = _f2_196;
                            float2 _packed_f32x2_32_1 = _f2_197;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_20)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_32_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_32_1)));
                        }
                        acc_1[8] = _packed_add_f32x2_20.x;
                        acc_1[9] = _packed_add_f32x2_20.y;
                        float2 _f2_198 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_20, _packed_add_f32x2_20, _f2_198).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_20, _packed_add_f32x2_20, _f2_198).y;
                        float2 _f2_199 = make_float2(acc_1[10], acc_1[11]);
                        float2 _f2_200 = make_float2(__uint_as_float(r_33_1[5] << 16), __uint_as_float(r_33_1[5] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_21;
                        {
                            float2 _packed_f32x2_33_0 = _f2_199;
                            float2 _packed_f32x2_33_1 = _f2_200;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_21)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_33_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_33_1)));
                        }
                        acc_1[10] = _packed_add_f32x2_21.x;
                        acc_1[11] = _packed_add_f32x2_21.y;
                        float2 _f2_201 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_21, _packed_add_f32x2_21, _f2_201).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_21, _packed_add_f32x2_21, _f2_201).y;
                        float2 _f2_202 = make_float2(acc_1[12], acc_1[13]);
                        float2 _f2_203 = make_float2(__uint_as_float(r_33_1[6] << 16), __uint_as_float(r_33_1[6] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_22;
                        {
                            float2 _packed_f32x2_34_0 = _f2_202;
                            float2 _packed_f32x2_34_1 = _f2_203;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_22)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_34_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_34_1)));
                        }
                        acc_1[12] = _packed_add_f32x2_22.x;
                        acc_1[13] = _packed_add_f32x2_22.y;
                        float2 _f2_204 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_22, _packed_add_f32x2_22, _f2_204).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_22, _packed_add_f32x2_22, _f2_204).y;
                        float2 _f2_205 = make_float2(acc_1[14], acc_1[15]);
                        float2 _f2_206 = make_float2(__uint_as_float(r_33_1[7] << 16), __uint_as_float(r_33_1[7] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_23;
                        {
                            float2 _packed_f32x2_35_0 = _f2_205;
                            float2 _packed_f32x2_35_1 = _f2_206;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_23)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_35_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_35_1)));
                        }
                        acc_1[14] = _packed_add_f32x2_23.x;
                        acc_1[15] = _packed_add_f32x2_23.y;
                        float2 _f2_207 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_23, _packed_add_f32x2_23, _f2_207).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_23, _packed_add_f32x2_23, _f2_207).y;
                        float2 _f2_208 = make_float2(acc_1[16], acc_1[17]);
                        float2 _f2_209 = make_float2(__uint_as_float(r_33_1[8] << 16), __uint_as_float(r_33_1[8] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_24;
                        {
                            float2 _packed_f32x2_36_0 = _f2_208;
                            float2 _packed_f32x2_36_1 = _f2_209;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_24)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_36_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_36_1)));
                        }
                        acc_1[16] = _packed_add_f32x2_24.x;
                        acc_1[17] = _packed_add_f32x2_24.y;
                        float2 _f2_210 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_24, _packed_add_f32x2_24, _f2_210).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_24, _packed_add_f32x2_24, _f2_210).y;
                        float2 _f2_211 = make_float2(acc_1[18], acc_1[19]);
                        float2 _f2_212 = make_float2(__uint_as_float(r_33_1[9] << 16), __uint_as_float(r_33_1[9] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_25;
                        {
                            float2 _packed_f32x2_37_0 = _f2_211;
                            float2 _packed_f32x2_37_1 = _f2_212;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_25)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_37_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_37_1)));
                        }
                        acc_1[18] = _packed_add_f32x2_25.x;
                        acc_1[19] = _packed_add_f32x2_25.y;
                        float2 _f2_213 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_25, _packed_add_f32x2_25, _f2_213).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_25, _packed_add_f32x2_25, _f2_213).y;
                        float2 _f2_214 = make_float2(acc_1[20], acc_1[21]);
                        float2 _f2_215 = make_float2(__uint_as_float(r_33_1[10] << 16), __uint_as_float(r_33_1[10] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_26;
                        {
                            float2 _packed_f32x2_38_0 = _f2_214;
                            float2 _packed_f32x2_38_1 = _f2_215;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_26)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_38_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_38_1)));
                        }
                        acc_1[20] = _packed_add_f32x2_26.x;
                        acc_1[21] = _packed_add_f32x2_26.y;
                        float2 _f2_216 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_26, _packed_add_f32x2_26, _f2_216).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_26, _packed_add_f32x2_26, _f2_216).y;
                        float2 _f2_217 = make_float2(acc_1[22], acc_1[23]);
                        float2 _f2_218 = make_float2(__uint_as_float(r_33_1[11] << 16), __uint_as_float(r_33_1[11] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_27;
                        {
                            float2 _packed_f32x2_39_0 = _f2_217;
                            float2 _packed_f32x2_39_1 = _f2_218;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_27)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_39_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_39_1)));
                        }
                        acc_1[22] = _packed_add_f32x2_27.x;
                        acc_1[23] = _packed_add_f32x2_27.y;
                        float2 _f2_219 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_27, _packed_add_f32x2_27, _f2_219).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_27, _packed_add_f32x2_27, _f2_219).y;
                        float2 _f2_220 = make_float2(acc_1[24], acc_1[25]);
                        float2 _f2_221 = make_float2(__uint_as_float(r_33_1[12] << 16), __uint_as_float(r_33_1[12] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_28;
                        {
                            float2 _packed_f32x2_40_0 = _f2_220;
                            float2 _packed_f32x2_40_1 = _f2_221;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_28)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_40_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_40_1)));
                        }
                        acc_1[24] = _packed_add_f32x2_28.x;
                        acc_1[25] = _packed_add_f32x2_28.y;
                        float2 _f2_222 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_28, _packed_add_f32x2_28, _f2_222).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_28, _packed_add_f32x2_28, _f2_222).y;
                        float2 _f2_223 = make_float2(acc_1[26], acc_1[27]);
                        float2 _f2_224 = make_float2(__uint_as_float(r_33_1[13] << 16), __uint_as_float(r_33_1[13] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_29;
                        {
                            float2 _packed_f32x2_41_0 = _f2_223;
                            float2 _packed_f32x2_41_1 = _f2_224;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_29)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_41_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_41_1)));
                        }
                        acc_1[26] = _packed_add_f32x2_29.x;
                        acc_1[27] = _packed_add_f32x2_29.y;
                        float2 _f2_225 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_29, _packed_add_f32x2_29, _f2_225).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_29, _packed_add_f32x2_29, _f2_225).y;
                        float2 _f2_226 = make_float2(acc_1[28], acc_1[29]);
                        float2 _f2_227 = make_float2(__uint_as_float(r_33_1[14] << 16), __uint_as_float(r_33_1[14] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_30;
                        {
                            float2 _packed_f32x2_42_0 = _f2_226;
                            float2 _packed_f32x2_42_1 = _f2_227;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_30)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_42_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_42_1)));
                        }
                        acc_1[28] = _packed_add_f32x2_30.x;
                        acc_1[29] = _packed_add_f32x2_30.y;
                        float2 _f2_228 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_30, _packed_add_f32x2_30, _f2_228).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_30, _packed_add_f32x2_30, _f2_228).y;
                        float2 _f2_229 = make_float2(acc_1[30], acc_1[31]);
                        float2 _f2_230 = make_float2(__uint_as_float(r_33_1[15] << 16), __uint_as_float(r_33_1[15] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_31;
                        {
                            float2 _packed_f32x2_43_0 = _f2_229;
                            float2 _packed_f32x2_43_1 = _f2_230;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_31)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_43_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_43_1)));
                        }
                        acc_1[30] = _packed_add_f32x2_31.x;
                        acc_1[31] = _packed_add_f32x2_31.y;
                        float2 _f2_231 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_31, _packed_add_f32x2_31, _f2_231).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_31, _packed_add_f32x2_31, _f2_231).y;
                        uint32_t acc_bf16_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_1[_lp*2 + 0], acc_1[_lp*2+1 + 0]));
                            acc_bf16_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 0)))), "r"(acc_bf16_1[0]), "r"(acc_bf16_1[1]), "r"(acc_bf16_1[2]), "r"(acc_bf16_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 1)))), "r"(acc_bf16_1[4]), "r"(acc_bf16_1[5]), "r"(acc_bf16_1[6]), "r"(acc_bf16_1[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 2)))), "r"(acc_bf16_1[8]), "r"(acc_bf16_1[9]), "r"(acc_bf16_1[10]), "r"(acc_bf16_1[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 3)))), "r"(acc_bf16_1[12]), "r"(acc_bf16_1[13]), "r"(acc_bf16_1[14]), "r"(acc_bf16_1[15]) : "memory");
                        __syncwarp();
                        int c_1 = lane_3;
                        int l_1 = c_1 >> 2;
                        unsigned int w4_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_1 + (c_1 & 3 ^ l_1 >> 1 & 3)))));
                        long long idx_1 = drow * 3072 + (long long)(4 * (128 + c_1));
                        {
                            int4 _iv4 = make_int4(w4_1[0 + 0], w4_1[0 + 1], w4_1[0 + 2], w4_1[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_1)) = _iv4;
                        }
                        int c_34_1 = lane_3 + 32;
                        int l_35_1 = c_34_1 >> 2;
                        unsigned int w4_36_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_36_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_1[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_35_1 + (c_34_1 & 3 ^ l_35_1 >> 1 & 3)))));
                        long long idx_37_1 = drow * 3072 + (long long)(4 * (128 + c_34_1));
                        {
                            int4 _iv4 = make_int4(w4_36_1[0 + 0], w4_36_1[0 + 1], w4_36_1[0 + 2], w4_36_1[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_37_1)) = _iv4;
                        }
                        int c_38_1 = lane_3 + 64;
                        int l_39_1 = c_38_1 >> 2;
                        unsigned int w4_40_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_40_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_1[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_39_1 + (c_38_1 & 3 ^ l_39_1 >> 1 & 3)))));
                        long long idx_41_1 = drow * 3072 + (long long)(4 * (128 + c_38_1));
                        {
                            int4 _iv4 = make_int4(w4_40_1[0 + 0], w4_40_1[0 + 1], w4_40_1[0 + 2], w4_40_1[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_41_1)) = _iv4;
                        }
                        int c_42_1 = lane_3 + 96;
                        int l_43_1 = c_42_1 >> 2;
                        unsigned int w4_44_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_44_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_1[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_43_1 + (c_42_1 & 3 ^ l_43_1 >> 1 & 3)))));
                        long long idx_45_1 = drow * 3072 + (long long)(4 * (128 + c_42_1));
                        {
                            int4 _iv4 = make_int4(w4_44_1[0 + 0], w4_44_1[0 + 1], w4_44_1[0 + 2], w4_44_1[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_45_1)) = _iv4;
                        }
                        __syncwarp();
                        zreg[0] = acc_1[0];
                        zreg[1] = acc_1[1];
                        zreg[2] = acc_1[2];
                        zreg[3] = acc_1[3];
                        zreg[4] = acc_1[4];
                        zreg[5] = acc_1[5];
                        zreg[6] = acc_1[6];
                        zreg[7] = acc_1[7];
                        zreg[8] = acc_1[8];
                        zreg[9] = acc_1[9];
                        zreg[10] = acc_1[10];
                        zreg[11] = acc_1[11];
                        zreg[12] = acc_1[12];
                        zreg[13] = acc_1[13];
                        zreg[14] = acc_1[14];
                        zreg[15] = acc_1[15];
                        zreg[16] = acc_1[16];
                        zreg[17] = acc_1[17];
                        zreg[18] = acc_1[18];
                        zreg[19] = acc_1[19];
                        zreg[20] = acc_1[20];
                        zreg[21] = acc_1[21];
                        zreg[22] = acc_1[22];
                        zreg[23] = acc_1[23];
                        zreg[24] = acc_1[24];
                        zreg[25] = acc_1[25];
                        zreg[26] = acc_1[26];
                        zreg[27] = acc_1[27];
                        zreg[28] = acc_1[28];
                        zreg[29] = acc_1[29];
                        zreg[30] = acc_1[30];
                        zreg[31] = acc_1[31];
                    }
                    {
                        unsigned int d_1 = 0;
                        unsigned int w_0_1 = pf_v[8];
                        d_1 = d_1 | w_0_1 & ~((w_0_1 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_1_1 = pf_v[9];
                        d_1 = d_1 | w_1_1 & ~((w_1_1 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_2_1 = pf_v[10];
                        d_1 = d_1 | w_2_1 & ~((w_2_1 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_3_1 = pf_v[11];
                        d_1 = d_1 | w_3_1 & ~((w_3_1 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int sc_1 = pf_s[2];
                        d_1 = d_1 | ((sc_1 & 255) + 1 | (sc_1 >> 8) + 1) & 256;
                        unsigned int w_4_1 = pf_v[12];
                        d_1 = d_1 | w_4_1 & ~((w_4_1 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_5_1 = pf_v[13];
                        d_1 = d_1 | w_5_1 & ~((w_5_1 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_6_1 = pf_v[14];
                        d_1 = d_1 | w_6_1 & ~((w_6_1 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_7_1 = pf_v[15];
                        d_1 = d_1 | w_7_1 & ~((w_7_1 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int sc_8 = pf_s[3];
                        d_1 = d_1 | ((sc_8 & 255) + 1 | (sc_8 >> 8) + 1) & 256;
                        if (d_1 != 0) {
                            uint32_t _fp4_wire_values_1[8];
                            uint32_t _fp4_wire_scales_1[2];
                            uint16_t _wire_h16_44_0;
                            uint16_t _wire_h16_44_1;
                            uint32_t _wire_vpend_44 = 3u;
                            uint32_t _wire_spend_44 = 3u;
                            do {
                                if (_wire_vpend_44 & 1u) {
                                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[0]), "=r"(_fp4_wire_values_1[1]), "=r"(_fp4_wire_values_1[2]), "=r"(_fp4_wire_values_1[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_3 + 64))))) : "memory");
                                }
                                if (_wire_vpend_44 & 2u) {
                                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[4]), "=r"(_fp4_wire_values_1[5]), "=r"(_fp4_wire_values_1[6]), "=r"(_fp4_wire_values_1[7]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_3 + 96))))) : "memory");
                                }
                                if (_wire_spend_44 & 1u) {
                                    asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_44_0) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_3 + 64))))) : "memory");
                                    _fp4_wire_scales_1[0] = (uint32_t)_wire_h16_44_0;
                                }
                                if (_wire_spend_44 & 2u) {
                                    asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_44_1) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_3 + 96))))) : "memory");
                                    _fp4_wire_scales_1[1] = (uint32_t)_wire_h16_44_1;
                                }
                                if (_wire_vpend_44 & 1u) {
                                    if (!((((_fp4_wire_values_1[0] & ~((_fp4_wire_values_1[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[1] & ~((_fp4_wire_values_1[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[2] & ~((_fp4_wire_values_1[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[3] & ~((_fp4_wire_values_1[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_44 &= ~1u;
                                }
                                if (_wire_vpend_44 & 2u) {
                                    if (!((((_fp4_wire_values_1[4] & ~((_fp4_wire_values_1[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[5] & ~((_fp4_wire_values_1[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[6] & ~((_fp4_wire_values_1[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[7] & ~((_fp4_wire_values_1[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_44 &= ~2u;
                                }
                                if (_wire_spend_44 & 1u) {
                                    if (!(((_fp4_wire_scales_1[0] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_1[0] >> 8) == 0xFFu))) _wire_spend_44 &= ~1u;
                                }
                                if (_wire_spend_44 & 2u) {
                                    if (!(((_fp4_wire_scales_1[1] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_1[1] >> 8) == 0xFFu))) _wire_spend_44 &= ~2u;
                                }
                            } while ((_wire_vpend_44 | _wire_spend_44) != 0u);
                            pf_v[8] = _fp4_wire_values_1[0];
                            pf_v[9] = _fp4_wire_values_1[1];
                            pf_v[10] = _fp4_wire_values_1[2];
                            pf_v[11] = _fp4_wire_values_1[3];
                            pf_s[2] = _fp4_wire_scales_1[0];
                            pf_v[12] = _fp4_wire_values_1[4];
                            pf_v[13] = _fp4_wire_values_1[5];
                            pf_v[14] = _fp4_wire_values_1[6];
                            pf_v[15] = _fp4_wire_values_1[7];
                            pf_s[3] = _fp4_wire_scales_1[1];
                        }
                    }
                    {
                        unsigned int own_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&own_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_2[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * (lane_3 + 64))));
                        unsigned int own_sc_2 = (unsigned int)wsm16[(dob_off + 3072 >> 1) + (lane_3 + 64)];
                        float acc_2[32];
                        acc_2[0] = 0.0f;
                        acc_2[1] = 0.0f;
                        acc_2[2] = 0.0f;
                        acc_2[3] = 0.0f;
                        acc_2[4] = 0.0f;
                        acc_2[5] = 0.0f;
                        acc_2[6] = 0.0f;
                        acc_2[7] = 0.0f;
                        acc_2[8] = 0.0f;
                        acc_2[9] = 0.0f;
                        acc_2[10] = 0.0f;
                        acc_2[11] = 0.0f;
                        acc_2[12] = 0.0f;
                        acc_2[13] = 0.0f;
                        acc_2[14] = 0.0f;
                        acc_2[15] = 0.0f;
                        acc_2[16] = 0.0f;
                        acc_2[17] = 0.0f;
                        acc_2[18] = 0.0f;
                        acc_2[19] = 0.0f;
                        acc_2[20] = 0.0f;
                        acc_2[21] = 0.0f;
                        acc_2[22] = 0.0f;
                        acc_2[23] = 0.0f;
                        acc_2[24] = 0.0f;
                        acc_2[25] = 0.0f;
                        acc_2[26] = 0.0f;
                        acc_2[27] = 0.0f;
                        acc_2[28] = 0.0f;
                        acc_2[29] = 0.0f;
                        acc_2[30] = 0.0f;
                        acc_2[31] = 0.0f;
                        float2 _fp8x2_decode_4;
                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                            "mov.b32 {lo, hi}, pair;\n"
                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                            : "=f"(_fp8x2_decode_4.x), "=f"(_fp8x2_decode_4.y) : "h"((uint16_t)pf_s[2]));
                        float _mul_8;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_fp8x2_decode_4.x), "f"(__uint_as_float(w_7)));
                        float s0_2 = _mul_8;
                        float _mul_9;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_fp8x2_decode_4.y), "f"(__uint_as_float(w_7)));
                        float s1_2 = _mul_9;
                        float2 _f2_232 = make_float2(s0_2, s0_2);
                        float2 _f2_233 = make_float2(s1_2, s1_2);
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
                        float2 _f2_234 = make_float2(acc_2[0], acc_2[1]);
                        float2 _f2_235 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                        float2 r_16 = fma_f32x2_rn_noftz(_f2_235, _f2_232, _f2_234);
                        acc_2[0] = r_16.x;
                        acc_2[1] = r_16.y;
                        float2 _f2_236 = make_float2(acc_2[2], acc_2[3]);
                        float2 _f2_237 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                        float2 r_0_2 = fma_f32x2_rn_noftz(_f2_237, _f2_232, _f2_236);
                        acc_2[2] = r_0_2.x;
                        acc_2[3] = r_0_2.y;
                        float2 _f2_238 = make_float2(acc_2[4], acc_2[5]);
                        float2 _f2_239 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                        float2 r_1_2 = fma_f32x2_rn_noftz(_f2_239, _f2_232, _f2_238);
                        acc_2[4] = r_1_2.x;
                        acc_2[5] = r_1_2.y;
                        float2 _f2_240 = make_float2(acc_2[6], acc_2[7]);
                        float2 _f2_241 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                        float2 r_2_2 = fma_f32x2_rn_noftz(_f2_241, _f2_232, _f2_240);
                        acc_2[6] = r_2_2.x;
                        acc_2[7] = r_2_2.y;
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
                        float2 _f2_242 = make_float2(acc_2[8], acc_2[9]);
                        float2 _f2_243 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                        float2 r_3_2 = fma_f32x2_rn_noftz(_f2_243, _f2_232, _f2_242);
                        acc_2[8] = r_3_2.x;
                        acc_2[9] = r_3_2.y;
                        float2 _f2_244 = make_float2(acc_2[10], acc_2[11]);
                        float2 _f2_245 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                        float2 r_4_2 = fma_f32x2_rn_noftz(_f2_245, _f2_232, _f2_244);
                        acc_2[10] = r_4_2.x;
                        acc_2[11] = r_4_2.y;
                        float2 _f2_246 = make_float2(acc_2[12], acc_2[13]);
                        float2 _f2_247 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                        float2 r_5_2 = fma_f32x2_rn_noftz(_f2_247, _f2_232, _f2_246);
                        acc_2[12] = r_5_2.x;
                        acc_2[13] = r_5_2.y;
                        float2 _f2_248 = make_float2(acc_2[14], acc_2[15]);
                        float2 _f2_249 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                        float2 r_6_2 = fma_f32x2_rn_noftz(_f2_249, _f2_232, _f2_248);
                        acc_2[14] = r_6_2.x;
                        acc_2[15] = r_6_2.y;
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
                        float2 _f2_250 = make_float2(acc_2[16], acc_2[17]);
                        float2 _f2_251 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                        float2 r_7_2 = fma_f32x2_rn_noftz(_f2_251, _f2_233, _f2_250);
                        acc_2[16] = r_7_2.x;
                        acc_2[17] = r_7_2.y;
                        float2 _f2_252 = make_float2(acc_2[18], acc_2[19]);
                        float2 _f2_253 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                        float2 r_8_2 = fma_f32x2_rn_noftz(_f2_253, _f2_233, _f2_252);
                        acc_2[18] = r_8_2.x;
                        acc_2[19] = r_8_2.y;
                        float2 _f2_254 = make_float2(acc_2[20], acc_2[21]);
                        float2 _f2_255 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                        float2 r_9_2 = fma_f32x2_rn_noftz(_f2_255, _f2_233, _f2_254);
                        acc_2[20] = r_9_2.x;
                        acc_2[21] = r_9_2.y;
                        float2 _f2_256 = make_float2(acc_2[22], acc_2[23]);
                        float2 _f2_257 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                        float2 r_10_2 = fma_f32x2_rn_noftz(_f2_257, _f2_233, _f2_256);
                        acc_2[22] = r_10_2.x;
                        acc_2[23] = r_10_2.y;
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
                        float2 _f2_258 = make_float2(acc_2[24], acc_2[25]);
                        float2 _f2_259 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                        float2 r_11_2 = fma_f32x2_rn_noftz(_f2_259, _f2_233, _f2_258);
                        acc_2[24] = r_11_2.x;
                        acc_2[25] = r_11_2.y;
                        float2 _f2_260 = make_float2(acc_2[26], acc_2[27]);
                        float2 _f2_261 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                        float2 r_12_2 = fma_f32x2_rn_noftz(_f2_261, _f2_233, _f2_260);
                        acc_2[26] = r_12_2.x;
                        acc_2[27] = r_12_2.y;
                        float2 _f2_262 = make_float2(acc_2[28], acc_2[29]);
                        float2 _f2_263 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                        float2 r_13_2 = fma_f32x2_rn_noftz(_f2_263, _f2_233, _f2_262);
                        acc_2[28] = r_13_2.x;
                        acc_2[29] = r_13_2.y;
                        float2 _f2_264 = make_float2(acc_2[30], acc_2[31]);
                        float2 _f2_265 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                        float2 r_14_2 = fma_f32x2_rn_noftz(_f2_265, _f2_233, _f2_264);
                        acc_2[30] = r_14_2.x;
                        acc_2[31] = r_14_2.y;
                        float2 _fp8x2_decode_5;
                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                            "mov.b32 {lo, hi}, pair;\n"
                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                            : "=f"(_fp8x2_decode_5.x), "=f"(_fp8x2_decode_5.y) : "h"((uint16_t)own_sc_2));
                        float _mul_10;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(_fp8x2_decode_5.x), "f"(gs_me));
                        float s0_15_2 = _mul_10;
                        float _mul_11;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_fp8x2_decode_5.y), "f"(gs_me));
                        float s1_16_2 = _mul_11;
                        float2 _f2_266 = make_float2(s0_15_2, s0_15_2);
                        float2 _f2_267 = make_float2(s1_16_2, s1_16_2);
                        float _unpack_fp4_20[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_2[0]));
                            (&_unpack_fp4_20[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_20[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_20[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_20[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_20[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_20[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_20[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_20[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_20[0]), "=f"(_unpack_fp4_20[1]), "=f"(_unpack_fp4_20[2]), "=f"(_unpack_fp4_20[3]), "=f"(_unpack_fp4_20[4]), "=f"(_unpack_fp4_20[5]), "=f"(_unpack_fp4_20[6]), "=f"(_unpack_fp4_20[7]) : "r"(own_2[0]));
                            #endif
                        }
                        float2 _f2_268 = make_float2(acc_2[0], acc_2[1]);
                        float2 _f2_269 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                        float2 r_17_2 = fma_f32x2_rn_noftz(_f2_269, _f2_266, _f2_268);
                        acc_2[0] = r_17_2.x;
                        acc_2[1] = r_17_2.y;
                        float2 _f2_270 = make_float2(acc_2[2], acc_2[3]);
                        float2 _f2_271 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                        float2 r_18_2 = fma_f32x2_rn_noftz(_f2_271, _f2_266, _f2_270);
                        acc_2[2] = r_18_2.x;
                        acc_2[3] = r_18_2.y;
                        float2 _f2_272 = make_float2(acc_2[4], acc_2[5]);
                        float2 _f2_273 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                        float2 r_19_2 = fma_f32x2_rn_noftz(_f2_273, _f2_266, _f2_272);
                        acc_2[4] = r_19_2.x;
                        acc_2[5] = r_19_2.y;
                        float2 _f2_274 = make_float2(acc_2[6], acc_2[7]);
                        float2 _f2_275 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                        float2 r_20_2 = fma_f32x2_rn_noftz(_f2_275, _f2_266, _f2_274);
                        acc_2[6] = r_20_2.x;
                        acc_2[7] = r_20_2.y;
                        float _unpack_fp4_21[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_2[1 + 0]));
                            (&_unpack_fp4_21[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_21[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_21[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_21[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_21[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_21[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_21[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_21[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_21[0]), "=f"(_unpack_fp4_21[1]), "=f"(_unpack_fp4_21[2]), "=f"(_unpack_fp4_21[3]), "=f"(_unpack_fp4_21[4]), "=f"(_unpack_fp4_21[5]), "=f"(_unpack_fp4_21[6]), "=f"(_unpack_fp4_21[7]) : "r"(own_2[1 + 0]));
                            #endif
                        }
                        float2 _f2_276 = make_float2(acc_2[8], acc_2[9]);
                        float2 _f2_277 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                        float2 r_21_2 = fma_f32x2_rn_noftz(_f2_277, _f2_266, _f2_276);
                        acc_2[8] = r_21_2.x;
                        acc_2[9] = r_21_2.y;
                        float2 _f2_278 = make_float2(acc_2[10], acc_2[11]);
                        float2 _f2_279 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                        float2 r_22_2 = fma_f32x2_rn_noftz(_f2_279, _f2_266, _f2_278);
                        acc_2[10] = r_22_2.x;
                        acc_2[11] = r_22_2.y;
                        float2 _f2_280 = make_float2(acc_2[12], acc_2[13]);
                        float2 _f2_281 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                        float2 r_23_2 = fma_f32x2_rn_noftz(_f2_281, _f2_266, _f2_280);
                        acc_2[12] = r_23_2.x;
                        acc_2[13] = r_23_2.y;
                        float2 _f2_282 = make_float2(acc_2[14], acc_2[15]);
                        float2 _f2_283 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                        float2 r_24_2 = fma_f32x2_rn_noftz(_f2_283, _f2_266, _f2_282);
                        acc_2[14] = r_24_2.x;
                        acc_2[15] = r_24_2.y;
                        float _unpack_fp4_22[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_2[2 + 0]));
                            (&_unpack_fp4_22[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_22[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_22[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_22[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_22[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_22[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_22[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_22[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_22[0]), "=f"(_unpack_fp4_22[1]), "=f"(_unpack_fp4_22[2]), "=f"(_unpack_fp4_22[3]), "=f"(_unpack_fp4_22[4]), "=f"(_unpack_fp4_22[5]), "=f"(_unpack_fp4_22[6]), "=f"(_unpack_fp4_22[7]) : "r"(own_2[2 + 0]));
                            #endif
                        }
                        float2 _f2_284 = make_float2(acc_2[16], acc_2[17]);
                        float2 _f2_285 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                        float2 r_25_2 = fma_f32x2_rn_noftz(_f2_285, _f2_267, _f2_284);
                        acc_2[16] = r_25_2.x;
                        acc_2[17] = r_25_2.y;
                        float2 _f2_286 = make_float2(acc_2[18], acc_2[19]);
                        float2 _f2_287 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                        float2 r_26_2 = fma_f32x2_rn_noftz(_f2_287, _f2_267, _f2_286);
                        acc_2[18] = r_26_2.x;
                        acc_2[19] = r_26_2.y;
                        float2 _f2_288 = make_float2(acc_2[20], acc_2[21]);
                        float2 _f2_289 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                        float2 r_27_2 = fma_f32x2_rn_noftz(_f2_289, _f2_267, _f2_288);
                        acc_2[20] = r_27_2.x;
                        acc_2[21] = r_27_2.y;
                        float2 _f2_290 = make_float2(acc_2[22], acc_2[23]);
                        float2 _f2_291 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                        float2 r_28_2 = fma_f32x2_rn_noftz(_f2_291, _f2_267, _f2_290);
                        acc_2[22] = r_28_2.x;
                        acc_2[23] = r_28_2.y;
                        float _unpack_fp4_23[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_2[3 + 0]));
                            (&_unpack_fp4_23[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_23[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_23[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_23[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_23[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_23[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_23[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_23[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_23[0]), "=f"(_unpack_fp4_23[1]), "=f"(_unpack_fp4_23[2]), "=f"(_unpack_fp4_23[3]), "=f"(_unpack_fp4_23[4]), "=f"(_unpack_fp4_23[5]), "=f"(_unpack_fp4_23[6]), "=f"(_unpack_fp4_23[7]) : "r"(own_2[3 + 0]));
                            #endif
                        }
                        float2 _f2_292 = make_float2(acc_2[24], acc_2[25]);
                        float2 _f2_293 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                        float2 r_29_2 = fma_f32x2_rn_noftz(_f2_293, _f2_267, _f2_292);
                        acc_2[24] = r_29_2.x;
                        acc_2[25] = r_29_2.y;
                        float2 _f2_294 = make_float2(acc_2[26], acc_2[27]);
                        float2 _f2_295 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                        float2 r_30_2 = fma_f32x2_rn_noftz(_f2_295, _f2_267, _f2_294);
                        acc_2[26] = r_30_2.x;
                        acc_2[27] = r_30_2.y;
                        float2 _f2_296 = make_float2(acc_2[28], acc_2[29]);
                        float2 _f2_297 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                        float2 r_31_2 = fma_f32x2_rn_noftz(_f2_297, _f2_267, _f2_296);
                        acc_2[28] = r_31_2.x;
                        acc_2[29] = r_31_2.y;
                        float2 _f2_298 = make_float2(acc_2[30], acc_2[31]);
                        float2 _f2_299 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                        float2 r_32_2 = fma_f32x2_rn_noftz(_f2_299, _f2_267, _f2_298);
                        acc_2[30] = r_32_2.x;
                        acc_2[31] = r_32_2.y;
                        unsigned int t_2[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 64)) + (unsigned int)(16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 64)) + (unsigned int)(16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 64)) + (unsigned int)(16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 64)) + (unsigned int)(16 * (f ^ 3))));
                        unsigned int s_2[16];
                        unsigned int r_33_2[16];
                        s_2[0] = (((f & 1) != 0) ? t_2[4] : t_2[0]);
                        s_2[4] = (((f & 1) != 0) ? t_2[0] : t_2[4]);
                        s_2[8] = (((f & 1) != 0) ? t_2[12] : t_2[8]);
                        s_2[12] = (((f & 1) != 0) ? t_2[8] : t_2[12]);
                        s_2[1] = (((f & 1) != 0) ? t_2[5] : t_2[1]);
                        s_2[5] = (((f & 1) != 0) ? t_2[1] : t_2[5]);
                        s_2[9] = (((f & 1) != 0) ? t_2[13] : t_2[9]);
                        s_2[13] = (((f & 1) != 0) ? t_2[9] : t_2[13]);
                        s_2[2] = (((f & 1) != 0) ? t_2[6] : t_2[2]);
                        s_2[6] = (((f & 1) != 0) ? t_2[2] : t_2[6]);
                        s_2[10] = (((f & 1) != 0) ? t_2[14] : t_2[10]);
                        s_2[14] = (((f & 1) != 0) ? t_2[10] : t_2[14]);
                        s_2[3] = (((f & 1) != 0) ? t_2[7] : t_2[3]);
                        s_2[7] = (((f & 1) != 0) ? t_2[3] : t_2[7]);
                        s_2[11] = (((f & 1) != 0) ? t_2[15] : t_2[11]);
                        s_2[15] = (((f & 1) != 0) ? t_2[11] : t_2[15]);
                        r_33_2[0] = (((f & 2) != 0) ? s_2[8] : s_2[0]);
                        r_33_2[4] = (((f & 2) != 0) ? s_2[12] : s_2[4]);
                        r_33_2[8] = (((f & 2) != 0) ? s_2[0] : s_2[8]);
                        r_33_2[12] = (((f & 2) != 0) ? s_2[4] : s_2[12]);
                        r_33_2[1] = (((f & 2) != 0) ? s_2[9] : s_2[1]);
                        r_33_2[5] = (((f & 2) != 0) ? s_2[13] : s_2[5]);
                        r_33_2[9] = (((f & 2) != 0) ? s_2[1] : s_2[9]);
                        r_33_2[13] = (((f & 2) != 0) ? s_2[5] : s_2[13]);
                        r_33_2[2] = (((f & 2) != 0) ? s_2[10] : s_2[2]);
                        r_33_2[6] = (((f & 2) != 0) ? s_2[14] : s_2[6]);
                        r_33_2[10] = (((f & 2) != 0) ? s_2[2] : s_2[10]);
                        r_33_2[14] = (((f & 2) != 0) ? s_2[6] : s_2[14]);
                        r_33_2[3] = (((f & 2) != 0) ? s_2[11] : s_2[3]);
                        r_33_2[7] = (((f & 2) != 0) ? s_2[15] : s_2[7]);
                        r_33_2[11] = (((f & 2) != 0) ? s_2[3] : s_2[11]);
                        r_33_2[15] = (((f & 2) != 0) ? s_2[7] : s_2[15]);
                        float2 _f2_300 = make_float2(acc_2[0], acc_2[1]);
                        float2 _f2_301 = make_float2(__uint_as_float(r_33_2[0] << 16), __uint_as_float(r_33_2[0] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_32;
                        {
                            float2 _packed_f32x2_45_0 = _f2_300;
                            float2 _packed_f32x2_45_1 = _f2_301;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_32)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_45_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_45_1)));
                        }
                        acc_2[0] = _packed_add_f32x2_32.x;
                        acc_2[1] = _packed_add_f32x2_32.y;
                        float2 _f2_302 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_32, _packed_add_f32x2_32, _f2_302).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_32, _packed_add_f32x2_32, _f2_302).y;
                        float2 _f2_303 = make_float2(acc_2[2], acc_2[3]);
                        float2 _f2_304 = make_float2(__uint_as_float(r_33_2[1] << 16), __uint_as_float(r_33_2[1] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_33;
                        {
                            float2 _packed_f32x2_46_0 = _f2_303;
                            float2 _packed_f32x2_46_1 = _f2_304;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_33)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_46_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_46_1)));
                        }
                        acc_2[2] = _packed_add_f32x2_33.x;
                        acc_2[3] = _packed_add_f32x2_33.y;
                        float2 _f2_305 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_33, _packed_add_f32x2_33, _f2_305).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_33, _packed_add_f32x2_33, _f2_305).y;
                        float2 _f2_306 = make_float2(acc_2[4], acc_2[5]);
                        float2 _f2_307 = make_float2(__uint_as_float(r_33_2[2] << 16), __uint_as_float(r_33_2[2] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_34;
                        {
                            float2 _packed_f32x2_47_0 = _f2_306;
                            float2 _packed_f32x2_47_1 = _f2_307;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_34)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_47_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_47_1)));
                        }
                        acc_2[4] = _packed_add_f32x2_34.x;
                        acc_2[5] = _packed_add_f32x2_34.y;
                        float2 _f2_308 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_34, _packed_add_f32x2_34, _f2_308).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_34, _packed_add_f32x2_34, _f2_308).y;
                        float2 _f2_309 = make_float2(acc_2[6], acc_2[7]);
                        float2 _f2_310 = make_float2(__uint_as_float(r_33_2[3] << 16), __uint_as_float(r_33_2[3] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_35;
                        {
                            float2 _packed_f32x2_48_0 = _f2_309;
                            float2 _packed_f32x2_48_1 = _f2_310;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_35)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_48_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_48_1)));
                        }
                        acc_2[6] = _packed_add_f32x2_35.x;
                        acc_2[7] = _packed_add_f32x2_35.y;
                        float2 _f2_311 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_35, _packed_add_f32x2_35, _f2_311).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_35, _packed_add_f32x2_35, _f2_311).y;
                        float2 _f2_312 = make_float2(acc_2[8], acc_2[9]);
                        float2 _f2_313 = make_float2(__uint_as_float(r_33_2[4] << 16), __uint_as_float(r_33_2[4] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_36;
                        {
                            float2 _packed_f32x2_49_0 = _f2_312;
                            float2 _packed_f32x2_49_1 = _f2_313;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_36)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_49_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_49_1)));
                        }
                        acc_2[8] = _packed_add_f32x2_36.x;
                        acc_2[9] = _packed_add_f32x2_36.y;
                        float2 _f2_314 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_36, _packed_add_f32x2_36, _f2_314).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_36, _packed_add_f32x2_36, _f2_314).y;
                        float2 _f2_315 = make_float2(acc_2[10], acc_2[11]);
                        float2 _f2_316 = make_float2(__uint_as_float(r_33_2[5] << 16), __uint_as_float(r_33_2[5] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_37;
                        {
                            float2 _packed_f32x2_50_0 = _f2_315;
                            float2 _packed_f32x2_50_1 = _f2_316;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_37)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_50_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_50_1)));
                        }
                        acc_2[10] = _packed_add_f32x2_37.x;
                        acc_2[11] = _packed_add_f32x2_37.y;
                        float2 _f2_317 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_37, _packed_add_f32x2_37, _f2_317).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_37, _packed_add_f32x2_37, _f2_317).y;
                        float2 _f2_318 = make_float2(acc_2[12], acc_2[13]);
                        float2 _f2_319 = make_float2(__uint_as_float(r_33_2[6] << 16), __uint_as_float(r_33_2[6] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_38;
                        {
                            float2 _packed_f32x2_51_0 = _f2_318;
                            float2 _packed_f32x2_51_1 = _f2_319;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_38)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_51_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_51_1)));
                        }
                        acc_2[12] = _packed_add_f32x2_38.x;
                        acc_2[13] = _packed_add_f32x2_38.y;
                        float2 _f2_320 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_38, _packed_add_f32x2_38, _f2_320).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_38, _packed_add_f32x2_38, _f2_320).y;
                        float2 _f2_321 = make_float2(acc_2[14], acc_2[15]);
                        float2 _f2_322 = make_float2(__uint_as_float(r_33_2[7] << 16), __uint_as_float(r_33_2[7] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_39;
                        {
                            float2 _packed_f32x2_52_0 = _f2_321;
                            float2 _packed_f32x2_52_1 = _f2_322;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_39)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_52_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_52_1)));
                        }
                        acc_2[14] = _packed_add_f32x2_39.x;
                        acc_2[15] = _packed_add_f32x2_39.y;
                        float2 _f2_323 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_39, _packed_add_f32x2_39, _f2_323).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_39, _packed_add_f32x2_39, _f2_323).y;
                        float2 _f2_324 = make_float2(acc_2[16], acc_2[17]);
                        float2 _f2_325 = make_float2(__uint_as_float(r_33_2[8] << 16), __uint_as_float(r_33_2[8] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_40;
                        {
                            float2 _packed_f32x2_53_0 = _f2_324;
                            float2 _packed_f32x2_53_1 = _f2_325;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_40)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_53_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_53_1)));
                        }
                        acc_2[16] = _packed_add_f32x2_40.x;
                        acc_2[17] = _packed_add_f32x2_40.y;
                        float2 _f2_326 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_40, _packed_add_f32x2_40, _f2_326).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_40, _packed_add_f32x2_40, _f2_326).y;
                        float2 _f2_327 = make_float2(acc_2[18], acc_2[19]);
                        float2 _f2_328 = make_float2(__uint_as_float(r_33_2[9] << 16), __uint_as_float(r_33_2[9] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_41;
                        {
                            float2 _packed_f32x2_54_0 = _f2_327;
                            float2 _packed_f32x2_54_1 = _f2_328;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_41)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_54_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_54_1)));
                        }
                        acc_2[18] = _packed_add_f32x2_41.x;
                        acc_2[19] = _packed_add_f32x2_41.y;
                        float2 _f2_329 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_41, _packed_add_f32x2_41, _f2_329).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_41, _packed_add_f32x2_41, _f2_329).y;
                        float2 _f2_330 = make_float2(acc_2[20], acc_2[21]);
                        float2 _f2_331 = make_float2(__uint_as_float(r_33_2[10] << 16), __uint_as_float(r_33_2[10] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_42;
                        {
                            float2 _packed_f32x2_55_0 = _f2_330;
                            float2 _packed_f32x2_55_1 = _f2_331;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_42)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_55_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_55_1)));
                        }
                        acc_2[20] = _packed_add_f32x2_42.x;
                        acc_2[21] = _packed_add_f32x2_42.y;
                        float2 _f2_332 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_42, _packed_add_f32x2_42, _f2_332).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_42, _packed_add_f32x2_42, _f2_332).y;
                        float2 _f2_333 = make_float2(acc_2[22], acc_2[23]);
                        float2 _f2_334 = make_float2(__uint_as_float(r_33_2[11] << 16), __uint_as_float(r_33_2[11] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_43;
                        {
                            float2 _packed_f32x2_56_0 = _f2_333;
                            float2 _packed_f32x2_56_1 = _f2_334;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_43)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_56_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_56_1)));
                        }
                        acc_2[22] = _packed_add_f32x2_43.x;
                        acc_2[23] = _packed_add_f32x2_43.y;
                        float2 _f2_335 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_43, _packed_add_f32x2_43, _f2_335).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_43, _packed_add_f32x2_43, _f2_335).y;
                        float2 _f2_336 = make_float2(acc_2[24], acc_2[25]);
                        float2 _f2_337 = make_float2(__uint_as_float(r_33_2[12] << 16), __uint_as_float(r_33_2[12] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_44;
                        {
                            float2 _packed_f32x2_57_0 = _f2_336;
                            float2 _packed_f32x2_57_1 = _f2_337;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_44)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_57_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_57_1)));
                        }
                        acc_2[24] = _packed_add_f32x2_44.x;
                        acc_2[25] = _packed_add_f32x2_44.y;
                        float2 _f2_338 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_44, _packed_add_f32x2_44, _f2_338).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_44, _packed_add_f32x2_44, _f2_338).y;
                        float2 _f2_339 = make_float2(acc_2[26], acc_2[27]);
                        float2 _f2_340 = make_float2(__uint_as_float(r_33_2[13] << 16), __uint_as_float(r_33_2[13] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_45;
                        {
                            float2 _packed_f32x2_58_0 = _f2_339;
                            float2 _packed_f32x2_58_1 = _f2_340;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_45)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_58_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_58_1)));
                        }
                        acc_2[26] = _packed_add_f32x2_45.x;
                        acc_2[27] = _packed_add_f32x2_45.y;
                        float2 _f2_341 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_45, _packed_add_f32x2_45, _f2_341).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_45, _packed_add_f32x2_45, _f2_341).y;
                        float2 _f2_342 = make_float2(acc_2[28], acc_2[29]);
                        float2 _f2_343 = make_float2(__uint_as_float(r_33_2[14] << 16), __uint_as_float(r_33_2[14] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_46;
                        {
                            float2 _packed_f32x2_59_0 = _f2_342;
                            float2 _packed_f32x2_59_1 = _f2_343;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_46)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_59_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_59_1)));
                        }
                        acc_2[28] = _packed_add_f32x2_46.x;
                        acc_2[29] = _packed_add_f32x2_46.y;
                        float2 _f2_344 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_46, _packed_add_f32x2_46, _f2_344).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_46, _packed_add_f32x2_46, _f2_344).y;
                        float2 _f2_345 = make_float2(acc_2[30], acc_2[31]);
                        float2 _f2_346 = make_float2(__uint_as_float(r_33_2[15] << 16), __uint_as_float(r_33_2[15] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_47;
                        {
                            float2 _packed_f32x2_60_0 = _f2_345;
                            float2 _packed_f32x2_60_1 = _f2_346;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_47)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_60_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_60_1)));
                        }
                        acc_2[30] = _packed_add_f32x2_47.x;
                        acc_2[31] = _packed_add_f32x2_47.y;
                        float2 _f2_347 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_47, _packed_add_f32x2_47, _f2_347).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_47, _packed_add_f32x2_47, _f2_347).y;
                        uint32_t acc_bf16_2[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_2[_lp*2 + 0], acc_2[_lp*2+1 + 0]));
                            acc_bf16_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 0)))), "r"(acc_bf16_2[0]), "r"(acc_bf16_2[1]), "r"(acc_bf16_2[2]), "r"(acc_bf16_2[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 1)))), "r"(acc_bf16_2[4]), "r"(acc_bf16_2[5]), "r"(acc_bf16_2[6]), "r"(acc_bf16_2[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 2)))), "r"(acc_bf16_2[8]), "r"(acc_bf16_2[9]), "r"(acc_bf16_2[10]), "r"(acc_bf16_2[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 3)))), "r"(acc_bf16_2[12]), "r"(acc_bf16_2[13]), "r"(acc_bf16_2[14]), "r"(acc_bf16_2[15]) : "memory");
                        __syncwarp();
                        int c_2 = lane_3;
                        int l_2 = c_2 >> 2;
                        unsigned int w4_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_2 + (c_2 & 3 ^ l_2 >> 1 & 3)))));
                        long long idx_2 = drow * 3072 + (long long)(4 * (256 + c_2));
                        {
                            int4 _iv4 = make_int4(w4_2[0 + 0], w4_2[0 + 1], w4_2[0 + 2], w4_2[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_2)) = _iv4;
                        }
                        int c_34_2 = lane_3 + 32;
                        int l_35_2 = c_34_2 >> 2;
                        unsigned int w4_36_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_36_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_2[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_35_2 + (c_34_2 & 3 ^ l_35_2 >> 1 & 3)))));
                        long long idx_37_2 = drow * 3072 + (long long)(4 * (256 + c_34_2));
                        {
                            int4 _iv4 = make_int4(w4_36_2[0 + 0], w4_36_2[0 + 1], w4_36_2[0 + 2], w4_36_2[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_37_2)) = _iv4;
                        }
                        int c_38_2 = lane_3 + 64;
                        int l_39_2 = c_38_2 >> 2;
                        unsigned int w4_40_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_40_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_2[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_39_2 + (c_38_2 & 3 ^ l_39_2 >> 1 & 3)))));
                        long long idx_41_2 = drow * 3072 + (long long)(4 * (256 + c_38_2));
                        {
                            int4 _iv4 = make_int4(w4_40_2[0 + 0], w4_40_2[0 + 1], w4_40_2[0 + 2], w4_40_2[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_41_2)) = _iv4;
                        }
                        int c_42_2 = lane_3 + 96;
                        int l_43_2 = c_42_2 >> 2;
                        unsigned int w4_44_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_44_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_2[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_43_2 + (c_42_2 & 3 ^ l_43_2 >> 1 & 3)))));
                        long long idx_45_2 = drow * 3072 + (long long)(4 * (256 + c_42_2));
                        {
                            int4 _iv4 = make_int4(w4_44_2[0 + 0], w4_44_2[0 + 1], w4_44_2[0 + 2], w4_44_2[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_45_2)) = _iv4;
                        }
                        __syncwarp();
                        zreg_b[0] = acc_2[0];
                        zreg_b[1] = acc_2[1];
                        zreg_b[2] = acc_2[2];
                        zreg_b[3] = acc_2[3];
                        zreg_b[4] = acc_2[4];
                        zreg_b[5] = acc_2[5];
                        zreg_b[6] = acc_2[6];
                        zreg_b[7] = acc_2[7];
                        zreg_b[8] = acc_2[8];
                        zreg_b[9] = acc_2[9];
                        zreg_b[10] = acc_2[10];
                        zreg_b[11] = acc_2[11];
                        zreg_b[12] = acc_2[12];
                        zreg_b[13] = acc_2[13];
                        zreg_b[14] = acc_2[14];
                        zreg_b[15] = acc_2[15];
                        zreg_b[16] = acc_2[16];
                        zreg_b[17] = acc_2[17];
                        zreg_b[18] = acc_2[18];
                        zreg_b[19] = acc_2[19];
                        zreg_b[20] = acc_2[20];
                        zreg_b[21] = acc_2[21];
                        zreg_b[22] = acc_2[22];
                        zreg_b[23] = acc_2[23];
                        zreg_b[24] = acc_2[24];
                        zreg_b[25] = acc_2[25];
                        zreg_b[26] = acc_2[26];
                        zreg_b[27] = acc_2[27];
                        zreg_b[28] = acc_2[28];
                        zreg_b[29] = acc_2[29];
                        zreg_b[30] = acc_2[30];
                        zreg_b[31] = acc_2[31];
                    }
                    {
                        unsigned int own_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&own_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_3[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * (lane_3 + 96))));
                        unsigned int own_sc_3 = (unsigned int)wsm16[(dob_off + 3072 >> 1) + (lane_3 + 96)];
                        float acc_3[32];
                        acc_3[0] = 0.0f;
                        acc_3[1] = 0.0f;
                        acc_3[2] = 0.0f;
                        acc_3[3] = 0.0f;
                        acc_3[4] = 0.0f;
                        acc_3[5] = 0.0f;
                        acc_3[6] = 0.0f;
                        acc_3[7] = 0.0f;
                        acc_3[8] = 0.0f;
                        acc_3[9] = 0.0f;
                        acc_3[10] = 0.0f;
                        acc_3[11] = 0.0f;
                        acc_3[12] = 0.0f;
                        acc_3[13] = 0.0f;
                        acc_3[14] = 0.0f;
                        acc_3[15] = 0.0f;
                        acc_3[16] = 0.0f;
                        acc_3[17] = 0.0f;
                        acc_3[18] = 0.0f;
                        acc_3[19] = 0.0f;
                        acc_3[20] = 0.0f;
                        acc_3[21] = 0.0f;
                        acc_3[22] = 0.0f;
                        acc_3[23] = 0.0f;
                        acc_3[24] = 0.0f;
                        acc_3[25] = 0.0f;
                        acc_3[26] = 0.0f;
                        acc_3[27] = 0.0f;
                        acc_3[28] = 0.0f;
                        acc_3[29] = 0.0f;
                        acc_3[30] = 0.0f;
                        acc_3[31] = 0.0f;
                        float2 _fp8x2_decode_6;
                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                            "mov.b32 {lo, hi}, pair;\n"
                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                            : "=f"(_fp8x2_decode_6.x), "=f"(_fp8x2_decode_6.y) : "h"((uint16_t)pf_s[3]));
                        float _mul_12;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(_fp8x2_decode_6.x), "f"(__uint_as_float(w_7)));
                        float s0_3 = _mul_12;
                        float _mul_13;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(_fp8x2_decode_6.y), "f"(__uint_as_float(w_7)));
                        float s1_3 = _mul_13;
                        float2 _f2_348 = make_float2(s0_3, s0_3);
                        float2 _f2_349 = make_float2(s1_3, s1_3);
                        float _unpack_fp4_24[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[12 + 0]));
                            (&_unpack_fp4_24[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_24[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_24[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_24[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_24[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_24[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_24[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_24[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_24[0]), "=f"(_unpack_fp4_24[1]), "=f"(_unpack_fp4_24[2]), "=f"(_unpack_fp4_24[3]), "=f"(_unpack_fp4_24[4]), "=f"(_unpack_fp4_24[5]), "=f"(_unpack_fp4_24[6]), "=f"(_unpack_fp4_24[7]) : "r"(pf_v[12 + 0]));
                            #endif
                        }
                        float2 _f2_350 = make_float2(acc_3[0], acc_3[1]);
                        float2 _f2_351 = make_float2(_unpack_fp4_24[0], _unpack_fp4_24[1]);
                        float2 r_34 = fma_f32x2_rn_noftz(_f2_351, _f2_348, _f2_350);
                        acc_3[0] = r_34.x;
                        acc_3[1] = r_34.y;
                        float2 _f2_352 = make_float2(acc_3[2], acc_3[3]);
                        float2 _f2_353 = make_float2(_unpack_fp4_24[2], _unpack_fp4_24[3]);
                        float2 r_0_3 = fma_f32x2_rn_noftz(_f2_353, _f2_348, _f2_352);
                        acc_3[2] = r_0_3.x;
                        acc_3[3] = r_0_3.y;
                        float2 _f2_354 = make_float2(acc_3[4], acc_3[5]);
                        float2 _f2_355 = make_float2(_unpack_fp4_24[4], _unpack_fp4_24[5]);
                        float2 r_1_3 = fma_f32x2_rn_noftz(_f2_355, _f2_348, _f2_354);
                        acc_3[4] = r_1_3.x;
                        acc_3[5] = r_1_3.y;
                        float2 _f2_356 = make_float2(acc_3[6], acc_3[7]);
                        float2 _f2_357 = make_float2(_unpack_fp4_24[6], _unpack_fp4_24[7]);
                        float2 r_2_3 = fma_f32x2_rn_noftz(_f2_357, _f2_348, _f2_356);
                        acc_3[6] = r_2_3.x;
                        acc_3[7] = r_2_3.y;
                        float _unpack_fp4_25[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[13 + 0]));
                            (&_unpack_fp4_25[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_25[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_25[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_25[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_25[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_25[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_25[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_25[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_25[0]), "=f"(_unpack_fp4_25[1]), "=f"(_unpack_fp4_25[2]), "=f"(_unpack_fp4_25[3]), "=f"(_unpack_fp4_25[4]), "=f"(_unpack_fp4_25[5]), "=f"(_unpack_fp4_25[6]), "=f"(_unpack_fp4_25[7]) : "r"(pf_v[13 + 0]));
                            #endif
                        }
                        float2 _f2_358 = make_float2(acc_3[8], acc_3[9]);
                        float2 _f2_359 = make_float2(_unpack_fp4_25[0], _unpack_fp4_25[1]);
                        float2 r_3_3 = fma_f32x2_rn_noftz(_f2_359, _f2_348, _f2_358);
                        acc_3[8] = r_3_3.x;
                        acc_3[9] = r_3_3.y;
                        float2 _f2_360 = make_float2(acc_3[10], acc_3[11]);
                        float2 _f2_361 = make_float2(_unpack_fp4_25[2], _unpack_fp4_25[3]);
                        float2 r_4_3 = fma_f32x2_rn_noftz(_f2_361, _f2_348, _f2_360);
                        acc_3[10] = r_4_3.x;
                        acc_3[11] = r_4_3.y;
                        float2 _f2_362 = make_float2(acc_3[12], acc_3[13]);
                        float2 _f2_363 = make_float2(_unpack_fp4_25[4], _unpack_fp4_25[5]);
                        float2 r_5_3 = fma_f32x2_rn_noftz(_f2_363, _f2_348, _f2_362);
                        acc_3[12] = r_5_3.x;
                        acc_3[13] = r_5_3.y;
                        float2 _f2_364 = make_float2(acc_3[14], acc_3[15]);
                        float2 _f2_365 = make_float2(_unpack_fp4_25[6], _unpack_fp4_25[7]);
                        float2 r_6_3 = fma_f32x2_rn_noftz(_f2_365, _f2_348, _f2_364);
                        acc_3[14] = r_6_3.x;
                        acc_3[15] = r_6_3.y;
                        float _unpack_fp4_26[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[14 + 0]));
                            (&_unpack_fp4_26[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_26[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_26[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_26[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_26[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_26[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_26[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_26[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_26[0]), "=f"(_unpack_fp4_26[1]), "=f"(_unpack_fp4_26[2]), "=f"(_unpack_fp4_26[3]), "=f"(_unpack_fp4_26[4]), "=f"(_unpack_fp4_26[5]), "=f"(_unpack_fp4_26[6]), "=f"(_unpack_fp4_26[7]) : "r"(pf_v[14 + 0]));
                            #endif
                        }
                        float2 _f2_366 = make_float2(acc_3[16], acc_3[17]);
                        float2 _f2_367 = make_float2(_unpack_fp4_26[0], _unpack_fp4_26[1]);
                        float2 r_7_3 = fma_f32x2_rn_noftz(_f2_367, _f2_349, _f2_366);
                        acc_3[16] = r_7_3.x;
                        acc_3[17] = r_7_3.y;
                        float2 _f2_368 = make_float2(acc_3[18], acc_3[19]);
                        float2 _f2_369 = make_float2(_unpack_fp4_26[2], _unpack_fp4_26[3]);
                        float2 r_8_3 = fma_f32x2_rn_noftz(_f2_369, _f2_349, _f2_368);
                        acc_3[18] = r_8_3.x;
                        acc_3[19] = r_8_3.y;
                        float2 _f2_370 = make_float2(acc_3[20], acc_3[21]);
                        float2 _f2_371 = make_float2(_unpack_fp4_26[4], _unpack_fp4_26[5]);
                        float2 r_9_3 = fma_f32x2_rn_noftz(_f2_371, _f2_349, _f2_370);
                        acc_3[20] = r_9_3.x;
                        acc_3[21] = r_9_3.y;
                        float2 _f2_372 = make_float2(acc_3[22], acc_3[23]);
                        float2 _f2_373 = make_float2(_unpack_fp4_26[6], _unpack_fp4_26[7]);
                        float2 r_10_3 = fma_f32x2_rn_noftz(_f2_373, _f2_349, _f2_372);
                        acc_3[22] = r_10_3.x;
                        acc_3[23] = r_10_3.y;
                        float _unpack_fp4_27[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[15 + 0]));
                            (&_unpack_fp4_27[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_27[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_27[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_27[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_27[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_27[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_27[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_27[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_27[0]), "=f"(_unpack_fp4_27[1]), "=f"(_unpack_fp4_27[2]), "=f"(_unpack_fp4_27[3]), "=f"(_unpack_fp4_27[4]), "=f"(_unpack_fp4_27[5]), "=f"(_unpack_fp4_27[6]), "=f"(_unpack_fp4_27[7]) : "r"(pf_v[15 + 0]));
                            #endif
                        }
                        float2 _f2_374 = make_float2(acc_3[24], acc_3[25]);
                        float2 _f2_375 = make_float2(_unpack_fp4_27[0], _unpack_fp4_27[1]);
                        float2 r_11_3 = fma_f32x2_rn_noftz(_f2_375, _f2_349, _f2_374);
                        acc_3[24] = r_11_3.x;
                        acc_3[25] = r_11_3.y;
                        float2 _f2_376 = make_float2(acc_3[26], acc_3[27]);
                        float2 _f2_377 = make_float2(_unpack_fp4_27[2], _unpack_fp4_27[3]);
                        float2 r_12_3 = fma_f32x2_rn_noftz(_f2_377, _f2_349, _f2_376);
                        acc_3[26] = r_12_3.x;
                        acc_3[27] = r_12_3.y;
                        float2 _f2_378 = make_float2(acc_3[28], acc_3[29]);
                        float2 _f2_379 = make_float2(_unpack_fp4_27[4], _unpack_fp4_27[5]);
                        float2 r_13_3 = fma_f32x2_rn_noftz(_f2_379, _f2_349, _f2_378);
                        acc_3[28] = r_13_3.x;
                        acc_3[29] = r_13_3.y;
                        float2 _f2_380 = make_float2(acc_3[30], acc_3[31]);
                        float2 _f2_381 = make_float2(_unpack_fp4_27[6], _unpack_fp4_27[7]);
                        float2 r_14_3 = fma_f32x2_rn_noftz(_f2_381, _f2_349, _f2_380);
                        acc_3[30] = r_14_3.x;
                        acc_3[31] = r_14_3.y;
                        float2 _fp8x2_decode_7;
                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                            "mov.b32 {lo, hi}, pair;\n"
                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                            : "=f"(_fp8x2_decode_7.x), "=f"(_fp8x2_decode_7.y) : "h"((uint16_t)own_sc_3));
                        float _mul_14;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(_fp8x2_decode_7.x), "f"(gs_me));
                        float s0_15_3 = _mul_14;
                        float _mul_15;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(_fp8x2_decode_7.y), "f"(gs_me));
                        float s1_16_3 = _mul_15;
                        float2 _f2_382 = make_float2(s0_15_3, s0_15_3);
                        float2 _f2_383 = make_float2(s1_16_3, s1_16_3);
                        float _unpack_fp4_28[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_3[0]));
                            (&_unpack_fp4_28[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_28[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_28[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_28[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_28[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_28[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_28[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_28[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_28[0]), "=f"(_unpack_fp4_28[1]), "=f"(_unpack_fp4_28[2]), "=f"(_unpack_fp4_28[3]), "=f"(_unpack_fp4_28[4]), "=f"(_unpack_fp4_28[5]), "=f"(_unpack_fp4_28[6]), "=f"(_unpack_fp4_28[7]) : "r"(own_3[0]));
                            #endif
                        }
                        float2 _f2_384 = make_float2(acc_3[0], acc_3[1]);
                        float2 _f2_385 = make_float2(_unpack_fp4_28[0], _unpack_fp4_28[1]);
                        float2 r_17_3 = fma_f32x2_rn_noftz(_f2_385, _f2_382, _f2_384);
                        acc_3[0] = r_17_3.x;
                        acc_3[1] = r_17_3.y;
                        float2 _f2_386 = make_float2(acc_3[2], acc_3[3]);
                        float2 _f2_387 = make_float2(_unpack_fp4_28[2], _unpack_fp4_28[3]);
                        float2 r_18_3 = fma_f32x2_rn_noftz(_f2_387, _f2_382, _f2_386);
                        acc_3[2] = r_18_3.x;
                        acc_3[3] = r_18_3.y;
                        float2 _f2_388 = make_float2(acc_3[4], acc_3[5]);
                        float2 _f2_389 = make_float2(_unpack_fp4_28[4], _unpack_fp4_28[5]);
                        float2 r_19_3 = fma_f32x2_rn_noftz(_f2_389, _f2_382, _f2_388);
                        acc_3[4] = r_19_3.x;
                        acc_3[5] = r_19_3.y;
                        float2 _f2_390 = make_float2(acc_3[6], acc_3[7]);
                        float2 _f2_391 = make_float2(_unpack_fp4_28[6], _unpack_fp4_28[7]);
                        float2 r_20_3 = fma_f32x2_rn_noftz(_f2_391, _f2_382, _f2_390);
                        acc_3[6] = r_20_3.x;
                        acc_3[7] = r_20_3.y;
                        float _unpack_fp4_29[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_3[1 + 0]));
                            (&_unpack_fp4_29[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_29[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_29[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_29[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_29[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_29[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_29[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_29[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_29[0]), "=f"(_unpack_fp4_29[1]), "=f"(_unpack_fp4_29[2]), "=f"(_unpack_fp4_29[3]), "=f"(_unpack_fp4_29[4]), "=f"(_unpack_fp4_29[5]), "=f"(_unpack_fp4_29[6]), "=f"(_unpack_fp4_29[7]) : "r"(own_3[1 + 0]));
                            #endif
                        }
                        float2 _f2_392 = make_float2(acc_3[8], acc_3[9]);
                        float2 _f2_393 = make_float2(_unpack_fp4_29[0], _unpack_fp4_29[1]);
                        float2 r_21_3 = fma_f32x2_rn_noftz(_f2_393, _f2_382, _f2_392);
                        acc_3[8] = r_21_3.x;
                        acc_3[9] = r_21_3.y;
                        float2 _f2_394 = make_float2(acc_3[10], acc_3[11]);
                        float2 _f2_395 = make_float2(_unpack_fp4_29[2], _unpack_fp4_29[3]);
                        float2 r_22_3 = fma_f32x2_rn_noftz(_f2_395, _f2_382, _f2_394);
                        acc_3[10] = r_22_3.x;
                        acc_3[11] = r_22_3.y;
                        float2 _f2_396 = make_float2(acc_3[12], acc_3[13]);
                        float2 _f2_397 = make_float2(_unpack_fp4_29[4], _unpack_fp4_29[5]);
                        float2 r_23_3 = fma_f32x2_rn_noftz(_f2_397, _f2_382, _f2_396);
                        acc_3[12] = r_23_3.x;
                        acc_3[13] = r_23_3.y;
                        float2 _f2_398 = make_float2(acc_3[14], acc_3[15]);
                        float2 _f2_399 = make_float2(_unpack_fp4_29[6], _unpack_fp4_29[7]);
                        float2 r_24_3 = fma_f32x2_rn_noftz(_f2_399, _f2_382, _f2_398);
                        acc_3[14] = r_24_3.x;
                        acc_3[15] = r_24_3.y;
                        float _unpack_fp4_30[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_3[2 + 0]));
                            (&_unpack_fp4_30[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_30[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_30[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_30[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_30[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_30[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_30[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_30[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_30[0]), "=f"(_unpack_fp4_30[1]), "=f"(_unpack_fp4_30[2]), "=f"(_unpack_fp4_30[3]), "=f"(_unpack_fp4_30[4]), "=f"(_unpack_fp4_30[5]), "=f"(_unpack_fp4_30[6]), "=f"(_unpack_fp4_30[7]) : "r"(own_3[2 + 0]));
                            #endif
                        }
                        float2 _f2_400 = make_float2(acc_3[16], acc_3[17]);
                        float2 _f2_401 = make_float2(_unpack_fp4_30[0], _unpack_fp4_30[1]);
                        float2 r_25_3 = fma_f32x2_rn_noftz(_f2_401, _f2_383, _f2_400);
                        acc_3[16] = r_25_3.x;
                        acc_3[17] = r_25_3.y;
                        float2 _f2_402 = make_float2(acc_3[18], acc_3[19]);
                        float2 _f2_403 = make_float2(_unpack_fp4_30[2], _unpack_fp4_30[3]);
                        float2 r_26_3 = fma_f32x2_rn_noftz(_f2_403, _f2_383, _f2_402);
                        acc_3[18] = r_26_3.x;
                        acc_3[19] = r_26_3.y;
                        float2 _f2_404 = make_float2(acc_3[20], acc_3[21]);
                        float2 _f2_405 = make_float2(_unpack_fp4_30[4], _unpack_fp4_30[5]);
                        float2 r_27_3 = fma_f32x2_rn_noftz(_f2_405, _f2_383, _f2_404);
                        acc_3[20] = r_27_3.x;
                        acc_3[21] = r_27_3.y;
                        float2 _f2_406 = make_float2(acc_3[22], acc_3[23]);
                        float2 _f2_407 = make_float2(_unpack_fp4_30[6], _unpack_fp4_30[7]);
                        float2 r_28_3 = fma_f32x2_rn_noftz(_f2_407, _f2_383, _f2_406);
                        acc_3[22] = r_28_3.x;
                        acc_3[23] = r_28_3.y;
                        float _unpack_fp4_31[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_3[3 + 0]));
                            (&_unpack_fp4_31[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_31[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_31[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_31[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_31[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_31[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_31[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_31[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_31[0]), "=f"(_unpack_fp4_31[1]), "=f"(_unpack_fp4_31[2]), "=f"(_unpack_fp4_31[3]), "=f"(_unpack_fp4_31[4]), "=f"(_unpack_fp4_31[5]), "=f"(_unpack_fp4_31[6]), "=f"(_unpack_fp4_31[7]) : "r"(own_3[3 + 0]));
                            #endif
                        }
                        float2 _f2_408 = make_float2(acc_3[24], acc_3[25]);
                        float2 _f2_409 = make_float2(_unpack_fp4_31[0], _unpack_fp4_31[1]);
                        float2 r_29_3 = fma_f32x2_rn_noftz(_f2_409, _f2_383, _f2_408);
                        acc_3[24] = r_29_3.x;
                        acc_3[25] = r_29_3.y;
                        float2 _f2_410 = make_float2(acc_3[26], acc_3[27]);
                        float2 _f2_411 = make_float2(_unpack_fp4_31[2], _unpack_fp4_31[3]);
                        float2 r_30_3 = fma_f32x2_rn_noftz(_f2_411, _f2_383, _f2_410);
                        acc_3[26] = r_30_3.x;
                        acc_3[27] = r_30_3.y;
                        float2 _f2_412 = make_float2(acc_3[28], acc_3[29]);
                        float2 _f2_413 = make_float2(_unpack_fp4_31[4], _unpack_fp4_31[5]);
                        float2 r_31_3 = fma_f32x2_rn_noftz(_f2_413, _f2_383, _f2_412);
                        acc_3[28] = r_31_3.x;
                        acc_3[29] = r_31_3.y;
                        float2 _f2_414 = make_float2(acc_3[30], acc_3[31]);
                        float2 _f2_415 = make_float2(_unpack_fp4_31[6], _unpack_fp4_31[7]);
                        float2 r_32_3 = fma_f32x2_rn_noftz(_f2_415, _f2_383, _f2_414);
                        acc_3[30] = r_32_3.x;
                        acc_3[31] = r_32_3.y;
                        unsigned int t_3[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 96)) + (unsigned int)(16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 96)) + (unsigned int)(16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 96)) + (unsigned int)(16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 96)) + (unsigned int)(16 * (f ^ 3))));
                        unsigned int s_3[16];
                        unsigned int r_33_3[16];
                        s_3[0] = (((f & 1) != 0) ? t_3[4] : t_3[0]);
                        s_3[4] = (((f & 1) != 0) ? t_3[0] : t_3[4]);
                        s_3[8] = (((f & 1) != 0) ? t_3[12] : t_3[8]);
                        s_3[12] = (((f & 1) != 0) ? t_3[8] : t_3[12]);
                        s_3[1] = (((f & 1) != 0) ? t_3[5] : t_3[1]);
                        s_3[5] = (((f & 1) != 0) ? t_3[1] : t_3[5]);
                        s_3[9] = (((f & 1) != 0) ? t_3[13] : t_3[9]);
                        s_3[13] = (((f & 1) != 0) ? t_3[9] : t_3[13]);
                        s_3[2] = (((f & 1) != 0) ? t_3[6] : t_3[2]);
                        s_3[6] = (((f & 1) != 0) ? t_3[2] : t_3[6]);
                        s_3[10] = (((f & 1) != 0) ? t_3[14] : t_3[10]);
                        s_3[14] = (((f & 1) != 0) ? t_3[10] : t_3[14]);
                        s_3[3] = (((f & 1) != 0) ? t_3[7] : t_3[3]);
                        s_3[7] = (((f & 1) != 0) ? t_3[3] : t_3[7]);
                        s_3[11] = (((f & 1) != 0) ? t_3[15] : t_3[11]);
                        s_3[15] = (((f & 1) != 0) ? t_3[11] : t_3[15]);
                        r_33_3[0] = (((f & 2) != 0) ? s_3[8] : s_3[0]);
                        r_33_3[4] = (((f & 2) != 0) ? s_3[12] : s_3[4]);
                        r_33_3[8] = (((f & 2) != 0) ? s_3[0] : s_3[8]);
                        r_33_3[12] = (((f & 2) != 0) ? s_3[4] : s_3[12]);
                        r_33_3[1] = (((f & 2) != 0) ? s_3[9] : s_3[1]);
                        r_33_3[5] = (((f & 2) != 0) ? s_3[13] : s_3[5]);
                        r_33_3[9] = (((f & 2) != 0) ? s_3[1] : s_3[9]);
                        r_33_3[13] = (((f & 2) != 0) ? s_3[5] : s_3[13]);
                        r_33_3[2] = (((f & 2) != 0) ? s_3[10] : s_3[2]);
                        r_33_3[6] = (((f & 2) != 0) ? s_3[14] : s_3[6]);
                        r_33_3[10] = (((f & 2) != 0) ? s_3[2] : s_3[10]);
                        r_33_3[14] = (((f & 2) != 0) ? s_3[6] : s_3[14]);
                        r_33_3[3] = (((f & 2) != 0) ? s_3[11] : s_3[3]);
                        r_33_3[7] = (((f & 2) != 0) ? s_3[15] : s_3[7]);
                        r_33_3[11] = (((f & 2) != 0) ? s_3[3] : s_3[11]);
                        r_33_3[15] = (((f & 2) != 0) ? s_3[7] : s_3[15]);
                        float2 _f2_416 = make_float2(acc_3[0], acc_3[1]);
                        float2 _f2_417 = make_float2(__uint_as_float(r_33_3[0] << 16), __uint_as_float(r_33_3[0] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_48;
                        {
                            float2 _packed_f32x2_61_0 = _f2_416;
                            float2 _packed_f32x2_61_1 = _f2_417;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_48)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_61_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_61_1)));
                        }
                        acc_3[0] = _packed_add_f32x2_48.x;
                        acc_3[1] = _packed_add_f32x2_48.y;
                        float2 _f2_418 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_48, _packed_add_f32x2_48, _f2_418).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_48, _packed_add_f32x2_48, _f2_418).y;
                        float2 _f2_419 = make_float2(acc_3[2], acc_3[3]);
                        float2 _f2_420 = make_float2(__uint_as_float(r_33_3[1] << 16), __uint_as_float(r_33_3[1] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_49;
                        {
                            float2 _packed_f32x2_62_0 = _f2_419;
                            float2 _packed_f32x2_62_1 = _f2_420;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_49)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_62_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_62_1)));
                        }
                        acc_3[2] = _packed_add_f32x2_49.x;
                        acc_3[3] = _packed_add_f32x2_49.y;
                        float2 _f2_421 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_49, _packed_add_f32x2_49, _f2_421).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_49, _packed_add_f32x2_49, _f2_421).y;
                        float2 _f2_422 = make_float2(acc_3[4], acc_3[5]);
                        float2 _f2_423 = make_float2(__uint_as_float(r_33_3[2] << 16), __uint_as_float(r_33_3[2] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_50;
                        {
                            float2 _packed_f32x2_63_0 = _f2_422;
                            float2 _packed_f32x2_63_1 = _f2_423;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_50)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_63_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_63_1)));
                        }
                        acc_3[4] = _packed_add_f32x2_50.x;
                        acc_3[5] = _packed_add_f32x2_50.y;
                        float2 _f2_424 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_50, _packed_add_f32x2_50, _f2_424).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_50, _packed_add_f32x2_50, _f2_424).y;
                        float2 _f2_425 = make_float2(acc_3[6], acc_3[7]);
                        float2 _f2_426 = make_float2(__uint_as_float(r_33_3[3] << 16), __uint_as_float(r_33_3[3] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_51;
                        {
                            float2 _packed_f32x2_64_0 = _f2_425;
                            float2 _packed_f32x2_64_1 = _f2_426;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_51)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_64_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_64_1)));
                        }
                        acc_3[6] = _packed_add_f32x2_51.x;
                        acc_3[7] = _packed_add_f32x2_51.y;
                        float2 _f2_427 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_51, _packed_add_f32x2_51, _f2_427).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_51, _packed_add_f32x2_51, _f2_427).y;
                        float2 _f2_428 = make_float2(acc_3[8], acc_3[9]);
                        float2 _f2_429 = make_float2(__uint_as_float(r_33_3[4] << 16), __uint_as_float(r_33_3[4] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_52;
                        {
                            float2 _packed_f32x2_65_0 = _f2_428;
                            float2 _packed_f32x2_65_1 = _f2_429;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_52)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_65_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_65_1)));
                        }
                        acc_3[8] = _packed_add_f32x2_52.x;
                        acc_3[9] = _packed_add_f32x2_52.y;
                        float2 _f2_430 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_52, _packed_add_f32x2_52, _f2_430).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_52, _packed_add_f32x2_52, _f2_430).y;
                        float2 _f2_431 = make_float2(acc_3[10], acc_3[11]);
                        float2 _f2_432 = make_float2(__uint_as_float(r_33_3[5] << 16), __uint_as_float(r_33_3[5] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_53;
                        {
                            float2 _packed_f32x2_66_0 = _f2_431;
                            float2 _packed_f32x2_66_1 = _f2_432;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_53)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_66_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_66_1)));
                        }
                        acc_3[10] = _packed_add_f32x2_53.x;
                        acc_3[11] = _packed_add_f32x2_53.y;
                        float2 _f2_433 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_53, _packed_add_f32x2_53, _f2_433).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_53, _packed_add_f32x2_53, _f2_433).y;
                        float2 _f2_434 = make_float2(acc_3[12], acc_3[13]);
                        float2 _f2_435 = make_float2(__uint_as_float(r_33_3[6] << 16), __uint_as_float(r_33_3[6] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_54;
                        {
                            float2 _packed_f32x2_67_0 = _f2_434;
                            float2 _packed_f32x2_67_1 = _f2_435;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_54)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_67_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_67_1)));
                        }
                        acc_3[12] = _packed_add_f32x2_54.x;
                        acc_3[13] = _packed_add_f32x2_54.y;
                        float2 _f2_436 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_54, _packed_add_f32x2_54, _f2_436).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_54, _packed_add_f32x2_54, _f2_436).y;
                        float2 _f2_437 = make_float2(acc_3[14], acc_3[15]);
                        float2 _f2_438 = make_float2(__uint_as_float(r_33_3[7] << 16), __uint_as_float(r_33_3[7] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_55;
                        {
                            float2 _packed_f32x2_68_0 = _f2_437;
                            float2 _packed_f32x2_68_1 = _f2_438;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_55)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_68_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_68_1)));
                        }
                        acc_3[14] = _packed_add_f32x2_55.x;
                        acc_3[15] = _packed_add_f32x2_55.y;
                        float2 _f2_439 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_55, _packed_add_f32x2_55, _f2_439).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_55, _packed_add_f32x2_55, _f2_439).y;
                        float2 _f2_440 = make_float2(acc_3[16], acc_3[17]);
                        float2 _f2_441 = make_float2(__uint_as_float(r_33_3[8] << 16), __uint_as_float(r_33_3[8] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_56;
                        {
                            float2 _packed_f32x2_69_0 = _f2_440;
                            float2 _packed_f32x2_69_1 = _f2_441;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_56)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_69_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_69_1)));
                        }
                        acc_3[16] = _packed_add_f32x2_56.x;
                        acc_3[17] = _packed_add_f32x2_56.y;
                        float2 _f2_442 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_56, _packed_add_f32x2_56, _f2_442).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_56, _packed_add_f32x2_56, _f2_442).y;
                        float2 _f2_443 = make_float2(acc_3[18], acc_3[19]);
                        float2 _f2_444 = make_float2(__uint_as_float(r_33_3[9] << 16), __uint_as_float(r_33_3[9] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_57;
                        {
                            float2 _packed_f32x2_70_0 = _f2_443;
                            float2 _packed_f32x2_70_1 = _f2_444;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_57)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_70_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_70_1)));
                        }
                        acc_3[18] = _packed_add_f32x2_57.x;
                        acc_3[19] = _packed_add_f32x2_57.y;
                        float2 _f2_445 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_57, _packed_add_f32x2_57, _f2_445).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_57, _packed_add_f32x2_57, _f2_445).y;
                        float2 _f2_446 = make_float2(acc_3[20], acc_3[21]);
                        float2 _f2_447 = make_float2(__uint_as_float(r_33_3[10] << 16), __uint_as_float(r_33_3[10] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_58;
                        {
                            float2 _packed_f32x2_71_0 = _f2_446;
                            float2 _packed_f32x2_71_1 = _f2_447;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_58)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_71_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_71_1)));
                        }
                        acc_3[20] = _packed_add_f32x2_58.x;
                        acc_3[21] = _packed_add_f32x2_58.y;
                        float2 _f2_448 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_58, _packed_add_f32x2_58, _f2_448).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_58, _packed_add_f32x2_58, _f2_448).y;
                        float2 _f2_449 = make_float2(acc_3[22], acc_3[23]);
                        float2 _f2_450 = make_float2(__uint_as_float(r_33_3[11] << 16), __uint_as_float(r_33_3[11] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_59;
                        {
                            float2 _packed_f32x2_72_0 = _f2_449;
                            float2 _packed_f32x2_72_1 = _f2_450;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_59)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_72_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_72_1)));
                        }
                        acc_3[22] = _packed_add_f32x2_59.x;
                        acc_3[23] = _packed_add_f32x2_59.y;
                        float2 _f2_451 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_59, _packed_add_f32x2_59, _f2_451).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_59, _packed_add_f32x2_59, _f2_451).y;
                        float2 _f2_452 = make_float2(acc_3[24], acc_3[25]);
                        float2 _f2_453 = make_float2(__uint_as_float(r_33_3[12] << 16), __uint_as_float(r_33_3[12] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_60;
                        {
                            float2 _packed_f32x2_73_0 = _f2_452;
                            float2 _packed_f32x2_73_1 = _f2_453;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_60)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_73_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_73_1)));
                        }
                        acc_3[24] = _packed_add_f32x2_60.x;
                        acc_3[25] = _packed_add_f32x2_60.y;
                        float2 _f2_454 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_60, _packed_add_f32x2_60, _f2_454).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_60, _packed_add_f32x2_60, _f2_454).y;
                        float2 _f2_455 = make_float2(acc_3[26], acc_3[27]);
                        float2 _f2_456 = make_float2(__uint_as_float(r_33_3[13] << 16), __uint_as_float(r_33_3[13] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_61;
                        {
                            float2 _packed_f32x2_74_0 = _f2_455;
                            float2 _packed_f32x2_74_1 = _f2_456;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_61)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_74_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_74_1)));
                        }
                        acc_3[26] = _packed_add_f32x2_61.x;
                        acc_3[27] = _packed_add_f32x2_61.y;
                        float2 _f2_457 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_61, _packed_add_f32x2_61, _f2_457).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_61, _packed_add_f32x2_61, _f2_457).y;
                        float2 _f2_458 = make_float2(acc_3[28], acc_3[29]);
                        float2 _f2_459 = make_float2(__uint_as_float(r_33_3[14] << 16), __uint_as_float(r_33_3[14] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_62;
                        {
                            float2 _packed_f32x2_75_0 = _f2_458;
                            float2 _packed_f32x2_75_1 = _f2_459;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_62)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_75_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_75_1)));
                        }
                        acc_3[28] = _packed_add_f32x2_62.x;
                        acc_3[29] = _packed_add_f32x2_62.y;
                        float2 _f2_460 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_62, _packed_add_f32x2_62, _f2_460).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_62, _packed_add_f32x2_62, _f2_460).y;
                        float2 _f2_461 = make_float2(acc_3[30], acc_3[31]);
                        float2 _f2_462 = make_float2(__uint_as_float(r_33_3[15] << 16), __uint_as_float(r_33_3[15] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_63;
                        {
                            float2 _packed_f32x2_76_0 = _f2_461;
                            float2 _packed_f32x2_76_1 = _f2_462;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_63)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_76_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_76_1)));
                        }
                        acc_3[30] = _packed_add_f32x2_63.x;
                        acc_3[31] = _packed_add_f32x2_63.y;
                        float2 _f2_463 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_63, _packed_add_f32x2_63, _f2_463).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_63, _packed_add_f32x2_63, _f2_463).y;
                        uint32_t acc_bf16_3[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_3[_lp*2 + 0], acc_3[_lp*2+1 + 0]));
                            acc_bf16_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 0)))), "r"(acc_bf16_3[0]), "r"(acc_bf16_3[1]), "r"(acc_bf16_3[2]), "r"(acc_bf16_3[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 1)))), "r"(acc_bf16_3[4]), "r"(acc_bf16_3[5]), "r"(acc_bf16_3[6]), "r"(acc_bf16_3[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 2)))), "r"(acc_bf16_3[8]), "r"(acc_bf16_3[9]), "r"(acc_bf16_3[10]), "r"(acc_bf16_3[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 3)))), "r"(acc_bf16_3[12]), "r"(acc_bf16_3[13]), "r"(acc_bf16_3[14]), "r"(acc_bf16_3[15]) : "memory");
                        __syncwarp();
                        int c_3 = lane_3;
                        int l_3 = c_3 >> 2;
                        unsigned int w4_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_3 + (c_3 & 3 ^ l_3 >> 1 & 3)))));
                        long long idx_3 = drow * 3072 + (long long)(4 * (384 + c_3));
                        {
                            int4 _iv4 = make_int4(w4_3[0 + 0], w4_3[0 + 1], w4_3[0 + 2], w4_3[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_3)) = _iv4;
                        }
                        int c_34_3 = lane_3 + 32;
                        int l_35_3 = c_34_3 >> 2;
                        unsigned int w4_36_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_36_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_3[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_35_3 + (c_34_3 & 3 ^ l_35_3 >> 1 & 3)))));
                        long long idx_37_3 = drow * 3072 + (long long)(4 * (384 + c_34_3));
                        {
                            int4 _iv4 = make_int4(w4_36_3[0 + 0], w4_36_3[0 + 1], w4_36_3[0 + 2], w4_36_3[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_37_3)) = _iv4;
                        }
                        int c_38_3 = lane_3 + 64;
                        int l_39_3 = c_38_3 >> 2;
                        unsigned int w4_40_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_40_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_3[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_39_3 + (c_38_3 & 3 ^ l_39_3 >> 1 & 3)))));
                        long long idx_41_3 = drow * 3072 + (long long)(4 * (384 + c_38_3));
                        {
                            int4 _iv4 = make_int4(w4_40_3[0 + 0], w4_40_3[0 + 1], w4_40_3[0 + 2], w4_40_3[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_41_3)) = _iv4;
                        }
                        int c_42_3 = lane_3 + 96;
                        int l_43_3 = c_42_3 >> 2;
                        unsigned int w4_44_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_44_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_3[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_43_3 + (c_42_3 & 3 ^ l_43_3 >> 1 & 3)))));
                        long long idx_45_3 = drow * 3072 + (long long)(4 * (384 + c_42_3));
                        {
                            int4 _iv4 = make_int4(w4_44_3[0 + 0], w4_44_3[0 + 1], w4_44_3[0 + 2], w4_44_3[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_45_3)) = _iv4;
                        }
                        __syncwarp();
                        int zlane_1 = rb_off + 128 * lane_3;
                        int l7_1 = lane_3 & 7;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_1 + 16 * (l7_1 ^ 0))), "r"(__as_u32(acc_3[0])), "r"(__as_u32(acc_3[1])), "r"(__as_u32(acc_3[2])), "r"(__as_u32(acc_3[3])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_1 + 16 * (l7_1 ^ 1))), "r"(__as_u32(acc_3[4])), "r"(__as_u32(acc_3[5])), "r"(__as_u32(acc_3[6])), "r"(__as_u32(acc_3[7])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_1 + 16 * (l7_1 ^ 2))), "r"(__as_u32(acc_3[8])), "r"(__as_u32(acc_3[9])), "r"(__as_u32(acc_3[10])), "r"(__as_u32(acc_3[11])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_1 + 16 * (l7_1 ^ 3))), "r"(__as_u32(acc_3[12])), "r"(__as_u32(acc_3[13])), "r"(__as_u32(acc_3[14])), "r"(__as_u32(acc_3[15])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_1 + 16 * (l7_1 ^ 4))), "r"(__as_u32(acc_3[16])), "r"(__as_u32(acc_3[17])), "r"(__as_u32(acc_3[18])), "r"(__as_u32(acc_3[19])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_1 + 16 * (l7_1 ^ 5))), "r"(__as_u32(acc_3[20])), "r"(__as_u32(acc_3[21])), "r"(__as_u32(acc_3[22])), "r"(__as_u32(acc_3[23])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_1 + 16 * (l7_1 ^ 6))), "r"(__as_u32(acc_3[24])), "r"(__as_u32(acc_3[25])), "r"(__as_u32(acc_3[26])), "r"(__as_u32(acc_3[27])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_1 + 16 * (l7_1 ^ 7))), "r"(__as_u32(acc_3[28])), "r"(__as_u32(acc_3[29])), "r"(__as_u32(acc_3[30])), "r"(__as_u32(acc_3[31])) : "memory");
                    }
                    {
                        unsigned int d_2 = 0;
                        unsigned int w_0_2 = pf_v[16];
                        d_2 = d_2 | w_0_2 & ~((w_0_2 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_1_2 = pf_v[17];
                        d_2 = d_2 | w_1_2 & ~((w_1_2 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_2_2 = pf_v[18];
                        d_2 = d_2 | w_2_2 & ~((w_2_2 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_3_2 = pf_v[19];
                        d_2 = d_2 | w_3_2 & ~((w_3_2 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int sc_2 = pf_s[4];
                        d_2 = d_2 | ((sc_2 & 255) + 1 | (sc_2 >> 8) + 1) & 256;
                        unsigned int w_4_2 = pf_v[20];
                        d_2 = d_2 | w_4_2 & ~((w_4_2 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_5_2 = pf_v[21];
                        d_2 = d_2 | w_5_2 & ~((w_5_2 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_6_2 = pf_v[22];
                        d_2 = d_2 | w_6_2 & ~((w_6_2 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int w_7_2 = pf_v[23];
                        d_2 = d_2 | w_7_2 & ~((w_7_2 & 2004318071) + 2004318071) & 2290649224u;
                        unsigned int sc_8_1 = pf_s[5];
                        d_2 = d_2 | ((sc_8_1 & 255) + 1 | (sc_8_1 >> 8) + 1) & 256;
                        if (d_2 != 0) {
                            uint32_t _fp4_wire_values_2[8];
                            uint32_t _fp4_wire_scales_2[2];
                            uint16_t _wire_h16_77_0;
                            uint16_t _wire_h16_77_1;
                            uint32_t _wire_vpend_77 = 3u;
                            uint32_t _wire_spend_77 = 3u;
                            do {
                                if (_wire_vpend_77 & 1u) {
                                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_2[0]), "=r"(_fp4_wire_values_2[1]), "=r"(_fp4_wire_values_2[2]), "=r"(_fp4_wire_values_2[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_3 + 128))))) : "memory");
                                }
                                if (_wire_vpend_77 & 2u) {
                                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_2[4]), "=r"(_fp4_wire_values_2[5]), "=r"(_fp4_wire_values_2[6]), "=r"(_fp4_wire_values_2[7]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_3 + 160))))) : "memory");
                                }
                                if (_wire_spend_77 & 1u) {
                                    asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_77_0) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_3 + 128))))) : "memory");
                                    _fp4_wire_scales_2[0] = (uint32_t)_wire_h16_77_0;
                                }
                                if (_wire_spend_77 & 2u) {
                                    asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_77_1) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_3 + 160))))) : "memory");
                                    _fp4_wire_scales_2[1] = (uint32_t)_wire_h16_77_1;
                                }
                                if (_wire_vpend_77 & 1u) {
                                    if (!((((_fp4_wire_values_2[0] & ~((_fp4_wire_values_2[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[1] & ~((_fp4_wire_values_2[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[2] & ~((_fp4_wire_values_2[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[3] & ~((_fp4_wire_values_2[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_77 &= ~1u;
                                }
                                if (_wire_vpend_77 & 2u) {
                                    if (!((((_fp4_wire_values_2[4] & ~((_fp4_wire_values_2[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[5] & ~((_fp4_wire_values_2[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[6] & ~((_fp4_wire_values_2[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_2[7] & ~((_fp4_wire_values_2[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_77 &= ~2u;
                                }
                                if (_wire_spend_77 & 1u) {
                                    if (!(((_fp4_wire_scales_2[0] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_2[0] >> 8) == 0xFFu))) _wire_spend_77 &= ~1u;
                                }
                                if (_wire_spend_77 & 2u) {
                                    if (!(((_fp4_wire_scales_2[1] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_2[1] >> 8) == 0xFFu))) _wire_spend_77 &= ~2u;
                                }
                            } while ((_wire_vpend_77 | _wire_spend_77) != 0u);
                            pf_v[16] = _fp4_wire_values_2[0];
                            pf_v[17] = _fp4_wire_values_2[1];
                            pf_v[18] = _fp4_wire_values_2[2];
                            pf_v[19] = _fp4_wire_values_2[3];
                            pf_s[4] = _fp4_wire_scales_2[0];
                            pf_v[20] = _fp4_wire_values_2[4];
                            pf_v[21] = _fp4_wire_values_2[5];
                            pf_v[22] = _fp4_wire_values_2[6];
                            pf_v[23] = _fp4_wire_values_2[7];
                            pf_s[5] = _fp4_wire_scales_2[1];
                        }
                    }
                    {
                        unsigned int own_4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&own_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_4[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * (lane_3 + 128))));
                        unsigned int own_sc_4 = (unsigned int)wsm16[(dob_off + 3072 >> 1) + (lane_3 + 128)];
                        float acc_4[32];
                        acc_4[0] = 0.0f;
                        acc_4[1] = 0.0f;
                        acc_4[2] = 0.0f;
                        acc_4[3] = 0.0f;
                        acc_4[4] = 0.0f;
                        acc_4[5] = 0.0f;
                        acc_4[6] = 0.0f;
                        acc_4[7] = 0.0f;
                        acc_4[8] = 0.0f;
                        acc_4[9] = 0.0f;
                        acc_4[10] = 0.0f;
                        acc_4[11] = 0.0f;
                        acc_4[12] = 0.0f;
                        acc_4[13] = 0.0f;
                        acc_4[14] = 0.0f;
                        acc_4[15] = 0.0f;
                        acc_4[16] = 0.0f;
                        acc_4[17] = 0.0f;
                        acc_4[18] = 0.0f;
                        acc_4[19] = 0.0f;
                        acc_4[20] = 0.0f;
                        acc_4[21] = 0.0f;
                        acc_4[22] = 0.0f;
                        acc_4[23] = 0.0f;
                        acc_4[24] = 0.0f;
                        acc_4[25] = 0.0f;
                        acc_4[26] = 0.0f;
                        acc_4[27] = 0.0f;
                        acc_4[28] = 0.0f;
                        acc_4[29] = 0.0f;
                        acc_4[30] = 0.0f;
                        acc_4[31] = 0.0f;
                        float2 _fp8x2_decode_8;
                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                            "mov.b32 {lo, hi}, pair;\n"
                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                            : "=f"(_fp8x2_decode_8.x), "=f"(_fp8x2_decode_8.y) : "h"((uint16_t)pf_s[4]));
                        float _mul_16;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_fp8x2_decode_8.x), "f"(__uint_as_float(w_7)));
                        float s0_4 = _mul_16;
                        float _mul_17;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_fp8x2_decode_8.y), "f"(__uint_as_float(w_7)));
                        float s1_4 = _mul_17;
                        float2 _f2_464 = make_float2(s0_4, s0_4);
                        float2 _f2_465 = make_float2(s1_4, s1_4);
                        float _unpack_fp4_32[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[16 + 0]));
                            (&_unpack_fp4_32[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_32[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_32[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_32[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_32[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_32[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_32[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_32[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_32[0]), "=f"(_unpack_fp4_32[1]), "=f"(_unpack_fp4_32[2]), "=f"(_unpack_fp4_32[3]), "=f"(_unpack_fp4_32[4]), "=f"(_unpack_fp4_32[5]), "=f"(_unpack_fp4_32[6]), "=f"(_unpack_fp4_32[7]) : "r"(pf_v[16 + 0]));
                            #endif
                        }
                        float2 _f2_466 = make_float2(acc_4[0], acc_4[1]);
                        float2 _f2_467 = make_float2(_unpack_fp4_32[0], _unpack_fp4_32[1]);
                        float2 r_35 = fma_f32x2_rn_noftz(_f2_467, _f2_464, _f2_466);
                        acc_4[0] = r_35.x;
                        acc_4[1] = r_35.y;
                        float2 _f2_468 = make_float2(acc_4[2], acc_4[3]);
                        float2 _f2_469 = make_float2(_unpack_fp4_32[2], _unpack_fp4_32[3]);
                        float2 r_0_4 = fma_f32x2_rn_noftz(_f2_469, _f2_464, _f2_468);
                        acc_4[2] = r_0_4.x;
                        acc_4[3] = r_0_4.y;
                        float2 _f2_470 = make_float2(acc_4[4], acc_4[5]);
                        float2 _f2_471 = make_float2(_unpack_fp4_32[4], _unpack_fp4_32[5]);
                        float2 r_1_4 = fma_f32x2_rn_noftz(_f2_471, _f2_464, _f2_470);
                        acc_4[4] = r_1_4.x;
                        acc_4[5] = r_1_4.y;
                        float2 _f2_472 = make_float2(acc_4[6], acc_4[7]);
                        float2 _f2_473 = make_float2(_unpack_fp4_32[6], _unpack_fp4_32[7]);
                        float2 r_2_4 = fma_f32x2_rn_noftz(_f2_473, _f2_464, _f2_472);
                        acc_4[6] = r_2_4.x;
                        acc_4[7] = r_2_4.y;
                        float _unpack_fp4_33[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[17 + 0]));
                            (&_unpack_fp4_33[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_33[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_33[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_33[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_33[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_33[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_33[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_33[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_33[0]), "=f"(_unpack_fp4_33[1]), "=f"(_unpack_fp4_33[2]), "=f"(_unpack_fp4_33[3]), "=f"(_unpack_fp4_33[4]), "=f"(_unpack_fp4_33[5]), "=f"(_unpack_fp4_33[6]), "=f"(_unpack_fp4_33[7]) : "r"(pf_v[17 + 0]));
                            #endif
                        }
                        float2 _f2_474 = make_float2(acc_4[8], acc_4[9]);
                        float2 _f2_475 = make_float2(_unpack_fp4_33[0], _unpack_fp4_33[1]);
                        float2 r_3_4 = fma_f32x2_rn_noftz(_f2_475, _f2_464, _f2_474);
                        acc_4[8] = r_3_4.x;
                        acc_4[9] = r_3_4.y;
                        float2 _f2_476 = make_float2(acc_4[10], acc_4[11]);
                        float2 _f2_477 = make_float2(_unpack_fp4_33[2], _unpack_fp4_33[3]);
                        float2 r_4_4 = fma_f32x2_rn_noftz(_f2_477, _f2_464, _f2_476);
                        acc_4[10] = r_4_4.x;
                        acc_4[11] = r_4_4.y;
                        float2 _f2_478 = make_float2(acc_4[12], acc_4[13]);
                        float2 _f2_479 = make_float2(_unpack_fp4_33[4], _unpack_fp4_33[5]);
                        float2 r_5_4 = fma_f32x2_rn_noftz(_f2_479, _f2_464, _f2_478);
                        acc_4[12] = r_5_4.x;
                        acc_4[13] = r_5_4.y;
                        float2 _f2_480 = make_float2(acc_4[14], acc_4[15]);
                        float2 _f2_481 = make_float2(_unpack_fp4_33[6], _unpack_fp4_33[7]);
                        float2 r_6_4 = fma_f32x2_rn_noftz(_f2_481, _f2_464, _f2_480);
                        acc_4[14] = r_6_4.x;
                        acc_4[15] = r_6_4.y;
                        float _unpack_fp4_34[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[18 + 0]));
                            (&_unpack_fp4_34[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_34[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_34[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_34[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_34[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_34[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_34[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_34[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_34[0]), "=f"(_unpack_fp4_34[1]), "=f"(_unpack_fp4_34[2]), "=f"(_unpack_fp4_34[3]), "=f"(_unpack_fp4_34[4]), "=f"(_unpack_fp4_34[5]), "=f"(_unpack_fp4_34[6]), "=f"(_unpack_fp4_34[7]) : "r"(pf_v[18 + 0]));
                            #endif
                        }
                        float2 _f2_482 = make_float2(acc_4[16], acc_4[17]);
                        float2 _f2_483 = make_float2(_unpack_fp4_34[0], _unpack_fp4_34[1]);
                        float2 r_7_4 = fma_f32x2_rn_noftz(_f2_483, _f2_465, _f2_482);
                        acc_4[16] = r_7_4.x;
                        acc_4[17] = r_7_4.y;
                        float2 _f2_484 = make_float2(acc_4[18], acc_4[19]);
                        float2 _f2_485 = make_float2(_unpack_fp4_34[2], _unpack_fp4_34[3]);
                        float2 r_8_4 = fma_f32x2_rn_noftz(_f2_485, _f2_465, _f2_484);
                        acc_4[18] = r_8_4.x;
                        acc_4[19] = r_8_4.y;
                        float2 _f2_486 = make_float2(acc_4[20], acc_4[21]);
                        float2 _f2_487 = make_float2(_unpack_fp4_34[4], _unpack_fp4_34[5]);
                        float2 r_9_4 = fma_f32x2_rn_noftz(_f2_487, _f2_465, _f2_486);
                        acc_4[20] = r_9_4.x;
                        acc_4[21] = r_9_4.y;
                        float2 _f2_488 = make_float2(acc_4[22], acc_4[23]);
                        float2 _f2_489 = make_float2(_unpack_fp4_34[6], _unpack_fp4_34[7]);
                        float2 r_10_4 = fma_f32x2_rn_noftz(_f2_489, _f2_465, _f2_488);
                        acc_4[22] = r_10_4.x;
                        acc_4[23] = r_10_4.y;
                        float _unpack_fp4_35[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[19 + 0]));
                            (&_unpack_fp4_35[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_35[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_35[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_35[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_35[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_35[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_35[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_35[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_35[0]), "=f"(_unpack_fp4_35[1]), "=f"(_unpack_fp4_35[2]), "=f"(_unpack_fp4_35[3]), "=f"(_unpack_fp4_35[4]), "=f"(_unpack_fp4_35[5]), "=f"(_unpack_fp4_35[6]), "=f"(_unpack_fp4_35[7]) : "r"(pf_v[19 + 0]));
                            #endif
                        }
                        float2 _f2_490 = make_float2(acc_4[24], acc_4[25]);
                        float2 _f2_491 = make_float2(_unpack_fp4_35[0], _unpack_fp4_35[1]);
                        float2 r_11_4 = fma_f32x2_rn_noftz(_f2_491, _f2_465, _f2_490);
                        acc_4[24] = r_11_4.x;
                        acc_4[25] = r_11_4.y;
                        float2 _f2_492 = make_float2(acc_4[26], acc_4[27]);
                        float2 _f2_493 = make_float2(_unpack_fp4_35[2], _unpack_fp4_35[3]);
                        float2 r_12_4 = fma_f32x2_rn_noftz(_f2_493, _f2_465, _f2_492);
                        acc_4[26] = r_12_4.x;
                        acc_4[27] = r_12_4.y;
                        float2 _f2_494 = make_float2(acc_4[28], acc_4[29]);
                        float2 _f2_495 = make_float2(_unpack_fp4_35[4], _unpack_fp4_35[5]);
                        float2 r_13_4 = fma_f32x2_rn_noftz(_f2_495, _f2_465, _f2_494);
                        acc_4[28] = r_13_4.x;
                        acc_4[29] = r_13_4.y;
                        float2 _f2_496 = make_float2(acc_4[30], acc_4[31]);
                        float2 _f2_497 = make_float2(_unpack_fp4_35[6], _unpack_fp4_35[7]);
                        float2 r_14_4 = fma_f32x2_rn_noftz(_f2_497, _f2_465, _f2_496);
                        acc_4[30] = r_14_4.x;
                        acc_4[31] = r_14_4.y;
                        float2 _fp8x2_decode_9;
                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                            "mov.b32 {lo, hi}, pair;\n"
                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                            : "=f"(_fp8x2_decode_9.x), "=f"(_fp8x2_decode_9.y) : "h"((uint16_t)own_sc_4));
                        float _mul_18;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(_fp8x2_decode_9.x), "f"(gs_me));
                        float s0_15_4 = _mul_18;
                        float _mul_19;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_fp8x2_decode_9.y), "f"(gs_me));
                        float s1_16_4 = _mul_19;
                        float2 _f2_498 = make_float2(s0_15_4, s0_15_4);
                        float2 _f2_499 = make_float2(s1_16_4, s1_16_4);
                        float _unpack_fp4_36[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_4[0]));
                            (&_unpack_fp4_36[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_36[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_36[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_36[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_36[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_36[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_36[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_36[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_36[0]), "=f"(_unpack_fp4_36[1]), "=f"(_unpack_fp4_36[2]), "=f"(_unpack_fp4_36[3]), "=f"(_unpack_fp4_36[4]), "=f"(_unpack_fp4_36[5]), "=f"(_unpack_fp4_36[6]), "=f"(_unpack_fp4_36[7]) : "r"(own_4[0]));
                            #endif
                        }
                        float2 _f2_500 = make_float2(acc_4[0], acc_4[1]);
                        float2 _f2_501 = make_float2(_unpack_fp4_36[0], _unpack_fp4_36[1]);
                        float2 r_17_4 = fma_f32x2_rn_noftz(_f2_501, _f2_498, _f2_500);
                        acc_4[0] = r_17_4.x;
                        acc_4[1] = r_17_4.y;
                        float2 _f2_502 = make_float2(acc_4[2], acc_4[3]);
                        float2 _f2_503 = make_float2(_unpack_fp4_36[2], _unpack_fp4_36[3]);
                        float2 r_18_4 = fma_f32x2_rn_noftz(_f2_503, _f2_498, _f2_502);
                        acc_4[2] = r_18_4.x;
                        acc_4[3] = r_18_4.y;
                        float2 _f2_504 = make_float2(acc_4[4], acc_4[5]);
                        float2 _f2_505 = make_float2(_unpack_fp4_36[4], _unpack_fp4_36[5]);
                        float2 r_19_4 = fma_f32x2_rn_noftz(_f2_505, _f2_498, _f2_504);
                        acc_4[4] = r_19_4.x;
                        acc_4[5] = r_19_4.y;
                        float2 _f2_506 = make_float2(acc_4[6], acc_4[7]);
                        float2 _f2_507 = make_float2(_unpack_fp4_36[6], _unpack_fp4_36[7]);
                        float2 r_20_4 = fma_f32x2_rn_noftz(_f2_507, _f2_498, _f2_506);
                        acc_4[6] = r_20_4.x;
                        acc_4[7] = r_20_4.y;
                        float _unpack_fp4_37[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_4[1 + 0]));
                            (&_unpack_fp4_37[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_37[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_37[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_37[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_37[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_37[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_37[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_37[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_37[0]), "=f"(_unpack_fp4_37[1]), "=f"(_unpack_fp4_37[2]), "=f"(_unpack_fp4_37[3]), "=f"(_unpack_fp4_37[4]), "=f"(_unpack_fp4_37[5]), "=f"(_unpack_fp4_37[6]), "=f"(_unpack_fp4_37[7]) : "r"(own_4[1 + 0]));
                            #endif
                        }
                        float2 _f2_508 = make_float2(acc_4[8], acc_4[9]);
                        float2 _f2_509 = make_float2(_unpack_fp4_37[0], _unpack_fp4_37[1]);
                        float2 r_21_4 = fma_f32x2_rn_noftz(_f2_509, _f2_498, _f2_508);
                        acc_4[8] = r_21_4.x;
                        acc_4[9] = r_21_4.y;
                        float2 _f2_510 = make_float2(acc_4[10], acc_4[11]);
                        float2 _f2_511 = make_float2(_unpack_fp4_37[2], _unpack_fp4_37[3]);
                        float2 r_22_4 = fma_f32x2_rn_noftz(_f2_511, _f2_498, _f2_510);
                        acc_4[10] = r_22_4.x;
                        acc_4[11] = r_22_4.y;
                        float2 _f2_512 = make_float2(acc_4[12], acc_4[13]);
                        float2 _f2_513 = make_float2(_unpack_fp4_37[4], _unpack_fp4_37[5]);
                        float2 r_23_4 = fma_f32x2_rn_noftz(_f2_513, _f2_498, _f2_512);
                        acc_4[12] = r_23_4.x;
                        acc_4[13] = r_23_4.y;
                        float2 _f2_514 = make_float2(acc_4[14], acc_4[15]);
                        float2 _f2_515 = make_float2(_unpack_fp4_37[6], _unpack_fp4_37[7]);
                        float2 r_24_4 = fma_f32x2_rn_noftz(_f2_515, _f2_498, _f2_514);
                        acc_4[14] = r_24_4.x;
                        acc_4[15] = r_24_4.y;
                        float _unpack_fp4_38[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_4[2 + 0]));
                            (&_unpack_fp4_38[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_38[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_38[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_38[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_38[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_38[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_38[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_38[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_38[0]), "=f"(_unpack_fp4_38[1]), "=f"(_unpack_fp4_38[2]), "=f"(_unpack_fp4_38[3]), "=f"(_unpack_fp4_38[4]), "=f"(_unpack_fp4_38[5]), "=f"(_unpack_fp4_38[6]), "=f"(_unpack_fp4_38[7]) : "r"(own_4[2 + 0]));
                            #endif
                        }
                        float2 _f2_516 = make_float2(acc_4[16], acc_4[17]);
                        float2 _f2_517 = make_float2(_unpack_fp4_38[0], _unpack_fp4_38[1]);
                        float2 r_25_4 = fma_f32x2_rn_noftz(_f2_517, _f2_499, _f2_516);
                        acc_4[16] = r_25_4.x;
                        acc_4[17] = r_25_4.y;
                        float2 _f2_518 = make_float2(acc_4[18], acc_4[19]);
                        float2 _f2_519 = make_float2(_unpack_fp4_38[2], _unpack_fp4_38[3]);
                        float2 r_26_4 = fma_f32x2_rn_noftz(_f2_519, _f2_499, _f2_518);
                        acc_4[18] = r_26_4.x;
                        acc_4[19] = r_26_4.y;
                        float2 _f2_520 = make_float2(acc_4[20], acc_4[21]);
                        float2 _f2_521 = make_float2(_unpack_fp4_38[4], _unpack_fp4_38[5]);
                        float2 r_27_4 = fma_f32x2_rn_noftz(_f2_521, _f2_499, _f2_520);
                        acc_4[20] = r_27_4.x;
                        acc_4[21] = r_27_4.y;
                        float2 _f2_522 = make_float2(acc_4[22], acc_4[23]);
                        float2 _f2_523 = make_float2(_unpack_fp4_38[6], _unpack_fp4_38[7]);
                        float2 r_28_4 = fma_f32x2_rn_noftz(_f2_523, _f2_499, _f2_522);
                        acc_4[22] = r_28_4.x;
                        acc_4[23] = r_28_4.y;
                        float _unpack_fp4_39[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_4[3 + 0]));
                            (&_unpack_fp4_39[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_39[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_39[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_39[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_39[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_39[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_39[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_39[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_39[0]), "=f"(_unpack_fp4_39[1]), "=f"(_unpack_fp4_39[2]), "=f"(_unpack_fp4_39[3]), "=f"(_unpack_fp4_39[4]), "=f"(_unpack_fp4_39[5]), "=f"(_unpack_fp4_39[6]), "=f"(_unpack_fp4_39[7]) : "r"(own_4[3 + 0]));
                            #endif
                        }
                        float2 _f2_524 = make_float2(acc_4[24], acc_4[25]);
                        float2 _f2_525 = make_float2(_unpack_fp4_39[0], _unpack_fp4_39[1]);
                        float2 r_29_4 = fma_f32x2_rn_noftz(_f2_525, _f2_499, _f2_524);
                        acc_4[24] = r_29_4.x;
                        acc_4[25] = r_29_4.y;
                        float2 _f2_526 = make_float2(acc_4[26], acc_4[27]);
                        float2 _f2_527 = make_float2(_unpack_fp4_39[2], _unpack_fp4_39[3]);
                        float2 r_30_4 = fma_f32x2_rn_noftz(_f2_527, _f2_499, _f2_526);
                        acc_4[26] = r_30_4.x;
                        acc_4[27] = r_30_4.y;
                        float2 _f2_528 = make_float2(acc_4[28], acc_4[29]);
                        float2 _f2_529 = make_float2(_unpack_fp4_39[4], _unpack_fp4_39[5]);
                        float2 r_31_4 = fma_f32x2_rn_noftz(_f2_529, _f2_499, _f2_528);
                        acc_4[28] = r_31_4.x;
                        acc_4[29] = r_31_4.y;
                        float2 _f2_530 = make_float2(acc_4[30], acc_4[31]);
                        float2 _f2_531 = make_float2(_unpack_fp4_39[6], _unpack_fp4_39[7]);
                        float2 r_32_4 = fma_f32x2_rn_noftz(_f2_531, _f2_499, _f2_530);
                        acc_4[30] = r_32_4.x;
                        acc_4[31] = r_32_4.y;
                        unsigned int t_4[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 128)) + (unsigned int)(16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 128)) + (unsigned int)(16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 128)) + (unsigned int)(16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 128)) + (unsigned int)(16 * (f ^ 3))));
                        unsigned int s_4[16];
                        unsigned int r_33_4[16];
                        s_4[0] = (((f & 1) != 0) ? t_4[4] : t_4[0]);
                        s_4[4] = (((f & 1) != 0) ? t_4[0] : t_4[4]);
                        s_4[8] = (((f & 1) != 0) ? t_4[12] : t_4[8]);
                        s_4[12] = (((f & 1) != 0) ? t_4[8] : t_4[12]);
                        s_4[1] = (((f & 1) != 0) ? t_4[5] : t_4[1]);
                        s_4[5] = (((f & 1) != 0) ? t_4[1] : t_4[5]);
                        s_4[9] = (((f & 1) != 0) ? t_4[13] : t_4[9]);
                        s_4[13] = (((f & 1) != 0) ? t_4[9] : t_4[13]);
                        s_4[2] = (((f & 1) != 0) ? t_4[6] : t_4[2]);
                        s_4[6] = (((f & 1) != 0) ? t_4[2] : t_4[6]);
                        s_4[10] = (((f & 1) != 0) ? t_4[14] : t_4[10]);
                        s_4[14] = (((f & 1) != 0) ? t_4[10] : t_4[14]);
                        s_4[3] = (((f & 1) != 0) ? t_4[7] : t_4[3]);
                        s_4[7] = (((f & 1) != 0) ? t_4[3] : t_4[7]);
                        s_4[11] = (((f & 1) != 0) ? t_4[15] : t_4[11]);
                        s_4[15] = (((f & 1) != 0) ? t_4[11] : t_4[15]);
                        r_33_4[0] = (((f & 2) != 0) ? s_4[8] : s_4[0]);
                        r_33_4[4] = (((f & 2) != 0) ? s_4[12] : s_4[4]);
                        r_33_4[8] = (((f & 2) != 0) ? s_4[0] : s_4[8]);
                        r_33_4[12] = (((f & 2) != 0) ? s_4[4] : s_4[12]);
                        r_33_4[1] = (((f & 2) != 0) ? s_4[9] : s_4[1]);
                        r_33_4[5] = (((f & 2) != 0) ? s_4[13] : s_4[5]);
                        r_33_4[9] = (((f & 2) != 0) ? s_4[1] : s_4[9]);
                        r_33_4[13] = (((f & 2) != 0) ? s_4[5] : s_4[13]);
                        r_33_4[2] = (((f & 2) != 0) ? s_4[10] : s_4[2]);
                        r_33_4[6] = (((f & 2) != 0) ? s_4[14] : s_4[6]);
                        r_33_4[10] = (((f & 2) != 0) ? s_4[2] : s_4[10]);
                        r_33_4[14] = (((f & 2) != 0) ? s_4[6] : s_4[14]);
                        r_33_4[3] = (((f & 2) != 0) ? s_4[11] : s_4[3]);
                        r_33_4[7] = (((f & 2) != 0) ? s_4[15] : s_4[7]);
                        r_33_4[11] = (((f & 2) != 0) ? s_4[3] : s_4[11]);
                        r_33_4[15] = (((f & 2) != 0) ? s_4[7] : s_4[15]);
                        float2 _f2_532 = make_float2(acc_4[0], acc_4[1]);
                        float2 _f2_533 = make_float2(__uint_as_float(r_33_4[0] << 16), __uint_as_float(r_33_4[0] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_64;
                        {
                            float2 _packed_f32x2_78_0 = _f2_532;
                            float2 _packed_f32x2_78_1 = _f2_533;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_64)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_78_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_78_1)));
                        }
                        acc_4[0] = _packed_add_f32x2_64.x;
                        acc_4[1] = _packed_add_f32x2_64.y;
                        float2 _f2_534 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_64, _packed_add_f32x2_64, _f2_534).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_64, _packed_add_f32x2_64, _f2_534).y;
                        float2 _f2_535 = make_float2(acc_4[2], acc_4[3]);
                        float2 _f2_536 = make_float2(__uint_as_float(r_33_4[1] << 16), __uint_as_float(r_33_4[1] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_65;
                        {
                            float2 _packed_f32x2_79_0 = _f2_535;
                            float2 _packed_f32x2_79_1 = _f2_536;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_65)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_79_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_79_1)));
                        }
                        acc_4[2] = _packed_add_f32x2_65.x;
                        acc_4[3] = _packed_add_f32x2_65.y;
                        float2 _f2_537 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_65, _packed_add_f32x2_65, _f2_537).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_65, _packed_add_f32x2_65, _f2_537).y;
                        float2 _f2_538 = make_float2(acc_4[4], acc_4[5]);
                        float2 _f2_539 = make_float2(__uint_as_float(r_33_4[2] << 16), __uint_as_float(r_33_4[2] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_66;
                        {
                            float2 _packed_f32x2_80_0 = _f2_538;
                            float2 _packed_f32x2_80_1 = _f2_539;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_66)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_80_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_80_1)));
                        }
                        acc_4[4] = _packed_add_f32x2_66.x;
                        acc_4[5] = _packed_add_f32x2_66.y;
                        float2 _f2_540 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_66, _packed_add_f32x2_66, _f2_540).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_66, _packed_add_f32x2_66, _f2_540).y;
                        float2 _f2_541 = make_float2(acc_4[6], acc_4[7]);
                        float2 _f2_542 = make_float2(__uint_as_float(r_33_4[3] << 16), __uint_as_float(r_33_4[3] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_67;
                        {
                            float2 _packed_f32x2_81_0 = _f2_541;
                            float2 _packed_f32x2_81_1 = _f2_542;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_67)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_81_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_81_1)));
                        }
                        acc_4[6] = _packed_add_f32x2_67.x;
                        acc_4[7] = _packed_add_f32x2_67.y;
                        float2 _f2_543 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_67, _packed_add_f32x2_67, _f2_543).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_67, _packed_add_f32x2_67, _f2_543).y;
                        float2 _f2_544 = make_float2(acc_4[8], acc_4[9]);
                        float2 _f2_545 = make_float2(__uint_as_float(r_33_4[4] << 16), __uint_as_float(r_33_4[4] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_68;
                        {
                            float2 _packed_f32x2_82_0 = _f2_544;
                            float2 _packed_f32x2_82_1 = _f2_545;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_68)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_82_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_82_1)));
                        }
                        acc_4[8] = _packed_add_f32x2_68.x;
                        acc_4[9] = _packed_add_f32x2_68.y;
                        float2 _f2_546 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_68, _packed_add_f32x2_68, _f2_546).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_68, _packed_add_f32x2_68, _f2_546).y;
                        float2 _f2_547 = make_float2(acc_4[10], acc_4[11]);
                        float2 _f2_548 = make_float2(__uint_as_float(r_33_4[5] << 16), __uint_as_float(r_33_4[5] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_69;
                        {
                            float2 _packed_f32x2_83_0 = _f2_547;
                            float2 _packed_f32x2_83_1 = _f2_548;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_69)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_83_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_83_1)));
                        }
                        acc_4[10] = _packed_add_f32x2_69.x;
                        acc_4[11] = _packed_add_f32x2_69.y;
                        float2 _f2_549 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_69, _packed_add_f32x2_69, _f2_549).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_69, _packed_add_f32x2_69, _f2_549).y;
                        float2 _f2_550 = make_float2(acc_4[12], acc_4[13]);
                        float2 _f2_551 = make_float2(__uint_as_float(r_33_4[6] << 16), __uint_as_float(r_33_4[6] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_70;
                        {
                            float2 _packed_f32x2_84_0 = _f2_550;
                            float2 _packed_f32x2_84_1 = _f2_551;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_70)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_84_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_84_1)));
                        }
                        acc_4[12] = _packed_add_f32x2_70.x;
                        acc_4[13] = _packed_add_f32x2_70.y;
                        float2 _f2_552 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_70, _packed_add_f32x2_70, _f2_552).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_70, _packed_add_f32x2_70, _f2_552).y;
                        float2 _f2_553 = make_float2(acc_4[14], acc_4[15]);
                        float2 _f2_554 = make_float2(__uint_as_float(r_33_4[7] << 16), __uint_as_float(r_33_4[7] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_71;
                        {
                            float2 _packed_f32x2_85_0 = _f2_553;
                            float2 _packed_f32x2_85_1 = _f2_554;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_71)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_85_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_85_1)));
                        }
                        acc_4[14] = _packed_add_f32x2_71.x;
                        acc_4[15] = _packed_add_f32x2_71.y;
                        float2 _f2_555 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_71, _packed_add_f32x2_71, _f2_555).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_71, _packed_add_f32x2_71, _f2_555).y;
                        float2 _f2_556 = make_float2(acc_4[16], acc_4[17]);
                        float2 _f2_557 = make_float2(__uint_as_float(r_33_4[8] << 16), __uint_as_float(r_33_4[8] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_72;
                        {
                            float2 _packed_f32x2_86_0 = _f2_556;
                            float2 _packed_f32x2_86_1 = _f2_557;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_72)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_86_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_86_1)));
                        }
                        acc_4[16] = _packed_add_f32x2_72.x;
                        acc_4[17] = _packed_add_f32x2_72.y;
                        float2 _f2_558 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_72, _packed_add_f32x2_72, _f2_558).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_72, _packed_add_f32x2_72, _f2_558).y;
                        float2 _f2_559 = make_float2(acc_4[18], acc_4[19]);
                        float2 _f2_560 = make_float2(__uint_as_float(r_33_4[9] << 16), __uint_as_float(r_33_4[9] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_73;
                        {
                            float2 _packed_f32x2_87_0 = _f2_559;
                            float2 _packed_f32x2_87_1 = _f2_560;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_73)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_87_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_87_1)));
                        }
                        acc_4[18] = _packed_add_f32x2_73.x;
                        acc_4[19] = _packed_add_f32x2_73.y;
                        float2 _f2_561 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_73, _packed_add_f32x2_73, _f2_561).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_73, _packed_add_f32x2_73, _f2_561).y;
                        float2 _f2_562 = make_float2(acc_4[20], acc_4[21]);
                        float2 _f2_563 = make_float2(__uint_as_float(r_33_4[10] << 16), __uint_as_float(r_33_4[10] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_74;
                        {
                            float2 _packed_f32x2_88_0 = _f2_562;
                            float2 _packed_f32x2_88_1 = _f2_563;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_74)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_88_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_88_1)));
                        }
                        acc_4[20] = _packed_add_f32x2_74.x;
                        acc_4[21] = _packed_add_f32x2_74.y;
                        float2 _f2_564 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_74, _packed_add_f32x2_74, _f2_564).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_74, _packed_add_f32x2_74, _f2_564).y;
                        float2 _f2_565 = make_float2(acc_4[22], acc_4[23]);
                        float2 _f2_566 = make_float2(__uint_as_float(r_33_4[11] << 16), __uint_as_float(r_33_4[11] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_75;
                        {
                            float2 _packed_f32x2_89_0 = _f2_565;
                            float2 _packed_f32x2_89_1 = _f2_566;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_75)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_89_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_89_1)));
                        }
                        acc_4[22] = _packed_add_f32x2_75.x;
                        acc_4[23] = _packed_add_f32x2_75.y;
                        float2 _f2_567 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_75, _packed_add_f32x2_75, _f2_567).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_75, _packed_add_f32x2_75, _f2_567).y;
                        float2 _f2_568 = make_float2(acc_4[24], acc_4[25]);
                        float2 _f2_569 = make_float2(__uint_as_float(r_33_4[12] << 16), __uint_as_float(r_33_4[12] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_76;
                        {
                            float2 _packed_f32x2_90_0 = _f2_568;
                            float2 _packed_f32x2_90_1 = _f2_569;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_76)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_90_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_90_1)));
                        }
                        acc_4[24] = _packed_add_f32x2_76.x;
                        acc_4[25] = _packed_add_f32x2_76.y;
                        float2 _f2_570 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_76, _packed_add_f32x2_76, _f2_570).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_76, _packed_add_f32x2_76, _f2_570).y;
                        float2 _f2_571 = make_float2(acc_4[26], acc_4[27]);
                        float2 _f2_572 = make_float2(__uint_as_float(r_33_4[13] << 16), __uint_as_float(r_33_4[13] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_77;
                        {
                            float2 _packed_f32x2_91_0 = _f2_571;
                            float2 _packed_f32x2_91_1 = _f2_572;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_77)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_91_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_91_1)));
                        }
                        acc_4[26] = _packed_add_f32x2_77.x;
                        acc_4[27] = _packed_add_f32x2_77.y;
                        float2 _f2_573 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_77, _packed_add_f32x2_77, _f2_573).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_77, _packed_add_f32x2_77, _f2_573).y;
                        float2 _f2_574 = make_float2(acc_4[28], acc_4[29]);
                        float2 _f2_575 = make_float2(__uint_as_float(r_33_4[14] << 16), __uint_as_float(r_33_4[14] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_78;
                        {
                            float2 _packed_f32x2_92_0 = _f2_574;
                            float2 _packed_f32x2_92_1 = _f2_575;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_78)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_92_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_92_1)));
                        }
                        acc_4[28] = _packed_add_f32x2_78.x;
                        acc_4[29] = _packed_add_f32x2_78.y;
                        float2 _f2_576 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_78, _packed_add_f32x2_78, _f2_576).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_78, _packed_add_f32x2_78, _f2_576).y;
                        float2 _f2_577 = make_float2(acc_4[30], acc_4[31]);
                        float2 _f2_578 = make_float2(__uint_as_float(r_33_4[15] << 16), __uint_as_float(r_33_4[15] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_79;
                        {
                            float2 _packed_f32x2_93_0 = _f2_577;
                            float2 _packed_f32x2_93_1 = _f2_578;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_79)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_93_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_93_1)));
                        }
                        acc_4[30] = _packed_add_f32x2_79.x;
                        acc_4[31] = _packed_add_f32x2_79.y;
                        float2 _f2_579 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_79, _packed_add_f32x2_79, _f2_579).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_79, _packed_add_f32x2_79, _f2_579).y;
                        uint32_t acc_bf16_4[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_4[_lp*2 + 0], acc_4[_lp*2+1 + 0]));
                            acc_bf16_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 0)))), "r"(acc_bf16_4[0]), "r"(acc_bf16_4[1]), "r"(acc_bf16_4[2]), "r"(acc_bf16_4[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 1)))), "r"(acc_bf16_4[4]), "r"(acc_bf16_4[5]), "r"(acc_bf16_4[6]), "r"(acc_bf16_4[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 2)))), "r"(acc_bf16_4[8]), "r"(acc_bf16_4[9]), "r"(acc_bf16_4[10]), "r"(acc_bf16_4[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 3)))), "r"(acc_bf16_4[12]), "r"(acc_bf16_4[13]), "r"(acc_bf16_4[14]), "r"(acc_bf16_4[15]) : "memory");
                        __syncwarp();
                        int c_4 = lane_3;
                        int l_4 = c_4 >> 2;
                        unsigned int w4_4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_4 + (c_4 & 3 ^ l_4 >> 1 & 3)))));
                        long long idx_4 = drow * 3072 + (long long)(4 * (512 + c_4));
                        {
                            int4 _iv4 = make_int4(w4_4[0 + 0], w4_4[0 + 1], w4_4[0 + 2], w4_4[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_4)) = _iv4;
                        }
                        int c_34_4 = lane_3 + 32;
                        int l_35_4 = c_34_4 >> 2;
                        unsigned int w4_36_4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_36_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_4[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_35_4 + (c_34_4 & 3 ^ l_35_4 >> 1 & 3)))));
                        long long idx_37_4 = drow * 3072 + (long long)(4 * (512 + c_34_4));
                        {
                            int4 _iv4 = make_int4(w4_36_4[0 + 0], w4_36_4[0 + 1], w4_36_4[0 + 2], w4_36_4[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_37_4)) = _iv4;
                        }
                        int c_38_4 = lane_3 + 64;
                        int l_39_4 = c_38_4 >> 2;
                        unsigned int w4_40_4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_40_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_4[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_39_4 + (c_38_4 & 3 ^ l_39_4 >> 1 & 3)))));
                        long long idx_41_4 = drow * 3072 + (long long)(4 * (512 + c_38_4));
                        {
                            int4 _iv4 = make_int4(w4_40_4[0 + 0], w4_40_4[0 + 1], w4_40_4[0 + 2], w4_40_4[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_41_4)) = _iv4;
                        }
                        int c_42_4 = lane_3 + 96;
                        int l_43_4 = c_42_4 >> 2;
                        unsigned int w4_44_4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_44_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_4[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_43_4 + (c_42_4 & 3 ^ l_43_4 >> 1 & 3)))));
                        long long idx_45_4 = drow * 3072 + (long long)(4 * (512 + c_42_4));
                        {
                            int4 _iv4 = make_int4(w4_44_4[0 + 0], w4_44_4[0 + 1], w4_44_4[0 + 2], w4_44_4[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_45_4)) = _iv4;
                        }
                        __syncwarp();
                        int zlane_2 = rb_off + 4096 + 128 * lane_3;
                        int l7_2 = lane_3 & 7;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_2 + 16 * (l7_2 ^ 0))), "r"(__as_u32(acc_4[0])), "r"(__as_u32(acc_4[1])), "r"(__as_u32(acc_4[2])), "r"(__as_u32(acc_4[3])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_2 + 16 * (l7_2 ^ 1))), "r"(__as_u32(acc_4[4])), "r"(__as_u32(acc_4[5])), "r"(__as_u32(acc_4[6])), "r"(__as_u32(acc_4[7])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_2 + 16 * (l7_2 ^ 2))), "r"(__as_u32(acc_4[8])), "r"(__as_u32(acc_4[9])), "r"(__as_u32(acc_4[10])), "r"(__as_u32(acc_4[11])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_2 + 16 * (l7_2 ^ 3))), "r"(__as_u32(acc_4[12])), "r"(__as_u32(acc_4[13])), "r"(__as_u32(acc_4[14])), "r"(__as_u32(acc_4[15])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_2 + 16 * (l7_2 ^ 4))), "r"(__as_u32(acc_4[16])), "r"(__as_u32(acc_4[17])), "r"(__as_u32(acc_4[18])), "r"(__as_u32(acc_4[19])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_2 + 16 * (l7_2 ^ 5))), "r"(__as_u32(acc_4[20])), "r"(__as_u32(acc_4[21])), "r"(__as_u32(acc_4[22])), "r"(__as_u32(acc_4[23])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_2 + 16 * (l7_2 ^ 6))), "r"(__as_u32(acc_4[24])), "r"(__as_u32(acc_4[25])), "r"(__as_u32(acc_4[26])), "r"(__as_u32(acc_4[27])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_2 + 16 * (l7_2 ^ 7))), "r"(__as_u32(acc_4[28])), "r"(__as_u32(acc_4[29])), "r"(__as_u32(acc_4[30])), "r"(__as_u32(acc_4[31])) : "memory");
                    }
                    {
                        unsigned int own_5[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&own_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_5[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * (lane_3 + 160))));
                        unsigned int own_sc_5 = (unsigned int)wsm16[(dob_off + 3072 >> 1) + (lane_3 + 160)];
                        float acc_5[32];
                        acc_5[0] = 0.0f;
                        acc_5[1] = 0.0f;
                        acc_5[2] = 0.0f;
                        acc_5[3] = 0.0f;
                        acc_5[4] = 0.0f;
                        acc_5[5] = 0.0f;
                        acc_5[6] = 0.0f;
                        acc_5[7] = 0.0f;
                        acc_5[8] = 0.0f;
                        acc_5[9] = 0.0f;
                        acc_5[10] = 0.0f;
                        acc_5[11] = 0.0f;
                        acc_5[12] = 0.0f;
                        acc_5[13] = 0.0f;
                        acc_5[14] = 0.0f;
                        acc_5[15] = 0.0f;
                        acc_5[16] = 0.0f;
                        acc_5[17] = 0.0f;
                        acc_5[18] = 0.0f;
                        acc_5[19] = 0.0f;
                        acc_5[20] = 0.0f;
                        acc_5[21] = 0.0f;
                        acc_5[22] = 0.0f;
                        acc_5[23] = 0.0f;
                        acc_5[24] = 0.0f;
                        acc_5[25] = 0.0f;
                        acc_5[26] = 0.0f;
                        acc_5[27] = 0.0f;
                        acc_5[28] = 0.0f;
                        acc_5[29] = 0.0f;
                        acc_5[30] = 0.0f;
                        acc_5[31] = 0.0f;
                        float2 _fp8x2_decode_10;
                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                            "mov.b32 {lo, hi}, pair;\n"
                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                            : "=f"(_fp8x2_decode_10.x), "=f"(_fp8x2_decode_10.y) : "h"((uint16_t)pf_s[5]));
                        float _mul_20;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(_fp8x2_decode_10.x), "f"(__uint_as_float(w_7)));
                        float s0_5 = _mul_20;
                        float _mul_21;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(_fp8x2_decode_10.y), "f"(__uint_as_float(w_7)));
                        float s1_5 = _mul_21;
                        float2 _f2_580 = make_float2(s0_5, s0_5);
                        float2 _f2_581 = make_float2(s1_5, s1_5);
                        float _unpack_fp4_40[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[20 + 0]));
                            (&_unpack_fp4_40[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_40[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_40[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_40[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_40[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_40[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_40[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_40[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_40[0]), "=f"(_unpack_fp4_40[1]), "=f"(_unpack_fp4_40[2]), "=f"(_unpack_fp4_40[3]), "=f"(_unpack_fp4_40[4]), "=f"(_unpack_fp4_40[5]), "=f"(_unpack_fp4_40[6]), "=f"(_unpack_fp4_40[7]) : "r"(pf_v[20 + 0]));
                            #endif
                        }
                        float2 _f2_582 = make_float2(acc_5[0], acc_5[1]);
                        float2 _f2_583 = make_float2(_unpack_fp4_40[0], _unpack_fp4_40[1]);
                        float2 r_36 = fma_f32x2_rn_noftz(_f2_583, _f2_580, _f2_582);
                        acc_5[0] = r_36.x;
                        acc_5[1] = r_36.y;
                        float2 _f2_584 = make_float2(acc_5[2], acc_5[3]);
                        float2 _f2_585 = make_float2(_unpack_fp4_40[2], _unpack_fp4_40[3]);
                        float2 r_0_5 = fma_f32x2_rn_noftz(_f2_585, _f2_580, _f2_584);
                        acc_5[2] = r_0_5.x;
                        acc_5[3] = r_0_5.y;
                        float2 _f2_586 = make_float2(acc_5[4], acc_5[5]);
                        float2 _f2_587 = make_float2(_unpack_fp4_40[4], _unpack_fp4_40[5]);
                        float2 r_1_5 = fma_f32x2_rn_noftz(_f2_587, _f2_580, _f2_586);
                        acc_5[4] = r_1_5.x;
                        acc_5[5] = r_1_5.y;
                        float2 _f2_588 = make_float2(acc_5[6], acc_5[7]);
                        float2 _f2_589 = make_float2(_unpack_fp4_40[6], _unpack_fp4_40[7]);
                        float2 r_2_5 = fma_f32x2_rn_noftz(_f2_589, _f2_580, _f2_588);
                        acc_5[6] = r_2_5.x;
                        acc_5[7] = r_2_5.y;
                        float _unpack_fp4_41[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[21 + 0]));
                            (&_unpack_fp4_41[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_41[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_41[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_41[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_41[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_41[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_41[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_41[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_41[0]), "=f"(_unpack_fp4_41[1]), "=f"(_unpack_fp4_41[2]), "=f"(_unpack_fp4_41[3]), "=f"(_unpack_fp4_41[4]), "=f"(_unpack_fp4_41[5]), "=f"(_unpack_fp4_41[6]), "=f"(_unpack_fp4_41[7]) : "r"(pf_v[21 + 0]));
                            #endif
                        }
                        float2 _f2_590 = make_float2(acc_5[8], acc_5[9]);
                        float2 _f2_591 = make_float2(_unpack_fp4_41[0], _unpack_fp4_41[1]);
                        float2 r_3_5 = fma_f32x2_rn_noftz(_f2_591, _f2_580, _f2_590);
                        acc_5[8] = r_3_5.x;
                        acc_5[9] = r_3_5.y;
                        float2 _f2_592 = make_float2(acc_5[10], acc_5[11]);
                        float2 _f2_593 = make_float2(_unpack_fp4_41[2], _unpack_fp4_41[3]);
                        float2 r_4_5 = fma_f32x2_rn_noftz(_f2_593, _f2_580, _f2_592);
                        acc_5[10] = r_4_5.x;
                        acc_5[11] = r_4_5.y;
                        float2 _f2_594 = make_float2(acc_5[12], acc_5[13]);
                        float2 _f2_595 = make_float2(_unpack_fp4_41[4], _unpack_fp4_41[5]);
                        float2 r_5_5 = fma_f32x2_rn_noftz(_f2_595, _f2_580, _f2_594);
                        acc_5[12] = r_5_5.x;
                        acc_5[13] = r_5_5.y;
                        float2 _f2_596 = make_float2(acc_5[14], acc_5[15]);
                        float2 _f2_597 = make_float2(_unpack_fp4_41[6], _unpack_fp4_41[7]);
                        float2 r_6_5 = fma_f32x2_rn_noftz(_f2_597, _f2_580, _f2_596);
                        acc_5[14] = r_6_5.x;
                        acc_5[15] = r_6_5.y;
                        float _unpack_fp4_42[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[22 + 0]));
                            (&_unpack_fp4_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_42[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_42[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_42[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_42[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_42[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_42[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_42[0]), "=f"(_unpack_fp4_42[1]), "=f"(_unpack_fp4_42[2]), "=f"(_unpack_fp4_42[3]), "=f"(_unpack_fp4_42[4]), "=f"(_unpack_fp4_42[5]), "=f"(_unpack_fp4_42[6]), "=f"(_unpack_fp4_42[7]) : "r"(pf_v[22 + 0]));
                            #endif
                        }
                        float2 _f2_598 = make_float2(acc_5[16], acc_5[17]);
                        float2 _f2_599 = make_float2(_unpack_fp4_42[0], _unpack_fp4_42[1]);
                        float2 r_7_5 = fma_f32x2_rn_noftz(_f2_599, _f2_581, _f2_598);
                        acc_5[16] = r_7_5.x;
                        acc_5[17] = r_7_5.y;
                        float2 _f2_600 = make_float2(acc_5[18], acc_5[19]);
                        float2 _f2_601 = make_float2(_unpack_fp4_42[2], _unpack_fp4_42[3]);
                        float2 r_8_5 = fma_f32x2_rn_noftz(_f2_601, _f2_581, _f2_600);
                        acc_5[18] = r_8_5.x;
                        acc_5[19] = r_8_5.y;
                        float2 _f2_602 = make_float2(acc_5[20], acc_5[21]);
                        float2 _f2_603 = make_float2(_unpack_fp4_42[4], _unpack_fp4_42[5]);
                        float2 r_9_5 = fma_f32x2_rn_noftz(_f2_603, _f2_581, _f2_602);
                        acc_5[20] = r_9_5.x;
                        acc_5[21] = r_9_5.y;
                        float2 _f2_604 = make_float2(acc_5[22], acc_5[23]);
                        float2 _f2_605 = make_float2(_unpack_fp4_42[6], _unpack_fp4_42[7]);
                        float2 r_10_5 = fma_f32x2_rn_noftz(_f2_605, _f2_581, _f2_604);
                        acc_5[22] = r_10_5.x;
                        acc_5[23] = r_10_5.y;
                        float _unpack_fp4_43[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(pf_v[23 + 0]));
                            (&_unpack_fp4_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_43[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_43[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_43[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_43[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_43[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_43[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_43[0]), "=f"(_unpack_fp4_43[1]), "=f"(_unpack_fp4_43[2]), "=f"(_unpack_fp4_43[3]), "=f"(_unpack_fp4_43[4]), "=f"(_unpack_fp4_43[5]), "=f"(_unpack_fp4_43[6]), "=f"(_unpack_fp4_43[7]) : "r"(pf_v[23 + 0]));
                            #endif
                        }
                        float2 _f2_606 = make_float2(acc_5[24], acc_5[25]);
                        float2 _f2_607 = make_float2(_unpack_fp4_43[0], _unpack_fp4_43[1]);
                        float2 r_11_5 = fma_f32x2_rn_noftz(_f2_607, _f2_581, _f2_606);
                        acc_5[24] = r_11_5.x;
                        acc_5[25] = r_11_5.y;
                        float2 _f2_608 = make_float2(acc_5[26], acc_5[27]);
                        float2 _f2_609 = make_float2(_unpack_fp4_43[2], _unpack_fp4_43[3]);
                        float2 r_12_5 = fma_f32x2_rn_noftz(_f2_609, _f2_581, _f2_608);
                        acc_5[26] = r_12_5.x;
                        acc_5[27] = r_12_5.y;
                        float2 _f2_610 = make_float2(acc_5[28], acc_5[29]);
                        float2 _f2_611 = make_float2(_unpack_fp4_43[4], _unpack_fp4_43[5]);
                        float2 r_13_5 = fma_f32x2_rn_noftz(_f2_611, _f2_581, _f2_610);
                        acc_5[28] = r_13_5.x;
                        acc_5[29] = r_13_5.y;
                        float2 _f2_612 = make_float2(acc_5[30], acc_5[31]);
                        float2 _f2_613 = make_float2(_unpack_fp4_43[6], _unpack_fp4_43[7]);
                        float2 r_14_5 = fma_f32x2_rn_noftz(_f2_613, _f2_581, _f2_612);
                        acc_5[30] = r_14_5.x;
                        acc_5[31] = r_14_5.y;
                        float2 _fp8x2_decode_11;
                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                            "mov.b32 {lo, hi}, pair;\n"
                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                            : "=f"(_fp8x2_decode_11.x), "=f"(_fp8x2_decode_11.y) : "h"((uint16_t)own_sc_5));
                        float _mul_22;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(_fp8x2_decode_11.x), "f"(gs_me));
                        float s0_15_5 = _mul_22;
                        float _mul_23;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_fp8x2_decode_11.y), "f"(gs_me));
                        float s1_16_5 = _mul_23;
                        float2 _f2_614 = make_float2(s0_15_5, s0_15_5);
                        float2 _f2_615 = make_float2(s1_16_5, s1_16_5);
                        float _unpack_fp4_44[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_5[0]));
                            (&_unpack_fp4_44[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_44[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_44[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_44[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_44[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_44[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_44[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_44[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_44[0]), "=f"(_unpack_fp4_44[1]), "=f"(_unpack_fp4_44[2]), "=f"(_unpack_fp4_44[3]), "=f"(_unpack_fp4_44[4]), "=f"(_unpack_fp4_44[5]), "=f"(_unpack_fp4_44[6]), "=f"(_unpack_fp4_44[7]) : "r"(own_5[0]));
                            #endif
                        }
                        float2 _f2_616 = make_float2(acc_5[0], acc_5[1]);
                        float2 _f2_617 = make_float2(_unpack_fp4_44[0], _unpack_fp4_44[1]);
                        float2 r_17_5 = fma_f32x2_rn_noftz(_f2_617, _f2_614, _f2_616);
                        acc_5[0] = r_17_5.x;
                        acc_5[1] = r_17_5.y;
                        float2 _f2_618 = make_float2(acc_5[2], acc_5[3]);
                        float2 _f2_619 = make_float2(_unpack_fp4_44[2], _unpack_fp4_44[3]);
                        float2 r_18_5 = fma_f32x2_rn_noftz(_f2_619, _f2_614, _f2_618);
                        acc_5[2] = r_18_5.x;
                        acc_5[3] = r_18_5.y;
                        float2 _f2_620 = make_float2(acc_5[4], acc_5[5]);
                        float2 _f2_621 = make_float2(_unpack_fp4_44[4], _unpack_fp4_44[5]);
                        float2 r_19_5 = fma_f32x2_rn_noftz(_f2_621, _f2_614, _f2_620);
                        acc_5[4] = r_19_5.x;
                        acc_5[5] = r_19_5.y;
                        float2 _f2_622 = make_float2(acc_5[6], acc_5[7]);
                        float2 _f2_623 = make_float2(_unpack_fp4_44[6], _unpack_fp4_44[7]);
                        float2 r_20_5 = fma_f32x2_rn_noftz(_f2_623, _f2_614, _f2_622);
                        acc_5[6] = r_20_5.x;
                        acc_5[7] = r_20_5.y;
                        float _unpack_fp4_45[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_5[1 + 0]));
                            (&_unpack_fp4_45[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_45[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_45[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_45[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_45[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_45[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_45[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_45[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_45[0]), "=f"(_unpack_fp4_45[1]), "=f"(_unpack_fp4_45[2]), "=f"(_unpack_fp4_45[3]), "=f"(_unpack_fp4_45[4]), "=f"(_unpack_fp4_45[5]), "=f"(_unpack_fp4_45[6]), "=f"(_unpack_fp4_45[7]) : "r"(own_5[1 + 0]));
                            #endif
                        }
                        float2 _f2_624 = make_float2(acc_5[8], acc_5[9]);
                        float2 _f2_625 = make_float2(_unpack_fp4_45[0], _unpack_fp4_45[1]);
                        float2 r_21_5 = fma_f32x2_rn_noftz(_f2_625, _f2_614, _f2_624);
                        acc_5[8] = r_21_5.x;
                        acc_5[9] = r_21_5.y;
                        float2 _f2_626 = make_float2(acc_5[10], acc_5[11]);
                        float2 _f2_627 = make_float2(_unpack_fp4_45[2], _unpack_fp4_45[3]);
                        float2 r_22_5 = fma_f32x2_rn_noftz(_f2_627, _f2_614, _f2_626);
                        acc_5[10] = r_22_5.x;
                        acc_5[11] = r_22_5.y;
                        float2 _f2_628 = make_float2(acc_5[12], acc_5[13]);
                        float2 _f2_629 = make_float2(_unpack_fp4_45[4], _unpack_fp4_45[5]);
                        float2 r_23_5 = fma_f32x2_rn_noftz(_f2_629, _f2_614, _f2_628);
                        acc_5[12] = r_23_5.x;
                        acc_5[13] = r_23_5.y;
                        float2 _f2_630 = make_float2(acc_5[14], acc_5[15]);
                        float2 _f2_631 = make_float2(_unpack_fp4_45[6], _unpack_fp4_45[7]);
                        float2 r_24_5 = fma_f32x2_rn_noftz(_f2_631, _f2_614, _f2_630);
                        acc_5[14] = r_24_5.x;
                        acc_5[15] = r_24_5.y;
                        float _unpack_fp4_46[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_5[2 + 0]));
                            (&_unpack_fp4_46[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_46[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_46[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_46[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_46[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_46[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_46[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_46[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_46[0]), "=f"(_unpack_fp4_46[1]), "=f"(_unpack_fp4_46[2]), "=f"(_unpack_fp4_46[3]), "=f"(_unpack_fp4_46[4]), "=f"(_unpack_fp4_46[5]), "=f"(_unpack_fp4_46[6]), "=f"(_unpack_fp4_46[7]) : "r"(own_5[2 + 0]));
                            #endif
                        }
                        float2 _f2_632 = make_float2(acc_5[16], acc_5[17]);
                        float2 _f2_633 = make_float2(_unpack_fp4_46[0], _unpack_fp4_46[1]);
                        float2 r_25_5 = fma_f32x2_rn_noftz(_f2_633, _f2_615, _f2_632);
                        acc_5[16] = r_25_5.x;
                        acc_5[17] = r_25_5.y;
                        float2 _f2_634 = make_float2(acc_5[18], acc_5[19]);
                        float2 _f2_635 = make_float2(_unpack_fp4_46[2], _unpack_fp4_46[3]);
                        float2 r_26_5 = fma_f32x2_rn_noftz(_f2_635, _f2_615, _f2_634);
                        acc_5[18] = r_26_5.x;
                        acc_5[19] = r_26_5.y;
                        float2 _f2_636 = make_float2(acc_5[20], acc_5[21]);
                        float2 _f2_637 = make_float2(_unpack_fp4_46[4], _unpack_fp4_46[5]);
                        float2 r_27_5 = fma_f32x2_rn_noftz(_f2_637, _f2_615, _f2_636);
                        acc_5[20] = r_27_5.x;
                        acc_5[21] = r_27_5.y;
                        float2 _f2_638 = make_float2(acc_5[22], acc_5[23]);
                        float2 _f2_639 = make_float2(_unpack_fp4_46[6], _unpack_fp4_46[7]);
                        float2 r_28_5 = fma_f32x2_rn_noftz(_f2_639, _f2_615, _f2_638);
                        acc_5[22] = r_28_5.x;
                        acc_5[23] = r_28_5.y;
                        float _unpack_fp4_47[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own_5[3 + 0]));
                            (&_unpack_fp4_47[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_47[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_47[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_47[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_47[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_47[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_47[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_47[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_47[0]), "=f"(_unpack_fp4_47[1]), "=f"(_unpack_fp4_47[2]), "=f"(_unpack_fp4_47[3]), "=f"(_unpack_fp4_47[4]), "=f"(_unpack_fp4_47[5]), "=f"(_unpack_fp4_47[6]), "=f"(_unpack_fp4_47[7]) : "r"(own_5[3 + 0]));
                            #endif
                        }
                        float2 _f2_640 = make_float2(acc_5[24], acc_5[25]);
                        float2 _f2_641 = make_float2(_unpack_fp4_47[0], _unpack_fp4_47[1]);
                        float2 r_29_5 = fma_f32x2_rn_noftz(_f2_641, _f2_615, _f2_640);
                        acc_5[24] = r_29_5.x;
                        acc_5[25] = r_29_5.y;
                        float2 _f2_642 = make_float2(acc_5[26], acc_5[27]);
                        float2 _f2_643 = make_float2(_unpack_fp4_47[2], _unpack_fp4_47[3]);
                        float2 r_30_5 = fma_f32x2_rn_noftz(_f2_643, _f2_615, _f2_642);
                        acc_5[26] = r_30_5.x;
                        acc_5[27] = r_30_5.y;
                        float2 _f2_644 = make_float2(acc_5[28], acc_5[29]);
                        float2 _f2_645 = make_float2(_unpack_fp4_47[4], _unpack_fp4_47[5]);
                        float2 r_31_5 = fma_f32x2_rn_noftz(_f2_645, _f2_615, _f2_644);
                        acc_5[28] = r_31_5.x;
                        acc_5[29] = r_31_5.y;
                        float2 _f2_646 = make_float2(acc_5[30], acc_5[31]);
                        float2 _f2_647 = make_float2(_unpack_fp4_47[6], _unpack_fp4_47[7]);
                        float2 r_32_5 = fma_f32x2_rn_noftz(_f2_647, _f2_615, _f2_646);
                        acc_5[30] = r_32_5.x;
                        acc_5[31] = r_32_5.y;
                        unsigned int t_5[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_5[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 160)) + (unsigned int)(16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_5[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 160)) + (unsigned int)(16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_5[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_5[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_5[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 160)) + (unsigned int)(16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_5[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_5[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_5[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)rb_off + (unsigned int)(64 * (lane_3 + 160)) + (unsigned int)(16 * (f ^ 3))));
                        unsigned int s_5[16];
                        unsigned int r_33_5[16];
                        s_5[0] = (((f & 1) != 0) ? t_5[4] : t_5[0]);
                        s_5[4] = (((f & 1) != 0) ? t_5[0] : t_5[4]);
                        s_5[8] = (((f & 1) != 0) ? t_5[12] : t_5[8]);
                        s_5[12] = (((f & 1) != 0) ? t_5[8] : t_5[12]);
                        s_5[1] = (((f & 1) != 0) ? t_5[5] : t_5[1]);
                        s_5[5] = (((f & 1) != 0) ? t_5[1] : t_5[5]);
                        s_5[9] = (((f & 1) != 0) ? t_5[13] : t_5[9]);
                        s_5[13] = (((f & 1) != 0) ? t_5[9] : t_5[13]);
                        s_5[2] = (((f & 1) != 0) ? t_5[6] : t_5[2]);
                        s_5[6] = (((f & 1) != 0) ? t_5[2] : t_5[6]);
                        s_5[10] = (((f & 1) != 0) ? t_5[14] : t_5[10]);
                        s_5[14] = (((f & 1) != 0) ? t_5[10] : t_5[14]);
                        s_5[3] = (((f & 1) != 0) ? t_5[7] : t_5[3]);
                        s_5[7] = (((f & 1) != 0) ? t_5[3] : t_5[7]);
                        s_5[11] = (((f & 1) != 0) ? t_5[15] : t_5[11]);
                        s_5[15] = (((f & 1) != 0) ? t_5[11] : t_5[15]);
                        r_33_5[0] = (((f & 2) != 0) ? s_5[8] : s_5[0]);
                        r_33_5[4] = (((f & 2) != 0) ? s_5[12] : s_5[4]);
                        r_33_5[8] = (((f & 2) != 0) ? s_5[0] : s_5[8]);
                        r_33_5[12] = (((f & 2) != 0) ? s_5[4] : s_5[12]);
                        r_33_5[1] = (((f & 2) != 0) ? s_5[9] : s_5[1]);
                        r_33_5[5] = (((f & 2) != 0) ? s_5[13] : s_5[5]);
                        r_33_5[9] = (((f & 2) != 0) ? s_5[1] : s_5[9]);
                        r_33_5[13] = (((f & 2) != 0) ? s_5[5] : s_5[13]);
                        r_33_5[2] = (((f & 2) != 0) ? s_5[10] : s_5[2]);
                        r_33_5[6] = (((f & 2) != 0) ? s_5[14] : s_5[6]);
                        r_33_5[10] = (((f & 2) != 0) ? s_5[2] : s_5[10]);
                        r_33_5[14] = (((f & 2) != 0) ? s_5[6] : s_5[14]);
                        r_33_5[3] = (((f & 2) != 0) ? s_5[11] : s_5[3]);
                        r_33_5[7] = (((f & 2) != 0) ? s_5[15] : s_5[7]);
                        r_33_5[11] = (((f & 2) != 0) ? s_5[3] : s_5[11]);
                        r_33_5[15] = (((f & 2) != 0) ? s_5[7] : s_5[15]);
                        float2 _f2_648 = make_float2(acc_5[0], acc_5[1]);
                        float2 _f2_649 = make_float2(__uint_as_float(r_33_5[0] << 16), __uint_as_float(r_33_5[0] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_80;
                        {
                            float2 _packed_f32x2_94_0 = _f2_648;
                            float2 _packed_f32x2_94_1 = _f2_649;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_80)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_94_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_94_1)));
                        }
                        acc_5[0] = _packed_add_f32x2_80.x;
                        acc_5[1] = _packed_add_f32x2_80.y;
                        float2 _f2_650 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_80, _packed_add_f32x2_80, _f2_650).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_80, _packed_add_f32x2_80, _f2_650).y;
                        float2 _f2_651 = make_float2(acc_5[2], acc_5[3]);
                        float2 _f2_652 = make_float2(__uint_as_float(r_33_5[1] << 16), __uint_as_float(r_33_5[1] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_81;
                        {
                            float2 _packed_f32x2_95_0 = _f2_651;
                            float2 _packed_f32x2_95_1 = _f2_652;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_81)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_95_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_95_1)));
                        }
                        acc_5[2] = _packed_add_f32x2_81.x;
                        acc_5[3] = _packed_add_f32x2_81.y;
                        float2 _f2_653 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_81, _packed_add_f32x2_81, _f2_653).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_81, _packed_add_f32x2_81, _f2_653).y;
                        float2 _f2_654 = make_float2(acc_5[4], acc_5[5]);
                        float2 _f2_655 = make_float2(__uint_as_float(r_33_5[2] << 16), __uint_as_float(r_33_5[2] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_82;
                        {
                            float2 _packed_f32x2_96_0 = _f2_654;
                            float2 _packed_f32x2_96_1 = _f2_655;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_82)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_96_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_96_1)));
                        }
                        acc_5[4] = _packed_add_f32x2_82.x;
                        acc_5[5] = _packed_add_f32x2_82.y;
                        float2 _f2_656 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_82, _packed_add_f32x2_82, _f2_656).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_82, _packed_add_f32x2_82, _f2_656).y;
                        float2 _f2_657 = make_float2(acc_5[6], acc_5[7]);
                        float2 _f2_658 = make_float2(__uint_as_float(r_33_5[3] << 16), __uint_as_float(r_33_5[3] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_83;
                        {
                            float2 _packed_f32x2_97_0 = _f2_657;
                            float2 _packed_f32x2_97_1 = _f2_658;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_83)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_97_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_97_1)));
                        }
                        acc_5[6] = _packed_add_f32x2_83.x;
                        acc_5[7] = _packed_add_f32x2_83.y;
                        float2 _f2_659 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_83, _packed_add_f32x2_83, _f2_659).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_83, _packed_add_f32x2_83, _f2_659).y;
                        float2 _f2_660 = make_float2(acc_5[8], acc_5[9]);
                        float2 _f2_661 = make_float2(__uint_as_float(r_33_5[4] << 16), __uint_as_float(r_33_5[4] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_84;
                        {
                            float2 _packed_f32x2_98_0 = _f2_660;
                            float2 _packed_f32x2_98_1 = _f2_661;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_84)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_98_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_98_1)));
                        }
                        acc_5[8] = _packed_add_f32x2_84.x;
                        acc_5[9] = _packed_add_f32x2_84.y;
                        float2 _f2_662 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_84, _packed_add_f32x2_84, _f2_662).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_84, _packed_add_f32x2_84, _f2_662).y;
                        float2 _f2_663 = make_float2(acc_5[10], acc_5[11]);
                        float2 _f2_664 = make_float2(__uint_as_float(r_33_5[5] << 16), __uint_as_float(r_33_5[5] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_85;
                        {
                            float2 _packed_f32x2_99_0 = _f2_663;
                            float2 _packed_f32x2_99_1 = _f2_664;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_85)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_99_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_99_1)));
                        }
                        acc_5[10] = _packed_add_f32x2_85.x;
                        acc_5[11] = _packed_add_f32x2_85.y;
                        float2 _f2_665 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_85, _packed_add_f32x2_85, _f2_665).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_85, _packed_add_f32x2_85, _f2_665).y;
                        float2 _f2_666 = make_float2(acc_5[12], acc_5[13]);
                        float2 _f2_667 = make_float2(__uint_as_float(r_33_5[6] << 16), __uint_as_float(r_33_5[6] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_86;
                        {
                            float2 _packed_f32x2_100_0 = _f2_666;
                            float2 _packed_f32x2_100_1 = _f2_667;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_86)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_100_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_100_1)));
                        }
                        acc_5[12] = _packed_add_f32x2_86.x;
                        acc_5[13] = _packed_add_f32x2_86.y;
                        float2 _f2_668 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_86, _packed_add_f32x2_86, _f2_668).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_86, _packed_add_f32x2_86, _f2_668).y;
                        float2 _f2_669 = make_float2(acc_5[14], acc_5[15]);
                        float2 _f2_670 = make_float2(__uint_as_float(r_33_5[7] << 16), __uint_as_float(r_33_5[7] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_87;
                        {
                            float2 _packed_f32x2_101_0 = _f2_669;
                            float2 _packed_f32x2_101_1 = _f2_670;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_87)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_101_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_101_1)));
                        }
                        acc_5[14] = _packed_add_f32x2_87.x;
                        acc_5[15] = _packed_add_f32x2_87.y;
                        float2 _f2_671 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_87, _packed_add_f32x2_87, _f2_671).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_87, _packed_add_f32x2_87, _f2_671).y;
                        float2 _f2_672 = make_float2(acc_5[16], acc_5[17]);
                        float2 _f2_673 = make_float2(__uint_as_float(r_33_5[8] << 16), __uint_as_float(r_33_5[8] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_88;
                        {
                            float2 _packed_f32x2_102_0 = _f2_672;
                            float2 _packed_f32x2_102_1 = _f2_673;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_88)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_102_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_102_1)));
                        }
                        acc_5[16] = _packed_add_f32x2_88.x;
                        acc_5[17] = _packed_add_f32x2_88.y;
                        float2 _f2_674 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_88, _packed_add_f32x2_88, _f2_674).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_88, _packed_add_f32x2_88, _f2_674).y;
                        float2 _f2_675 = make_float2(acc_5[18], acc_5[19]);
                        float2 _f2_676 = make_float2(__uint_as_float(r_33_5[9] << 16), __uint_as_float(r_33_5[9] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_89;
                        {
                            float2 _packed_f32x2_103_0 = _f2_675;
                            float2 _packed_f32x2_103_1 = _f2_676;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_89)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_103_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_103_1)));
                        }
                        acc_5[18] = _packed_add_f32x2_89.x;
                        acc_5[19] = _packed_add_f32x2_89.y;
                        float2 _f2_677 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_89, _packed_add_f32x2_89, _f2_677).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_89, _packed_add_f32x2_89, _f2_677).y;
                        float2 _f2_678 = make_float2(acc_5[20], acc_5[21]);
                        float2 _f2_679 = make_float2(__uint_as_float(r_33_5[10] << 16), __uint_as_float(r_33_5[10] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_90;
                        {
                            float2 _packed_f32x2_104_0 = _f2_678;
                            float2 _packed_f32x2_104_1 = _f2_679;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_90)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_104_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_104_1)));
                        }
                        acc_5[20] = _packed_add_f32x2_90.x;
                        acc_5[21] = _packed_add_f32x2_90.y;
                        float2 _f2_680 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_90, _packed_add_f32x2_90, _f2_680).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_90, _packed_add_f32x2_90, _f2_680).y;
                        float2 _f2_681 = make_float2(acc_5[22], acc_5[23]);
                        float2 _f2_682 = make_float2(__uint_as_float(r_33_5[11] << 16), __uint_as_float(r_33_5[11] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_91;
                        {
                            float2 _packed_f32x2_105_0 = _f2_681;
                            float2 _packed_f32x2_105_1 = _f2_682;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_91)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_105_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_105_1)));
                        }
                        acc_5[22] = _packed_add_f32x2_91.x;
                        acc_5[23] = _packed_add_f32x2_91.y;
                        float2 _f2_683 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_91, _packed_add_f32x2_91, _f2_683).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_91, _packed_add_f32x2_91, _f2_683).y;
                        float2 _f2_684 = make_float2(acc_5[24], acc_5[25]);
                        float2 _f2_685 = make_float2(__uint_as_float(r_33_5[12] << 16), __uint_as_float(r_33_5[12] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_92;
                        {
                            float2 _packed_f32x2_106_0 = _f2_684;
                            float2 _packed_f32x2_106_1 = _f2_685;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_92)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_106_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_106_1)));
                        }
                        acc_5[24] = _packed_add_f32x2_92.x;
                        acc_5[25] = _packed_add_f32x2_92.y;
                        float2 _f2_686 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_92, _packed_add_f32x2_92, _f2_686).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_92, _packed_add_f32x2_92, _f2_686).y;
                        float2 _f2_687 = make_float2(acc_5[26], acc_5[27]);
                        float2 _f2_688 = make_float2(__uint_as_float(r_33_5[13] << 16), __uint_as_float(r_33_5[13] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_93;
                        {
                            float2 _packed_f32x2_107_0 = _f2_687;
                            float2 _packed_f32x2_107_1 = _f2_688;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_93)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_107_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_107_1)));
                        }
                        acc_5[26] = _packed_add_f32x2_93.x;
                        acc_5[27] = _packed_add_f32x2_93.y;
                        float2 _f2_689 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_93, _packed_add_f32x2_93, _f2_689).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_93, _packed_add_f32x2_93, _f2_689).y;
                        float2 _f2_690 = make_float2(acc_5[28], acc_5[29]);
                        float2 _f2_691 = make_float2(__uint_as_float(r_33_5[14] << 16), __uint_as_float(r_33_5[14] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_94;
                        {
                            float2 _packed_f32x2_108_0 = _f2_690;
                            float2 _packed_f32x2_108_1 = _f2_691;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_94)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_108_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_108_1)));
                        }
                        acc_5[28] = _packed_add_f32x2_94.x;
                        acc_5[29] = _packed_add_f32x2_94.y;
                        float2 _f2_692 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_94, _packed_add_f32x2_94, _f2_692).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_94, _packed_add_f32x2_94, _f2_692).y;
                        float2 _f2_693 = make_float2(acc_5[30], acc_5[31]);
                        float2 _f2_694 = make_float2(__uint_as_float(r_33_5[15] << 16), __uint_as_float(r_33_5[15] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_95;
                        {
                            float2 _packed_f32x2_109_0 = _f2_693;
                            float2 _packed_f32x2_109_1 = _f2_694;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_95)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_109_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_109_1)));
                        }
                        acc_5[30] = _packed_add_f32x2_95.x;
                        acc_5[31] = _packed_add_f32x2_95.y;
                        float2 _f2_695 = make_float2(ssa[0], ssa[1]);
                        ssa[0] = fma_f32x2_rn_noftz(_packed_add_f32x2_95, _packed_add_f32x2_95, _f2_695).x;
                        ssa[1] = fma_f32x2_rn_noftz(_packed_add_f32x2_95, _packed_add_f32x2_95, _f2_695).y;
                        uint32_t acc_bf16_5[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_5[_lp*2 + 0], acc_5[_lp*2+1 + 0]));
                            acc_bf16_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 0)))), "r"(acc_bf16_5[0]), "r"(acc_bf16_5[1]), "r"(acc_bf16_5[2]), "r"(acc_bf16_5[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 1)))), "r"(acc_bf16_5[4]), "r"(acc_bf16_5[5]), "r"(acc_bf16_5[6]), "r"(acc_bf16_5[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 2)))), "r"(acc_bf16_5[8]), "r"(acc_bf16_5[9]), "r"(acc_bf16_5[10]), "r"(acc_bf16_5[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + (f ^ 3)))), "r"(acc_bf16_5[12]), "r"(acc_bf16_5[13]), "r"(acc_bf16_5[14]), "r"(acc_bf16_5[15]) : "memory");
                        __syncwarp();
                        int c_5 = lane_3;
                        int l_5 = c_5 >> 2;
                        unsigned int w4_5[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_5 + (c_5 & 3 ^ l_5 >> 1 & 3)))));
                        long long idx_5 = drow * 3072 + (long long)(4 * (640 + c_5));
                        {
                            int4 _iv4 = make_int4(w4_5[0 + 0], w4_5[0 + 1], w4_5[0 + 2], w4_5[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_5)) = _iv4;
                        }
                        int c_34_5 = lane_3 + 32;
                        int l_35_5 = c_34_5 >> 2;
                        unsigned int w4_36_5[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_36_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36_5[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_35_5 + (c_34_5 & 3 ^ l_35_5 >> 1 & 3)))));
                        long long idx_37_5 = drow * 3072 + (long long)(4 * (640 + c_34_5));
                        {
                            int4 _iv4 = make_int4(w4_36_5[0 + 0], w4_36_5[0 + 1], w4_36_5[0 + 2], w4_36_5[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_37_5)) = _iv4;
                        }
                        int c_38_5 = lane_3 + 64;
                        int l_39_5 = c_38_5 >> 2;
                        unsigned int w4_40_5[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_40_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40_5[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_39_5 + (c_38_5 & 3 ^ l_39_5 >> 1 & 3)))));
                        long long idx_41_5 = drow * 3072 + (long long)(4 * (640 + c_38_5));
                        {
                            int4 _iv4 = make_int4(w4_40_5[0 + 0], w4_40_5[0 + 1], w4_40_5[0 + 2], w4_40_5[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_41_5)) = _iv4;
                        }
                        int c_42_5 = lane_3 + 96;
                        int l_43_5 = c_42_5 >> 2;
                        unsigned int w4_44_5[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_44_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44_5[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_43_5 + (c_42_5 & 3 ^ l_43_5 >> 1 & 3)))));
                        long long idx_45_5 = drow * 3072 + (long long)(4 * (640 + c_42_5));
                        {
                            int4 _iv4 = make_int4(w4_44_5[0 + 0], w4_44_5[0 + 1], w4_44_5[0 + 2], w4_44_5[0 + 3]);
                            *reinterpret_cast<int4*>(resout_i32 + (idx_45_5)) = _iv4;
                        }
                        __syncwarp();
                        int zlane_3 = rb_off + 8192 + 128 * lane_3;
                        int l7_3 = lane_3 & 7;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_3 + 16 * (l7_3 ^ 0))), "r"(__as_u32(acc_5[0])), "r"(__as_u32(acc_5[1])), "r"(__as_u32(acc_5[2])), "r"(__as_u32(acc_5[3])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_3 + 16 * (l7_3 ^ 1))), "r"(__as_u32(acc_5[4])), "r"(__as_u32(acc_5[5])), "r"(__as_u32(acc_5[6])), "r"(__as_u32(acc_5[7])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_3 + 16 * (l7_3 ^ 2))), "r"(__as_u32(acc_5[8])), "r"(__as_u32(acc_5[9])), "r"(__as_u32(acc_5[10])), "r"(__as_u32(acc_5[11])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_3 + 16 * (l7_3 ^ 3))), "r"(__as_u32(acc_5[12])), "r"(__as_u32(acc_5[13])), "r"(__as_u32(acc_5[14])), "r"(__as_u32(acc_5[15])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_3 + 16 * (l7_3 ^ 4))), "r"(__as_u32(acc_5[16])), "r"(__as_u32(acc_5[17])), "r"(__as_u32(acc_5[18])), "r"(__as_u32(acc_5[19])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_3 + 16 * (l7_3 ^ 5))), "r"(__as_u32(acc_5[20])), "r"(__as_u32(acc_5[21])), "r"(__as_u32(acc_5[22])), "r"(__as_u32(acc_5[23])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_3 + 16 * (l7_3 ^ 6))), "r"(__as_u32(acc_5[24])), "r"(__as_u32(acc_5[25])), "r"(__as_u32(acc_5[26])), "r"(__as_u32(acc_5[27])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(zlane_3 + 16 * (l7_3 ^ 7))), "r"(__as_u32(acc_5[28])), "r"(__as_u32(acc_5[29])), "r"(__as_u32(acc_5[30])), "r"(__as_u32(acc_5[31])) : "memory");
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
                        float2 _f2_696 = make_float2(_rsqrt_0, _rsqrt_0);
                        float one_0 = 1.0f;
                        float2 _f2_697 = make_float2(one_0, one_0);
                        int pp = lane_3 + 32;
                        unsigned int wc[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&wc[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc[(0) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp)));
                        float yv[8];
                        float2 _f2_698 = make_float2(zreg[0], zreg[1]);
                        float2 _mul_f32x2_0;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_698), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_699 = make_float2(__uint_as_float(wc[0] << 16), __uint_as_float(wc[0] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_96;
                        {
                            float2 _packed_f32x2_110_0 = _f2_697;
                            float2 _packed_f32x2_110_1 = _f2_699;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_96)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_110_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_110_1)));
                        }
                        float2 _mul_f32x2_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_mul_f32x2_0), "l"(*(const unsigned long long*)&_packed_add_f32x2_96));
                        yv[0] = _mul_f32x2_1.x;
                        yv[1] = _mul_f32x2_1.y;
                        float2 _f2_700 = make_float2(zreg[2], zreg[3]);
                        float2 _mul_f32x2_2;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_700), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_701 = make_float2(__uint_as_float(wc[1] << 16), __uint_as_float(wc[1] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_97;
                        {
                            float2 _packed_f32x2_111_0 = _f2_697;
                            float2 _packed_f32x2_111_1 = _f2_701;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_97)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_111_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_111_1)));
                        }
                        float2 _mul_f32x2_3;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_mul_f32x2_2), "l"(*(const unsigned long long*)&_packed_add_f32x2_97));
                        yv[2] = _mul_f32x2_3.x;
                        yv[3] = _mul_f32x2_3.y;
                        float2 _f2_702 = make_float2(zreg[4], zreg[5]);
                        float2 _mul_f32x2_4;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_702), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_703 = make_float2(__uint_as_float(wc[2] << 16), __uint_as_float(wc[2] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_98;
                        {
                            float2 _packed_f32x2_112_0 = _f2_697;
                            float2 _packed_f32x2_112_1 = _f2_703;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_98)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_112_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_112_1)));
                        }
                        float2 _mul_f32x2_5;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_mul_f32x2_4), "l"(*(const unsigned long long*)&_packed_add_f32x2_98));
                        yv[4] = _mul_f32x2_5.x;
                        yv[5] = _mul_f32x2_5.y;
                        float2 _f2_704 = make_float2(zreg[6], zreg[7]);
                        float2 _mul_f32x2_6;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&_f2_704), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_705 = make_float2(__uint_as_float(wc[3] << 16), __uint_as_float(wc[3] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_99;
                        {
                            float2 _packed_f32x2_113_0 = _f2_697;
                            float2 _packed_f32x2_113_1 = _f2_705;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_99)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_113_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_113_1)));
                        }
                        float2 _mul_f32x2_7;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&_mul_f32x2_6), "l"(*(const unsigned long long*)&_packed_add_f32x2_99));
                        yv[6] = _mul_f32x2_7.x;
                        yv[7] = _mul_f32x2_7.y;
                        uint32_t yv_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv[_lp*2 + 0], yv[_lp*2+1 + 0]));
                            yv_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3))), "r"(yv_bf16[0]), "r"(yv_bf16[1]), "r"(yv_bf16[2]), "r"(yv_bf16[3]) : "memory");
                        unsigned int wc_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&wc_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_1[(0) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp) + 16));
                        float yv_2[8];
                        float2 _f2_706 = make_float2(zreg[8], zreg[9]);
                        float2 _mul_f32x2_8;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_8) : "l"(*(const unsigned long long*)&_f2_706), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_707 = make_float2(__uint_as_float(wc_1[0] << 16), __uint_as_float(wc_1[0] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_100;
                        {
                            float2 _packed_f32x2_114_0 = _f2_697;
                            float2 _packed_f32x2_114_1 = _f2_707;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_100)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_114_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_114_1)));
                        }
                        float2 _mul_f32x2_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&_mul_f32x2_8), "l"(*(const unsigned long long*)&_packed_add_f32x2_100));
                        yv_2[0] = _mul_f32x2_9.x;
                        yv_2[1] = _mul_f32x2_9.y;
                        float2 _f2_708 = make_float2(zreg[10], zreg[11]);
                        float2 _mul_f32x2_10;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_10) : "l"(*(const unsigned long long*)&_f2_708), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_709 = make_float2(__uint_as_float(wc_1[1] << 16), __uint_as_float(wc_1[1] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_101;
                        {
                            float2 _packed_f32x2_115_0 = _f2_697;
                            float2 _packed_f32x2_115_1 = _f2_709;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_101)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_115_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_115_1)));
                        }
                        float2 _mul_f32x2_11;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_11) : "l"(*(const unsigned long long*)&_mul_f32x2_10), "l"(*(const unsigned long long*)&_packed_add_f32x2_101));
                        yv_2[2] = _mul_f32x2_11.x;
                        yv_2[3] = _mul_f32x2_11.y;
                        float2 _f2_710 = make_float2(zreg[12], zreg[13]);
                        float2 _mul_f32x2_12;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_12) : "l"(*(const unsigned long long*)&_f2_710), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_711 = make_float2(__uint_as_float(wc_1[2] << 16), __uint_as_float(wc_1[2] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_102;
                        {
                            float2 _packed_f32x2_116_0 = _f2_697;
                            float2 _packed_f32x2_116_1 = _f2_711;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_102)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_116_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_116_1)));
                        }
                        float2 _mul_f32x2_13;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_13) : "l"(*(const unsigned long long*)&_mul_f32x2_12), "l"(*(const unsigned long long*)&_packed_add_f32x2_102));
                        yv_2[4] = _mul_f32x2_13.x;
                        yv_2[5] = _mul_f32x2_13.y;
                        float2 _f2_712 = make_float2(zreg[14], zreg[15]);
                        float2 _mul_f32x2_14;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&_f2_712), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_713 = make_float2(__uint_as_float(wc_1[3] << 16), __uint_as_float(wc_1[3] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_103;
                        {
                            float2 _packed_f32x2_117_0 = _f2_697;
                            float2 _packed_f32x2_117_1 = _f2_713;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_103)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_117_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_117_1)));
                        }
                        float2 _mul_f32x2_15;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&_mul_f32x2_14), "l"(*(const unsigned long long*)&_packed_add_f32x2_103));
                        yv_2[6] = _mul_f32x2_15.x;
                        yv_2[7] = _mul_f32x2_15.y;
                        uint32_t yv_2_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_2[_lp*2 + 0], yv_2[_lp*2+1 + 0]));
                            yv_2_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + 1))), "r"(yv_2_bf16[0]), "r"(yv_2_bf16[1]), "r"(yv_2_bf16[2]), "r"(yv_2_bf16[3]) : "memory");
                        unsigned int wc_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&wc_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_3[(0) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp) + 32));
                        float yv_4[8];
                        float2 _f2_714 = make_float2(zreg[16], zreg[17]);
                        float2 _mul_f32x2_16;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_16) : "l"(*(const unsigned long long*)&_f2_714), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_715 = make_float2(__uint_as_float(wc_3[0] << 16), __uint_as_float(wc_3[0] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_104;
                        {
                            float2 _packed_f32x2_118_0 = _f2_697;
                            float2 _packed_f32x2_118_1 = _f2_715;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_104)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_118_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_118_1)));
                        }
                        float2 _mul_f32x2_17;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_17) : "l"(*(const unsigned long long*)&_mul_f32x2_16), "l"(*(const unsigned long long*)&_packed_add_f32x2_104));
                        yv_4[0] = _mul_f32x2_17.x;
                        yv_4[1] = _mul_f32x2_17.y;
                        float2 _f2_716 = make_float2(zreg[18], zreg[19]);
                        float2 _mul_f32x2_18;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_18) : "l"(*(const unsigned long long*)&_f2_716), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_717 = make_float2(__uint_as_float(wc_3[1] << 16), __uint_as_float(wc_3[1] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_105;
                        {
                            float2 _packed_f32x2_119_0 = _f2_697;
                            float2 _packed_f32x2_119_1 = _f2_717;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_105)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_119_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_119_1)));
                        }
                        float2 _mul_f32x2_19;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_19) : "l"(*(const unsigned long long*)&_mul_f32x2_18), "l"(*(const unsigned long long*)&_packed_add_f32x2_105));
                        yv_4[2] = _mul_f32x2_19.x;
                        yv_4[3] = _mul_f32x2_19.y;
                        float2 _f2_718 = make_float2(zreg[20], zreg[21]);
                        float2 _mul_f32x2_20;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_20) : "l"(*(const unsigned long long*)&_f2_718), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_719 = make_float2(__uint_as_float(wc_3[2] << 16), __uint_as_float(wc_3[2] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_106;
                        {
                            float2 _packed_f32x2_120_0 = _f2_697;
                            float2 _packed_f32x2_120_1 = _f2_719;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_106)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_120_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_120_1)));
                        }
                        float2 _mul_f32x2_21;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_21) : "l"(*(const unsigned long long*)&_mul_f32x2_20), "l"(*(const unsigned long long*)&_packed_add_f32x2_106));
                        yv_4[4] = _mul_f32x2_21.x;
                        yv_4[5] = _mul_f32x2_21.y;
                        float2 _f2_720 = make_float2(zreg[22], zreg[23]);
                        float2 _mul_f32x2_22;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_22) : "l"(*(const unsigned long long*)&_f2_720), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_721 = make_float2(__uint_as_float(wc_3[3] << 16), __uint_as_float(wc_3[3] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_107;
                        {
                            float2 _packed_f32x2_121_0 = _f2_697;
                            float2 _packed_f32x2_121_1 = _f2_721;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_107)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_121_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_121_1)));
                        }
                        float2 _mul_f32x2_23;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_23) : "l"(*(const unsigned long long*)&_mul_f32x2_22), "l"(*(const unsigned long long*)&_packed_add_f32x2_107));
                        yv_4[6] = _mul_f32x2_23.x;
                        yv_4[7] = _mul_f32x2_23.y;
                        uint32_t yv_4_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_4[_lp*2 + 0], yv_4[_lp*2+1 + 0]));
                            yv_4_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + 2))), "r"(yv_4_bf16[0]), "r"(yv_4_bf16[1]), "r"(yv_4_bf16[2]), "r"(yv_4_bf16[3]) : "memory");
                        unsigned int wc_5[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&wc_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_5[(0) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp) + 48));
                        float yv_6[8];
                        float2 _f2_722 = make_float2(zreg[24], zreg[25]);
                        float2 _mul_f32x2_24;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_24) : "l"(*(const unsigned long long*)&_f2_722), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_723 = make_float2(__uint_as_float(wc_5[0] << 16), __uint_as_float(wc_5[0] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_108;
                        {
                            float2 _packed_f32x2_122_0 = _f2_697;
                            float2 _packed_f32x2_122_1 = _f2_723;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_108)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_122_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_122_1)));
                        }
                        float2 _mul_f32x2_25;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_25) : "l"(*(const unsigned long long*)&_mul_f32x2_24), "l"(*(const unsigned long long*)&_packed_add_f32x2_108));
                        yv_6[0] = _mul_f32x2_25.x;
                        yv_6[1] = _mul_f32x2_25.y;
                        float2 _f2_724 = make_float2(zreg[26], zreg[27]);
                        float2 _mul_f32x2_26;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_26) : "l"(*(const unsigned long long*)&_f2_724), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_725 = make_float2(__uint_as_float(wc_5[1] << 16), __uint_as_float(wc_5[1] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_109;
                        {
                            float2 _packed_f32x2_123_0 = _f2_697;
                            float2 _packed_f32x2_123_1 = _f2_725;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_109)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_123_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_123_1)));
                        }
                        float2 _mul_f32x2_27;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_27) : "l"(*(const unsigned long long*)&_mul_f32x2_26), "l"(*(const unsigned long long*)&_packed_add_f32x2_109));
                        yv_6[2] = _mul_f32x2_27.x;
                        yv_6[3] = _mul_f32x2_27.y;
                        float2 _f2_726 = make_float2(zreg[28], zreg[29]);
                        float2 _mul_f32x2_28;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_28) : "l"(*(const unsigned long long*)&_f2_726), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_727 = make_float2(__uint_as_float(wc_5[2] << 16), __uint_as_float(wc_5[2] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_110;
                        {
                            float2 _packed_f32x2_124_0 = _f2_697;
                            float2 _packed_f32x2_124_1 = _f2_727;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_110)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_124_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_124_1)));
                        }
                        float2 _mul_f32x2_29;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_29) : "l"(*(const unsigned long long*)&_mul_f32x2_28), "l"(*(const unsigned long long*)&_packed_add_f32x2_110));
                        yv_6[4] = _mul_f32x2_29.x;
                        yv_6[5] = _mul_f32x2_29.y;
                        float2 _f2_728 = make_float2(zreg[30], zreg[31]);
                        float2 _mul_f32x2_30;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_30) : "l"(*(const unsigned long long*)&_f2_728), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_729 = make_float2(__uint_as_float(wc_5[3] << 16), __uint_as_float(wc_5[3] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_111;
                        {
                            float2 _packed_f32x2_125_0 = _f2_697;
                            float2 _packed_f32x2_125_1 = _f2_729;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_111)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_125_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_125_1)));
                        }
                        float2 _mul_f32x2_31;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_31) : "l"(*(const unsigned long long*)&_mul_f32x2_30), "l"(*(const unsigned long long*)&_packed_add_f32x2_111));
                        yv_6[6] = _mul_f32x2_31.x;
                        yv_6[7] = _mul_f32x2_31.y;
                        uint32_t yv_6_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_6[_lp*2 + 0], yv_6[_lp*2+1 + 0]));
                            yv_6_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + 3))), "r"(yv_6_bf16[0]), "r"(yv_6_bf16[1]), "r"(yv_6_bf16[2]), "r"(yv_6_bf16[3]) : "memory");
                        __syncwarp();
                        unsigned int w4_6[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * lane_3)));
                        long long idx_6 = drow * 3072 + (long long)(4 * (128 + lane_3));
                        {
                            int4 _iv4 = make_int4(w4_6[0 + 0], w4_6[0 + 1], w4_6[0 + 2], w4_6[0 + 3]);
                            *reinterpret_cast<int4*>(out_i32 + (idx_6)) = _iv4;
                        }
                        unsigned int w4_7[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 32))));
                        long long idx_8 = drow * 3072 + (long long)(4 * (128 + lane_3 + 32));
                        {
                            int4 _iv4 = make_int4(w4_7[0 + 0], w4_7[0 + 1], w4_7[0 + 2], w4_7[0 + 3]);
                            *reinterpret_cast<int4*>(out_i32 + (idx_8)) = _iv4;
                        }
                        unsigned int w4_9[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 64))));
                        long long idx_10 = drow * 3072 + (long long)(4 * (128 + lane_3 + 64));
                        {
                            int4 _iv4 = make_int4(w4_9[0 + 0], w4_9[0 + 1], w4_9[0 + 2], w4_9[0 + 3]);
                            *reinterpret_cast<int4*>(out_i32 + (idx_10)) = _iv4;
                        }
                        unsigned int w4_11[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_11[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 96))));
                        long long idx_12 = drow * 3072 + (long long)(4 * (128 + lane_3 + 96));
                        {
                            int4 _iv4 = make_int4(w4_11[0 + 0], w4_11[0 + 1], w4_11[0 + 2], w4_11[0 + 3]);
                            *reinterpret_cast<int4*>(out_i32 + (idx_12)) = _iv4;
                        }
                        __syncwarp();
                        int pp_13 = lane_3 + 64;
                        unsigned int wc_14[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&wc_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_14[(0) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp_13)));
                        float yv_15[8];
                        float2 _f2_730 = make_float2(zreg_b[0], zreg_b[1]);
                        float2 _mul_f32x2_32;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_32) : "l"(*(const unsigned long long*)&_f2_730), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_731 = make_float2(__uint_as_float(wc_14[0] << 16), __uint_as_float(wc_14[0] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_112;
                        {
                            float2 _packed_f32x2_126_0 = _f2_697;
                            float2 _packed_f32x2_126_1 = _f2_731;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_112)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_126_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_126_1)));
                        }
                        float2 _mul_f32x2_33;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_33) : "l"(*(const unsigned long long*)&_mul_f32x2_32), "l"(*(const unsigned long long*)&_packed_add_f32x2_112));
                        yv_15[0] = _mul_f32x2_33.x;
                        yv_15[1] = _mul_f32x2_33.y;
                        float2 _f2_732 = make_float2(zreg_b[2], zreg_b[3]);
                        float2 _mul_f32x2_34;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_34) : "l"(*(const unsigned long long*)&_f2_732), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_733 = make_float2(__uint_as_float(wc_14[1] << 16), __uint_as_float(wc_14[1] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_113;
                        {
                            float2 _packed_f32x2_127_0 = _f2_697;
                            float2 _packed_f32x2_127_1 = _f2_733;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_113)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_127_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_127_1)));
                        }
                        float2 _mul_f32x2_35;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_35) : "l"(*(const unsigned long long*)&_mul_f32x2_34), "l"(*(const unsigned long long*)&_packed_add_f32x2_113));
                        yv_15[2] = _mul_f32x2_35.x;
                        yv_15[3] = _mul_f32x2_35.y;
                        float2 _f2_734 = make_float2(zreg_b[4], zreg_b[5]);
                        float2 _mul_f32x2_36;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_36) : "l"(*(const unsigned long long*)&_f2_734), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_735 = make_float2(__uint_as_float(wc_14[2] << 16), __uint_as_float(wc_14[2] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_114;
                        {
                            float2 _packed_f32x2_128_0 = _f2_697;
                            float2 _packed_f32x2_128_1 = _f2_735;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_114)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_128_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_128_1)));
                        }
                        float2 _mul_f32x2_37;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_37) : "l"(*(const unsigned long long*)&_mul_f32x2_36), "l"(*(const unsigned long long*)&_packed_add_f32x2_114));
                        yv_15[4] = _mul_f32x2_37.x;
                        yv_15[5] = _mul_f32x2_37.y;
                        float2 _f2_736 = make_float2(zreg_b[6], zreg_b[7]);
                        float2 _mul_f32x2_38;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_38) : "l"(*(const unsigned long long*)&_f2_736), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_737 = make_float2(__uint_as_float(wc_14[3] << 16), __uint_as_float(wc_14[3] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_115;
                        {
                            float2 _packed_f32x2_129_0 = _f2_697;
                            float2 _packed_f32x2_129_1 = _f2_737;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_115)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_129_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_129_1)));
                        }
                        float2 _mul_f32x2_39;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_39) : "l"(*(const unsigned long long*)&_mul_f32x2_38), "l"(*(const unsigned long long*)&_packed_add_f32x2_115));
                        yv_15[6] = _mul_f32x2_39.x;
                        yv_15[7] = _mul_f32x2_39.y;
                        uint32_t yv_15_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_15[_lp*2 + 0], yv_15[_lp*2+1 + 0]));
                            yv_15_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3))), "r"(yv_15_bf16[0]), "r"(yv_15_bf16[1]), "r"(yv_15_bf16[2]), "r"(yv_15_bf16[3]) : "memory");
                        unsigned int wc_16[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&wc_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_16[(0) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp_13) + 16));
                        float yv_17[8];
                        float2 _f2_738 = make_float2(zreg_b[8], zreg_b[9]);
                        float2 _mul_f32x2_40;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_40) : "l"(*(const unsigned long long*)&_f2_738), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_739 = make_float2(__uint_as_float(wc_16[0] << 16), __uint_as_float(wc_16[0] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_116;
                        {
                            float2 _packed_f32x2_130_0 = _f2_697;
                            float2 _packed_f32x2_130_1 = _f2_739;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_116)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_130_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_130_1)));
                        }
                        float2 _mul_f32x2_41;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_41) : "l"(*(const unsigned long long*)&_mul_f32x2_40), "l"(*(const unsigned long long*)&_packed_add_f32x2_116));
                        yv_17[0] = _mul_f32x2_41.x;
                        yv_17[1] = _mul_f32x2_41.y;
                        float2 _f2_740 = make_float2(zreg_b[10], zreg_b[11]);
                        float2 _mul_f32x2_42;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_42) : "l"(*(const unsigned long long*)&_f2_740), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_741 = make_float2(__uint_as_float(wc_16[1] << 16), __uint_as_float(wc_16[1] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_117;
                        {
                            float2 _packed_f32x2_131_0 = _f2_697;
                            float2 _packed_f32x2_131_1 = _f2_741;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_117)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_131_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_131_1)));
                        }
                        float2 _mul_f32x2_43;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_43) : "l"(*(const unsigned long long*)&_mul_f32x2_42), "l"(*(const unsigned long long*)&_packed_add_f32x2_117));
                        yv_17[2] = _mul_f32x2_43.x;
                        yv_17[3] = _mul_f32x2_43.y;
                        float2 _f2_742 = make_float2(zreg_b[12], zreg_b[13]);
                        float2 _mul_f32x2_44;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_44) : "l"(*(const unsigned long long*)&_f2_742), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_743 = make_float2(__uint_as_float(wc_16[2] << 16), __uint_as_float(wc_16[2] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_118;
                        {
                            float2 _packed_f32x2_132_0 = _f2_697;
                            float2 _packed_f32x2_132_1 = _f2_743;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_118)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_132_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_132_1)));
                        }
                        float2 _mul_f32x2_45;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_45) : "l"(*(const unsigned long long*)&_mul_f32x2_44), "l"(*(const unsigned long long*)&_packed_add_f32x2_118));
                        yv_17[4] = _mul_f32x2_45.x;
                        yv_17[5] = _mul_f32x2_45.y;
                        float2 _f2_744 = make_float2(zreg_b[14], zreg_b[15]);
                        float2 _mul_f32x2_46;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_46) : "l"(*(const unsigned long long*)&_f2_744), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_745 = make_float2(__uint_as_float(wc_16[3] << 16), __uint_as_float(wc_16[3] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_119;
                        {
                            float2 _packed_f32x2_133_0 = _f2_697;
                            float2 _packed_f32x2_133_1 = _f2_745;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_119)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_133_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_133_1)));
                        }
                        float2 _mul_f32x2_47;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_47) : "l"(*(const unsigned long long*)&_mul_f32x2_46), "l"(*(const unsigned long long*)&_packed_add_f32x2_119));
                        yv_17[6] = _mul_f32x2_47.x;
                        yv_17[7] = _mul_f32x2_47.y;
                        uint32_t yv_17_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_17[_lp*2 + 0], yv_17[_lp*2+1 + 0]));
                            yv_17_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + 1))), "r"(yv_17_bf16[0]), "r"(yv_17_bf16[1]), "r"(yv_17_bf16[2]), "r"(yv_17_bf16[3]) : "memory");
                        unsigned int wc_18[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&wc_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_18[(0) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp_13) + 32));
                        float yv_19[8];
                        float2 _f2_746 = make_float2(zreg_b[16], zreg_b[17]);
                        float2 _mul_f32x2_48;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_48) : "l"(*(const unsigned long long*)&_f2_746), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_747 = make_float2(__uint_as_float(wc_18[0] << 16), __uint_as_float(wc_18[0] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_120;
                        {
                            float2 _packed_f32x2_134_0 = _f2_697;
                            float2 _packed_f32x2_134_1 = _f2_747;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_120)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_134_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_134_1)));
                        }
                        float2 _mul_f32x2_49;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_49) : "l"(*(const unsigned long long*)&_mul_f32x2_48), "l"(*(const unsigned long long*)&_packed_add_f32x2_120));
                        yv_19[0] = _mul_f32x2_49.x;
                        yv_19[1] = _mul_f32x2_49.y;
                        float2 _f2_748 = make_float2(zreg_b[18], zreg_b[19]);
                        float2 _mul_f32x2_50;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_50) : "l"(*(const unsigned long long*)&_f2_748), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_749 = make_float2(__uint_as_float(wc_18[1] << 16), __uint_as_float(wc_18[1] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_121;
                        {
                            float2 _packed_f32x2_135_0 = _f2_697;
                            float2 _packed_f32x2_135_1 = _f2_749;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_121)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_135_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_135_1)));
                        }
                        float2 _mul_f32x2_51;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_51) : "l"(*(const unsigned long long*)&_mul_f32x2_50), "l"(*(const unsigned long long*)&_packed_add_f32x2_121));
                        yv_19[2] = _mul_f32x2_51.x;
                        yv_19[3] = _mul_f32x2_51.y;
                        float2 _f2_750 = make_float2(zreg_b[20], zreg_b[21]);
                        float2 _mul_f32x2_52;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_52) : "l"(*(const unsigned long long*)&_f2_750), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_751 = make_float2(__uint_as_float(wc_18[2] << 16), __uint_as_float(wc_18[2] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_122;
                        {
                            float2 _packed_f32x2_136_0 = _f2_697;
                            float2 _packed_f32x2_136_1 = _f2_751;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_122)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_136_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_136_1)));
                        }
                        float2 _mul_f32x2_53;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_53) : "l"(*(const unsigned long long*)&_mul_f32x2_52), "l"(*(const unsigned long long*)&_packed_add_f32x2_122));
                        yv_19[4] = _mul_f32x2_53.x;
                        yv_19[5] = _mul_f32x2_53.y;
                        float2 _f2_752 = make_float2(zreg_b[22], zreg_b[23]);
                        float2 _mul_f32x2_54;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_54) : "l"(*(const unsigned long long*)&_f2_752), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_753 = make_float2(__uint_as_float(wc_18[3] << 16), __uint_as_float(wc_18[3] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_123;
                        {
                            float2 _packed_f32x2_137_0 = _f2_697;
                            float2 _packed_f32x2_137_1 = _f2_753;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_123)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_137_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_137_1)));
                        }
                        float2 _mul_f32x2_55;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_55) : "l"(*(const unsigned long long*)&_mul_f32x2_54), "l"(*(const unsigned long long*)&_packed_add_f32x2_123));
                        yv_19[6] = _mul_f32x2_55.x;
                        yv_19[7] = _mul_f32x2_55.y;
                        uint32_t yv_19_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_19[_lp*2 + 0], yv_19[_lp*2+1 + 0]));
                            yv_19_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + 2))), "r"(yv_19_bf16[0]), "r"(yv_19_bf16[1]), "r"(yv_19_bf16[2]), "r"(yv_19_bf16[3]) : "memory");
                        unsigned int wc_20[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&wc_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_20[(0) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp_13) + 48));
                        float yv_21[8];
                        float2 _f2_754 = make_float2(zreg_b[24], zreg_b[25]);
                        float2 _mul_f32x2_56;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_56) : "l"(*(const unsigned long long*)&_f2_754), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_755 = make_float2(__uint_as_float(wc_20[0] << 16), __uint_as_float(wc_20[0] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_124;
                        {
                            float2 _packed_f32x2_138_0 = _f2_697;
                            float2 _packed_f32x2_138_1 = _f2_755;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_124)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_138_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_138_1)));
                        }
                        float2 _mul_f32x2_57;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_57) : "l"(*(const unsigned long long*)&_mul_f32x2_56), "l"(*(const unsigned long long*)&_packed_add_f32x2_124));
                        yv_21[0] = _mul_f32x2_57.x;
                        yv_21[1] = _mul_f32x2_57.y;
                        float2 _f2_756 = make_float2(zreg_b[26], zreg_b[27]);
                        float2 _mul_f32x2_58;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_58) : "l"(*(const unsigned long long*)&_f2_756), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_757 = make_float2(__uint_as_float(wc_20[1] << 16), __uint_as_float(wc_20[1] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_125;
                        {
                            float2 _packed_f32x2_139_0 = _f2_697;
                            float2 _packed_f32x2_139_1 = _f2_757;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_125)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_139_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_139_1)));
                        }
                        float2 _mul_f32x2_59;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_59) : "l"(*(const unsigned long long*)&_mul_f32x2_58), "l"(*(const unsigned long long*)&_packed_add_f32x2_125));
                        yv_21[2] = _mul_f32x2_59.x;
                        yv_21[3] = _mul_f32x2_59.y;
                        float2 _f2_758 = make_float2(zreg_b[28], zreg_b[29]);
                        float2 _mul_f32x2_60;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_60) : "l"(*(const unsigned long long*)&_f2_758), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_759 = make_float2(__uint_as_float(wc_20[2] << 16), __uint_as_float(wc_20[2] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_126;
                        {
                            float2 _packed_f32x2_140_0 = _f2_697;
                            float2 _packed_f32x2_140_1 = _f2_759;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_126)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_140_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_140_1)));
                        }
                        float2 _mul_f32x2_61;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_61) : "l"(*(const unsigned long long*)&_mul_f32x2_60), "l"(*(const unsigned long long*)&_packed_add_f32x2_126));
                        yv_21[4] = _mul_f32x2_61.x;
                        yv_21[5] = _mul_f32x2_61.y;
                        float2 _f2_760 = make_float2(zreg_b[30], zreg_b[31]);
                        float2 _mul_f32x2_62;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_62) : "l"(*(const unsigned long long*)&_f2_760), "l"(*(const unsigned long long*)&_f2_696));
                        float2 _f2_761 = make_float2(__uint_as_float(wc_20[3] << 16), __uint_as_float(wc_20[3] & 4294901760u));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_add_f32x2_127;
                        {
                            float2 _packed_f32x2_141_0 = _f2_697;
                            float2 _packed_f32x2_141_1 = _f2_761;
                            asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_127)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_141_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_141_1)));
                        }
                        float2 _mul_f32x2_63;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_63) : "l"(*(const unsigned long long*)&_mul_f32x2_62), "l"(*(const unsigned long long*)&_packed_add_f32x2_127));
                        yv_21[6] = _mul_f32x2_63.x;
                        yv_21[7] = _mul_f32x2_63.y;
                        uint32_t yv_21_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_21[_lp*2 + 0], yv_21[_lp*2+1 + 0]));
                            yv_21_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + 3))), "r"(yv_21_bf16[0]), "r"(yv_21_bf16[1]), "r"(yv_21_bf16[2]), "r"(yv_21_bf16[3]) : "memory");
                        __syncwarp();
                        unsigned int w4_22[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * lane_3)));
                        long long idx_23 = drow * 3072 + (long long)(4 * (256 + lane_3));
                        {
                            int4 _iv4 = make_int4(w4_22[0 + 0], w4_22[0 + 1], w4_22[0 + 2], w4_22[0 + 3]);
                            *reinterpret_cast<int4*>(out_i32 + (idx_23)) = _iv4;
                        }
                        unsigned int w4_24[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 32))));
                        long long idx_25 = drow * 3072 + (long long)(4 * (256 + lane_3 + 32));
                        {
                            int4 _iv4 = make_int4(w4_24[0 + 0], w4_24[0 + 1], w4_24[0 + 2], w4_24[0 + 3]);
                            *reinterpret_cast<int4*>(out_i32 + (idx_25)) = _iv4;
                        }
                        unsigned int w4_26[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_26[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 64))));
                        long long idx_27 = drow * 3072 + (long long)(4 * (256 + lane_3 + 64));
                        {
                            int4 _iv4 = make_int4(w4_26[0 + 0], w4_26[0 + 1], w4_26[0 + 2], w4_26[0 + 3]);
                            *reinterpret_cast<int4*>(out_i32 + (idx_27)) = _iv4;
                        }
                        unsigned int w4_28[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_28[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_28[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_28[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 96))));
                        long long idx_29 = drow * 3072 + (long long)(4 * (256 + lane_3 + 96));
                        {
                            int4 _iv4 = make_int4(w4_28[0 + 0], w4_28[0 + 1], w4_28[0 + 2], w4_28[0 + 3]);
                            *reinterpret_cast<int4*>(out_i32 + (idx_29)) = _iv4;
                        }
                        __syncwarp();
                        #pragma unroll 1
                        for (int j = 0; j < 1; j++) {
                            int l7_4 = lane_3 & 7;
                            int pp_0 = lane_3 + 32 * j;
                            int zlane_4 = zb_off + 4096 * j + 128 * lane_3;
                            int kw = f ^ 0;
                            unsigned int wc_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&wc_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_2[(0) + 3]))
                                : "r"(wrow_addr + (unsigned int)(64 * pp_0) + (unsigned int)(16 * kw)));
                            float zc[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc[0])), "=r"(*reinterpret_cast<uint32_t*>(&zc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_4 + (unsigned int)(16 * (l7_4 ^ 2 * kw))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc[4])), "=r"(*reinterpret_cast<uint32_t*>(&zc[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_4 + (unsigned int)(16 * (l7_4 ^ 2 * kw + 1))));
                            float yv_3[8];
                            float2 _f2_762 = make_float2(zc[0], zc[1]);
                            float2 _mul_f32x2_64;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_64) : "l"(*(const unsigned long long*)&_f2_762), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_763 = make_float2(__uint_as_float(wc_2[0] << 16), __uint_as_float(wc_2[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_128;
                            {
                                float2 _packed_f32x2_142_0 = _f2_697;
                                float2 _packed_f32x2_142_1 = _f2_763;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_128)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_142_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_142_1)));
                            }
                            float2 _mul_f32x2_65;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_65) : "l"(*(const unsigned long long*)&_mul_f32x2_64), "l"(*(const unsigned long long*)&_packed_add_f32x2_128));
                            yv_3[0] = _mul_f32x2_65.x;
                            yv_3[1] = _mul_f32x2_65.y;
                            float2 _f2_764 = make_float2(zc[2], zc[3]);
                            float2 _mul_f32x2_66;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_66) : "l"(*(const unsigned long long*)&_f2_764), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_765 = make_float2(__uint_as_float(wc_2[1] << 16), __uint_as_float(wc_2[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_129;
                            {
                                float2 _packed_f32x2_143_0 = _f2_697;
                                float2 _packed_f32x2_143_1 = _f2_765;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_129)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_143_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_143_1)));
                            }
                            float2 _mul_f32x2_67;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_67) : "l"(*(const unsigned long long*)&_mul_f32x2_66), "l"(*(const unsigned long long*)&_packed_add_f32x2_129));
                            yv_3[2] = _mul_f32x2_67.x;
                            yv_3[3] = _mul_f32x2_67.y;
                            float2 _f2_766 = make_float2(zc[4], zc[5]);
                            float2 _mul_f32x2_68;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_68) : "l"(*(const unsigned long long*)&_f2_766), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_767 = make_float2(__uint_as_float(wc_2[2] << 16), __uint_as_float(wc_2[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_130;
                            {
                                float2 _packed_f32x2_144_0 = _f2_697;
                                float2 _packed_f32x2_144_1 = _f2_767;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_130)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_144_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_144_1)));
                            }
                            float2 _mul_f32x2_69;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_69) : "l"(*(const unsigned long long*)&_mul_f32x2_68), "l"(*(const unsigned long long*)&_packed_add_f32x2_130));
                            yv_3[4] = _mul_f32x2_69.x;
                            yv_3[5] = _mul_f32x2_69.y;
                            float2 _f2_768 = make_float2(zc[6], zc[7]);
                            float2 _mul_f32x2_70;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_70) : "l"(*(const unsigned long long*)&_f2_768), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_769 = make_float2(__uint_as_float(wc_2[3] << 16), __uint_as_float(wc_2[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_131;
                            {
                                float2 _packed_f32x2_145_0 = _f2_697;
                                float2 _packed_f32x2_145_1 = _f2_769;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_131)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_145_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_145_1)));
                            }
                            float2 _mul_f32x2_71;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_71) : "l"(*(const unsigned long long*)&_mul_f32x2_70), "l"(*(const unsigned long long*)&_packed_add_f32x2_131));
                            yv_3[6] = _mul_f32x2_71.x;
                            yv_3[7] = _mul_f32x2_71.y;
                            uint32_t yv_3_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_3[_lp*2 + 0], yv_3[_lp*2+1 + 0]));
                                yv_3_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + kw))), "r"(yv_3_bf16[0]), "r"(yv_3_bf16[1]), "r"(yv_3_bf16[2]), "r"(yv_3_bf16[3]) : "memory");
                            int kw_4 = f ^ 1;
                            unsigned int wc_6[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&wc_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_6[(0) + 3]))
                                : "r"(wrow_addr + (unsigned int)(64 * pp_0) + (unsigned int)(16 * kw_4)));
                            float zc_7[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&zc_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_7[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_4 + (unsigned int)(16 * (l7_4 ^ 2 * kw_4))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&zc_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_7[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_4 + (unsigned int)(16 * (l7_4 ^ 2 * kw_4 + 1))));
                            float yv_8[8];
                            float2 _f2_770 = make_float2(zc_7[0], zc_7[1]);
                            float2 _mul_f32x2_72;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_72) : "l"(*(const unsigned long long*)&_f2_770), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_771 = make_float2(__uint_as_float(wc_6[0] << 16), __uint_as_float(wc_6[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_132;
                            {
                                float2 _packed_f32x2_146_0 = _f2_697;
                                float2 _packed_f32x2_146_1 = _f2_771;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_132)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_146_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_146_1)));
                            }
                            float2 _mul_f32x2_73;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_73) : "l"(*(const unsigned long long*)&_mul_f32x2_72), "l"(*(const unsigned long long*)&_packed_add_f32x2_132));
                            yv_8[0] = _mul_f32x2_73.x;
                            yv_8[1] = _mul_f32x2_73.y;
                            float2 _f2_772 = make_float2(zc_7[2], zc_7[3]);
                            float2 _mul_f32x2_74;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_74) : "l"(*(const unsigned long long*)&_f2_772), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_773 = make_float2(__uint_as_float(wc_6[1] << 16), __uint_as_float(wc_6[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_133;
                            {
                                float2 _packed_f32x2_147_0 = _f2_697;
                                float2 _packed_f32x2_147_1 = _f2_773;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_133)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_147_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_147_1)));
                            }
                            float2 _mul_f32x2_75;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_75) : "l"(*(const unsigned long long*)&_mul_f32x2_74), "l"(*(const unsigned long long*)&_packed_add_f32x2_133));
                            yv_8[2] = _mul_f32x2_75.x;
                            yv_8[3] = _mul_f32x2_75.y;
                            float2 _f2_774 = make_float2(zc_7[4], zc_7[5]);
                            float2 _mul_f32x2_76;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_76) : "l"(*(const unsigned long long*)&_f2_774), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_775 = make_float2(__uint_as_float(wc_6[2] << 16), __uint_as_float(wc_6[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_134;
                            {
                                float2 _packed_f32x2_148_0 = _f2_697;
                                float2 _packed_f32x2_148_1 = _f2_775;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_134)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_148_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_148_1)));
                            }
                            float2 _mul_f32x2_77;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_77) : "l"(*(const unsigned long long*)&_mul_f32x2_76), "l"(*(const unsigned long long*)&_packed_add_f32x2_134));
                            yv_8[4] = _mul_f32x2_77.x;
                            yv_8[5] = _mul_f32x2_77.y;
                            float2 _f2_776 = make_float2(zc_7[6], zc_7[7]);
                            float2 _mul_f32x2_78;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_78) : "l"(*(const unsigned long long*)&_f2_776), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_777 = make_float2(__uint_as_float(wc_6[3] << 16), __uint_as_float(wc_6[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_135;
                            {
                                float2 _packed_f32x2_149_0 = _f2_697;
                                float2 _packed_f32x2_149_1 = _f2_777;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_135)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_149_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_149_1)));
                            }
                            float2 _mul_f32x2_79;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_79) : "l"(*(const unsigned long long*)&_mul_f32x2_78), "l"(*(const unsigned long long*)&_packed_add_f32x2_135));
                            yv_8[6] = _mul_f32x2_79.x;
                            yv_8[7] = _mul_f32x2_79.y;
                            uint32_t yv_8_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_8[_lp*2 + 0], yv_8[_lp*2+1 + 0]));
                                yv_8_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + kw_4))), "r"(yv_8_bf16[0]), "r"(yv_8_bf16[1]), "r"(yv_8_bf16[2]), "r"(yv_8_bf16[3]) : "memory");
                            int kw_9 = f ^ 2;
                            unsigned int wc_10[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&wc_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_10[(0) + 3]))
                                : "r"(wrow_addr + (unsigned int)(64 * pp_0) + (unsigned int)(16 * kw_9)));
                            float zc_11[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&zc_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_11[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_4 + (unsigned int)(16 * (l7_4 ^ 2 * kw_9))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&zc_11[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_11[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_11[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_4 + (unsigned int)(16 * (l7_4 ^ 2 * kw_9 + 1))));
                            float yv_12[8];
                            float2 _f2_778 = make_float2(zc_11[0], zc_11[1]);
                            float2 _mul_f32x2_80;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_80) : "l"(*(const unsigned long long*)&_f2_778), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_779 = make_float2(__uint_as_float(wc_10[0] << 16), __uint_as_float(wc_10[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_136;
                            {
                                float2 _packed_f32x2_150_0 = _f2_697;
                                float2 _packed_f32x2_150_1 = _f2_779;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_136)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_150_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_150_1)));
                            }
                            float2 _mul_f32x2_81;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_81) : "l"(*(const unsigned long long*)&_mul_f32x2_80), "l"(*(const unsigned long long*)&_packed_add_f32x2_136));
                            yv_12[0] = _mul_f32x2_81.x;
                            yv_12[1] = _mul_f32x2_81.y;
                            float2 _f2_780 = make_float2(zc_11[2], zc_11[3]);
                            float2 _mul_f32x2_82;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_82) : "l"(*(const unsigned long long*)&_f2_780), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_781 = make_float2(__uint_as_float(wc_10[1] << 16), __uint_as_float(wc_10[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_137;
                            {
                                float2 _packed_f32x2_151_0 = _f2_697;
                                float2 _packed_f32x2_151_1 = _f2_781;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_137)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_151_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_151_1)));
                            }
                            float2 _mul_f32x2_83;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_83) : "l"(*(const unsigned long long*)&_mul_f32x2_82), "l"(*(const unsigned long long*)&_packed_add_f32x2_137));
                            yv_12[2] = _mul_f32x2_83.x;
                            yv_12[3] = _mul_f32x2_83.y;
                            float2 _f2_782 = make_float2(zc_11[4], zc_11[5]);
                            float2 _mul_f32x2_84;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_84) : "l"(*(const unsigned long long*)&_f2_782), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_783 = make_float2(__uint_as_float(wc_10[2] << 16), __uint_as_float(wc_10[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_138;
                            {
                                float2 _packed_f32x2_152_0 = _f2_697;
                                float2 _packed_f32x2_152_1 = _f2_783;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_138)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_152_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_152_1)));
                            }
                            float2 _mul_f32x2_85;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_85) : "l"(*(const unsigned long long*)&_mul_f32x2_84), "l"(*(const unsigned long long*)&_packed_add_f32x2_138));
                            yv_12[4] = _mul_f32x2_85.x;
                            yv_12[5] = _mul_f32x2_85.y;
                            float2 _f2_784 = make_float2(zc_11[6], zc_11[7]);
                            float2 _mul_f32x2_86;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_86) : "l"(*(const unsigned long long*)&_f2_784), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_785 = make_float2(__uint_as_float(wc_10[3] << 16), __uint_as_float(wc_10[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_139;
                            {
                                float2 _packed_f32x2_153_0 = _f2_697;
                                float2 _packed_f32x2_153_1 = _f2_785;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_139)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_153_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_153_1)));
                            }
                            float2 _mul_f32x2_87;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_87) : "l"(*(const unsigned long long*)&_mul_f32x2_86), "l"(*(const unsigned long long*)&_packed_add_f32x2_139));
                            yv_12[6] = _mul_f32x2_87.x;
                            yv_12[7] = _mul_f32x2_87.y;
                            uint32_t yv_12_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_12[_lp*2 + 0], yv_12[_lp*2+1 + 0]));
                                yv_12_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + kw_9))), "r"(yv_12_bf16[0]), "r"(yv_12_bf16[1]), "r"(yv_12_bf16[2]), "r"(yv_12_bf16[3]) : "memory");
                            int kw_13 = f ^ 3;
                            unsigned int wc_15[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&wc_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_15[(0) + 3]))
                                : "r"(wrow_addr + (unsigned int)(64 * pp_0) + (unsigned int)(16 * kw_13)));
                            float zc_16[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&zc_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_16[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_4 + (unsigned int)(16 * (l7_4 ^ 2 * kw_13))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&zc_16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_16[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_4 + (unsigned int)(16 * (l7_4 ^ 2 * kw_13 + 1))));
                            float yv_18[8];
                            float2 _f2_786 = make_float2(zc_16[0], zc_16[1]);
                            float2 _mul_f32x2_88;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_88) : "l"(*(const unsigned long long*)&_f2_786), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_787 = make_float2(__uint_as_float(wc_15[0] << 16), __uint_as_float(wc_15[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_140;
                            {
                                float2 _packed_f32x2_154_0 = _f2_697;
                                float2 _packed_f32x2_154_1 = _f2_787;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_140)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_154_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_154_1)));
                            }
                            float2 _mul_f32x2_89;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_89) : "l"(*(const unsigned long long*)&_mul_f32x2_88), "l"(*(const unsigned long long*)&_packed_add_f32x2_140));
                            yv_18[0] = _mul_f32x2_89.x;
                            yv_18[1] = _mul_f32x2_89.y;
                            float2 _f2_788 = make_float2(zc_16[2], zc_16[3]);
                            float2 _mul_f32x2_90;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_90) : "l"(*(const unsigned long long*)&_f2_788), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_789 = make_float2(__uint_as_float(wc_15[1] << 16), __uint_as_float(wc_15[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_141;
                            {
                                float2 _packed_f32x2_155_0 = _f2_697;
                                float2 _packed_f32x2_155_1 = _f2_789;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_141)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_155_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_155_1)));
                            }
                            float2 _mul_f32x2_91;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_91) : "l"(*(const unsigned long long*)&_mul_f32x2_90), "l"(*(const unsigned long long*)&_packed_add_f32x2_141));
                            yv_18[2] = _mul_f32x2_91.x;
                            yv_18[3] = _mul_f32x2_91.y;
                            float2 _f2_790 = make_float2(zc_16[4], zc_16[5]);
                            float2 _mul_f32x2_92;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_92) : "l"(*(const unsigned long long*)&_f2_790), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_791 = make_float2(__uint_as_float(wc_15[2] << 16), __uint_as_float(wc_15[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_142;
                            {
                                float2 _packed_f32x2_156_0 = _f2_697;
                                float2 _packed_f32x2_156_1 = _f2_791;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_142)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_156_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_156_1)));
                            }
                            float2 _mul_f32x2_93;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_93) : "l"(*(const unsigned long long*)&_mul_f32x2_92), "l"(*(const unsigned long long*)&_packed_add_f32x2_142));
                            yv_18[4] = _mul_f32x2_93.x;
                            yv_18[5] = _mul_f32x2_93.y;
                            float2 _f2_792 = make_float2(zc_16[6], zc_16[7]);
                            float2 _mul_f32x2_94;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_94) : "l"(*(const unsigned long long*)&_f2_792), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_793 = make_float2(__uint_as_float(wc_15[3] << 16), __uint_as_float(wc_15[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_143;
                            {
                                float2 _packed_f32x2_157_0 = _f2_697;
                                float2 _packed_f32x2_157_1 = _f2_793;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_143)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_157_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_157_1)));
                            }
                            float2 _mul_f32x2_95;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_95) : "l"(*(const unsigned long long*)&_mul_f32x2_94), "l"(*(const unsigned long long*)&_packed_add_f32x2_143));
                            yv_18[6] = _mul_f32x2_95.x;
                            yv_18[7] = _mul_f32x2_95.y;
                            uint32_t yv_18_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_18[_lp*2 + 0], yv_18[_lp*2+1 + 0]));
                                yv_18_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + kw_13))), "r"(yv_18_bf16[0]), "r"(yv_18_bf16[1]), "r"(yv_18_bf16[2]), "r"(yv_18_bf16[3]) : "memory");
                            __syncwarp();
                            unsigned int w4_19[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * lane_3)));
                            long long idx_20 = drow * 3072 + (long long)(4 * (128 * j + lane_3));
                            {
                                int4 _iv4 = make_int4(w4_19[0 + 0], w4_19[0 + 1], w4_19[0 + 2], w4_19[0 + 3]);
                                *reinterpret_cast<int4*>(out_i32 + (idx_20)) = _iv4;
                            }
                            unsigned int w4_21[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 32))));
                            long long idx_22 = drow * 3072 + (long long)(4 * (128 * j + lane_3 + 32));
                            {
                                int4 _iv4 = make_int4(w4_21[0 + 0], w4_21[0 + 1], w4_21[0 + 2], w4_21[0 + 3]);
                                *reinterpret_cast<int4*>(out_i32 + (idx_22)) = _iv4;
                            }
                            unsigned int w4_23[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 64))));
                            long long idx_24 = drow * 3072 + (long long)(4 * (128 * j + lane_3 + 64));
                            {
                                int4 _iv4 = make_int4(w4_23[0 + 0], w4_23[0 + 1], w4_23[0 + 2], w4_23[0 + 3]);
                                *reinterpret_cast<int4*>(out_i32 + (idx_24)) = _iv4;
                            }
                            unsigned int w4_25[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_25[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_25[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_25[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 96))));
                            long long idx_26 = drow * 3072 + (long long)(4 * (128 * j + lane_3 + 96));
                            {
                                int4 _iv4 = make_int4(w4_25[0 + 0], w4_25[0 + 1], w4_25[0 + 2], w4_25[0 + 3]);
                                *reinterpret_cast<int4*>(out_i32 + (idx_26)) = _iv4;
                            }
                            __syncwarp();
                        }
                        #pragma unroll 1
                        for (int j_1 = 3; j_1 < 6; j_1++) {
                            int l7_5 = lane_3 & 7;
                            int pp_0_1 = lane_3 + 32 * j_1;
                            int zlane_5 = rb_off + 4096 * (j_1 - 3) + 128 * lane_3;
                            int kw_1 = f ^ 0;
                            unsigned int wc_2_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&wc_2_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_2_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_2_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_2_1[(0) + 3]))
                                : "r"(wrow_addr + (unsigned int)(64 * pp_0_1) + (unsigned int)(16 * kw_1)));
                            float zc_1[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&zc_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_1[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_5 + (unsigned int)(16 * (l7_5 ^ 2 * kw_1))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&zc_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_1[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_5 + (unsigned int)(16 * (l7_5 ^ 2 * kw_1 + 1))));
                            float yv_3_1[8];
                            float2 _f2_794 = make_float2(zc_1[0], zc_1[1]);
                            float2 _mul_f32x2_96;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_96) : "l"(*(const unsigned long long*)&_f2_794), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_795 = make_float2(__uint_as_float(wc_2_1[0] << 16), __uint_as_float(wc_2_1[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_144;
                            {
                                float2 _packed_f32x2_158_0 = _f2_697;
                                float2 _packed_f32x2_158_1 = _f2_795;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_144)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_158_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_158_1)));
                            }
                            float2 _mul_f32x2_97;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_97) : "l"(*(const unsigned long long*)&_mul_f32x2_96), "l"(*(const unsigned long long*)&_packed_add_f32x2_144));
                            yv_3_1[0] = _mul_f32x2_97.x;
                            yv_3_1[1] = _mul_f32x2_97.y;
                            float2 _f2_796 = make_float2(zc_1[2], zc_1[3]);
                            float2 _mul_f32x2_98;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_98) : "l"(*(const unsigned long long*)&_f2_796), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_797 = make_float2(__uint_as_float(wc_2_1[1] << 16), __uint_as_float(wc_2_1[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_145;
                            {
                                float2 _packed_f32x2_159_0 = _f2_697;
                                float2 _packed_f32x2_159_1 = _f2_797;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_145)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_159_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_159_1)));
                            }
                            float2 _mul_f32x2_99;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_99) : "l"(*(const unsigned long long*)&_mul_f32x2_98), "l"(*(const unsigned long long*)&_packed_add_f32x2_145));
                            yv_3_1[2] = _mul_f32x2_99.x;
                            yv_3_1[3] = _mul_f32x2_99.y;
                            float2 _f2_798 = make_float2(zc_1[4], zc_1[5]);
                            float2 _mul_f32x2_100;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_100) : "l"(*(const unsigned long long*)&_f2_798), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_799 = make_float2(__uint_as_float(wc_2_1[2] << 16), __uint_as_float(wc_2_1[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_146;
                            {
                                float2 _packed_f32x2_160_0 = _f2_697;
                                float2 _packed_f32x2_160_1 = _f2_799;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_146)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_160_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_160_1)));
                            }
                            float2 _mul_f32x2_101;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_101) : "l"(*(const unsigned long long*)&_mul_f32x2_100), "l"(*(const unsigned long long*)&_packed_add_f32x2_146));
                            yv_3_1[4] = _mul_f32x2_101.x;
                            yv_3_1[5] = _mul_f32x2_101.y;
                            float2 _f2_800 = make_float2(zc_1[6], zc_1[7]);
                            float2 _mul_f32x2_102;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_102) : "l"(*(const unsigned long long*)&_f2_800), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_801 = make_float2(__uint_as_float(wc_2_1[3] << 16), __uint_as_float(wc_2_1[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_147;
                            {
                                float2 _packed_f32x2_161_0 = _f2_697;
                                float2 _packed_f32x2_161_1 = _f2_801;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_147)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_161_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_161_1)));
                            }
                            float2 _mul_f32x2_103;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_103) : "l"(*(const unsigned long long*)&_mul_f32x2_102), "l"(*(const unsigned long long*)&_packed_add_f32x2_147));
                            yv_3_1[6] = _mul_f32x2_103.x;
                            yv_3_1[7] = _mul_f32x2_103.y;
                            uint32_t yv_3_bf16_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_3_1[_lp*2 + 0], yv_3_1[_lp*2+1 + 0]));
                                yv_3_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + kw_1))), "r"(yv_3_bf16_1[0]), "r"(yv_3_bf16_1[1]), "r"(yv_3_bf16_1[2]), "r"(yv_3_bf16_1[3]) : "memory");
                            int kw_4_1 = f ^ 1;
                            unsigned int wc_6_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&wc_6_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_6_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_6_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_6_1[(0) + 3]))
                                : "r"(wrow_addr + (unsigned int)(64 * pp_0_1) + (unsigned int)(16 * kw_4_1)));
                            float zc_7_1[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_7_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&zc_7_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_7_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_7_1[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_5 + (unsigned int)(16 * (l7_5 ^ 2 * kw_4_1))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_7_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&zc_7_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_7_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_7_1[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_5 + (unsigned int)(16 * (l7_5 ^ 2 * kw_4_1 + 1))));
                            float yv_8_1[8];
                            float2 _f2_802 = make_float2(zc_7_1[0], zc_7_1[1]);
                            float2 _mul_f32x2_104;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_104) : "l"(*(const unsigned long long*)&_f2_802), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_803 = make_float2(__uint_as_float(wc_6_1[0] << 16), __uint_as_float(wc_6_1[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_148;
                            {
                                float2 _packed_f32x2_162_0 = _f2_697;
                                float2 _packed_f32x2_162_1 = _f2_803;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_148)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_162_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_162_1)));
                            }
                            float2 _mul_f32x2_105;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_105) : "l"(*(const unsigned long long*)&_mul_f32x2_104), "l"(*(const unsigned long long*)&_packed_add_f32x2_148));
                            yv_8_1[0] = _mul_f32x2_105.x;
                            yv_8_1[1] = _mul_f32x2_105.y;
                            float2 _f2_804 = make_float2(zc_7_1[2], zc_7_1[3]);
                            float2 _mul_f32x2_106;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_106) : "l"(*(const unsigned long long*)&_f2_804), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_805 = make_float2(__uint_as_float(wc_6_1[1] << 16), __uint_as_float(wc_6_1[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_149;
                            {
                                float2 _packed_f32x2_163_0 = _f2_697;
                                float2 _packed_f32x2_163_1 = _f2_805;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_149)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_163_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_163_1)));
                            }
                            float2 _mul_f32x2_107;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_107) : "l"(*(const unsigned long long*)&_mul_f32x2_106), "l"(*(const unsigned long long*)&_packed_add_f32x2_149));
                            yv_8_1[2] = _mul_f32x2_107.x;
                            yv_8_1[3] = _mul_f32x2_107.y;
                            float2 _f2_806 = make_float2(zc_7_1[4], zc_7_1[5]);
                            float2 _mul_f32x2_108;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_108) : "l"(*(const unsigned long long*)&_f2_806), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_807 = make_float2(__uint_as_float(wc_6_1[2] << 16), __uint_as_float(wc_6_1[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_150;
                            {
                                float2 _packed_f32x2_164_0 = _f2_697;
                                float2 _packed_f32x2_164_1 = _f2_807;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_150)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_164_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_164_1)));
                            }
                            float2 _mul_f32x2_109;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_109) : "l"(*(const unsigned long long*)&_mul_f32x2_108), "l"(*(const unsigned long long*)&_packed_add_f32x2_150));
                            yv_8_1[4] = _mul_f32x2_109.x;
                            yv_8_1[5] = _mul_f32x2_109.y;
                            float2 _f2_808 = make_float2(zc_7_1[6], zc_7_1[7]);
                            float2 _mul_f32x2_110;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_110) : "l"(*(const unsigned long long*)&_f2_808), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_809 = make_float2(__uint_as_float(wc_6_1[3] << 16), __uint_as_float(wc_6_1[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_151;
                            {
                                float2 _packed_f32x2_165_0 = _f2_697;
                                float2 _packed_f32x2_165_1 = _f2_809;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_151)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_165_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_165_1)));
                            }
                            float2 _mul_f32x2_111;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_111) : "l"(*(const unsigned long long*)&_mul_f32x2_110), "l"(*(const unsigned long long*)&_packed_add_f32x2_151));
                            yv_8_1[6] = _mul_f32x2_111.x;
                            yv_8_1[7] = _mul_f32x2_111.y;
                            uint32_t yv_8_bf16_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_8_1[_lp*2 + 0], yv_8_1[_lp*2+1 + 0]));
                                yv_8_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + kw_4_1))), "r"(yv_8_bf16_1[0]), "r"(yv_8_bf16_1[1]), "r"(yv_8_bf16_1[2]), "r"(yv_8_bf16_1[3]) : "memory");
                            int kw_9_1 = f ^ 2;
                            unsigned int wc_10_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&wc_10_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_10_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_10_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_10_1[(0) + 3]))
                                : "r"(wrow_addr + (unsigned int)(64 * pp_0_1) + (unsigned int)(16 * kw_9_1)));
                            float zc_11_1[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_11_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&zc_11_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_11_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_11_1[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_5 + (unsigned int)(16 * (l7_5 ^ 2 * kw_9_1))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_11_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&zc_11_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_11_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_11_1[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_5 + (unsigned int)(16 * (l7_5 ^ 2 * kw_9_1 + 1))));
                            float yv_12_1[8];
                            float2 _f2_810 = make_float2(zc_11_1[0], zc_11_1[1]);
                            float2 _mul_f32x2_112;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_112) : "l"(*(const unsigned long long*)&_f2_810), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_811 = make_float2(__uint_as_float(wc_10_1[0] << 16), __uint_as_float(wc_10_1[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_152;
                            {
                                float2 _packed_f32x2_166_0 = _f2_697;
                                float2 _packed_f32x2_166_1 = _f2_811;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_152)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_166_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_166_1)));
                            }
                            float2 _mul_f32x2_113;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_113) : "l"(*(const unsigned long long*)&_mul_f32x2_112), "l"(*(const unsigned long long*)&_packed_add_f32x2_152));
                            yv_12_1[0] = _mul_f32x2_113.x;
                            yv_12_1[1] = _mul_f32x2_113.y;
                            float2 _f2_812 = make_float2(zc_11_1[2], zc_11_1[3]);
                            float2 _mul_f32x2_114;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_114) : "l"(*(const unsigned long long*)&_f2_812), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_813 = make_float2(__uint_as_float(wc_10_1[1] << 16), __uint_as_float(wc_10_1[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_153;
                            {
                                float2 _packed_f32x2_167_0 = _f2_697;
                                float2 _packed_f32x2_167_1 = _f2_813;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_153)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_167_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_167_1)));
                            }
                            float2 _mul_f32x2_115;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_115) : "l"(*(const unsigned long long*)&_mul_f32x2_114), "l"(*(const unsigned long long*)&_packed_add_f32x2_153));
                            yv_12_1[2] = _mul_f32x2_115.x;
                            yv_12_1[3] = _mul_f32x2_115.y;
                            float2 _f2_814 = make_float2(zc_11_1[4], zc_11_1[5]);
                            float2 _mul_f32x2_116;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_116) : "l"(*(const unsigned long long*)&_f2_814), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_815 = make_float2(__uint_as_float(wc_10_1[2] << 16), __uint_as_float(wc_10_1[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_154;
                            {
                                float2 _packed_f32x2_168_0 = _f2_697;
                                float2 _packed_f32x2_168_1 = _f2_815;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_154)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_168_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_168_1)));
                            }
                            float2 _mul_f32x2_117;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_117) : "l"(*(const unsigned long long*)&_mul_f32x2_116), "l"(*(const unsigned long long*)&_packed_add_f32x2_154));
                            yv_12_1[4] = _mul_f32x2_117.x;
                            yv_12_1[5] = _mul_f32x2_117.y;
                            float2 _f2_816 = make_float2(zc_11_1[6], zc_11_1[7]);
                            float2 _mul_f32x2_118;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_118) : "l"(*(const unsigned long long*)&_f2_816), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_817 = make_float2(__uint_as_float(wc_10_1[3] << 16), __uint_as_float(wc_10_1[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_155;
                            {
                                float2 _packed_f32x2_169_0 = _f2_697;
                                float2 _packed_f32x2_169_1 = _f2_817;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_155)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_169_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_169_1)));
                            }
                            float2 _mul_f32x2_119;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_119) : "l"(*(const unsigned long long*)&_mul_f32x2_118), "l"(*(const unsigned long long*)&_packed_add_f32x2_155));
                            yv_12_1[6] = _mul_f32x2_119.x;
                            yv_12_1[7] = _mul_f32x2_119.y;
                            uint32_t yv_12_bf16_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_12_1[_lp*2 + 0], yv_12_1[_lp*2+1 + 0]));
                                yv_12_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + kw_9_1))), "r"(yv_12_bf16_1[0]), "r"(yv_12_bf16_1[1]), "r"(yv_12_bf16_1[2]), "r"(yv_12_bf16_1[3]) : "memory");
                            int kw_13_1 = f ^ 3;
                            unsigned int wc_15_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&wc_15_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&wc_15_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wc_15_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wc_15_1[(0) + 3]))
                                : "r"(wrow_addr + (unsigned int)(64 * pp_0_1) + (unsigned int)(16 * kw_13_1)));
                            float zc_16_1[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_16_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&zc_16_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_16_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_16_1[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_5 + (unsigned int)(16 * (l7_5 ^ 2 * kw_13_1))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&zc_16_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&zc_16_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&zc_16_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&zc_16_1[(4) + 3]))
                                : "r"(wsm_addr + (unsigned int)zlane_5 + (unsigned int)(16 * (l7_5 ^ 2 * kw_13_1 + 1))));
                            float yv_18_1[8];
                            float2 _f2_818 = make_float2(zc_16_1[0], zc_16_1[1]);
                            float2 _mul_f32x2_120;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_120) : "l"(*(const unsigned long long*)&_f2_818), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_819 = make_float2(__uint_as_float(wc_15_1[0] << 16), __uint_as_float(wc_15_1[0] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_156;
                            {
                                float2 _packed_f32x2_170_0 = _f2_697;
                                float2 _packed_f32x2_170_1 = _f2_819;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_156)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_170_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_170_1)));
                            }
                            float2 _mul_f32x2_121;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_121) : "l"(*(const unsigned long long*)&_mul_f32x2_120), "l"(*(const unsigned long long*)&_packed_add_f32x2_156));
                            yv_18_1[0] = _mul_f32x2_121.x;
                            yv_18_1[1] = _mul_f32x2_121.y;
                            float2 _f2_820 = make_float2(zc_16_1[2], zc_16_1[3]);
                            float2 _mul_f32x2_122;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_122) : "l"(*(const unsigned long long*)&_f2_820), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_821 = make_float2(__uint_as_float(wc_15_1[1] << 16), __uint_as_float(wc_15_1[1] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_157;
                            {
                                float2 _packed_f32x2_171_0 = _f2_697;
                                float2 _packed_f32x2_171_1 = _f2_821;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_157)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_171_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_171_1)));
                            }
                            float2 _mul_f32x2_123;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_123) : "l"(*(const unsigned long long*)&_mul_f32x2_122), "l"(*(const unsigned long long*)&_packed_add_f32x2_157));
                            yv_18_1[2] = _mul_f32x2_123.x;
                            yv_18_1[3] = _mul_f32x2_123.y;
                            float2 _f2_822 = make_float2(zc_16_1[4], zc_16_1[5]);
                            float2 _mul_f32x2_124;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_124) : "l"(*(const unsigned long long*)&_f2_822), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_823 = make_float2(__uint_as_float(wc_15_1[2] << 16), __uint_as_float(wc_15_1[2] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_158;
                            {
                                float2 _packed_f32x2_172_0 = _f2_697;
                                float2 _packed_f32x2_172_1 = _f2_823;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_158)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_172_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_172_1)));
                            }
                            float2 _mul_f32x2_125;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_125) : "l"(*(const unsigned long long*)&_mul_f32x2_124), "l"(*(const unsigned long long*)&_packed_add_f32x2_158));
                            yv_18_1[4] = _mul_f32x2_125.x;
                            yv_18_1[5] = _mul_f32x2_125.y;
                            float2 _f2_824 = make_float2(zc_16_1[6], zc_16_1[7]);
                            float2 _mul_f32x2_126;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_126) : "l"(*(const unsigned long long*)&_f2_824), "l"(*(const unsigned long long*)&_f2_696));
                            float2 _f2_825 = make_float2(__uint_as_float(wc_15_1[3] << 16), __uint_as_float(wc_15_1[3] & 4294901760u));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_add_f32x2_159;
                            {
                                float2 _packed_f32x2_173_0 = _f2_697;
                                float2 _packed_f32x2_173_1 = _f2_825;
                                asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_159)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_173_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_173_1)));
                            }
                            float2 _mul_f32x2_127;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_127) : "l"(*(const unsigned long long*)&_mul_f32x2_126), "l"(*(const unsigned long long*)&_packed_add_f32x2_159));
                            yv_18_1[6] = _mul_f32x2_127.x;
                            yv_18_1[7] = _mul_f32x2_127.y;
                            uint32_t yv_18_bf16_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yv_18_1[_lp*2 + 0], yv_18_1[_lp*2+1 + 0]));
                                yv_18_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_3 + kw_13_1))), "r"(yv_18_bf16_1[0]), "r"(yv_18_bf16_1[1]), "r"(yv_18_bf16_1[2]), "r"(yv_18_bf16_1[3]) : "memory");
                            __syncwarp();
                            unsigned int w4_19_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_19_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19_1[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * lane_3)));
                            long long idx_20_1 = drow * 3072 + (long long)(4 * (128 * j_1 + lane_3));
                            {
                                int4 _iv4 = make_int4(w4_19_1[0 + 0], w4_19_1[0 + 1], w4_19_1[0 + 2], w4_19_1[0 + 3]);
                                *reinterpret_cast<int4*>(out_i32 + (idx_20_1)) = _iv4;
                            }
                            unsigned int w4_21_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_21_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21_1[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 32))));
                            long long idx_22_1 = drow * 3072 + (long long)(4 * (128 * j_1 + lane_3 + 32));
                            {
                                int4 _iv4 = make_int4(w4_21_1[0 + 0], w4_21_1[0 + 1], w4_21_1[0 + 2], w4_21_1[0 + 3]);
                                *reinterpret_cast<int4*>(out_i32 + (idx_22_1)) = _iv4;
                            }
                            unsigned int w4_23_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_23_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23_1[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 64))));
                            long long idx_24_1 = drow * 3072 + (long long)(4 * (128 * j_1 + lane_3 + 64));
                            {
                                int4 _iv4 = make_int4(w4_23_1[0 + 0], w4_23_1[0 + 1], w4_23_1[0 + 2], w4_23_1[0 + 3]);
                                *reinterpret_cast<int4*>(out_i32 + (idx_24_1)) = _iv4;
                            }
                            unsigned int w4_25_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w4_25_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_25_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_25_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_25_1[(0) + 3]))
                                : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (lane_3 + 96))));
                            long long idx_26_1 = drow * 3072 + (long long)(4 * (128 * j_1 + lane_3 + 96));
                            {
                                int4 _iv4 = make_int4(w4_25_1[0 + 0], w4_25_1[0 + 1], w4_25_1[0 + 2], w4_25_1[0 + 3]);
                                *reinterpret_cast<int4*>(out_i32 + (idx_26_1)) = _iv4;
                            }
                            __syncwarp();
                        }
                    }
                    if (n > kd + 1) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (lane_3 == 0) {
                            long long nrow = (long long)(r0 + warp_2 + 7 * (kd + 1));
                            mbarrier_arrive_expect_tx(rb_full_addr + (warp_2) * 8, 12288);
                            cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)rb_off, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(residual_in) + ((unsigned long long)(nrow * 6144) * (unsigned long long)2)), 12288, rb_full_addr + (warp_2) * 8);
                        }
                    }
                }
            }
            if (lane_3 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            asm volatile("barrier.sync 6, 256;" ::: "memory");
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
    if (warp == 7) {
        { // reset_warp_main
            int lane_0 = (int)lane;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("barrier.sync 0, 256;" ::: "memory");
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
                unsigned int w4_8[4];
                w4_8[0] = sv;
                w4_8[1] = sv;
                w4_8[2] = sv;
                w4_8[3] = sv;
                long long plane = pb64 * 113770496;
                long long u = pr64 * 3072 / 16;
                long long lo = u * bid64_1 / g64_1 * 16;
                long long hi = u * (bid64_1 + 1) / g64_1 * 16;
                #pragma unroll 1
                for (long long o = lo + 16 * lane64; o < hi; o += 512) {
                    long long widx = plane + o >> 2;
                    {
                        int4 _iv4 = make_int4(w4_8[0 + 0], w4_8[0 + 1], w4_8[0 + 2], w4_8[0 + 3]);
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
                unsigned int w4_5_1[4];
                w4_5_1[0] = sr;
                w4_5_1[1] = sr;
                w4_5_1[2] = sr;
                w4_5_1[3] = sr;
                long long plane_6 = pb64 * 113770496 + 113246208;
                long long u_7 = pr64 * 16 / 16;
                long long lo_8 = u_7 * bid64_1 / g64_1 * 16;
                long long hi_9 = u_7 * (bid64_1 + 1) / g64_1 * 16;
                #pragma unroll 1
                for (long long o_2 = lo_8 + 16 * lane64; o_2 < hi_9; o_2 += 512) {
                    long long widx_2 = plane_6 + o_2 >> 2;
                    {
                        int4 _iv4 = make_int4(w4_5_1[0 + 0], w4_5_1[0 + 1], w4_5_1[0 + 2], w4_5_1[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx_2)) = _iv4;
                    }
                }
            }
            asm volatile("barrier.sync 6, 256;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
