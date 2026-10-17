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
kernel_fp4_allreduce_tp4_twoshot_bf16_fused_bv0_r0_cap16384(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, unsigned int* __restrict__ ctrl, int rows, const unsigned long long* __restrict__ workspace)
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
                mbarrier_arrive_expect_tx(xb_full_addr + (stage) * 8, 12288);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)xb_off, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(x) + ((unsigned long long)(first_row * 6144) * (unsigned long long)2)), 12288, xb_full_addr + (stage) * 8);
            }
            #pragma unroll 1
            for (int k = 0; k < n; k++) {
                int ob_off = own_off + (k & 1) * 3472;
                if (k >= 2) {
                    if (lane_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncwarp();
                }
                float rs = 1.0f;
                mbarrier_wait(xb_full_addr + (2 * warp_0 + (k & 1)) * 8, k >> 1 & 1);
                unsigned int bmp[6];
                float am = 0.0f;
                int pp = lane_1;
                int base = xb_off + 64 * pp;
                unsigned int a = 0;
                unsigned int b = 0;
                unsigned int c0[4];
                unsigned int c1[4];
                unsigned int c2[4];
                unsigned int c3[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c0[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base + (unsigned int)(16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base + (unsigned int)(16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base + (unsigned int)(16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base + (unsigned int)(16 * (f ^ 3))));
                uint32_t _bf16x2_abs_0;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(c0[0]));
                uint32_t _bf16x2_max_0;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(a), "r"(_bf16x2_abs_0));
                a = _bf16x2_max_0;
                uint32_t _bf16x2_abs_1;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(c0[1]));
                uint32_t _bf16x2_max_1;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(a), "r"(_bf16x2_abs_1));
                a = _bf16x2_max_1;
                uint32_t _bf16x2_abs_2;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(c0[2]));
                uint32_t _bf16x2_max_2;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(a), "r"(_bf16x2_abs_2));
                a = _bf16x2_max_2;
                uint32_t _bf16x2_abs_3;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(c0[3]));
                uint32_t _bf16x2_max_3;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(a), "r"(_bf16x2_abs_3));
                a = _bf16x2_max_3;
                uint32_t _bf16x2_abs_4;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_4) : "r"(c1[0]));
                uint32_t _bf16x2_max_4;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(a), "r"(_bf16x2_abs_4));
                a = _bf16x2_max_4;
                uint32_t _bf16x2_abs_5;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_5) : "r"(c1[1]));
                uint32_t _bf16x2_max_5;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(a), "r"(_bf16x2_abs_5));
                a = _bf16x2_max_5;
                uint32_t _bf16x2_abs_6;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_6) : "r"(c1[2]));
                uint32_t _bf16x2_max_6;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_6) : "r"(a), "r"(_bf16x2_abs_6));
                a = _bf16x2_max_6;
                uint32_t _bf16x2_abs_7;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_7) : "r"(c1[3]));
                uint32_t _bf16x2_max_7;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_7) : "r"(a), "r"(_bf16x2_abs_7));
                a = _bf16x2_max_7;
                uint32_t _bf16x2_abs_8;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_8) : "r"(c2[0]));
                uint32_t _bf16x2_max_8;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_8) : "r"(b), "r"(_bf16x2_abs_8));
                b = _bf16x2_max_8;
                uint32_t _bf16x2_abs_9;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_9) : "r"(c2[1]));
                uint32_t _bf16x2_max_9;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_9) : "r"(b), "r"(_bf16x2_abs_9));
                b = _bf16x2_max_9;
                uint32_t _bf16x2_abs_10;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_10) : "r"(c2[2]));
                uint32_t _bf16x2_max_10;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_10) : "r"(b), "r"(_bf16x2_abs_10));
                b = _bf16x2_max_10;
                uint32_t _bf16x2_abs_11;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_11) : "r"(c2[3]));
                uint32_t _bf16x2_max_11;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_11) : "r"(b), "r"(_bf16x2_abs_11));
                b = _bf16x2_max_11;
                uint32_t _bf16x2_abs_12;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_12) : "r"(c3[0]));
                uint32_t _bf16x2_max_12;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_12) : "r"(b), "r"(_bf16x2_abs_12));
                b = _bf16x2_max_12;
                uint32_t _bf16x2_abs_13;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_13) : "r"(c3[1]));
                uint32_t _bf16x2_max_13;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_13) : "r"(b), "r"(_bf16x2_abs_13));
                b = _bf16x2_max_13;
                uint32_t _bf16x2_abs_14;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_14) : "r"(c3[2]));
                uint32_t _bf16x2_max_14;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_14) : "r"(b), "r"(_bf16x2_abs_14));
                b = _bf16x2_max_14;
                uint32_t _bf16x2_abs_15;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_15) : "r"(c3[3]));
                uint32_t _bf16x2_max_15;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_15) : "r"(b), "r"(_bf16x2_abs_15));
                b = _bf16x2_max_15;
                unsigned int _max_0 = ((a & 65535) > (a >> 16) ? (a & 65535) : (a >> 16));
                unsigned int xa = _max_0;
                unsigned int _max_1 = ((b & 65535) > (b >> 16) ? (b & 65535) : (b >> 16));
                unsigned int xb = _max_1;
                bmp[0] = (((f & 2) != 0) ? xb | xa << 16 : xa | xb << 16);
                unsigned int _max_2 = ((xa) > (xb) ? (xa) : (xb));
                float _fmax_0 = fmaxf(am, __uint_as_float(_max_2 << 16));
                am = _fmax_0;
                int pp_0 = lane_1 + 32;
                int base_1 = xb_off + 64 * pp_0;
                unsigned int a_2 = 0;
                unsigned int b_3 = 0;
                unsigned int c0_4[4];
                unsigned int c1_5[4];
                unsigned int c2_6[4];
                unsigned int c3_7[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c0_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_4[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_1 + (unsigned int)(16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_5[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_1 + (unsigned int)(16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_6[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_1 + (unsigned int)(16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_7[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_1 + (unsigned int)(16 * (f ^ 3))));
                uint32_t _bf16x2_abs_16;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16) : "r"(c0_4[0]));
                uint32_t _bf16x2_max_16;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_16) : "r"(a_2), "r"(_bf16x2_abs_16));
                a_2 = _bf16x2_max_16;
                uint32_t _bf16x2_abs_17;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_17) : "r"(c0_4[1]));
                uint32_t _bf16x2_max_17;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_17) : "r"(a_2), "r"(_bf16x2_abs_17));
                a_2 = _bf16x2_max_17;
                uint32_t _bf16x2_abs_18;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_18) : "r"(c0_4[2]));
                uint32_t _bf16x2_max_18;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_18) : "r"(a_2), "r"(_bf16x2_abs_18));
                a_2 = _bf16x2_max_18;
                uint32_t _bf16x2_abs_19;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_19) : "r"(c0_4[3]));
                uint32_t _bf16x2_max_19;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_19) : "r"(a_2), "r"(_bf16x2_abs_19));
                a_2 = _bf16x2_max_19;
                uint32_t _bf16x2_abs_20;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_20) : "r"(c1_5[0]));
                uint32_t _bf16x2_max_20;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_20) : "r"(a_2), "r"(_bf16x2_abs_20));
                a_2 = _bf16x2_max_20;
                uint32_t _bf16x2_abs_21;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_21) : "r"(c1_5[1]));
                uint32_t _bf16x2_max_21;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_21) : "r"(a_2), "r"(_bf16x2_abs_21));
                a_2 = _bf16x2_max_21;
                uint32_t _bf16x2_abs_22;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_22) : "r"(c1_5[2]));
                uint32_t _bf16x2_max_22;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_22) : "r"(a_2), "r"(_bf16x2_abs_22));
                a_2 = _bf16x2_max_22;
                uint32_t _bf16x2_abs_23;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_23) : "r"(c1_5[3]));
                uint32_t _bf16x2_max_23;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_23) : "r"(a_2), "r"(_bf16x2_abs_23));
                a_2 = _bf16x2_max_23;
                uint32_t _bf16x2_abs_24;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_24) : "r"(c2_6[0]));
                uint32_t _bf16x2_max_24;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_24) : "r"(b_3), "r"(_bf16x2_abs_24));
                b_3 = _bf16x2_max_24;
                uint32_t _bf16x2_abs_25;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_25) : "r"(c2_6[1]));
                uint32_t _bf16x2_max_25;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_25) : "r"(b_3), "r"(_bf16x2_abs_25));
                b_3 = _bf16x2_max_25;
                uint32_t _bf16x2_abs_26;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_26) : "r"(c2_6[2]));
                uint32_t _bf16x2_max_26;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_26) : "r"(b_3), "r"(_bf16x2_abs_26));
                b_3 = _bf16x2_max_26;
                uint32_t _bf16x2_abs_27;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_27) : "r"(c2_6[3]));
                uint32_t _bf16x2_max_27;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_27) : "r"(b_3), "r"(_bf16x2_abs_27));
                b_3 = _bf16x2_max_27;
                uint32_t _bf16x2_abs_28;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_28) : "r"(c3_7[0]));
                uint32_t _bf16x2_max_28;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_28) : "r"(b_3), "r"(_bf16x2_abs_28));
                b_3 = _bf16x2_max_28;
                uint32_t _bf16x2_abs_29;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_29) : "r"(c3_7[1]));
                uint32_t _bf16x2_max_29;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_29) : "r"(b_3), "r"(_bf16x2_abs_29));
                b_3 = _bf16x2_max_29;
                uint32_t _bf16x2_abs_30;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_30) : "r"(c3_7[2]));
                uint32_t _bf16x2_max_30;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_30) : "r"(b_3), "r"(_bf16x2_abs_30));
                b_3 = _bf16x2_max_30;
                uint32_t _bf16x2_abs_31;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_31) : "r"(c3_7[3]));
                uint32_t _bf16x2_max_31;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_31) : "r"(b_3), "r"(_bf16x2_abs_31));
                b_3 = _bf16x2_max_31;
                unsigned int _max_3 = ((a_2 & 65535) > (a_2 >> 16) ? (a_2 & 65535) : (a_2 >> 16));
                unsigned int xa_8 = _max_3;
                unsigned int _max_4 = ((b_3 & 65535) > (b_3 >> 16) ? (b_3 & 65535) : (b_3 >> 16));
                unsigned int xb_9 = _max_4;
                bmp[1] = (((f & 2) != 0) ? xb_9 | xa_8 << 16 : xa_8 | xb_9 << 16);
                unsigned int _max_5 = ((xa_8) > (xb_9) ? (xa_8) : (xb_9));
                float _fmax_1 = fmaxf(am, __uint_as_float(_max_5 << 16));
                am = _fmax_1;
                int pp_10 = lane_1 + 64;
                int base_11 = xb_off + 64 * pp_10;
                unsigned int a_12 = 0;
                unsigned int b_13 = 0;
                unsigned int c0_14[4];
                unsigned int c1_15[4];
                unsigned int c2_16[4];
                unsigned int c3_17[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c0_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_14[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_11 + (unsigned int)(16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_15[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_11 + (unsigned int)(16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_16[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_11 + (unsigned int)(16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_17[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_11 + (unsigned int)(16 * (f ^ 3))));
                uint32_t _bf16x2_abs_32;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_32) : "r"(c0_14[0]));
                uint32_t _bf16x2_max_32;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_32) : "r"(a_12), "r"(_bf16x2_abs_32));
                a_12 = _bf16x2_max_32;
                uint32_t _bf16x2_abs_33;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_33) : "r"(c0_14[1]));
                uint32_t _bf16x2_max_33;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_33) : "r"(a_12), "r"(_bf16x2_abs_33));
                a_12 = _bf16x2_max_33;
                uint32_t _bf16x2_abs_34;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_34) : "r"(c0_14[2]));
                uint32_t _bf16x2_max_34;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_34) : "r"(a_12), "r"(_bf16x2_abs_34));
                a_12 = _bf16x2_max_34;
                uint32_t _bf16x2_abs_35;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_35) : "r"(c0_14[3]));
                uint32_t _bf16x2_max_35;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_35) : "r"(a_12), "r"(_bf16x2_abs_35));
                a_12 = _bf16x2_max_35;
                uint32_t _bf16x2_abs_36;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_36) : "r"(c1_15[0]));
                uint32_t _bf16x2_max_36;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_36) : "r"(a_12), "r"(_bf16x2_abs_36));
                a_12 = _bf16x2_max_36;
                uint32_t _bf16x2_abs_37;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_37) : "r"(c1_15[1]));
                uint32_t _bf16x2_max_37;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_37) : "r"(a_12), "r"(_bf16x2_abs_37));
                a_12 = _bf16x2_max_37;
                uint32_t _bf16x2_abs_38;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_38) : "r"(c1_15[2]));
                uint32_t _bf16x2_max_38;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_38) : "r"(a_12), "r"(_bf16x2_abs_38));
                a_12 = _bf16x2_max_38;
                uint32_t _bf16x2_abs_39;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_39) : "r"(c1_15[3]));
                uint32_t _bf16x2_max_39;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_39) : "r"(a_12), "r"(_bf16x2_abs_39));
                a_12 = _bf16x2_max_39;
                uint32_t _bf16x2_abs_40;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_40) : "r"(c2_16[0]));
                uint32_t _bf16x2_max_40;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_40) : "r"(b_13), "r"(_bf16x2_abs_40));
                b_13 = _bf16x2_max_40;
                uint32_t _bf16x2_abs_41;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_41) : "r"(c2_16[1]));
                uint32_t _bf16x2_max_41;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_41) : "r"(b_13), "r"(_bf16x2_abs_41));
                b_13 = _bf16x2_max_41;
                uint32_t _bf16x2_abs_42;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_42) : "r"(c2_16[2]));
                uint32_t _bf16x2_max_42;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_42) : "r"(b_13), "r"(_bf16x2_abs_42));
                b_13 = _bf16x2_max_42;
                uint32_t _bf16x2_abs_43;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_43) : "r"(c2_16[3]));
                uint32_t _bf16x2_max_43;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_43) : "r"(b_13), "r"(_bf16x2_abs_43));
                b_13 = _bf16x2_max_43;
                uint32_t _bf16x2_abs_44;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_44) : "r"(c3_17[0]));
                uint32_t _bf16x2_max_44;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_44) : "r"(b_13), "r"(_bf16x2_abs_44));
                b_13 = _bf16x2_max_44;
                uint32_t _bf16x2_abs_45;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_45) : "r"(c3_17[1]));
                uint32_t _bf16x2_max_45;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_45) : "r"(b_13), "r"(_bf16x2_abs_45));
                b_13 = _bf16x2_max_45;
                uint32_t _bf16x2_abs_46;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_46) : "r"(c3_17[2]));
                uint32_t _bf16x2_max_46;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_46) : "r"(b_13), "r"(_bf16x2_abs_46));
                b_13 = _bf16x2_max_46;
                uint32_t _bf16x2_abs_47;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_47) : "r"(c3_17[3]));
                uint32_t _bf16x2_max_47;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_47) : "r"(b_13), "r"(_bf16x2_abs_47));
                b_13 = _bf16x2_max_47;
                unsigned int _max_6 = ((a_12 & 65535) > (a_12 >> 16) ? (a_12 & 65535) : (a_12 >> 16));
                unsigned int xa_18 = _max_6;
                unsigned int _max_7 = ((b_13 & 65535) > (b_13 >> 16) ? (b_13 & 65535) : (b_13 >> 16));
                unsigned int xb_19 = _max_7;
                bmp[2] = (((f & 2) != 0) ? xb_19 | xa_18 << 16 : xa_18 | xb_19 << 16);
                unsigned int _max_8 = ((xa_18) > (xb_19) ? (xa_18) : (xb_19));
                float _fmax_2 = fmaxf(am, __uint_as_float(_max_8 << 16));
                am = _fmax_2;
                int pp_20 = lane_1 + 96;
                int base_21 = xb_off + 64 * pp_20;
                unsigned int a_22 = 0;
                unsigned int b_23 = 0;
                unsigned int c0_24[4];
                unsigned int c1_25[4];
                unsigned int c2_26[4];
                unsigned int c3_27[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c0_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_24[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_21 + (unsigned int)(16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_25[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_25[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_25[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_21 + (unsigned int)(16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_26[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_21 + (unsigned int)(16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_27[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_21 + (unsigned int)(16 * (f ^ 3))));
                uint32_t _bf16x2_abs_48;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_48) : "r"(c0_24[0]));
                uint32_t _bf16x2_max_48;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_48) : "r"(a_22), "r"(_bf16x2_abs_48));
                a_22 = _bf16x2_max_48;
                uint32_t _bf16x2_abs_49;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_49) : "r"(c0_24[1]));
                uint32_t _bf16x2_max_49;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_49) : "r"(a_22), "r"(_bf16x2_abs_49));
                a_22 = _bf16x2_max_49;
                uint32_t _bf16x2_abs_50;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_50) : "r"(c0_24[2]));
                uint32_t _bf16x2_max_50;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_50) : "r"(a_22), "r"(_bf16x2_abs_50));
                a_22 = _bf16x2_max_50;
                uint32_t _bf16x2_abs_51;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_51) : "r"(c0_24[3]));
                uint32_t _bf16x2_max_51;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_51) : "r"(a_22), "r"(_bf16x2_abs_51));
                a_22 = _bf16x2_max_51;
                uint32_t _bf16x2_abs_52;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_52) : "r"(c1_25[0]));
                uint32_t _bf16x2_max_52;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_52) : "r"(a_22), "r"(_bf16x2_abs_52));
                a_22 = _bf16x2_max_52;
                uint32_t _bf16x2_abs_53;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_53) : "r"(c1_25[1]));
                uint32_t _bf16x2_max_53;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_53) : "r"(a_22), "r"(_bf16x2_abs_53));
                a_22 = _bf16x2_max_53;
                uint32_t _bf16x2_abs_54;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_54) : "r"(c1_25[2]));
                uint32_t _bf16x2_max_54;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_54) : "r"(a_22), "r"(_bf16x2_abs_54));
                a_22 = _bf16x2_max_54;
                uint32_t _bf16x2_abs_55;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_55) : "r"(c1_25[3]));
                uint32_t _bf16x2_max_55;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_55) : "r"(a_22), "r"(_bf16x2_abs_55));
                a_22 = _bf16x2_max_55;
                uint32_t _bf16x2_abs_56;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_56) : "r"(c2_26[0]));
                uint32_t _bf16x2_max_56;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_56) : "r"(b_23), "r"(_bf16x2_abs_56));
                b_23 = _bf16x2_max_56;
                uint32_t _bf16x2_abs_57;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_57) : "r"(c2_26[1]));
                uint32_t _bf16x2_max_57;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_57) : "r"(b_23), "r"(_bf16x2_abs_57));
                b_23 = _bf16x2_max_57;
                uint32_t _bf16x2_abs_58;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_58) : "r"(c2_26[2]));
                uint32_t _bf16x2_max_58;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_58) : "r"(b_23), "r"(_bf16x2_abs_58));
                b_23 = _bf16x2_max_58;
                uint32_t _bf16x2_abs_59;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_59) : "r"(c2_26[3]));
                uint32_t _bf16x2_max_59;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_59) : "r"(b_23), "r"(_bf16x2_abs_59));
                b_23 = _bf16x2_max_59;
                uint32_t _bf16x2_abs_60;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_60) : "r"(c3_27[0]));
                uint32_t _bf16x2_max_60;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_60) : "r"(b_23), "r"(_bf16x2_abs_60));
                b_23 = _bf16x2_max_60;
                uint32_t _bf16x2_abs_61;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_61) : "r"(c3_27[1]));
                uint32_t _bf16x2_max_61;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_61) : "r"(b_23), "r"(_bf16x2_abs_61));
                b_23 = _bf16x2_max_61;
                uint32_t _bf16x2_abs_62;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_62) : "r"(c3_27[2]));
                uint32_t _bf16x2_max_62;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_62) : "r"(b_23), "r"(_bf16x2_abs_62));
                b_23 = _bf16x2_max_62;
                uint32_t _bf16x2_abs_63;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_63) : "r"(c3_27[3]));
                uint32_t _bf16x2_max_63;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_63) : "r"(b_23), "r"(_bf16x2_abs_63));
                b_23 = _bf16x2_max_63;
                unsigned int _max_9 = ((a_22 & 65535) > (a_22 >> 16) ? (a_22 & 65535) : (a_22 >> 16));
                unsigned int xa_28 = _max_9;
                unsigned int _max_10 = ((b_23 & 65535) > (b_23 >> 16) ? (b_23 & 65535) : (b_23 >> 16));
                unsigned int xb_29 = _max_10;
                bmp[3] = (((f & 2) != 0) ? xb_29 | xa_28 << 16 : xa_28 | xb_29 << 16);
                unsigned int _max_11 = ((xa_28) > (xb_29) ? (xa_28) : (xb_29));
                float _fmax_3 = fmaxf(am, __uint_as_float(_max_11 << 16));
                am = _fmax_3;
                int pp_30 = lane_1 + 128;
                int base_31 = xb_off + 64 * pp_30;
                unsigned int a_32 = 0;
                unsigned int b_33 = 0;
                unsigned int c0_34[4];
                unsigned int c1_35[4];
                unsigned int c2_36[4];
                unsigned int c3_37[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c0_34[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_34[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_34[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_34[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_31 + (unsigned int)(16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1_35[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_35[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_35[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_35[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_31 + (unsigned int)(16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_36[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_36[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_36[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_31 + (unsigned int)(16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_37[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_37[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_37[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_31 + (unsigned int)(16 * (f ^ 3))));
                uint32_t _bf16x2_abs_64;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_64) : "r"(c0_34[0]));
                uint32_t _bf16x2_max_64;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_64) : "r"(a_32), "r"(_bf16x2_abs_64));
                a_32 = _bf16x2_max_64;
                uint32_t _bf16x2_abs_65;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_65) : "r"(c0_34[1]));
                uint32_t _bf16x2_max_65;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_65) : "r"(a_32), "r"(_bf16x2_abs_65));
                a_32 = _bf16x2_max_65;
                uint32_t _bf16x2_abs_66;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_66) : "r"(c0_34[2]));
                uint32_t _bf16x2_max_66;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_66) : "r"(a_32), "r"(_bf16x2_abs_66));
                a_32 = _bf16x2_max_66;
                uint32_t _bf16x2_abs_67;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_67) : "r"(c0_34[3]));
                uint32_t _bf16x2_max_67;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_67) : "r"(a_32), "r"(_bf16x2_abs_67));
                a_32 = _bf16x2_max_67;
                uint32_t _bf16x2_abs_68;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_68) : "r"(c1_35[0]));
                uint32_t _bf16x2_max_68;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_68) : "r"(a_32), "r"(_bf16x2_abs_68));
                a_32 = _bf16x2_max_68;
                uint32_t _bf16x2_abs_69;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_69) : "r"(c1_35[1]));
                uint32_t _bf16x2_max_69;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_69) : "r"(a_32), "r"(_bf16x2_abs_69));
                a_32 = _bf16x2_max_69;
                uint32_t _bf16x2_abs_70;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_70) : "r"(c1_35[2]));
                uint32_t _bf16x2_max_70;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_70) : "r"(a_32), "r"(_bf16x2_abs_70));
                a_32 = _bf16x2_max_70;
                uint32_t _bf16x2_abs_71;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_71) : "r"(c1_35[3]));
                uint32_t _bf16x2_max_71;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_71) : "r"(a_32), "r"(_bf16x2_abs_71));
                a_32 = _bf16x2_max_71;
                uint32_t _bf16x2_abs_72;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_72) : "r"(c2_36[0]));
                uint32_t _bf16x2_max_72;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_72) : "r"(b_33), "r"(_bf16x2_abs_72));
                b_33 = _bf16x2_max_72;
                uint32_t _bf16x2_abs_73;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_73) : "r"(c2_36[1]));
                uint32_t _bf16x2_max_73;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_73) : "r"(b_33), "r"(_bf16x2_abs_73));
                b_33 = _bf16x2_max_73;
                uint32_t _bf16x2_abs_74;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_74) : "r"(c2_36[2]));
                uint32_t _bf16x2_max_74;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_74) : "r"(b_33), "r"(_bf16x2_abs_74));
                b_33 = _bf16x2_max_74;
                uint32_t _bf16x2_abs_75;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_75) : "r"(c2_36[3]));
                uint32_t _bf16x2_max_75;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_75) : "r"(b_33), "r"(_bf16x2_abs_75));
                b_33 = _bf16x2_max_75;
                uint32_t _bf16x2_abs_76;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_76) : "r"(c3_37[0]));
                uint32_t _bf16x2_max_76;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_76) : "r"(b_33), "r"(_bf16x2_abs_76));
                b_33 = _bf16x2_max_76;
                uint32_t _bf16x2_abs_77;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_77) : "r"(c3_37[1]));
                uint32_t _bf16x2_max_77;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_77) : "r"(b_33), "r"(_bf16x2_abs_77));
                b_33 = _bf16x2_max_77;
                uint32_t _bf16x2_abs_78;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_78) : "r"(c3_37[2]));
                uint32_t _bf16x2_max_78;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_78) : "r"(b_33), "r"(_bf16x2_abs_78));
                b_33 = _bf16x2_max_78;
                uint32_t _bf16x2_abs_79;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_79) : "r"(c3_37[3]));
                uint32_t _bf16x2_max_79;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_79) : "r"(b_33), "r"(_bf16x2_abs_79));
                b_33 = _bf16x2_max_79;
                unsigned int _max_12 = ((a_32 & 65535) > (a_32 >> 16) ? (a_32 & 65535) : (a_32 >> 16));
                unsigned int xa_38 = _max_12;
                unsigned int _max_13 = ((b_33 & 65535) > (b_33 >> 16) ? (b_33 & 65535) : (b_33 >> 16));
                unsigned int xb_39 = _max_13;
                bmp[4] = (((f & 2) != 0) ? xb_39 | xa_38 << 16 : xa_38 | xb_39 << 16);
                unsigned int _max_14 = ((xa_38) > (xb_39) ? (xa_38) : (xb_39));
                float _fmax_4 = fmaxf(am, __uint_as_float(_max_14 << 16));
                am = _fmax_4;
                int pp_40 = lane_1 + 160;
                int base_41 = xb_off + 64 * pp_40;
                unsigned int a_42 = 0;
                unsigned int b_43 = 0;
                unsigned int c0_44[4];
                unsigned int c1_45[4];
                unsigned int c2_46[4];
                unsigned int c3_47[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c0_44[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_44[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_44[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_44[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_41 + (unsigned int)(16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1_45[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_45[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_45[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_45[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_41 + (unsigned int)(16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2_46[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_46[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_46[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_46[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_41 + (unsigned int)(16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3_47[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_47[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_47[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_47[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base_41 + (unsigned int)(16 * (f ^ 3))));
                uint32_t _bf16x2_abs_80;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_80) : "r"(c0_44[0]));
                uint32_t _bf16x2_max_80;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_80) : "r"(a_42), "r"(_bf16x2_abs_80));
                a_42 = _bf16x2_max_80;
                uint32_t _bf16x2_abs_81;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_81) : "r"(c0_44[1]));
                uint32_t _bf16x2_max_81;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_81) : "r"(a_42), "r"(_bf16x2_abs_81));
                a_42 = _bf16x2_max_81;
                uint32_t _bf16x2_abs_82;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_82) : "r"(c0_44[2]));
                uint32_t _bf16x2_max_82;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_82) : "r"(a_42), "r"(_bf16x2_abs_82));
                a_42 = _bf16x2_max_82;
                uint32_t _bf16x2_abs_83;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_83) : "r"(c0_44[3]));
                uint32_t _bf16x2_max_83;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_83) : "r"(a_42), "r"(_bf16x2_abs_83));
                a_42 = _bf16x2_max_83;
                uint32_t _bf16x2_abs_84;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_84) : "r"(c1_45[0]));
                uint32_t _bf16x2_max_84;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_84) : "r"(a_42), "r"(_bf16x2_abs_84));
                a_42 = _bf16x2_max_84;
                uint32_t _bf16x2_abs_85;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_85) : "r"(c1_45[1]));
                uint32_t _bf16x2_max_85;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_85) : "r"(a_42), "r"(_bf16x2_abs_85));
                a_42 = _bf16x2_max_85;
                uint32_t _bf16x2_abs_86;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_86) : "r"(c1_45[2]));
                uint32_t _bf16x2_max_86;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_86) : "r"(a_42), "r"(_bf16x2_abs_86));
                a_42 = _bf16x2_max_86;
                uint32_t _bf16x2_abs_87;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_87) : "r"(c1_45[3]));
                uint32_t _bf16x2_max_87;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_87) : "r"(a_42), "r"(_bf16x2_abs_87));
                a_42 = _bf16x2_max_87;
                uint32_t _bf16x2_abs_88;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_88) : "r"(c2_46[0]));
                uint32_t _bf16x2_max_88;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_88) : "r"(b_43), "r"(_bf16x2_abs_88));
                b_43 = _bf16x2_max_88;
                uint32_t _bf16x2_abs_89;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_89) : "r"(c2_46[1]));
                uint32_t _bf16x2_max_89;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_89) : "r"(b_43), "r"(_bf16x2_abs_89));
                b_43 = _bf16x2_max_89;
                uint32_t _bf16x2_abs_90;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_90) : "r"(c2_46[2]));
                uint32_t _bf16x2_max_90;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_90) : "r"(b_43), "r"(_bf16x2_abs_90));
                b_43 = _bf16x2_max_90;
                uint32_t _bf16x2_abs_91;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_91) : "r"(c2_46[3]));
                uint32_t _bf16x2_max_91;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_91) : "r"(b_43), "r"(_bf16x2_abs_91));
                b_43 = _bf16x2_max_91;
                uint32_t _bf16x2_abs_92;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_92) : "r"(c3_47[0]));
                uint32_t _bf16x2_max_92;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_92) : "r"(b_43), "r"(_bf16x2_abs_92));
                b_43 = _bf16x2_max_92;
                uint32_t _bf16x2_abs_93;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_93) : "r"(c3_47[1]));
                uint32_t _bf16x2_max_93;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_93) : "r"(b_43), "r"(_bf16x2_abs_93));
                b_43 = _bf16x2_max_93;
                uint32_t _bf16x2_abs_94;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_94) : "r"(c3_47[2]));
                uint32_t _bf16x2_max_94;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_94) : "r"(b_43), "r"(_bf16x2_abs_94));
                b_43 = _bf16x2_max_94;
                uint32_t _bf16x2_abs_95;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_95) : "r"(c3_47[3]));
                uint32_t _bf16x2_max_95;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_95) : "r"(b_43), "r"(_bf16x2_abs_95));
                b_43 = _bf16x2_max_95;
                unsigned int _max_15 = ((a_42 & 65535) > (a_42 >> 16) ? (a_42 & 65535) : (a_42 >> 16));
                unsigned int xa_48 = _max_15;
                unsigned int _max_16 = ((b_43 & 65535) > (b_43 >> 16) ? (b_43 & 65535) : (b_43 >> 16));
                unsigned int xb_49 = _max_16;
                bmp[5] = (((f & 2) != 0) ? xb_49 | xa_48 << 16 : xa_48 | xb_49 << 16);
                unsigned int _max_17 = ((xa_48) > (xb_49) ? (xa_48) : (xb_49));
                float _fmax_5 = fmaxf(am, __uint_as_float(_max_17 << 16));
                am = _fmax_5;
                float _warp_reduce_0 = am;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 = fmaxf(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                am = _warp_reduce_0;
                float zero = 0.0f;
                float target = 2688.0f;
                float _fdiv_rn_0 = __fdiv_rn(target, am);
                float inv = ((am > zero) ? _fdiv_rn_0 : zero);
                float _fdiv_rn_1 = __fdiv_rn(am, target);
                float rs_50 = _fdiv_rn_1;
                float zero_51 = 0.0f;
                float six = 6.0f;
                int pp2 = lane_1;
                int base2 = xb_off + 64 * pp2;
                float bm0 = __uint_as_float(bmp[0] << 16);
                float bm1 = __uint_as_float(bmp[0] & 4294901760u);
                float _mul_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(bm0), "f"(inv));
                float _fdiv_rn_2 = __fdiv_rn(_mul_0, six);
                float _mul_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(bm1), "f"(inv));
                float _fdiv_rn_3 = __fdiv_rn(_mul_1, six);
                uint16_t _e4m3x2_f32_0;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_fdiv_rn_3), "f"(_fdiv_rn_2));
                float2 _fp8x2_decode_0;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_0.x), "=f"(_fp8x2_decode_0.y) : "h"(_e4m3x2_f32_0));
                float _fdiv_rn_4 = __fdiv_rn(inv, _fp8x2_decode_0.x);
                float _fdiv_rn_5 = __fdiv_rn(inv, _fp8x2_decode_0.y);
                float m0 = ((zero_51 < _fp8x2_decode_0.x) ? _fdiv_rn_4 : zero_51);
                float m1 = ((zero_51 < _fp8x2_decode_0.y) ? _fdiv_rn_5 : zero_51);
                unsigned int w[4];
                int cf = f ^ 0;
                unsigned int v4[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2 + (unsigned int)(16 * cf)));
                float m = (((cf & 2) != 0) ? m1 : m0);
                float2 _f2_0 = make_float2(m, m);
                float2 _f2_1 = make_float2(__uint_as_float(v4[0] << 16), __uint_as_float(v4[0] & 4294901760u));
                float2 _mul_f32x2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_1), "l"(*(const unsigned long long*)&_f2_0));
                uint32_t _fp4_pair_0;
                uint16_t _e2m1_pair_h_0;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_0) : "f"(_mul_f32x2_0.x), "f"(_mul_f32x2_0.y));
                _fp4_pair_0 = (uint32_t)_e2m1_pair_h_0;
                float2 _f2_2 = make_float2(__uint_as_float(v4[1] << 16), __uint_as_float(v4[1] & 4294901760u));
                float2 _mul_f32x2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_f2_2), "l"(*(const unsigned long long*)&_f2_0));
                uint32_t _fp4_pair_1;
                uint16_t _e2m1_pair_h_1;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_1) : "f"(_mul_f32x2_1.x), "f"(_mul_f32x2_1.y));
                _fp4_pair_1 = (uint32_t)_e2m1_pair_h_1;
                float2 _f2_3 = make_float2(__uint_as_float(v4[2] << 16), __uint_as_float(v4[2] & 4294901760u));
                float2 _mul_f32x2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_3), "l"(*(const unsigned long long*)&_f2_0));
                uint32_t _fp4_pair_2;
                uint16_t _e2m1_pair_h_2;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_2) : "f"(_mul_f32x2_2.x), "f"(_mul_f32x2_2.y));
                _fp4_pair_2 = (uint32_t)_e2m1_pair_h_2;
                float2 _f2_4 = make_float2(__uint_as_float(v4[3] << 16), __uint_as_float(v4[3] & 4294901760u));
                float2 _mul_f32x2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_f2_4), "l"(*(const unsigned long long*)&_f2_0));
                uint32_t _fp4_pair_3;
                uint16_t _e2m1_pair_h_3;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_3) : "f"(_mul_f32x2_3.x), "f"(_mul_f32x2_3.y));
                _fp4_pair_3 = (uint32_t)_e2m1_pair_h_3;
                unsigned int word = _fp4_pair_0 | _fp4_pair_1 << 8 | _fp4_pair_2 << 16 | _fp4_pair_3 << 24;
                uint32_t _e2m1_canon_0;
                {
                    const uint32_t _canon_w_4 = word;
                    const uint32_t _canon_s_4 = (_canon_w_4 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_0 = _canon_w_4 & (_canon_s_4 | 0x77777777u);
                }
                w[0] = _e2m1_canon_0;
                int cf_52 = f ^ 1;
                unsigned int v4_53[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_53[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_53[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_53[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_53[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2 + (unsigned int)(16 * cf_52)));
                float m_54 = (((cf_52 & 2) != 0) ? m1 : m0);
                float2 _f2_5 = make_float2(m_54, m_54);
                float2 _f2_6 = make_float2(__uint_as_float(v4_53[0] << 16), __uint_as_float(v4_53[0] & 4294901760u));
                float2 _mul_f32x2_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_6), "l"(*(const unsigned long long*)&_f2_5));
                uint32_t _fp4_pair_4;
                uint16_t _e2m1_pair_h_5;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_5) : "f"(_mul_f32x2_4.x), "f"(_mul_f32x2_4.y));
                _fp4_pair_4 = (uint32_t)_e2m1_pair_h_5;
                float2 _f2_7 = make_float2(__uint_as_float(v4_53[1] << 16), __uint_as_float(v4_53[1] & 4294901760u));
                float2 _mul_f32x2_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_f2_7), "l"(*(const unsigned long long*)&_f2_5));
                uint32_t _fp4_pair_5;
                uint16_t _e2m1_pair_h_6;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_6) : "f"(_mul_f32x2_5.x), "f"(_mul_f32x2_5.y));
                _fp4_pair_5 = (uint32_t)_e2m1_pair_h_6;
                float2 _f2_8 = make_float2(__uint_as_float(v4_53[2] << 16), __uint_as_float(v4_53[2] & 4294901760u));
                float2 _mul_f32x2_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&_f2_8), "l"(*(const unsigned long long*)&_f2_5));
                uint32_t _fp4_pair_6;
                uint16_t _e2m1_pair_h_7;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_7) : "f"(_mul_f32x2_6.x), "f"(_mul_f32x2_6.y));
                _fp4_pair_6 = (uint32_t)_e2m1_pair_h_7;
                float2 _f2_9 = make_float2(__uint_as_float(v4_53[3] << 16), __uint_as_float(v4_53[3] & 4294901760u));
                float2 _mul_f32x2_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&_f2_9), "l"(*(const unsigned long long*)&_f2_5));
                uint32_t _fp4_pair_7;
                uint16_t _e2m1_pair_h_8;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_8) : "f"(_mul_f32x2_7.x), "f"(_mul_f32x2_7.y));
                _fp4_pair_7 = (uint32_t)_e2m1_pair_h_8;
                unsigned int word_55 = _fp4_pair_4 | _fp4_pair_5 << 8 | _fp4_pair_6 << 16 | _fp4_pair_7 << 24;
                uint32_t _e2m1_canon_1;
                {
                    const uint32_t _canon_w_9 = word_55;
                    const uint32_t _canon_s_9 = (_canon_w_9 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_1 = _canon_w_9 & (_canon_s_9 | 0x77777777u);
                }
                w[1] = _e2m1_canon_1;
                int cf_56 = f ^ 2;
                unsigned int v4_57[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_57[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_57[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_57[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_57[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2 + (unsigned int)(16 * cf_56)));
                float m_58 = (((cf_56 & 2) != 0) ? m1 : m0);
                float2 _f2_10 = make_float2(m_58, m_58);
                float2 _f2_11 = make_float2(__uint_as_float(v4_57[0] << 16), __uint_as_float(v4_57[0] & 4294901760u));
                float2 _mul_f32x2_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_8) : "l"(*(const unsigned long long*)&_f2_11), "l"(*(const unsigned long long*)&_f2_10));
                uint32_t _fp4_pair_8;
                uint16_t _e2m1_pair_h_10;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_10) : "f"(_mul_f32x2_8.x), "f"(_mul_f32x2_8.y));
                _fp4_pair_8 = (uint32_t)_e2m1_pair_h_10;
                float2 _f2_12 = make_float2(__uint_as_float(v4_57[1] << 16), __uint_as_float(v4_57[1] & 4294901760u));
                float2 _mul_f32x2_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&_f2_12), "l"(*(const unsigned long long*)&_f2_10));
                uint32_t _fp4_pair_9;
                uint16_t _e2m1_pair_h_11;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_11) : "f"(_mul_f32x2_9.x), "f"(_mul_f32x2_9.y));
                _fp4_pair_9 = (uint32_t)_e2m1_pair_h_11;
                float2 _f2_13 = make_float2(__uint_as_float(v4_57[2] << 16), __uint_as_float(v4_57[2] & 4294901760u));
                float2 _mul_f32x2_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_10) : "l"(*(const unsigned long long*)&_f2_13), "l"(*(const unsigned long long*)&_f2_10));
                uint32_t _fp4_pair_10;
                uint16_t _e2m1_pair_h_12;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_12) : "f"(_mul_f32x2_10.x), "f"(_mul_f32x2_10.y));
                _fp4_pair_10 = (uint32_t)_e2m1_pair_h_12;
                float2 _f2_14 = make_float2(__uint_as_float(v4_57[3] << 16), __uint_as_float(v4_57[3] & 4294901760u));
                float2 _mul_f32x2_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_11) : "l"(*(const unsigned long long*)&_f2_14), "l"(*(const unsigned long long*)&_f2_10));
                uint32_t _fp4_pair_11;
                uint16_t _e2m1_pair_h_13;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_13) : "f"(_mul_f32x2_11.x), "f"(_mul_f32x2_11.y));
                _fp4_pair_11 = (uint32_t)_e2m1_pair_h_13;
                unsigned int word_59 = _fp4_pair_8 | _fp4_pair_9 << 8 | _fp4_pair_10 << 16 | _fp4_pair_11 << 24;
                uint32_t _e2m1_canon_2;
                {
                    const uint32_t _canon_w_14 = word_59;
                    const uint32_t _canon_s_14 = (_canon_w_14 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_2 = _canon_w_14 & (_canon_s_14 | 0x77777777u);
                }
                w[2] = _e2m1_canon_2;
                int cf_60 = f ^ 3;
                unsigned int v4_61[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_61[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_61[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_61[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_61[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2 + (unsigned int)(16 * cf_60)));
                float m_62 = (((cf_60 & 2) != 0) ? m1 : m0);
                float2 _f2_15 = make_float2(m_62, m_62);
                float2 _f2_16 = make_float2(__uint_as_float(v4_61[0] << 16), __uint_as_float(v4_61[0] & 4294901760u));
                float2 _mul_f32x2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_12) : "l"(*(const unsigned long long*)&_f2_16), "l"(*(const unsigned long long*)&_f2_15));
                uint32_t _fp4_pair_12;
                uint16_t _e2m1_pair_h_15;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_15) : "f"(_mul_f32x2_12.x), "f"(_mul_f32x2_12.y));
                _fp4_pair_12 = (uint32_t)_e2m1_pair_h_15;
                float2 _f2_17 = make_float2(__uint_as_float(v4_61[1] << 16), __uint_as_float(v4_61[1] & 4294901760u));
                float2 _mul_f32x2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_13) : "l"(*(const unsigned long long*)&_f2_17), "l"(*(const unsigned long long*)&_f2_15));
                uint32_t _fp4_pair_13;
                uint16_t _e2m1_pair_h_16;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_16) : "f"(_mul_f32x2_13.x), "f"(_mul_f32x2_13.y));
                _fp4_pair_13 = (uint32_t)_e2m1_pair_h_16;
                float2 _f2_18 = make_float2(__uint_as_float(v4_61[2] << 16), __uint_as_float(v4_61[2] & 4294901760u));
                float2 _mul_f32x2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&_f2_18), "l"(*(const unsigned long long*)&_f2_15));
                uint32_t _fp4_pair_14;
                uint16_t _e2m1_pair_h_17;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_17) : "f"(_mul_f32x2_14.x), "f"(_mul_f32x2_14.y));
                _fp4_pair_14 = (uint32_t)_e2m1_pair_h_17;
                float2 _f2_19 = make_float2(__uint_as_float(v4_61[3] << 16), __uint_as_float(v4_61[3] & 4294901760u));
                float2 _mul_f32x2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&_f2_19), "l"(*(const unsigned long long*)&_f2_15));
                uint32_t _fp4_pair_15;
                uint16_t _e2m1_pair_h_18;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_18) : "f"(_mul_f32x2_15.x), "f"(_mul_f32x2_15.y));
                _fp4_pair_15 = (uint32_t)_e2m1_pair_h_18;
                unsigned int word_63 = _fp4_pair_12 | _fp4_pair_13 << 8 | _fp4_pair_14 << 16 | _fp4_pair_15 << 24;
                uint32_t _e2m1_canon_3;
                {
                    const uint32_t _canon_w_19 = word_63;
                    const uint32_t _canon_s_19 = (_canon_w_19 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_3 = _canon_w_19 & (_canon_s_19 | 0x77777777u);
                }
                w[3] = _e2m1_canon_3;
                unsigned int t0 = (((f & 1) != 0) ? w[1] : w[0]);
                unsigned int t1 = (((f & 1) != 0) ? w[0] : w[1]);
                unsigned int t2 = (((f & 1) != 0) ? w[3] : w[2]);
                unsigned int t3 = (((f & 1) != 0) ? w[2] : w[3]);
                unsigned int q0 = (((f & 2) != 0) ? t2 : t0);
                unsigned int q1 = (((f & 2) != 0) ? t3 : t1);
                unsigned int q2 = (((f & 2) != 0) ? t0 : t2);
                unsigned int q3 = (((f & 2) != 0) ? t1 : t3);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2)), "r"(q0), "r"(q1), "r"(q2), "r"(q3) : "memory");
                wsm16[(ob_off + 3072 >> 1) + pp2] = _e4m3x2_f32_0;
                int pp2_64 = lane_1 + 32;
                int base2_65 = xb_off + 64 * pp2_64;
                float bm0_66 = __uint_as_float(bmp[1] << 16);
                float bm1_67 = __uint_as_float(bmp[1] & 4294901760u);
                float _mul_2;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(bm0_66), "f"(inv));
                float _fdiv_rn_6 = __fdiv_rn(_mul_2, six);
                float _mul_3;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(bm1_67), "f"(inv));
                float _fdiv_rn_7 = __fdiv_rn(_mul_3, six);
                uint16_t _e4m3x2_f32_1;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_fdiv_rn_7), "f"(_fdiv_rn_6));
                float2 _fp8x2_decode_1;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_1.x), "=f"(_fp8x2_decode_1.y) : "h"(_e4m3x2_f32_1));
                float _fdiv_rn_8 = __fdiv_rn(inv, _fp8x2_decode_1.x);
                float _fdiv_rn_9 = __fdiv_rn(inv, _fp8x2_decode_1.y);
                float m0_68 = ((zero_51 < _fp8x2_decode_1.x) ? _fdiv_rn_8 : zero_51);
                float m1_69 = ((zero_51 < _fp8x2_decode_1.y) ? _fdiv_rn_9 : zero_51);
                unsigned int w_70[4];
                int cf_71 = f ^ 0;
                unsigned int v4_72[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_72[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_72[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_72[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_72[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_65 + (unsigned int)(16 * cf_71)));
                float m_73 = (((cf_71 & 2) != 0) ? m1_69 : m0_68);
                float2 _f2_20 = make_float2(m_73, m_73);
                float2 _f2_21 = make_float2(__uint_as_float(v4_72[0] << 16), __uint_as_float(v4_72[0] & 4294901760u));
                float2 _mul_f32x2_16;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_16) : "l"(*(const unsigned long long*)&_f2_21), "l"(*(const unsigned long long*)&_f2_20));
                uint32_t _fp4_pair_16;
                uint16_t _e2m1_pair_h_20;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_20) : "f"(_mul_f32x2_16.x), "f"(_mul_f32x2_16.y));
                _fp4_pair_16 = (uint32_t)_e2m1_pair_h_20;
                float2 _f2_22 = make_float2(__uint_as_float(v4_72[1] << 16), __uint_as_float(v4_72[1] & 4294901760u));
                float2 _mul_f32x2_17;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_17) : "l"(*(const unsigned long long*)&_f2_22), "l"(*(const unsigned long long*)&_f2_20));
                uint32_t _fp4_pair_17;
                uint16_t _e2m1_pair_h_21;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_21) : "f"(_mul_f32x2_17.x), "f"(_mul_f32x2_17.y));
                _fp4_pair_17 = (uint32_t)_e2m1_pair_h_21;
                float2 _f2_23 = make_float2(__uint_as_float(v4_72[2] << 16), __uint_as_float(v4_72[2] & 4294901760u));
                float2 _mul_f32x2_18;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_18) : "l"(*(const unsigned long long*)&_f2_23), "l"(*(const unsigned long long*)&_f2_20));
                uint32_t _fp4_pair_18;
                uint16_t _e2m1_pair_h_22;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_22) : "f"(_mul_f32x2_18.x), "f"(_mul_f32x2_18.y));
                _fp4_pair_18 = (uint32_t)_e2m1_pair_h_22;
                float2 _f2_24 = make_float2(__uint_as_float(v4_72[3] << 16), __uint_as_float(v4_72[3] & 4294901760u));
                float2 _mul_f32x2_19;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_19) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_20));
                uint32_t _fp4_pair_19;
                uint16_t _e2m1_pair_h_23;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_23) : "f"(_mul_f32x2_19.x), "f"(_mul_f32x2_19.y));
                _fp4_pair_19 = (uint32_t)_e2m1_pair_h_23;
                unsigned int word_74 = _fp4_pair_16 | _fp4_pair_17 << 8 | _fp4_pair_18 << 16 | _fp4_pair_19 << 24;
                uint32_t _e2m1_canon_4;
                {
                    const uint32_t _canon_w_24 = word_74;
                    const uint32_t _canon_s_24 = (_canon_w_24 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_4 = _canon_w_24 & (_canon_s_24 | 0x77777777u);
                }
                w_70[0] = _e2m1_canon_4;
                int cf_75 = f ^ 1;
                unsigned int v4_76[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_76[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_76[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_76[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_76[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_65 + (unsigned int)(16 * cf_75)));
                float m_77 = (((cf_75 & 2) != 0) ? m1_69 : m0_68);
                float2 _f2_25 = make_float2(m_77, m_77);
                float2 _f2_26 = make_float2(__uint_as_float(v4_76[0] << 16), __uint_as_float(v4_76[0] & 4294901760u));
                float2 _mul_f32x2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_20) : "l"(*(const unsigned long long*)&_f2_26), "l"(*(const unsigned long long*)&_f2_25));
                uint32_t _fp4_pair_20;
                uint16_t _e2m1_pair_h_25;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_25) : "f"(_mul_f32x2_20.x), "f"(_mul_f32x2_20.y));
                _fp4_pair_20 = (uint32_t)_e2m1_pair_h_25;
                float2 _f2_27 = make_float2(__uint_as_float(v4_76[1] << 16), __uint_as_float(v4_76[1] & 4294901760u));
                float2 _mul_f32x2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_21) : "l"(*(const unsigned long long*)&_f2_27), "l"(*(const unsigned long long*)&_f2_25));
                uint32_t _fp4_pair_21;
                uint16_t _e2m1_pair_h_26;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_26) : "f"(_mul_f32x2_21.x), "f"(_mul_f32x2_21.y));
                _fp4_pair_21 = (uint32_t)_e2m1_pair_h_26;
                float2 _f2_28 = make_float2(__uint_as_float(v4_76[2] << 16), __uint_as_float(v4_76[2] & 4294901760u));
                float2 _mul_f32x2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_22) : "l"(*(const unsigned long long*)&_f2_28), "l"(*(const unsigned long long*)&_f2_25));
                uint32_t _fp4_pair_22;
                uint16_t _e2m1_pair_h_27;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_27) : "f"(_mul_f32x2_22.x), "f"(_mul_f32x2_22.y));
                _fp4_pair_22 = (uint32_t)_e2m1_pair_h_27;
                float2 _f2_29 = make_float2(__uint_as_float(v4_76[3] << 16), __uint_as_float(v4_76[3] & 4294901760u));
                float2 _mul_f32x2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_23) : "l"(*(const unsigned long long*)&_f2_29), "l"(*(const unsigned long long*)&_f2_25));
                uint32_t _fp4_pair_23;
                uint16_t _e2m1_pair_h_28;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_28) : "f"(_mul_f32x2_23.x), "f"(_mul_f32x2_23.y));
                _fp4_pair_23 = (uint32_t)_e2m1_pair_h_28;
                unsigned int word_78 = _fp4_pair_20 | _fp4_pair_21 << 8 | _fp4_pair_22 << 16 | _fp4_pair_23 << 24;
                uint32_t _e2m1_canon_5;
                {
                    const uint32_t _canon_w_29 = word_78;
                    const uint32_t _canon_s_29 = (_canon_w_29 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_5 = _canon_w_29 & (_canon_s_29 | 0x77777777u);
                }
                w_70[1] = _e2m1_canon_5;
                int cf_79 = f ^ 2;
                unsigned int v4_80[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_80[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_80[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_80[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_80[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_65 + (unsigned int)(16 * cf_79)));
                float m_81 = (((cf_79 & 2) != 0) ? m1_69 : m0_68);
                float2 _f2_30 = make_float2(m_81, m_81);
                float2 _f2_31 = make_float2(__uint_as_float(v4_80[0] << 16), __uint_as_float(v4_80[0] & 4294901760u));
                float2 _mul_f32x2_24;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_24) : "l"(*(const unsigned long long*)&_f2_31), "l"(*(const unsigned long long*)&_f2_30));
                uint32_t _fp4_pair_24;
                uint16_t _e2m1_pair_h_30;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_30) : "f"(_mul_f32x2_24.x), "f"(_mul_f32x2_24.y));
                _fp4_pair_24 = (uint32_t)_e2m1_pair_h_30;
                float2 _f2_32 = make_float2(__uint_as_float(v4_80[1] << 16), __uint_as_float(v4_80[1] & 4294901760u));
                float2 _mul_f32x2_25;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_25) : "l"(*(const unsigned long long*)&_f2_32), "l"(*(const unsigned long long*)&_f2_30));
                uint32_t _fp4_pair_25;
                uint16_t _e2m1_pair_h_31;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_31) : "f"(_mul_f32x2_25.x), "f"(_mul_f32x2_25.y));
                _fp4_pair_25 = (uint32_t)_e2m1_pair_h_31;
                float2 _f2_33 = make_float2(__uint_as_float(v4_80[2] << 16), __uint_as_float(v4_80[2] & 4294901760u));
                float2 _mul_f32x2_26;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_26) : "l"(*(const unsigned long long*)&_f2_33), "l"(*(const unsigned long long*)&_f2_30));
                uint32_t _fp4_pair_26;
                uint16_t _e2m1_pair_h_32;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_32) : "f"(_mul_f32x2_26.x), "f"(_mul_f32x2_26.y));
                _fp4_pair_26 = (uint32_t)_e2m1_pair_h_32;
                float2 _f2_34 = make_float2(__uint_as_float(v4_80[3] << 16), __uint_as_float(v4_80[3] & 4294901760u));
                float2 _mul_f32x2_27;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_27) : "l"(*(const unsigned long long*)&_f2_34), "l"(*(const unsigned long long*)&_f2_30));
                uint32_t _fp4_pair_27;
                uint16_t _e2m1_pair_h_33;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_33) : "f"(_mul_f32x2_27.x), "f"(_mul_f32x2_27.y));
                _fp4_pair_27 = (uint32_t)_e2m1_pair_h_33;
                unsigned int word_82 = _fp4_pair_24 | _fp4_pair_25 << 8 | _fp4_pair_26 << 16 | _fp4_pair_27 << 24;
                uint32_t _e2m1_canon_6;
                {
                    const uint32_t _canon_w_34 = word_82;
                    const uint32_t _canon_s_34 = (_canon_w_34 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_6 = _canon_w_34 & (_canon_s_34 | 0x77777777u);
                }
                w_70[2] = _e2m1_canon_6;
                int cf_83 = f ^ 3;
                unsigned int v4_84[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_84[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_84[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_84[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_84[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_65 + (unsigned int)(16 * cf_83)));
                float m_85 = (((cf_83 & 2) != 0) ? m1_69 : m0_68);
                float2 _f2_35 = make_float2(m_85, m_85);
                float2 _f2_36 = make_float2(__uint_as_float(v4_84[0] << 16), __uint_as_float(v4_84[0] & 4294901760u));
                float2 _mul_f32x2_28;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_28) : "l"(*(const unsigned long long*)&_f2_36), "l"(*(const unsigned long long*)&_f2_35));
                uint32_t _fp4_pair_28;
                uint16_t _e2m1_pair_h_35;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_35) : "f"(_mul_f32x2_28.x), "f"(_mul_f32x2_28.y));
                _fp4_pair_28 = (uint32_t)_e2m1_pair_h_35;
                float2 _f2_37 = make_float2(__uint_as_float(v4_84[1] << 16), __uint_as_float(v4_84[1] & 4294901760u));
                float2 _mul_f32x2_29;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_29) : "l"(*(const unsigned long long*)&_f2_37), "l"(*(const unsigned long long*)&_f2_35));
                uint32_t _fp4_pair_29;
                uint16_t _e2m1_pair_h_36;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_36) : "f"(_mul_f32x2_29.x), "f"(_mul_f32x2_29.y));
                _fp4_pair_29 = (uint32_t)_e2m1_pair_h_36;
                float2 _f2_38 = make_float2(__uint_as_float(v4_84[2] << 16), __uint_as_float(v4_84[2] & 4294901760u));
                float2 _mul_f32x2_30;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_30) : "l"(*(const unsigned long long*)&_f2_38), "l"(*(const unsigned long long*)&_f2_35));
                uint32_t _fp4_pair_30;
                uint16_t _e2m1_pair_h_37;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_37) : "f"(_mul_f32x2_30.x), "f"(_mul_f32x2_30.y));
                _fp4_pair_30 = (uint32_t)_e2m1_pair_h_37;
                float2 _f2_39 = make_float2(__uint_as_float(v4_84[3] << 16), __uint_as_float(v4_84[3] & 4294901760u));
                float2 _mul_f32x2_31;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_31) : "l"(*(const unsigned long long*)&_f2_39), "l"(*(const unsigned long long*)&_f2_35));
                uint32_t _fp4_pair_31;
                uint16_t _e2m1_pair_h_38;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_38) : "f"(_mul_f32x2_31.x), "f"(_mul_f32x2_31.y));
                _fp4_pair_31 = (uint32_t)_e2m1_pair_h_38;
                unsigned int word_86 = _fp4_pair_28 | _fp4_pair_29 << 8 | _fp4_pair_30 << 16 | _fp4_pair_31 << 24;
                uint32_t _e2m1_canon_7;
                {
                    const uint32_t _canon_w_39 = word_86;
                    const uint32_t _canon_s_39 = (_canon_w_39 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_7 = _canon_w_39 & (_canon_s_39 | 0x77777777u);
                }
                w_70[3] = _e2m1_canon_7;
                unsigned int t0_87 = (((f & 1) != 0) ? w_70[1] : w_70[0]);
                unsigned int t1_88 = (((f & 1) != 0) ? w_70[0] : w_70[1]);
                unsigned int t2_89 = (((f & 1) != 0) ? w_70[3] : w_70[2]);
                unsigned int t3_90 = (((f & 1) != 0) ? w_70[2] : w_70[3]);
                unsigned int q0_91 = (((f & 2) != 0) ? t2_89 : t0_87);
                unsigned int q1_92 = (((f & 2) != 0) ? t3_90 : t1_88);
                unsigned int q2_93 = (((f & 2) != 0) ? t0_87 : t2_89);
                unsigned int q3_94 = (((f & 2) != 0) ? t1_88 : t3_90);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_64)), "r"(q0_91), "r"(q1_92), "r"(q2_93), "r"(q3_94) : "memory");
                wsm16[(ob_off + 3072 >> 1) + pp2_64] = _e4m3x2_f32_1;
                int pp2_95 = lane_1 + 64;
                int base2_96 = xb_off + 64 * pp2_95;
                float bm0_97 = __uint_as_float(bmp[2] << 16);
                float bm1_98 = __uint_as_float(bmp[2] & 4294901760u);
                float _mul_4;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(bm0_97), "f"(inv));
                float _fdiv_rn_10 = __fdiv_rn(_mul_4, six);
                float _mul_5;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(bm1_98), "f"(inv));
                float _fdiv_rn_11 = __fdiv_rn(_mul_5, six);
                uint16_t _e4m3x2_f32_2;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(_fdiv_rn_11), "f"(_fdiv_rn_10));
                float2 _fp8x2_decode_2;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_2.x), "=f"(_fp8x2_decode_2.y) : "h"(_e4m3x2_f32_2));
                float _fdiv_rn_12 = __fdiv_rn(inv, _fp8x2_decode_2.x);
                float _fdiv_rn_13 = __fdiv_rn(inv, _fp8x2_decode_2.y);
                float m0_99 = ((zero_51 < _fp8x2_decode_2.x) ? _fdiv_rn_12 : zero_51);
                float m1_100 = ((zero_51 < _fp8x2_decode_2.y) ? _fdiv_rn_13 : zero_51);
                unsigned int w_101[4];
                int cf_102 = f ^ 0;
                unsigned int v4_103[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_103[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_103[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_103[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_103[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_96 + (unsigned int)(16 * cf_102)));
                float m_104 = (((cf_102 & 2) != 0) ? m1_100 : m0_99);
                float2 _f2_40 = make_float2(m_104, m_104);
                float2 _f2_41 = make_float2(__uint_as_float(v4_103[0] << 16), __uint_as_float(v4_103[0] & 4294901760u));
                float2 _mul_f32x2_32;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_32) : "l"(*(const unsigned long long*)&_f2_41), "l"(*(const unsigned long long*)&_f2_40));
                uint32_t _fp4_pair_32;
                uint16_t _e2m1_pair_h_40;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_40) : "f"(_mul_f32x2_32.x), "f"(_mul_f32x2_32.y));
                _fp4_pair_32 = (uint32_t)_e2m1_pair_h_40;
                float2 _f2_42 = make_float2(__uint_as_float(v4_103[1] << 16), __uint_as_float(v4_103[1] & 4294901760u));
                float2 _mul_f32x2_33;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_33) : "l"(*(const unsigned long long*)&_f2_42), "l"(*(const unsigned long long*)&_f2_40));
                uint32_t _fp4_pair_33;
                uint16_t _e2m1_pair_h_41;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_41) : "f"(_mul_f32x2_33.x), "f"(_mul_f32x2_33.y));
                _fp4_pair_33 = (uint32_t)_e2m1_pair_h_41;
                float2 _f2_43 = make_float2(__uint_as_float(v4_103[2] << 16), __uint_as_float(v4_103[2] & 4294901760u));
                float2 _mul_f32x2_34;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_34) : "l"(*(const unsigned long long*)&_f2_43), "l"(*(const unsigned long long*)&_f2_40));
                uint32_t _fp4_pair_34;
                uint16_t _e2m1_pair_h_42;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_42) : "f"(_mul_f32x2_34.x), "f"(_mul_f32x2_34.y));
                _fp4_pair_34 = (uint32_t)_e2m1_pair_h_42;
                float2 _f2_44 = make_float2(__uint_as_float(v4_103[3] << 16), __uint_as_float(v4_103[3] & 4294901760u));
                float2 _mul_f32x2_35;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_35) : "l"(*(const unsigned long long*)&_f2_44), "l"(*(const unsigned long long*)&_f2_40));
                uint32_t _fp4_pair_35;
                uint16_t _e2m1_pair_h_43;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_43) : "f"(_mul_f32x2_35.x), "f"(_mul_f32x2_35.y));
                _fp4_pair_35 = (uint32_t)_e2m1_pair_h_43;
                unsigned int word_105 = _fp4_pair_32 | _fp4_pair_33 << 8 | _fp4_pair_34 << 16 | _fp4_pair_35 << 24;
                uint32_t _e2m1_canon_8;
                {
                    const uint32_t _canon_w_44 = word_105;
                    const uint32_t _canon_s_44 = (_canon_w_44 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_8 = _canon_w_44 & (_canon_s_44 | 0x77777777u);
                }
                w_101[0] = _e2m1_canon_8;
                int cf_106 = f ^ 1;
                unsigned int v4_107[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_107[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_107[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_107[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_107[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_96 + (unsigned int)(16 * cf_106)));
                float m_108 = (((cf_106 & 2) != 0) ? m1_100 : m0_99);
                float2 _f2_45 = make_float2(m_108, m_108);
                float2 _f2_46 = make_float2(__uint_as_float(v4_107[0] << 16), __uint_as_float(v4_107[0] & 4294901760u));
                float2 _mul_f32x2_36;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_36) : "l"(*(const unsigned long long*)&_f2_46), "l"(*(const unsigned long long*)&_f2_45));
                uint32_t _fp4_pair_36;
                uint16_t _e2m1_pair_h_45;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_45) : "f"(_mul_f32x2_36.x), "f"(_mul_f32x2_36.y));
                _fp4_pair_36 = (uint32_t)_e2m1_pair_h_45;
                float2 _f2_47 = make_float2(__uint_as_float(v4_107[1] << 16), __uint_as_float(v4_107[1] & 4294901760u));
                float2 _mul_f32x2_37;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_37) : "l"(*(const unsigned long long*)&_f2_47), "l"(*(const unsigned long long*)&_f2_45));
                uint32_t _fp4_pair_37;
                uint16_t _e2m1_pair_h_46;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_46) : "f"(_mul_f32x2_37.x), "f"(_mul_f32x2_37.y));
                _fp4_pair_37 = (uint32_t)_e2m1_pair_h_46;
                float2 _f2_48 = make_float2(__uint_as_float(v4_107[2] << 16), __uint_as_float(v4_107[2] & 4294901760u));
                float2 _mul_f32x2_38;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_38) : "l"(*(const unsigned long long*)&_f2_48), "l"(*(const unsigned long long*)&_f2_45));
                uint32_t _fp4_pair_38;
                uint16_t _e2m1_pair_h_47;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_47) : "f"(_mul_f32x2_38.x), "f"(_mul_f32x2_38.y));
                _fp4_pair_38 = (uint32_t)_e2m1_pair_h_47;
                float2 _f2_49 = make_float2(__uint_as_float(v4_107[3] << 16), __uint_as_float(v4_107[3] & 4294901760u));
                float2 _mul_f32x2_39;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_39) : "l"(*(const unsigned long long*)&_f2_49), "l"(*(const unsigned long long*)&_f2_45));
                uint32_t _fp4_pair_39;
                uint16_t _e2m1_pair_h_48;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_48) : "f"(_mul_f32x2_39.x), "f"(_mul_f32x2_39.y));
                _fp4_pair_39 = (uint32_t)_e2m1_pair_h_48;
                unsigned int word_109 = _fp4_pair_36 | _fp4_pair_37 << 8 | _fp4_pair_38 << 16 | _fp4_pair_39 << 24;
                uint32_t _e2m1_canon_9;
                {
                    const uint32_t _canon_w_49 = word_109;
                    const uint32_t _canon_s_49 = (_canon_w_49 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_9 = _canon_w_49 & (_canon_s_49 | 0x77777777u);
                }
                w_101[1] = _e2m1_canon_9;
                int cf_110 = f ^ 2;
                unsigned int v4_111[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_111[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_111[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_111[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_111[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_96 + (unsigned int)(16 * cf_110)));
                float m_112 = (((cf_110 & 2) != 0) ? m1_100 : m0_99);
                float2 _f2_50 = make_float2(m_112, m_112);
                float2 _f2_51 = make_float2(__uint_as_float(v4_111[0] << 16), __uint_as_float(v4_111[0] & 4294901760u));
                float2 _mul_f32x2_40;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_40) : "l"(*(const unsigned long long*)&_f2_51), "l"(*(const unsigned long long*)&_f2_50));
                uint32_t _fp4_pair_40;
                uint16_t _e2m1_pair_h_50;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_50) : "f"(_mul_f32x2_40.x), "f"(_mul_f32x2_40.y));
                _fp4_pair_40 = (uint32_t)_e2m1_pair_h_50;
                float2 _f2_52 = make_float2(__uint_as_float(v4_111[1] << 16), __uint_as_float(v4_111[1] & 4294901760u));
                float2 _mul_f32x2_41;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_41) : "l"(*(const unsigned long long*)&_f2_52), "l"(*(const unsigned long long*)&_f2_50));
                uint32_t _fp4_pair_41;
                uint16_t _e2m1_pair_h_51;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_51) : "f"(_mul_f32x2_41.x), "f"(_mul_f32x2_41.y));
                _fp4_pair_41 = (uint32_t)_e2m1_pair_h_51;
                float2 _f2_53 = make_float2(__uint_as_float(v4_111[2] << 16), __uint_as_float(v4_111[2] & 4294901760u));
                float2 _mul_f32x2_42;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_42) : "l"(*(const unsigned long long*)&_f2_53), "l"(*(const unsigned long long*)&_f2_50));
                uint32_t _fp4_pair_42;
                uint16_t _e2m1_pair_h_52;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_52) : "f"(_mul_f32x2_42.x), "f"(_mul_f32x2_42.y));
                _fp4_pair_42 = (uint32_t)_e2m1_pair_h_52;
                float2 _f2_54 = make_float2(__uint_as_float(v4_111[3] << 16), __uint_as_float(v4_111[3] & 4294901760u));
                float2 _mul_f32x2_43;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_43) : "l"(*(const unsigned long long*)&_f2_54), "l"(*(const unsigned long long*)&_f2_50));
                uint32_t _fp4_pair_43;
                uint16_t _e2m1_pair_h_53;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_53) : "f"(_mul_f32x2_43.x), "f"(_mul_f32x2_43.y));
                _fp4_pair_43 = (uint32_t)_e2m1_pair_h_53;
                unsigned int word_113 = _fp4_pair_40 | _fp4_pair_41 << 8 | _fp4_pair_42 << 16 | _fp4_pair_43 << 24;
                uint32_t _e2m1_canon_10;
                {
                    const uint32_t _canon_w_54 = word_113;
                    const uint32_t _canon_s_54 = (_canon_w_54 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_10 = _canon_w_54 & (_canon_s_54 | 0x77777777u);
                }
                w_101[2] = _e2m1_canon_10;
                int cf_114 = f ^ 3;
                unsigned int v4_115[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_115[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_115[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_115[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_115[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_96 + (unsigned int)(16 * cf_114)));
                float m_116 = (((cf_114 & 2) != 0) ? m1_100 : m0_99);
                float2 _f2_55 = make_float2(m_116, m_116);
                float2 _f2_56 = make_float2(__uint_as_float(v4_115[0] << 16), __uint_as_float(v4_115[0] & 4294901760u));
                float2 _mul_f32x2_44;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_44) : "l"(*(const unsigned long long*)&_f2_56), "l"(*(const unsigned long long*)&_f2_55));
                uint32_t _fp4_pair_44;
                uint16_t _e2m1_pair_h_55;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_55) : "f"(_mul_f32x2_44.x), "f"(_mul_f32x2_44.y));
                _fp4_pair_44 = (uint32_t)_e2m1_pair_h_55;
                float2 _f2_57 = make_float2(__uint_as_float(v4_115[1] << 16), __uint_as_float(v4_115[1] & 4294901760u));
                float2 _mul_f32x2_45;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_45) : "l"(*(const unsigned long long*)&_f2_57), "l"(*(const unsigned long long*)&_f2_55));
                uint32_t _fp4_pair_45;
                uint16_t _e2m1_pair_h_56;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_56) : "f"(_mul_f32x2_45.x), "f"(_mul_f32x2_45.y));
                _fp4_pair_45 = (uint32_t)_e2m1_pair_h_56;
                float2 _f2_58 = make_float2(__uint_as_float(v4_115[2] << 16), __uint_as_float(v4_115[2] & 4294901760u));
                float2 _mul_f32x2_46;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_46) : "l"(*(const unsigned long long*)&_f2_58), "l"(*(const unsigned long long*)&_f2_55));
                uint32_t _fp4_pair_46;
                uint16_t _e2m1_pair_h_57;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_57) : "f"(_mul_f32x2_46.x), "f"(_mul_f32x2_46.y));
                _fp4_pair_46 = (uint32_t)_e2m1_pair_h_57;
                float2 _f2_59 = make_float2(__uint_as_float(v4_115[3] << 16), __uint_as_float(v4_115[3] & 4294901760u));
                float2 _mul_f32x2_47;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_47) : "l"(*(const unsigned long long*)&_f2_59), "l"(*(const unsigned long long*)&_f2_55));
                uint32_t _fp4_pair_47;
                uint16_t _e2m1_pair_h_58;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_58) : "f"(_mul_f32x2_47.x), "f"(_mul_f32x2_47.y));
                _fp4_pair_47 = (uint32_t)_e2m1_pair_h_58;
                unsigned int word_117 = _fp4_pair_44 | _fp4_pair_45 << 8 | _fp4_pair_46 << 16 | _fp4_pair_47 << 24;
                uint32_t _e2m1_canon_11;
                {
                    const uint32_t _canon_w_59 = word_117;
                    const uint32_t _canon_s_59 = (_canon_w_59 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_11 = _canon_w_59 & (_canon_s_59 | 0x77777777u);
                }
                w_101[3] = _e2m1_canon_11;
                unsigned int t0_118 = (((f & 1) != 0) ? w_101[1] : w_101[0]);
                unsigned int t1_119 = (((f & 1) != 0) ? w_101[0] : w_101[1]);
                unsigned int t2_120 = (((f & 1) != 0) ? w_101[3] : w_101[2]);
                unsigned int t3_121 = (((f & 1) != 0) ? w_101[2] : w_101[3]);
                unsigned int q0_122 = (((f & 2) != 0) ? t2_120 : t0_118);
                unsigned int q1_123 = (((f & 2) != 0) ? t3_121 : t1_119);
                unsigned int q2_124 = (((f & 2) != 0) ? t0_118 : t2_120);
                unsigned int q3_125 = (((f & 2) != 0) ? t1_119 : t3_121);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_95)), "r"(q0_122), "r"(q1_123), "r"(q2_124), "r"(q3_125) : "memory");
                wsm16[(ob_off + 3072 >> 1) + pp2_95] = _e4m3x2_f32_2;
                int pp2_126 = lane_1 + 96;
                int base2_127 = xb_off + 64 * pp2_126;
                float bm0_128 = __uint_as_float(bmp[3] << 16);
                float bm1_129 = __uint_as_float(bmp[3] & 4294901760u);
                float _mul_6;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(bm0_128), "f"(inv));
                float _fdiv_rn_14 = __fdiv_rn(_mul_6, six);
                float _mul_7;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(bm1_129), "f"(inv));
                float _fdiv_rn_15 = __fdiv_rn(_mul_7, six);
                uint16_t _e4m3x2_f32_3;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(_fdiv_rn_15), "f"(_fdiv_rn_14));
                float2 _fp8x2_decode_3;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_3.x), "=f"(_fp8x2_decode_3.y) : "h"(_e4m3x2_f32_3));
                float _fdiv_rn_16 = __fdiv_rn(inv, _fp8x2_decode_3.x);
                float _fdiv_rn_17 = __fdiv_rn(inv, _fp8x2_decode_3.y);
                float m0_130 = ((zero_51 < _fp8x2_decode_3.x) ? _fdiv_rn_16 : zero_51);
                float m1_131 = ((zero_51 < _fp8x2_decode_3.y) ? _fdiv_rn_17 : zero_51);
                unsigned int w_132[4];
                int cf_133 = f ^ 0;
                unsigned int v4_134[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_134[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_134[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_134[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_134[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_127 + (unsigned int)(16 * cf_133)));
                float m_135 = (((cf_133 & 2) != 0) ? m1_131 : m0_130);
                float2 _f2_60 = make_float2(m_135, m_135);
                float2 _f2_61 = make_float2(__uint_as_float(v4_134[0] << 16), __uint_as_float(v4_134[0] & 4294901760u));
                float2 _mul_f32x2_48;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_48) : "l"(*(const unsigned long long*)&_f2_61), "l"(*(const unsigned long long*)&_f2_60));
                uint32_t _fp4_pair_48;
                uint16_t _e2m1_pair_h_60;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_60) : "f"(_mul_f32x2_48.x), "f"(_mul_f32x2_48.y));
                _fp4_pair_48 = (uint32_t)_e2m1_pair_h_60;
                float2 _f2_62 = make_float2(__uint_as_float(v4_134[1] << 16), __uint_as_float(v4_134[1] & 4294901760u));
                float2 _mul_f32x2_49;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_49) : "l"(*(const unsigned long long*)&_f2_62), "l"(*(const unsigned long long*)&_f2_60));
                uint32_t _fp4_pair_49;
                uint16_t _e2m1_pair_h_61;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_61) : "f"(_mul_f32x2_49.x), "f"(_mul_f32x2_49.y));
                _fp4_pair_49 = (uint32_t)_e2m1_pair_h_61;
                float2 _f2_63 = make_float2(__uint_as_float(v4_134[2] << 16), __uint_as_float(v4_134[2] & 4294901760u));
                float2 _mul_f32x2_50;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_50) : "l"(*(const unsigned long long*)&_f2_63), "l"(*(const unsigned long long*)&_f2_60));
                uint32_t _fp4_pair_50;
                uint16_t _e2m1_pair_h_62;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_62) : "f"(_mul_f32x2_50.x), "f"(_mul_f32x2_50.y));
                _fp4_pair_50 = (uint32_t)_e2m1_pair_h_62;
                float2 _f2_64 = make_float2(__uint_as_float(v4_134[3] << 16), __uint_as_float(v4_134[3] & 4294901760u));
                float2 _mul_f32x2_51;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_51) : "l"(*(const unsigned long long*)&_f2_64), "l"(*(const unsigned long long*)&_f2_60));
                uint32_t _fp4_pair_51;
                uint16_t _e2m1_pair_h_63;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_63) : "f"(_mul_f32x2_51.x), "f"(_mul_f32x2_51.y));
                _fp4_pair_51 = (uint32_t)_e2m1_pair_h_63;
                unsigned int word_136 = _fp4_pair_48 | _fp4_pair_49 << 8 | _fp4_pair_50 << 16 | _fp4_pair_51 << 24;
                uint32_t _e2m1_canon_12;
                {
                    const uint32_t _canon_w_64 = word_136;
                    const uint32_t _canon_s_64 = (_canon_w_64 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_12 = _canon_w_64 & (_canon_s_64 | 0x77777777u);
                }
                w_132[0] = _e2m1_canon_12;
                int cf_137 = f ^ 1;
                unsigned int v4_138[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_138[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_138[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_138[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_138[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_127 + (unsigned int)(16 * cf_137)));
                float m_139 = (((cf_137 & 2) != 0) ? m1_131 : m0_130);
                float2 _f2_65 = make_float2(m_139, m_139);
                float2 _f2_66 = make_float2(__uint_as_float(v4_138[0] << 16), __uint_as_float(v4_138[0] & 4294901760u));
                float2 _mul_f32x2_52;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_52) : "l"(*(const unsigned long long*)&_f2_66), "l"(*(const unsigned long long*)&_f2_65));
                uint32_t _fp4_pair_52;
                uint16_t _e2m1_pair_h_65;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_65) : "f"(_mul_f32x2_52.x), "f"(_mul_f32x2_52.y));
                _fp4_pair_52 = (uint32_t)_e2m1_pair_h_65;
                float2 _f2_67 = make_float2(__uint_as_float(v4_138[1] << 16), __uint_as_float(v4_138[1] & 4294901760u));
                float2 _mul_f32x2_53;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_53) : "l"(*(const unsigned long long*)&_f2_67), "l"(*(const unsigned long long*)&_f2_65));
                uint32_t _fp4_pair_53;
                uint16_t _e2m1_pair_h_66;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_66) : "f"(_mul_f32x2_53.x), "f"(_mul_f32x2_53.y));
                _fp4_pair_53 = (uint32_t)_e2m1_pair_h_66;
                float2 _f2_68 = make_float2(__uint_as_float(v4_138[2] << 16), __uint_as_float(v4_138[2] & 4294901760u));
                float2 _mul_f32x2_54;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_54) : "l"(*(const unsigned long long*)&_f2_68), "l"(*(const unsigned long long*)&_f2_65));
                uint32_t _fp4_pair_54;
                uint16_t _e2m1_pair_h_67;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_67) : "f"(_mul_f32x2_54.x), "f"(_mul_f32x2_54.y));
                _fp4_pair_54 = (uint32_t)_e2m1_pair_h_67;
                float2 _f2_69 = make_float2(__uint_as_float(v4_138[3] << 16), __uint_as_float(v4_138[3] & 4294901760u));
                float2 _mul_f32x2_55;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_55) : "l"(*(const unsigned long long*)&_f2_69), "l"(*(const unsigned long long*)&_f2_65));
                uint32_t _fp4_pair_55;
                uint16_t _e2m1_pair_h_68;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_68) : "f"(_mul_f32x2_55.x), "f"(_mul_f32x2_55.y));
                _fp4_pair_55 = (uint32_t)_e2m1_pair_h_68;
                unsigned int word_140 = _fp4_pair_52 | _fp4_pair_53 << 8 | _fp4_pair_54 << 16 | _fp4_pair_55 << 24;
                uint32_t _e2m1_canon_13;
                {
                    const uint32_t _canon_w_69 = word_140;
                    const uint32_t _canon_s_69 = (_canon_w_69 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_13 = _canon_w_69 & (_canon_s_69 | 0x77777777u);
                }
                w_132[1] = _e2m1_canon_13;
                int cf_141 = f ^ 2;
                unsigned int v4_142[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_142[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_142[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_142[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_142[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_127 + (unsigned int)(16 * cf_141)));
                float m_143 = (((cf_141 & 2) != 0) ? m1_131 : m0_130);
                float2 _f2_70 = make_float2(m_143, m_143);
                float2 _f2_71 = make_float2(__uint_as_float(v4_142[0] << 16), __uint_as_float(v4_142[0] & 4294901760u));
                float2 _mul_f32x2_56;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_56) : "l"(*(const unsigned long long*)&_f2_71), "l"(*(const unsigned long long*)&_f2_70));
                uint32_t _fp4_pair_56;
                uint16_t _e2m1_pair_h_70;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_70) : "f"(_mul_f32x2_56.x), "f"(_mul_f32x2_56.y));
                _fp4_pair_56 = (uint32_t)_e2m1_pair_h_70;
                float2 _f2_72 = make_float2(__uint_as_float(v4_142[1] << 16), __uint_as_float(v4_142[1] & 4294901760u));
                float2 _mul_f32x2_57;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_57) : "l"(*(const unsigned long long*)&_f2_72), "l"(*(const unsigned long long*)&_f2_70));
                uint32_t _fp4_pair_57;
                uint16_t _e2m1_pair_h_71;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_71) : "f"(_mul_f32x2_57.x), "f"(_mul_f32x2_57.y));
                _fp4_pair_57 = (uint32_t)_e2m1_pair_h_71;
                float2 _f2_73 = make_float2(__uint_as_float(v4_142[2] << 16), __uint_as_float(v4_142[2] & 4294901760u));
                float2 _mul_f32x2_58;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_58) : "l"(*(const unsigned long long*)&_f2_73), "l"(*(const unsigned long long*)&_f2_70));
                uint32_t _fp4_pair_58;
                uint16_t _e2m1_pair_h_72;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_72) : "f"(_mul_f32x2_58.x), "f"(_mul_f32x2_58.y));
                _fp4_pair_58 = (uint32_t)_e2m1_pair_h_72;
                float2 _f2_74 = make_float2(__uint_as_float(v4_142[3] << 16), __uint_as_float(v4_142[3] & 4294901760u));
                float2 _mul_f32x2_59;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_59) : "l"(*(const unsigned long long*)&_f2_74), "l"(*(const unsigned long long*)&_f2_70));
                uint32_t _fp4_pair_59;
                uint16_t _e2m1_pair_h_73;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_73) : "f"(_mul_f32x2_59.x), "f"(_mul_f32x2_59.y));
                _fp4_pair_59 = (uint32_t)_e2m1_pair_h_73;
                unsigned int word_144 = _fp4_pair_56 | _fp4_pair_57 << 8 | _fp4_pair_58 << 16 | _fp4_pair_59 << 24;
                uint32_t _e2m1_canon_14;
                {
                    const uint32_t _canon_w_74 = word_144;
                    const uint32_t _canon_s_74 = (_canon_w_74 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_14 = _canon_w_74 & (_canon_s_74 | 0x77777777u);
                }
                w_132[2] = _e2m1_canon_14;
                int cf_145 = f ^ 3;
                unsigned int v4_146[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_146[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_146[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_146[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_146[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_127 + (unsigned int)(16 * cf_145)));
                float m_147 = (((cf_145 & 2) != 0) ? m1_131 : m0_130);
                float2 _f2_75 = make_float2(m_147, m_147);
                float2 _f2_76 = make_float2(__uint_as_float(v4_146[0] << 16), __uint_as_float(v4_146[0] & 4294901760u));
                float2 _mul_f32x2_60;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_60) : "l"(*(const unsigned long long*)&_f2_76), "l"(*(const unsigned long long*)&_f2_75));
                uint32_t _fp4_pair_60;
                uint16_t _e2m1_pair_h_75;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_75) : "f"(_mul_f32x2_60.x), "f"(_mul_f32x2_60.y));
                _fp4_pair_60 = (uint32_t)_e2m1_pair_h_75;
                float2 _f2_77 = make_float2(__uint_as_float(v4_146[1] << 16), __uint_as_float(v4_146[1] & 4294901760u));
                float2 _mul_f32x2_61;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_61) : "l"(*(const unsigned long long*)&_f2_77), "l"(*(const unsigned long long*)&_f2_75));
                uint32_t _fp4_pair_61;
                uint16_t _e2m1_pair_h_76;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_76) : "f"(_mul_f32x2_61.x), "f"(_mul_f32x2_61.y));
                _fp4_pair_61 = (uint32_t)_e2m1_pair_h_76;
                float2 _f2_78 = make_float2(__uint_as_float(v4_146[2] << 16), __uint_as_float(v4_146[2] & 4294901760u));
                float2 _mul_f32x2_62;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_62) : "l"(*(const unsigned long long*)&_f2_78), "l"(*(const unsigned long long*)&_f2_75));
                uint32_t _fp4_pair_62;
                uint16_t _e2m1_pair_h_77;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_77) : "f"(_mul_f32x2_62.x), "f"(_mul_f32x2_62.y));
                _fp4_pair_62 = (uint32_t)_e2m1_pair_h_77;
                float2 _f2_79 = make_float2(__uint_as_float(v4_146[3] << 16), __uint_as_float(v4_146[3] & 4294901760u));
                float2 _mul_f32x2_63;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_63) : "l"(*(const unsigned long long*)&_f2_79), "l"(*(const unsigned long long*)&_f2_75));
                uint32_t _fp4_pair_63;
                uint16_t _e2m1_pair_h_78;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_78) : "f"(_mul_f32x2_63.x), "f"(_mul_f32x2_63.y));
                _fp4_pair_63 = (uint32_t)_e2m1_pair_h_78;
                unsigned int word_148 = _fp4_pair_60 | _fp4_pair_61 << 8 | _fp4_pair_62 << 16 | _fp4_pair_63 << 24;
                uint32_t _e2m1_canon_15;
                {
                    const uint32_t _canon_w_79 = word_148;
                    const uint32_t _canon_s_79 = (_canon_w_79 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_15 = _canon_w_79 & (_canon_s_79 | 0x77777777u);
                }
                w_132[3] = _e2m1_canon_15;
                unsigned int t0_149 = (((f & 1) != 0) ? w_132[1] : w_132[0]);
                unsigned int t1_150 = (((f & 1) != 0) ? w_132[0] : w_132[1]);
                unsigned int t2_151 = (((f & 1) != 0) ? w_132[3] : w_132[2]);
                unsigned int t3_152 = (((f & 1) != 0) ? w_132[2] : w_132[3]);
                unsigned int q0_153 = (((f & 2) != 0) ? t2_151 : t0_149);
                unsigned int q1_154 = (((f & 2) != 0) ? t3_152 : t1_150);
                unsigned int q2_155 = (((f & 2) != 0) ? t0_149 : t2_151);
                unsigned int q3_156 = (((f & 2) != 0) ? t1_150 : t3_152);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_126)), "r"(q0_153), "r"(q1_154), "r"(q2_155), "r"(q3_156) : "memory");
                wsm16[(ob_off + 3072 >> 1) + pp2_126] = _e4m3x2_f32_3;
                int pp2_157 = lane_1 + 128;
                int base2_158 = xb_off + 64 * pp2_157;
                float bm0_159 = __uint_as_float(bmp[4] << 16);
                float bm1_160 = __uint_as_float(bmp[4] & 4294901760u);
                float _mul_8;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(bm0_159), "f"(inv));
                float _fdiv_rn_18 = __fdiv_rn(_mul_8, six);
                float _mul_9;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(bm1_160), "f"(inv));
                float _fdiv_rn_19 = __fdiv_rn(_mul_9, six);
                uint16_t _e4m3x2_f32_4;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_4) : "f"(_fdiv_rn_19), "f"(_fdiv_rn_18));
                float2 _fp8x2_decode_4;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_4.x), "=f"(_fp8x2_decode_4.y) : "h"(_e4m3x2_f32_4));
                float _fdiv_rn_20 = __fdiv_rn(inv, _fp8x2_decode_4.x);
                float _fdiv_rn_21 = __fdiv_rn(inv, _fp8x2_decode_4.y);
                float m0_161 = ((zero_51 < _fp8x2_decode_4.x) ? _fdiv_rn_20 : zero_51);
                float m1_162 = ((zero_51 < _fp8x2_decode_4.y) ? _fdiv_rn_21 : zero_51);
                unsigned int w_163[4];
                int cf_164 = f ^ 0;
                unsigned int v4_165[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_165[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_165[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_165[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_165[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_158 + (unsigned int)(16 * cf_164)));
                float m_166 = (((cf_164 & 2) != 0) ? m1_162 : m0_161);
                float2 _f2_80 = make_float2(m_166, m_166);
                float2 _f2_81 = make_float2(__uint_as_float(v4_165[0] << 16), __uint_as_float(v4_165[0] & 4294901760u));
                float2 _mul_f32x2_64;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_64) : "l"(*(const unsigned long long*)&_f2_81), "l"(*(const unsigned long long*)&_f2_80));
                uint32_t _fp4_pair_64;
                uint16_t _e2m1_pair_h_80;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_80) : "f"(_mul_f32x2_64.x), "f"(_mul_f32x2_64.y));
                _fp4_pair_64 = (uint32_t)_e2m1_pair_h_80;
                float2 _f2_82 = make_float2(__uint_as_float(v4_165[1] << 16), __uint_as_float(v4_165[1] & 4294901760u));
                float2 _mul_f32x2_65;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_65) : "l"(*(const unsigned long long*)&_f2_82), "l"(*(const unsigned long long*)&_f2_80));
                uint32_t _fp4_pair_65;
                uint16_t _e2m1_pair_h_81;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_81) : "f"(_mul_f32x2_65.x), "f"(_mul_f32x2_65.y));
                _fp4_pair_65 = (uint32_t)_e2m1_pair_h_81;
                float2 _f2_83 = make_float2(__uint_as_float(v4_165[2] << 16), __uint_as_float(v4_165[2] & 4294901760u));
                float2 _mul_f32x2_66;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_66) : "l"(*(const unsigned long long*)&_f2_83), "l"(*(const unsigned long long*)&_f2_80));
                uint32_t _fp4_pair_66;
                uint16_t _e2m1_pair_h_82;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_82) : "f"(_mul_f32x2_66.x), "f"(_mul_f32x2_66.y));
                _fp4_pair_66 = (uint32_t)_e2m1_pair_h_82;
                float2 _f2_84 = make_float2(__uint_as_float(v4_165[3] << 16), __uint_as_float(v4_165[3] & 4294901760u));
                float2 _mul_f32x2_67;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_67) : "l"(*(const unsigned long long*)&_f2_84), "l"(*(const unsigned long long*)&_f2_80));
                uint32_t _fp4_pair_67;
                uint16_t _e2m1_pair_h_83;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_83) : "f"(_mul_f32x2_67.x), "f"(_mul_f32x2_67.y));
                _fp4_pair_67 = (uint32_t)_e2m1_pair_h_83;
                unsigned int word_167 = _fp4_pair_64 | _fp4_pair_65 << 8 | _fp4_pair_66 << 16 | _fp4_pair_67 << 24;
                uint32_t _e2m1_canon_16;
                {
                    const uint32_t _canon_w_84 = word_167;
                    const uint32_t _canon_s_84 = (_canon_w_84 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_16 = _canon_w_84 & (_canon_s_84 | 0x77777777u);
                }
                w_163[0] = _e2m1_canon_16;
                int cf_168 = f ^ 1;
                unsigned int v4_169[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_169[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_169[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_169[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_169[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_158 + (unsigned int)(16 * cf_168)));
                float m_170 = (((cf_168 & 2) != 0) ? m1_162 : m0_161);
                float2 _f2_85 = make_float2(m_170, m_170);
                float2 _f2_86 = make_float2(__uint_as_float(v4_169[0] << 16), __uint_as_float(v4_169[0] & 4294901760u));
                float2 _mul_f32x2_68;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_68) : "l"(*(const unsigned long long*)&_f2_86), "l"(*(const unsigned long long*)&_f2_85));
                uint32_t _fp4_pair_68;
                uint16_t _e2m1_pair_h_85;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_85) : "f"(_mul_f32x2_68.x), "f"(_mul_f32x2_68.y));
                _fp4_pair_68 = (uint32_t)_e2m1_pair_h_85;
                float2 _f2_87 = make_float2(__uint_as_float(v4_169[1] << 16), __uint_as_float(v4_169[1] & 4294901760u));
                float2 _mul_f32x2_69;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_69) : "l"(*(const unsigned long long*)&_f2_87), "l"(*(const unsigned long long*)&_f2_85));
                uint32_t _fp4_pair_69;
                uint16_t _e2m1_pair_h_86;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_86) : "f"(_mul_f32x2_69.x), "f"(_mul_f32x2_69.y));
                _fp4_pair_69 = (uint32_t)_e2m1_pair_h_86;
                float2 _f2_88 = make_float2(__uint_as_float(v4_169[2] << 16), __uint_as_float(v4_169[2] & 4294901760u));
                float2 _mul_f32x2_70;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_70) : "l"(*(const unsigned long long*)&_f2_88), "l"(*(const unsigned long long*)&_f2_85));
                uint32_t _fp4_pair_70;
                uint16_t _e2m1_pair_h_87;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_87) : "f"(_mul_f32x2_70.x), "f"(_mul_f32x2_70.y));
                _fp4_pair_70 = (uint32_t)_e2m1_pair_h_87;
                float2 _f2_89 = make_float2(__uint_as_float(v4_169[3] << 16), __uint_as_float(v4_169[3] & 4294901760u));
                float2 _mul_f32x2_71;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_71) : "l"(*(const unsigned long long*)&_f2_89), "l"(*(const unsigned long long*)&_f2_85));
                uint32_t _fp4_pair_71;
                uint16_t _e2m1_pair_h_88;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_88) : "f"(_mul_f32x2_71.x), "f"(_mul_f32x2_71.y));
                _fp4_pair_71 = (uint32_t)_e2m1_pair_h_88;
                unsigned int word_171 = _fp4_pair_68 | _fp4_pair_69 << 8 | _fp4_pair_70 << 16 | _fp4_pair_71 << 24;
                uint32_t _e2m1_canon_17;
                {
                    const uint32_t _canon_w_89 = word_171;
                    const uint32_t _canon_s_89 = (_canon_w_89 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_17 = _canon_w_89 & (_canon_s_89 | 0x77777777u);
                }
                w_163[1] = _e2m1_canon_17;
                int cf_172 = f ^ 2;
                unsigned int v4_173[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_173[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_173[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_173[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_173[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_158 + (unsigned int)(16 * cf_172)));
                float m_174 = (((cf_172 & 2) != 0) ? m1_162 : m0_161);
                float2 _f2_90 = make_float2(m_174, m_174);
                float2 _f2_91 = make_float2(__uint_as_float(v4_173[0] << 16), __uint_as_float(v4_173[0] & 4294901760u));
                float2 _mul_f32x2_72;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_72) : "l"(*(const unsigned long long*)&_f2_91), "l"(*(const unsigned long long*)&_f2_90));
                uint32_t _fp4_pair_72;
                uint16_t _e2m1_pair_h_90;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_90) : "f"(_mul_f32x2_72.x), "f"(_mul_f32x2_72.y));
                _fp4_pair_72 = (uint32_t)_e2m1_pair_h_90;
                float2 _f2_92 = make_float2(__uint_as_float(v4_173[1] << 16), __uint_as_float(v4_173[1] & 4294901760u));
                float2 _mul_f32x2_73;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_73) : "l"(*(const unsigned long long*)&_f2_92), "l"(*(const unsigned long long*)&_f2_90));
                uint32_t _fp4_pair_73;
                uint16_t _e2m1_pair_h_91;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_91) : "f"(_mul_f32x2_73.x), "f"(_mul_f32x2_73.y));
                _fp4_pair_73 = (uint32_t)_e2m1_pair_h_91;
                float2 _f2_93 = make_float2(__uint_as_float(v4_173[2] << 16), __uint_as_float(v4_173[2] & 4294901760u));
                float2 _mul_f32x2_74;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_74) : "l"(*(const unsigned long long*)&_f2_93), "l"(*(const unsigned long long*)&_f2_90));
                uint32_t _fp4_pair_74;
                uint16_t _e2m1_pair_h_92;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_92) : "f"(_mul_f32x2_74.x), "f"(_mul_f32x2_74.y));
                _fp4_pair_74 = (uint32_t)_e2m1_pair_h_92;
                float2 _f2_94 = make_float2(__uint_as_float(v4_173[3] << 16), __uint_as_float(v4_173[3] & 4294901760u));
                float2 _mul_f32x2_75;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_75) : "l"(*(const unsigned long long*)&_f2_94), "l"(*(const unsigned long long*)&_f2_90));
                uint32_t _fp4_pair_75;
                uint16_t _e2m1_pair_h_93;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_93) : "f"(_mul_f32x2_75.x), "f"(_mul_f32x2_75.y));
                _fp4_pair_75 = (uint32_t)_e2m1_pair_h_93;
                unsigned int word_175 = _fp4_pair_72 | _fp4_pair_73 << 8 | _fp4_pair_74 << 16 | _fp4_pair_75 << 24;
                uint32_t _e2m1_canon_18;
                {
                    const uint32_t _canon_w_94 = word_175;
                    const uint32_t _canon_s_94 = (_canon_w_94 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_18 = _canon_w_94 & (_canon_s_94 | 0x77777777u);
                }
                w_163[2] = _e2m1_canon_18;
                int cf_176 = f ^ 3;
                unsigned int v4_177[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_177[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_177[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_177[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_177[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_158 + (unsigned int)(16 * cf_176)));
                float m_178 = (((cf_176 & 2) != 0) ? m1_162 : m0_161);
                float2 _f2_95 = make_float2(m_178, m_178);
                float2 _f2_96 = make_float2(__uint_as_float(v4_177[0] << 16), __uint_as_float(v4_177[0] & 4294901760u));
                float2 _mul_f32x2_76;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_76) : "l"(*(const unsigned long long*)&_f2_96), "l"(*(const unsigned long long*)&_f2_95));
                uint32_t _fp4_pair_76;
                uint16_t _e2m1_pair_h_95;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_95) : "f"(_mul_f32x2_76.x), "f"(_mul_f32x2_76.y));
                _fp4_pair_76 = (uint32_t)_e2m1_pair_h_95;
                float2 _f2_97 = make_float2(__uint_as_float(v4_177[1] << 16), __uint_as_float(v4_177[1] & 4294901760u));
                float2 _mul_f32x2_77;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_77) : "l"(*(const unsigned long long*)&_f2_97), "l"(*(const unsigned long long*)&_f2_95));
                uint32_t _fp4_pair_77;
                uint16_t _e2m1_pair_h_96;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_96) : "f"(_mul_f32x2_77.x), "f"(_mul_f32x2_77.y));
                _fp4_pair_77 = (uint32_t)_e2m1_pair_h_96;
                float2 _f2_98 = make_float2(__uint_as_float(v4_177[2] << 16), __uint_as_float(v4_177[2] & 4294901760u));
                float2 _mul_f32x2_78;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_78) : "l"(*(const unsigned long long*)&_f2_98), "l"(*(const unsigned long long*)&_f2_95));
                uint32_t _fp4_pair_78;
                uint16_t _e2m1_pair_h_97;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_97) : "f"(_mul_f32x2_78.x), "f"(_mul_f32x2_78.y));
                _fp4_pair_78 = (uint32_t)_e2m1_pair_h_97;
                float2 _f2_99 = make_float2(__uint_as_float(v4_177[3] << 16), __uint_as_float(v4_177[3] & 4294901760u));
                float2 _mul_f32x2_79;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_79) : "l"(*(const unsigned long long*)&_f2_99), "l"(*(const unsigned long long*)&_f2_95));
                uint32_t _fp4_pair_79;
                uint16_t _e2m1_pair_h_98;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_98) : "f"(_mul_f32x2_79.x), "f"(_mul_f32x2_79.y));
                _fp4_pair_79 = (uint32_t)_e2m1_pair_h_98;
                unsigned int word_179 = _fp4_pair_76 | _fp4_pair_77 << 8 | _fp4_pair_78 << 16 | _fp4_pair_79 << 24;
                uint32_t _e2m1_canon_19;
                {
                    const uint32_t _canon_w_99 = word_179;
                    const uint32_t _canon_s_99 = (_canon_w_99 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_19 = _canon_w_99 & (_canon_s_99 | 0x77777777u);
                }
                w_163[3] = _e2m1_canon_19;
                unsigned int t0_180 = (((f & 1) != 0) ? w_163[1] : w_163[0]);
                unsigned int t1_181 = (((f & 1) != 0) ? w_163[0] : w_163[1]);
                unsigned int t2_182 = (((f & 1) != 0) ? w_163[3] : w_163[2]);
                unsigned int t3_183 = (((f & 1) != 0) ? w_163[2] : w_163[3]);
                unsigned int q0_184 = (((f & 2) != 0) ? t2_182 : t0_180);
                unsigned int q1_185 = (((f & 2) != 0) ? t3_183 : t1_181);
                unsigned int q2_186 = (((f & 2) != 0) ? t0_180 : t2_182);
                unsigned int q3_187 = (((f & 2) != 0) ? t1_181 : t3_183);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_157)), "r"(q0_184), "r"(q1_185), "r"(q2_186), "r"(q3_187) : "memory");
                wsm16[(ob_off + 3072 >> 1) + pp2_157] = _e4m3x2_f32_4;
                int pp2_188 = lane_1 + 160;
                int base2_189 = xb_off + 64 * pp2_188;
                float bm0_190 = __uint_as_float(bmp[5] << 16);
                float bm1_191 = __uint_as_float(bmp[5] & 4294901760u);
                float _mul_10;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(bm0_190), "f"(inv));
                float _fdiv_rn_22 = __fdiv_rn(_mul_10, six);
                float _mul_11;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(bm1_191), "f"(inv));
                float _fdiv_rn_23 = __fdiv_rn(_mul_11, six);
                uint16_t _e4m3x2_f32_5;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_5) : "f"(_fdiv_rn_23), "f"(_fdiv_rn_22));
                float2 _fp8x2_decode_5;
                asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                    "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                    "mov.b32 {lo, hi}, pair;\n"
                    "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                    : "=f"(_fp8x2_decode_5.x), "=f"(_fp8x2_decode_5.y) : "h"(_e4m3x2_f32_5));
                float _fdiv_rn_24 = __fdiv_rn(inv, _fp8x2_decode_5.x);
                float _fdiv_rn_25 = __fdiv_rn(inv, _fp8x2_decode_5.y);
                float m0_192 = ((zero_51 < _fp8x2_decode_5.x) ? _fdiv_rn_24 : zero_51);
                float m1_193 = ((zero_51 < _fp8x2_decode_5.y) ? _fdiv_rn_25 : zero_51);
                unsigned int w_194[4];
                int cf_195 = f ^ 0;
                unsigned int v4_196[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_196[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_196[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_196[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_196[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_189 + (unsigned int)(16 * cf_195)));
                float m_197 = (((cf_195 & 2) != 0) ? m1_193 : m0_192);
                float2 _f2_100 = make_float2(m_197, m_197);
                float2 _f2_101 = make_float2(__uint_as_float(v4_196[0] << 16), __uint_as_float(v4_196[0] & 4294901760u));
                float2 _mul_f32x2_80;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_80) : "l"(*(const unsigned long long*)&_f2_101), "l"(*(const unsigned long long*)&_f2_100));
                uint32_t _fp4_pair_80;
                uint16_t _e2m1_pair_h_100;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_100) : "f"(_mul_f32x2_80.x), "f"(_mul_f32x2_80.y));
                _fp4_pair_80 = (uint32_t)_e2m1_pair_h_100;
                float2 _f2_102 = make_float2(__uint_as_float(v4_196[1] << 16), __uint_as_float(v4_196[1] & 4294901760u));
                float2 _mul_f32x2_81;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_81) : "l"(*(const unsigned long long*)&_f2_102), "l"(*(const unsigned long long*)&_f2_100));
                uint32_t _fp4_pair_81;
                uint16_t _e2m1_pair_h_101;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_101) : "f"(_mul_f32x2_81.x), "f"(_mul_f32x2_81.y));
                _fp4_pair_81 = (uint32_t)_e2m1_pair_h_101;
                float2 _f2_103 = make_float2(__uint_as_float(v4_196[2] << 16), __uint_as_float(v4_196[2] & 4294901760u));
                float2 _mul_f32x2_82;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_82) : "l"(*(const unsigned long long*)&_f2_103), "l"(*(const unsigned long long*)&_f2_100));
                uint32_t _fp4_pair_82;
                uint16_t _e2m1_pair_h_102;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_102) : "f"(_mul_f32x2_82.x), "f"(_mul_f32x2_82.y));
                _fp4_pair_82 = (uint32_t)_e2m1_pair_h_102;
                float2 _f2_104 = make_float2(__uint_as_float(v4_196[3] << 16), __uint_as_float(v4_196[3] & 4294901760u));
                float2 _mul_f32x2_83;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_83) : "l"(*(const unsigned long long*)&_f2_104), "l"(*(const unsigned long long*)&_f2_100));
                uint32_t _fp4_pair_83;
                uint16_t _e2m1_pair_h_103;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_103) : "f"(_mul_f32x2_83.x), "f"(_mul_f32x2_83.y));
                _fp4_pair_83 = (uint32_t)_e2m1_pair_h_103;
                unsigned int word_198 = _fp4_pair_80 | _fp4_pair_81 << 8 | _fp4_pair_82 << 16 | _fp4_pair_83 << 24;
                uint32_t _e2m1_canon_20;
                {
                    const uint32_t _canon_w_104 = word_198;
                    const uint32_t _canon_s_104 = (_canon_w_104 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_20 = _canon_w_104 & (_canon_s_104 | 0x77777777u);
                }
                w_194[0] = _e2m1_canon_20;
                int cf_199 = f ^ 1;
                unsigned int v4_200[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_200[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_200[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_200[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_200[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_189 + (unsigned int)(16 * cf_199)));
                float m_201 = (((cf_199 & 2) != 0) ? m1_193 : m0_192);
                float2 _f2_105 = make_float2(m_201, m_201);
                float2 _f2_106 = make_float2(__uint_as_float(v4_200[0] << 16), __uint_as_float(v4_200[0] & 4294901760u));
                float2 _mul_f32x2_84;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_84) : "l"(*(const unsigned long long*)&_f2_106), "l"(*(const unsigned long long*)&_f2_105));
                uint32_t _fp4_pair_84;
                uint16_t _e2m1_pair_h_105;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_105) : "f"(_mul_f32x2_84.x), "f"(_mul_f32x2_84.y));
                _fp4_pair_84 = (uint32_t)_e2m1_pair_h_105;
                float2 _f2_107 = make_float2(__uint_as_float(v4_200[1] << 16), __uint_as_float(v4_200[1] & 4294901760u));
                float2 _mul_f32x2_85;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_85) : "l"(*(const unsigned long long*)&_f2_107), "l"(*(const unsigned long long*)&_f2_105));
                uint32_t _fp4_pair_85;
                uint16_t _e2m1_pair_h_106;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_106) : "f"(_mul_f32x2_85.x), "f"(_mul_f32x2_85.y));
                _fp4_pair_85 = (uint32_t)_e2m1_pair_h_106;
                float2 _f2_108 = make_float2(__uint_as_float(v4_200[2] << 16), __uint_as_float(v4_200[2] & 4294901760u));
                float2 _mul_f32x2_86;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_86) : "l"(*(const unsigned long long*)&_f2_108), "l"(*(const unsigned long long*)&_f2_105));
                uint32_t _fp4_pair_86;
                uint16_t _e2m1_pair_h_107;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_107) : "f"(_mul_f32x2_86.x), "f"(_mul_f32x2_86.y));
                _fp4_pair_86 = (uint32_t)_e2m1_pair_h_107;
                float2 _f2_109 = make_float2(__uint_as_float(v4_200[3] << 16), __uint_as_float(v4_200[3] & 4294901760u));
                float2 _mul_f32x2_87;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_87) : "l"(*(const unsigned long long*)&_f2_109), "l"(*(const unsigned long long*)&_f2_105));
                uint32_t _fp4_pair_87;
                uint16_t _e2m1_pair_h_108;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_108) : "f"(_mul_f32x2_87.x), "f"(_mul_f32x2_87.y));
                _fp4_pair_87 = (uint32_t)_e2m1_pair_h_108;
                unsigned int word_202 = _fp4_pair_84 | _fp4_pair_85 << 8 | _fp4_pair_86 << 16 | _fp4_pair_87 << 24;
                uint32_t _e2m1_canon_21;
                {
                    const uint32_t _canon_w_109 = word_202;
                    const uint32_t _canon_s_109 = (_canon_w_109 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_21 = _canon_w_109 & (_canon_s_109 | 0x77777777u);
                }
                w_194[1] = _e2m1_canon_21;
                int cf_203 = f ^ 2;
                unsigned int v4_204[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_204[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_204[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_204[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_204[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_189 + (unsigned int)(16 * cf_203)));
                float m_205 = (((cf_203 & 2) != 0) ? m1_193 : m0_192);
                float2 _f2_110 = make_float2(m_205, m_205);
                float2 _f2_111 = make_float2(__uint_as_float(v4_204[0] << 16), __uint_as_float(v4_204[0] & 4294901760u));
                float2 _mul_f32x2_88;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_88) : "l"(*(const unsigned long long*)&_f2_111), "l"(*(const unsigned long long*)&_f2_110));
                uint32_t _fp4_pair_88;
                uint16_t _e2m1_pair_h_110;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_110) : "f"(_mul_f32x2_88.x), "f"(_mul_f32x2_88.y));
                _fp4_pair_88 = (uint32_t)_e2m1_pair_h_110;
                float2 _f2_112 = make_float2(__uint_as_float(v4_204[1] << 16), __uint_as_float(v4_204[1] & 4294901760u));
                float2 _mul_f32x2_89;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_89) : "l"(*(const unsigned long long*)&_f2_112), "l"(*(const unsigned long long*)&_f2_110));
                uint32_t _fp4_pair_89;
                uint16_t _e2m1_pair_h_111;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_111) : "f"(_mul_f32x2_89.x), "f"(_mul_f32x2_89.y));
                _fp4_pair_89 = (uint32_t)_e2m1_pair_h_111;
                float2 _f2_113 = make_float2(__uint_as_float(v4_204[2] << 16), __uint_as_float(v4_204[2] & 4294901760u));
                float2 _mul_f32x2_90;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_90) : "l"(*(const unsigned long long*)&_f2_113), "l"(*(const unsigned long long*)&_f2_110));
                uint32_t _fp4_pair_90;
                uint16_t _e2m1_pair_h_112;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_112) : "f"(_mul_f32x2_90.x), "f"(_mul_f32x2_90.y));
                _fp4_pair_90 = (uint32_t)_e2m1_pair_h_112;
                float2 _f2_114 = make_float2(__uint_as_float(v4_204[3] << 16), __uint_as_float(v4_204[3] & 4294901760u));
                float2 _mul_f32x2_91;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_91) : "l"(*(const unsigned long long*)&_f2_114), "l"(*(const unsigned long long*)&_f2_110));
                uint32_t _fp4_pair_91;
                uint16_t _e2m1_pair_h_113;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_113) : "f"(_mul_f32x2_91.x), "f"(_mul_f32x2_91.y));
                _fp4_pair_91 = (uint32_t)_e2m1_pair_h_113;
                unsigned int word_206 = _fp4_pair_88 | _fp4_pair_89 << 8 | _fp4_pair_90 << 16 | _fp4_pair_91 << 24;
                uint32_t _e2m1_canon_22;
                {
                    const uint32_t _canon_w_114 = word_206;
                    const uint32_t _canon_s_114 = (_canon_w_114 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_22 = _canon_w_114 & (_canon_s_114 | 0x77777777u);
                }
                w_194[2] = _e2m1_canon_22;
                int cf_207 = f ^ 3;
                unsigned int v4_208[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_208[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_208[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_208[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_208[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)base2_189 + (unsigned int)(16 * cf_207)));
                float m_209 = (((cf_207 & 2) != 0) ? m1_193 : m0_192);
                float2 _f2_115 = make_float2(m_209, m_209);
                float2 _f2_116 = make_float2(__uint_as_float(v4_208[0] << 16), __uint_as_float(v4_208[0] & 4294901760u));
                float2 _mul_f32x2_92;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_92) : "l"(*(const unsigned long long*)&_f2_116), "l"(*(const unsigned long long*)&_f2_115));
                uint32_t _fp4_pair_92;
                uint16_t _e2m1_pair_h_115;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_115) : "f"(_mul_f32x2_92.x), "f"(_mul_f32x2_92.y));
                _fp4_pair_92 = (uint32_t)_e2m1_pair_h_115;
                float2 _f2_117 = make_float2(__uint_as_float(v4_208[1] << 16), __uint_as_float(v4_208[1] & 4294901760u));
                float2 _mul_f32x2_93;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_93) : "l"(*(const unsigned long long*)&_f2_117), "l"(*(const unsigned long long*)&_f2_115));
                uint32_t _fp4_pair_93;
                uint16_t _e2m1_pair_h_116;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_116) : "f"(_mul_f32x2_93.x), "f"(_mul_f32x2_93.y));
                _fp4_pair_93 = (uint32_t)_e2m1_pair_h_116;
                float2 _f2_118 = make_float2(__uint_as_float(v4_208[2] << 16), __uint_as_float(v4_208[2] & 4294901760u));
                float2 _mul_f32x2_94;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_94) : "l"(*(const unsigned long long*)&_f2_118), "l"(*(const unsigned long long*)&_f2_115));
                uint32_t _fp4_pair_94;
                uint16_t _e2m1_pair_h_117;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_117) : "f"(_mul_f32x2_94.x), "f"(_mul_f32x2_94.y));
                _fp4_pair_94 = (uint32_t)_e2m1_pair_h_117;
                float2 _f2_119 = make_float2(__uint_as_float(v4_208[3] << 16), __uint_as_float(v4_208[3] & 4294901760u));
                float2 _mul_f32x2_95;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_95) : "l"(*(const unsigned long long*)&_f2_119), "l"(*(const unsigned long long*)&_f2_115));
                uint32_t _fp4_pair_95;
                uint16_t _e2m1_pair_h_118;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_118) : "f"(_mul_f32x2_95.x), "f"(_mul_f32x2_95.y));
                _fp4_pair_95 = (uint32_t)_e2m1_pair_h_118;
                unsigned int word_210 = _fp4_pair_92 | _fp4_pair_93 << 8 | _fp4_pair_94 << 16 | _fp4_pair_95 << 24;
                uint32_t _e2m1_canon_23;
                {
                    const uint32_t _canon_w_119 = word_210;
                    const uint32_t _canon_s_119 = (_canon_w_119 & 0x77777777u) + 0x77777777u;
                    _e2m1_canon_23 = _canon_w_119 & (_canon_s_119 | 0x77777777u);
                }
                w_194[3] = _e2m1_canon_23;
                unsigned int t0_211 = (((f & 1) != 0) ? w_194[1] : w_194[0]);
                unsigned int t1_212 = (((f & 1) != 0) ? w_194[0] : w_194[1]);
                unsigned int t2_213 = (((f & 1) != 0) ? w_194[3] : w_194[2]);
                unsigned int t3_214 = (((f & 1) != 0) ? w_194[2] : w_194[3]);
                unsigned int q0_215 = (((f & 2) != 0) ? t2_213 : t0_211);
                unsigned int q1_216 = (((f & 2) != 0) ? t3_214 : t1_212);
                unsigned int q2_217 = (((f & 2) != 0) ? t0_211 : t2_213);
                unsigned int q3_218 = (((f & 2) != 0) ? t1_212 : t3_214);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_188)), "r"(q0_215), "r"(q1_216), "r"(q2_217), "r"(q3_218) : "memory");
                wsm16[(ob_off + 3072 >> 1) + pp2_188] = _e4m3x2_f32_5;
                rs = rs_50;
                __syncwarp();
                if (n > k + 1 && lane_1 == 0) {
                    long long next_row2 = (long long)(r0 + warp_0 + 10 * (k + 1));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int stage_1 = 2 * warp_0 + (k + 1 & 1);
                    mbarrier_arrive_expect_tx(xb_full_addr + (stage_1) * 8, 12288);
                    cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)xb_off, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(x) + ((unsigned long long)(next_row2 * 6144) * (unsigned long long)2)), 12288, xb_full_addr + (stage_1) * 8);
                }
                if (lane_1 < 4) {
                    wsm32[(ob_off + 3072 + 384 >> 2) + lane_1] = rs;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane_1 == 0) {
                    long long row64 = (long long)(r0 + warp_0 + 10 * k);
                    long long owner = row64 % 4;
                    long long ix64 = row64 / 4;
                    if (owner == 0) {
                        {
                            void* _cpbulk_dst_120 = reinterpret_cast<void*>(peer_base + (rs_base + ix64 * 3072));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_120), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_121 = reinterpret_cast<void*>(peer_base + (rs_base + 50331648 + ix64 * 384));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_121), "r"(wsm_addr + (unsigned int)ob_off + 3072), "r"((uint32_t)(384))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_122 = reinterpret_cast<void*>(peer_base + (rs_base + 56623104 + ix64 * 16));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_122), "r"(wsm_addr + (unsigned int)ob_off + 3072 + 384), "r"((uint32_t)(16))
                                : "memory");
                        }
                    }
                    if (owner == 1) {
                        {
                            void* _cpbulk_dst_123 = reinterpret_cast<void*>(peer_base_2 + (rs_base + ix64 * 3072));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_123), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_124 = reinterpret_cast<void*>(peer_base_2 + (rs_base + 50331648 + ix64 * 384));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_124), "r"(wsm_addr + (unsigned int)ob_off + 3072), "r"((uint32_t)(384))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_125 = reinterpret_cast<void*>(peer_base_2 + (rs_base + 56623104 + ix64 * 16));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_125), "r"(wsm_addr + (unsigned int)ob_off + 3072 + 384), "r"((uint32_t)(16))
                                : "memory");
                        }
                    }
                    if (owner == 2) {
                        {
                            void* _cpbulk_dst_126 = reinterpret_cast<void*>(peer_base_3 + (rs_base + ix64 * 3072));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_126), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_127 = reinterpret_cast<void*>(peer_base_3 + (rs_base + 50331648 + ix64 * 384));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_127), "r"(wsm_addr + (unsigned int)ob_off + 3072), "r"((uint32_t)(384))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_128 = reinterpret_cast<void*>(peer_base_3 + (rs_base + 56623104 + ix64 * 16));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_128), "r"(wsm_addr + (unsigned int)ob_off + 3072 + 384), "r"((uint32_t)(16))
                                : "memory");
                        }
                    }
                    if (owner == 3) {
                        {
                            void* _cpbulk_dst_129 = reinterpret_cast<void*>(peer_base_4 + (rs_base + ix64 * 3072));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_129), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_130 = reinterpret_cast<void*>(peer_base_4 + (rs_base + 50331648 + ix64 * 384));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_130), "r"(wsm_addr + (unsigned int)ob_off + 3072), "r"((uint32_t)(384))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_131 = reinterpret_cast<void*>(peer_base_4 + (rs_base + 56623104 + ix64 * 16));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_131), "r"(wsm_addr + (unsigned int)ob_off + 3072 + 384), "r"((uint32_t)(16))
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
            int nC = nrows - nB;
            int wB = (warp_0 - nrows % 10 + 10) % 10;
            int wC = (warp_0 - (nrows + nB) % 10 + 10) % 10;
            int kb = 0;
            #pragma unroll 1
            for (int i = wB; i < nB; i += 10) {
                int brow = rb0 + 4 * i;
                long long bix64 = (long long)(brow / 4);
                long long brow64 = (long long)brow;
                int bob_off = xb_off + (kb & 1) * 3472;
                if (kb >= 2) {
                    if (lane_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncwarp();
                }
                uint32_t _sysv_poll32_group_0[1];
                do {
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[0]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + 56623104 + bix64 * 16))) : "memory");
                } while (
                    (_sysv_poll32_group_0[0] == 0x80000000u)
                );
                uint32_t _sysv_poll32_group_1[1];
                do {
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_1[0]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + 56623104 + (4096 + bix64) * 16))) : "memory");
                } while (
                    (_sysv_poll32_group_1[0] == 0x80000000u)
                );
                uint32_t _sysv_poll32_group_2[1];
                do {
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_2[0]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + 56623104 + (8192 + bix64) * 16))) : "memory");
                } while (
                    (_sysv_poll32_group_2[0] == 0x80000000u)
                );
                uint32_t _sysv_poll32_group_3[1];
                do {
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_3[0]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + 56623104 + (12288 + bix64) * 16))) : "memory");
                } while (
                    (_sysv_poll32_group_3[0] == 0x80000000u)
                );
                float am_1 = 0.0f;
                #pragma unroll 1
                for (int jj = 0; jj < 3; jj++) {
                    int pp0 = lane_1 + 64 * jj;
                    int pp1 = pp0 + 32;
                    uint32_t _fp4_wire_values_0[32];
                    uint32_t _fp4_wire_scales_0[8];
                    uint16_t _wire_h16_136_0;
                    uint16_t _wire_h16_136_1;
                    uint16_t _wire_h16_136_2;
                    uint16_t _wire_h16_136_3;
                    uint16_t _wire_h16_136_4;
                    uint16_t _wire_h16_136_5;
                    uint16_t _wire_h16_136_6;
                    uint16_t _wire_h16_136_7;
                    uint32_t _wire_vpend_136 = 255u;
                    uint32_t _wire_spend_136 = 255u;
                    do {
                        if (_wire_vpend_136 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + bix64 * 3072 + (long long)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_136 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + (4096 + bix64) * 3072 + (long long)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_136 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[8]), "=r"(_fp4_wire_values_0[9]), "=r"(_fp4_wire_values_0[10]), "=r"(_fp4_wire_values_0[11]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + (8192 + bix64) * 3072 + (long long)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_136 & 8u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[12]), "=r"(_fp4_wire_values_0[13]), "=r"(_fp4_wire_values_0[14]), "=r"(_fp4_wire_values_0[15]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + (12288 + bix64) * 3072 + (long long)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_136 & 16u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[16]), "=r"(_fp4_wire_values_0[17]), "=r"(_fp4_wire_values_0[18]), "=r"(_fp4_wire_values_0[19]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + bix64 * 3072 + (long long)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_136 & 32u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[20]), "=r"(_fp4_wire_values_0[21]), "=r"(_fp4_wire_values_0[22]), "=r"(_fp4_wire_values_0[23]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + (4096 + bix64) * 3072 + (long long)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_136 & 64u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[24]), "=r"(_fp4_wire_values_0[25]), "=r"(_fp4_wire_values_0[26]), "=r"(_fp4_wire_values_0[27]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + (8192 + bix64) * 3072 + (long long)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_136 & 128u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[28]), "=r"(_fp4_wire_values_0[29]), "=r"(_fp4_wire_values_0[30]), "=r"(_fp4_wire_values_0[31]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + (12288 + bix64) * 3072 + (long long)(16 * pp1)))) : "memory");
                        }
                        if (_wire_spend_136 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_136_0) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (rs_base + 50331648 + bix64 * 384 + (long long)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_136_0;
                        }
                        if (_wire_spend_136 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_136_1) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (rs_base + 50331648 + (4096 + bix64) * 384 + (long long)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_136_1;
                        }
                        if (_wire_spend_136 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_136_2) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (rs_base + 50331648 + (8192 + bix64) * 384 + (long long)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_0[2] = (uint32_t)_wire_h16_136_2;
                        }
                        if (_wire_spend_136 & 8u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_136_3) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (rs_base + 50331648 + (12288 + bix64) * 384 + (long long)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_0[3] = (uint32_t)_wire_h16_136_3;
                        }
                        if (_wire_spend_136 & 16u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_136_4) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (rs_base + 50331648 + bix64 * 384 + (long long)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_0[4] = (uint32_t)_wire_h16_136_4;
                        }
                        if (_wire_spend_136 & 32u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_136_5) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (rs_base + 50331648 + (4096 + bix64) * 384 + (long long)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_0[5] = (uint32_t)_wire_h16_136_5;
                        }
                        if (_wire_spend_136 & 64u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_136_6) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (rs_base + 50331648 + (8192 + bix64) * 384 + (long long)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_0[6] = (uint32_t)_wire_h16_136_6;
                        }
                        if (_wire_spend_136 & 128u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_136_7) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (rs_base + 50331648 + (12288 + bix64) * 384 + (long long)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_0[7] = (uint32_t)_wire_h16_136_7;
                        }
                        if (_wire_vpend_136 & 1u) {
                            if (!((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_136 &= ~1u;
                        }
                        if (_wire_vpend_136 & 2u) {
                            if (!((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_136 &= ~2u;
                        }
                        if (_wire_vpend_136 & 4u) {
                            if (!((((_fp4_wire_values_0[8] & ~((_fp4_wire_values_0[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[9] & ~((_fp4_wire_values_0[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[10] & ~((_fp4_wire_values_0[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[11] & ~((_fp4_wire_values_0[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_136 &= ~4u;
                        }
                        if (_wire_vpend_136 & 8u) {
                            if (!((((_fp4_wire_values_0[12] & ~((_fp4_wire_values_0[12] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[13] & ~((_fp4_wire_values_0[13] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[14] & ~((_fp4_wire_values_0[14] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[15] & ~((_fp4_wire_values_0[15] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_136 &= ~8u;
                        }
                        if (_wire_vpend_136 & 16u) {
                            if (!((((_fp4_wire_values_0[16] & ~((_fp4_wire_values_0[16] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[17] & ~((_fp4_wire_values_0[17] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[18] & ~((_fp4_wire_values_0[18] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[19] & ~((_fp4_wire_values_0[19] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_136 &= ~16u;
                        }
                        if (_wire_vpend_136 & 32u) {
                            if (!((((_fp4_wire_values_0[20] & ~((_fp4_wire_values_0[20] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[21] & ~((_fp4_wire_values_0[21] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[22] & ~((_fp4_wire_values_0[22] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[23] & ~((_fp4_wire_values_0[23] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_136 &= ~32u;
                        }
                        if (_wire_vpend_136 & 64u) {
                            if (!((((_fp4_wire_values_0[24] & ~((_fp4_wire_values_0[24] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[25] & ~((_fp4_wire_values_0[25] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[26] & ~((_fp4_wire_values_0[26] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[27] & ~((_fp4_wire_values_0[27] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_136 &= ~64u;
                        }
                        if (_wire_vpend_136 & 128u) {
                            if (!((((_fp4_wire_values_0[28] & ~((_fp4_wire_values_0[28] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[29] & ~((_fp4_wire_values_0[29] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[30] & ~((_fp4_wire_values_0[30] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[31] & ~((_fp4_wire_values_0[31] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_136 &= ~128u;
                        }
                        if (_wire_spend_136 & 1u) {
                            if (!(((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[0] >> 8) == 0xFFu))) _wire_spend_136 &= ~1u;
                        }
                        if (_wire_spend_136 & 2u) {
                            if (!(((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[1] >> 8) == 0xFFu))) _wire_spend_136 &= ~2u;
                        }
                        if (_wire_spend_136 & 4u) {
                            if (!(((_fp4_wire_scales_0[2] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[2] >> 8) == 0xFFu))) _wire_spend_136 &= ~4u;
                        }
                        if (_wire_spend_136 & 8u) {
                            if (!(((_fp4_wire_scales_0[3] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[3] >> 8) == 0xFFu))) _wire_spend_136 &= ~8u;
                        }
                        if (_wire_spend_136 & 16u) {
                            if (!(((_fp4_wire_scales_0[4] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[4] >> 8) == 0xFFu))) _wire_spend_136 &= ~16u;
                        }
                        if (_wire_spend_136 & 32u) {
                            if (!(((_fp4_wire_scales_0[5] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[5] >> 8) == 0xFFu))) _wire_spend_136 &= ~32u;
                        }
                        if (_wire_spend_136 & 64u) {
                            if (!(((_fp4_wire_scales_0[6] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[6] >> 8) == 0xFFu))) _wire_spend_136 &= ~64u;
                        }
                        if (_wire_spend_136 & 128u) {
                            if (!(((_fp4_wire_scales_0[7] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[7] >> 8) == 0xFFu))) _wire_spend_136 &= ~128u;
                        }
                    } while ((_wire_vpend_136 | _wire_spend_136) != 0u);
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
                    float2 _fp8x2_decode_6;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_6.x), "=f"(_fp8x2_decode_6.y) : "h"((uint16_t)_fp4_wire_scales_0[0]));
                    float _mul_12;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(_fp8x2_decode_6.x), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                    float s0 = _mul_12;
                    float _mul_13;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(_fp8x2_decode_6.y), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                    float s1 = _mul_13;
                    float2 _f2_120 = make_float2(s0, s0);
                    float2 _f2_121 = make_float2(s1, s1);
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
                    float2 _f2_122 = make_float2(acc[0], acc[1]);
                    float2 _f2_123 = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                    float2 r = fma_f32x2_rn_noftz(_f2_123, _f2_120, _f2_122);
                    acc[0] = r.x;
                    acc[1] = r.y;
                    float2 _f2_124 = make_float2(acc[2], acc[3]);
                    float2 _f2_125 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                    float2 r_0 = fma_f32x2_rn_noftz(_f2_125, _f2_120, _f2_124);
                    acc[2] = r_0.x;
                    acc[3] = r_0.y;
                    float2 _f2_126 = make_float2(acc[4], acc[5]);
                    float2 _f2_127 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                    float2 r_1 = fma_f32x2_rn_noftz(_f2_127, _f2_120, _f2_126);
                    acc[4] = r_1.x;
                    acc[5] = r_1.y;
                    float2 _f2_128 = make_float2(acc[6], acc[7]);
                    float2 _f2_129 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                    float2 r_2 = fma_f32x2_rn_noftz(_f2_129, _f2_120, _f2_128);
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
                    float2 _f2_130 = make_float2(acc[8], acc[9]);
                    float2 _f2_131 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                    float2 r_3 = fma_f32x2_rn_noftz(_f2_131, _f2_120, _f2_130);
                    acc[8] = r_3.x;
                    acc[9] = r_3.y;
                    float2 _f2_132 = make_float2(acc[10], acc[11]);
                    float2 _f2_133 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                    float2 r_4 = fma_f32x2_rn_noftz(_f2_133, _f2_120, _f2_132);
                    acc[10] = r_4.x;
                    acc[11] = r_4.y;
                    float2 _f2_134 = make_float2(acc[12], acc[13]);
                    float2 _f2_135 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                    float2 r_5 = fma_f32x2_rn_noftz(_f2_135, _f2_120, _f2_134);
                    acc[12] = r_5.x;
                    acc[13] = r_5.y;
                    float2 _f2_136 = make_float2(acc[14], acc[15]);
                    float2 _f2_137 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                    float2 r_6 = fma_f32x2_rn_noftz(_f2_137, _f2_120, _f2_136);
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
                    float2 _f2_138 = make_float2(acc[16], acc[17]);
                    float2 _f2_139 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                    float2 r_7 = fma_f32x2_rn_noftz(_f2_139, _f2_121, _f2_138);
                    acc[16] = r_7.x;
                    acc[17] = r_7.y;
                    float2 _f2_140 = make_float2(acc[18], acc[19]);
                    float2 _f2_141 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                    float2 r_8 = fma_f32x2_rn_noftz(_f2_141, _f2_121, _f2_140);
                    acc[18] = r_8.x;
                    acc[19] = r_8.y;
                    float2 _f2_142 = make_float2(acc[20], acc[21]);
                    float2 _f2_143 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                    float2 r_9 = fma_f32x2_rn_noftz(_f2_143, _f2_121, _f2_142);
                    acc[20] = r_9.x;
                    acc[21] = r_9.y;
                    float2 _f2_144 = make_float2(acc[22], acc[23]);
                    float2 _f2_145 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                    float2 r_10 = fma_f32x2_rn_noftz(_f2_145, _f2_121, _f2_144);
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
                    float2 _f2_146 = make_float2(acc[24], acc[25]);
                    float2 _f2_147 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                    float2 r_11 = fma_f32x2_rn_noftz(_f2_147, _f2_121, _f2_146);
                    acc[24] = r_11.x;
                    acc[25] = r_11.y;
                    float2 _f2_148 = make_float2(acc[26], acc[27]);
                    float2 _f2_149 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                    float2 r_12 = fma_f32x2_rn_noftz(_f2_149, _f2_121, _f2_148);
                    acc[26] = r_12.x;
                    acc[27] = r_12.y;
                    float2 _f2_150 = make_float2(acc[28], acc[29]);
                    float2 _f2_151 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                    float2 r_13 = fma_f32x2_rn_noftz(_f2_151, _f2_121, _f2_150);
                    acc[28] = r_13.x;
                    acc[29] = r_13.y;
                    float2 _f2_152 = make_float2(acc[30], acc[31]);
                    float2 _f2_153 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                    float2 r_14 = fma_f32x2_rn_noftz(_f2_153, _f2_121, _f2_152);
                    acc[30] = r_14.x;
                    acc[31] = r_14.y;
                    float2 _fp8x2_decode_7;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_7.x), "=f"(_fp8x2_decode_7.y) : "h"((uint16_t)_fp4_wire_scales_0[1]));
                    float _mul_14;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(_fp8x2_decode_7.x), "f"(__uint_as_float(_sysv_poll32_group_1[0])));
                    float s0_15 = _mul_14;
                    float _mul_15;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(_fp8x2_decode_7.y), "f"(__uint_as_float(_sysv_poll32_group_1[0])));
                    float s1_16 = _mul_15;
                    float2 _f2_154 = make_float2(s0_15, s0_15);
                    float2 _f2_155 = make_float2(s1_16, s1_16);
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
                    float2 _f2_156 = make_float2(acc[0], acc[1]);
                    float2 _f2_157 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                    float2 r_17 = fma_f32x2_rn_noftz(_f2_157, _f2_154, _f2_156);
                    acc[0] = r_17.x;
                    acc[1] = r_17.y;
                    float2 _f2_158 = make_float2(acc[2], acc[3]);
                    float2 _f2_159 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                    float2 r_18 = fma_f32x2_rn_noftz(_f2_159, _f2_154, _f2_158);
                    acc[2] = r_18.x;
                    acc[3] = r_18.y;
                    float2 _f2_160 = make_float2(acc[4], acc[5]);
                    float2 _f2_161 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                    float2 r_19 = fma_f32x2_rn_noftz(_f2_161, _f2_154, _f2_160);
                    acc[4] = r_19.x;
                    acc[5] = r_19.y;
                    float2 _f2_162 = make_float2(acc[6], acc[7]);
                    float2 _f2_163 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                    float2 r_20 = fma_f32x2_rn_noftz(_f2_163, _f2_154, _f2_162);
                    acc[6] = r_20.x;
                    acc[7] = r_20.y;
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
                    float2 _f2_164 = make_float2(acc[8], acc[9]);
                    float2 _f2_165 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                    float2 r_21 = fma_f32x2_rn_noftz(_f2_165, _f2_154, _f2_164);
                    acc[8] = r_21.x;
                    acc[9] = r_21.y;
                    float2 _f2_166 = make_float2(acc[10], acc[11]);
                    float2 _f2_167 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                    float2 r_22 = fma_f32x2_rn_noftz(_f2_167, _f2_154, _f2_166);
                    acc[10] = r_22.x;
                    acc[11] = r_22.y;
                    float2 _f2_168 = make_float2(acc[12], acc[13]);
                    float2 _f2_169 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                    float2 r_23 = fma_f32x2_rn_noftz(_f2_169, _f2_154, _f2_168);
                    acc[12] = r_23.x;
                    acc[13] = r_23.y;
                    float2 _f2_170 = make_float2(acc[14], acc[15]);
                    float2 _f2_171 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                    float2 r_24 = fma_f32x2_rn_noftz(_f2_171, _f2_154, _f2_170);
                    acc[14] = r_24.x;
                    acc[15] = r_24.y;
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
                    float2 _f2_172 = make_float2(acc[16], acc[17]);
                    float2 _f2_173 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                    float2 r_25 = fma_f32x2_rn_noftz(_f2_173, _f2_155, _f2_172);
                    acc[16] = r_25.x;
                    acc[17] = r_25.y;
                    float2 _f2_174 = make_float2(acc[18], acc[19]);
                    float2 _f2_175 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                    float2 r_26 = fma_f32x2_rn_noftz(_f2_175, _f2_155, _f2_174);
                    acc[18] = r_26.x;
                    acc[19] = r_26.y;
                    float2 _f2_176 = make_float2(acc[20], acc[21]);
                    float2 _f2_177 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                    float2 r_27 = fma_f32x2_rn_noftz(_f2_177, _f2_155, _f2_176);
                    acc[20] = r_27.x;
                    acc[21] = r_27.y;
                    float2 _f2_178 = make_float2(acc[22], acc[23]);
                    float2 _f2_179 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                    float2 r_28 = fma_f32x2_rn_noftz(_f2_179, _f2_155, _f2_178);
                    acc[22] = r_28.x;
                    acc[23] = r_28.y;
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
                    float2 _f2_180 = make_float2(acc[24], acc[25]);
                    float2 _f2_181 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                    float2 r_29 = fma_f32x2_rn_noftz(_f2_181, _f2_155, _f2_180);
                    acc[24] = r_29.x;
                    acc[25] = r_29.y;
                    float2 _f2_182 = make_float2(acc[26], acc[27]);
                    float2 _f2_183 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                    float2 r_30 = fma_f32x2_rn_noftz(_f2_183, _f2_155, _f2_182);
                    acc[26] = r_30.x;
                    acc[27] = r_30.y;
                    float2 _f2_184 = make_float2(acc[28], acc[29]);
                    float2 _f2_185 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                    float2 r_31 = fma_f32x2_rn_noftz(_f2_185, _f2_155, _f2_184);
                    acc[28] = r_31.x;
                    acc[29] = r_31.y;
                    float2 _f2_186 = make_float2(acc[30], acc[31]);
                    float2 _f2_187 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                    float2 r_32 = fma_f32x2_rn_noftz(_f2_187, _f2_155, _f2_186);
                    acc[30] = r_32.x;
                    acc[31] = r_32.y;
                    float2 _fp8x2_decode_8;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_8.x), "=f"(_fp8x2_decode_8.y) : "h"((uint16_t)_fp4_wire_scales_0[2]));
                    float _mul_16;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_fp8x2_decode_8.x), "f"(__uint_as_float(_sysv_poll32_group_2[0])));
                    float s0_33 = _mul_16;
                    float _mul_17;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_fp8x2_decode_8.y), "f"(__uint_as_float(_sysv_poll32_group_2[0])));
                    float s1_34 = _mul_17;
                    float2 _f2_188 = make_float2(s0_33, s0_33);
                    float2 _f2_189 = make_float2(s1_34, s1_34);
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
                    float2 _f2_190 = make_float2(acc[0], acc[1]);
                    float2 _f2_191 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                    float2 r_35 = fma_f32x2_rn_noftz(_f2_191, _f2_188, _f2_190);
                    acc[0] = r_35.x;
                    acc[1] = r_35.y;
                    float2 _f2_192 = make_float2(acc[2], acc[3]);
                    float2 _f2_193 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                    float2 r_36 = fma_f32x2_rn_noftz(_f2_193, _f2_188, _f2_192);
                    acc[2] = r_36.x;
                    acc[3] = r_36.y;
                    float2 _f2_194 = make_float2(acc[4], acc[5]);
                    float2 _f2_195 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                    float2 r_37 = fma_f32x2_rn_noftz(_f2_195, _f2_188, _f2_194);
                    acc[4] = r_37.x;
                    acc[5] = r_37.y;
                    float2 _f2_196 = make_float2(acc[6], acc[7]);
                    float2 _f2_197 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                    float2 r_38 = fma_f32x2_rn_noftz(_f2_197, _f2_188, _f2_196);
                    acc[6] = r_38.x;
                    acc[7] = r_38.y;
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
                    float2 _f2_198 = make_float2(acc[8], acc[9]);
                    float2 _f2_199 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                    float2 r_39 = fma_f32x2_rn_noftz(_f2_199, _f2_188, _f2_198);
                    acc[8] = r_39.x;
                    acc[9] = r_39.y;
                    float2 _f2_200 = make_float2(acc[10], acc[11]);
                    float2 _f2_201 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                    float2 r_40 = fma_f32x2_rn_noftz(_f2_201, _f2_188, _f2_200);
                    acc[10] = r_40.x;
                    acc[11] = r_40.y;
                    float2 _f2_202 = make_float2(acc[12], acc[13]);
                    float2 _f2_203 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                    float2 r_41 = fma_f32x2_rn_noftz(_f2_203, _f2_188, _f2_202);
                    acc[12] = r_41.x;
                    acc[13] = r_41.y;
                    float2 _f2_204 = make_float2(acc[14], acc[15]);
                    float2 _f2_205 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                    float2 r_42 = fma_f32x2_rn_noftz(_f2_205, _f2_188, _f2_204);
                    acc[14] = r_42.x;
                    acc[15] = r_42.y;
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
                    float2 _f2_206 = make_float2(acc[16], acc[17]);
                    float2 _f2_207 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                    float2 r_43 = fma_f32x2_rn_noftz(_f2_207, _f2_189, _f2_206);
                    acc[16] = r_43.x;
                    acc[17] = r_43.y;
                    float2 _f2_208 = make_float2(acc[18], acc[19]);
                    float2 _f2_209 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                    float2 r_44 = fma_f32x2_rn_noftz(_f2_209, _f2_189, _f2_208);
                    acc[18] = r_44.x;
                    acc[19] = r_44.y;
                    float2 _f2_210 = make_float2(acc[20], acc[21]);
                    float2 _f2_211 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                    float2 r_45 = fma_f32x2_rn_noftz(_f2_211, _f2_189, _f2_210);
                    acc[20] = r_45.x;
                    acc[21] = r_45.y;
                    float2 _f2_212 = make_float2(acc[22], acc[23]);
                    float2 _f2_213 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                    float2 r_46 = fma_f32x2_rn_noftz(_f2_213, _f2_189, _f2_212);
                    acc[22] = r_46.x;
                    acc[23] = r_46.y;
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
                    float2 _f2_214 = make_float2(acc[24], acc[25]);
                    float2 _f2_215 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                    float2 r_47 = fma_f32x2_rn_noftz(_f2_215, _f2_189, _f2_214);
                    acc[24] = r_47.x;
                    acc[25] = r_47.y;
                    float2 _f2_216 = make_float2(acc[26], acc[27]);
                    float2 _f2_217 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                    float2 r_48 = fma_f32x2_rn_noftz(_f2_217, _f2_189, _f2_216);
                    acc[26] = r_48.x;
                    acc[27] = r_48.y;
                    float2 _f2_218 = make_float2(acc[28], acc[29]);
                    float2 _f2_219 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                    float2 r_49 = fma_f32x2_rn_noftz(_f2_219, _f2_189, _f2_218);
                    acc[28] = r_49.x;
                    acc[29] = r_49.y;
                    float2 _f2_220 = make_float2(acc[30], acc[31]);
                    float2 _f2_221 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                    float2 r_50 = fma_f32x2_rn_noftz(_f2_221, _f2_189, _f2_220);
                    acc[30] = r_50.x;
                    acc[31] = r_50.y;
                    float2 _fp8x2_decode_9;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_9.x), "=f"(_fp8x2_decode_9.y) : "h"((uint16_t)_fp4_wire_scales_0[3]));
                    float _mul_18;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(_fp8x2_decode_9.x), "f"(__uint_as_float(_sysv_poll32_group_3[0])));
                    float s0_51 = _mul_18;
                    float _mul_19;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_fp8x2_decode_9.y), "f"(__uint_as_float(_sysv_poll32_group_3[0])));
                    float s1_52 = _mul_19;
                    float2 _f2_222 = make_float2(s0_51, s0_51);
                    float2 _f2_223 = make_float2(s1_52, s1_52);
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
                    float2 _f2_224 = make_float2(acc[0], acc[1]);
                    float2 _f2_225 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                    float2 r_53 = fma_f32x2_rn_noftz(_f2_225, _f2_222, _f2_224);
                    acc[0] = r_53.x;
                    acc[1] = r_53.y;
                    float2 _f2_226 = make_float2(acc[2], acc[3]);
                    float2 _f2_227 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                    float2 r_54 = fma_f32x2_rn_noftz(_f2_227, _f2_222, _f2_226);
                    acc[2] = r_54.x;
                    acc[3] = r_54.y;
                    float2 _f2_228 = make_float2(acc[4], acc[5]);
                    float2 _f2_229 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                    float2 r_55 = fma_f32x2_rn_noftz(_f2_229, _f2_222, _f2_228);
                    acc[4] = r_55.x;
                    acc[5] = r_55.y;
                    float2 _f2_230 = make_float2(acc[6], acc[7]);
                    float2 _f2_231 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                    float2 r_56 = fma_f32x2_rn_noftz(_f2_231, _f2_222, _f2_230);
                    acc[6] = r_56.x;
                    acc[7] = r_56.y;
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
                    float2 _f2_232 = make_float2(acc[8], acc[9]);
                    float2 _f2_233 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                    float2 r_57 = fma_f32x2_rn_noftz(_f2_233, _f2_222, _f2_232);
                    acc[8] = r_57.x;
                    acc[9] = r_57.y;
                    float2 _f2_234 = make_float2(acc[10], acc[11]);
                    float2 _f2_235 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                    float2 r_58 = fma_f32x2_rn_noftz(_f2_235, _f2_222, _f2_234);
                    acc[10] = r_58.x;
                    acc[11] = r_58.y;
                    float2 _f2_236 = make_float2(acc[12], acc[13]);
                    float2 _f2_237 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                    float2 r_59 = fma_f32x2_rn_noftz(_f2_237, _f2_222, _f2_236);
                    acc[12] = r_59.x;
                    acc[13] = r_59.y;
                    float2 _f2_238 = make_float2(acc[14], acc[15]);
                    float2 _f2_239 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                    float2 r_60 = fma_f32x2_rn_noftz(_f2_239, _f2_222, _f2_238);
                    acc[14] = r_60.x;
                    acc[15] = r_60.y;
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
                    float2 _f2_240 = make_float2(acc[16], acc[17]);
                    float2 _f2_241 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                    float2 r_61 = fma_f32x2_rn_noftz(_f2_241, _f2_223, _f2_240);
                    acc[16] = r_61.x;
                    acc[17] = r_61.y;
                    float2 _f2_242 = make_float2(acc[18], acc[19]);
                    float2 _f2_243 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                    float2 r_62 = fma_f32x2_rn_noftz(_f2_243, _f2_223, _f2_242);
                    acc[18] = r_62.x;
                    acc[19] = r_62.y;
                    float2 _f2_244 = make_float2(acc[20], acc[21]);
                    float2 _f2_245 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                    float2 r_63 = fma_f32x2_rn_noftz(_f2_245, _f2_223, _f2_244);
                    acc[20] = r_63.x;
                    acc[21] = r_63.y;
                    float2 _f2_246 = make_float2(acc[22], acc[23]);
                    float2 _f2_247 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                    float2 r_64 = fma_f32x2_rn_noftz(_f2_247, _f2_223, _f2_246);
                    acc[22] = r_64.x;
                    acc[23] = r_64.y;
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
                    float2 _f2_248 = make_float2(acc[24], acc[25]);
                    float2 _f2_249 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                    float2 r_65 = fma_f32x2_rn_noftz(_f2_249, _f2_223, _f2_248);
                    acc[24] = r_65.x;
                    acc[25] = r_65.y;
                    float2 _f2_250 = make_float2(acc[26], acc[27]);
                    float2 _f2_251 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                    float2 r_66 = fma_f32x2_rn_noftz(_f2_251, _f2_223, _f2_250);
                    acc[26] = r_66.x;
                    acc[27] = r_66.y;
                    float2 _f2_252 = make_float2(acc[28], acc[29]);
                    float2 _f2_253 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                    float2 r_67 = fma_f32x2_rn_noftz(_f2_253, _f2_223, _f2_252);
                    acc[28] = r_67.x;
                    acc[29] = r_67.y;
                    float2 _f2_254 = make_float2(acc[30], acc[31]);
                    float2 _f2_255 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                    float2 r_68 = fma_f32x2_rn_noftz(_f2_255, _f2_223, _f2_254);
                    acc[30] = r_68.x;
                    acc[31] = r_68.y;
                    float cur = am_1;
                    float _fabs_0 = fabsf(acc[0]);
                    float _fmax_6 = fmaxf(cur, _fabs_0);
                    cur = _fmax_6;
                    float _fabs_1 = fabsf(acc[1]);
                    float _fmax_7 = fmaxf(cur, _fabs_1);
                    cur = _fmax_7;
                    float _fabs_2 = fabsf(acc[2]);
                    float _fmax_8 = fmaxf(cur, _fabs_2);
                    cur = _fmax_8;
                    float _fabs_3 = fabsf(acc[3]);
                    float _fmax_9 = fmaxf(cur, _fabs_3);
                    cur = _fmax_9;
                    float _fabs_4 = fabsf(acc[4]);
                    float _fmax_10 = fmaxf(cur, _fabs_4);
                    cur = _fmax_10;
                    float _fabs_5 = fabsf(acc[5]);
                    float _fmax_11 = fmaxf(cur, _fabs_5);
                    cur = _fmax_11;
                    float _fabs_6 = fabsf(acc[6]);
                    float _fmax_12 = fmaxf(cur, _fabs_6);
                    cur = _fmax_12;
                    float _fabs_7 = fabsf(acc[7]);
                    float _fmax_13 = fmaxf(cur, _fabs_7);
                    cur = _fmax_13;
                    float _fabs_8 = fabsf(acc[8]);
                    float _fmax_14 = fmaxf(cur, _fabs_8);
                    cur = _fmax_14;
                    float _fabs_9 = fabsf(acc[9]);
                    float _fmax_15 = fmaxf(cur, _fabs_9);
                    cur = _fmax_15;
                    float _fabs_10 = fabsf(acc[10]);
                    float _fmax_16 = fmaxf(cur, _fabs_10);
                    cur = _fmax_16;
                    float _fabs_11 = fabsf(acc[11]);
                    float _fmax_17 = fmaxf(cur, _fabs_11);
                    cur = _fmax_17;
                    float _fabs_12 = fabsf(acc[12]);
                    float _fmax_18 = fmaxf(cur, _fabs_12);
                    cur = _fmax_18;
                    float _fabs_13 = fabsf(acc[13]);
                    float _fmax_19 = fmaxf(cur, _fabs_13);
                    cur = _fmax_19;
                    float _fabs_14 = fabsf(acc[14]);
                    float _fmax_20 = fmaxf(cur, _fabs_14);
                    cur = _fmax_20;
                    float _fabs_15 = fabsf(acc[15]);
                    float _fmax_21 = fmaxf(cur, _fabs_15);
                    cur = _fmax_21;
                    float _fabs_16 = fabsf(acc[16]);
                    float _fmax_22 = fmaxf(cur, _fabs_16);
                    cur = _fmax_22;
                    float _fabs_17 = fabsf(acc[17]);
                    float _fmax_23 = fmaxf(cur, _fabs_17);
                    cur = _fmax_23;
                    float _fabs_18 = fabsf(acc[18]);
                    float _fmax_24 = fmaxf(cur, _fabs_18);
                    cur = _fmax_24;
                    float _fabs_19 = fabsf(acc[19]);
                    float _fmax_25 = fmaxf(cur, _fabs_19);
                    cur = _fmax_25;
                    float _fabs_20 = fabsf(acc[20]);
                    float _fmax_26 = fmaxf(cur, _fabs_20);
                    cur = _fmax_26;
                    float _fabs_21 = fabsf(acc[21]);
                    float _fmax_27 = fmaxf(cur, _fabs_21);
                    cur = _fmax_27;
                    float _fabs_22 = fabsf(acc[22]);
                    float _fmax_28 = fmaxf(cur, _fabs_22);
                    cur = _fmax_28;
                    float _fabs_23 = fabsf(acc[23]);
                    float _fmax_29 = fmaxf(cur, _fabs_23);
                    cur = _fmax_29;
                    float _fabs_24 = fabsf(acc[24]);
                    float _fmax_30 = fmaxf(cur, _fabs_24);
                    cur = _fmax_30;
                    float _fabs_25 = fabsf(acc[25]);
                    float _fmax_31 = fmaxf(cur, _fabs_25);
                    cur = _fmax_31;
                    float _fabs_26 = fabsf(acc[26]);
                    float _fmax_32 = fmaxf(cur, _fabs_26);
                    cur = _fmax_32;
                    float _fabs_27 = fabsf(acc[27]);
                    float _fmax_33 = fmaxf(cur, _fabs_27);
                    cur = _fmax_33;
                    float _fabs_28 = fabsf(acc[28]);
                    float _fmax_34 = fmaxf(cur, _fabs_28);
                    cur = _fmax_34;
                    float _fabs_29 = fabsf(acc[29]);
                    float _fmax_35 = fmaxf(cur, _fabs_29);
                    cur = _fmax_35;
                    float _fabs_30 = fabsf(acc[30]);
                    float _fmax_36 = fmaxf(cur, _fabs_30);
                    cur = _fmax_36;
                    float _fabs_31 = fabsf(acc[31]);
                    float _fmax_37 = fmaxf(cur, _fabs_31);
                    cur = _fmax_37;
                    am_1 = cur;
                    float acc_69[32];
                    acc_69[0] = 0.0f;
                    acc_69[1] = 0.0f;
                    acc_69[2] = 0.0f;
                    acc_69[3] = 0.0f;
                    acc_69[4] = 0.0f;
                    acc_69[5] = 0.0f;
                    acc_69[6] = 0.0f;
                    acc_69[7] = 0.0f;
                    acc_69[8] = 0.0f;
                    acc_69[9] = 0.0f;
                    acc_69[10] = 0.0f;
                    acc_69[11] = 0.0f;
                    acc_69[12] = 0.0f;
                    acc_69[13] = 0.0f;
                    acc_69[14] = 0.0f;
                    acc_69[15] = 0.0f;
                    acc_69[16] = 0.0f;
                    acc_69[17] = 0.0f;
                    acc_69[18] = 0.0f;
                    acc_69[19] = 0.0f;
                    acc_69[20] = 0.0f;
                    acc_69[21] = 0.0f;
                    acc_69[22] = 0.0f;
                    acc_69[23] = 0.0f;
                    acc_69[24] = 0.0f;
                    acc_69[25] = 0.0f;
                    acc_69[26] = 0.0f;
                    acc_69[27] = 0.0f;
                    acc_69[28] = 0.0f;
                    acc_69[29] = 0.0f;
                    acc_69[30] = 0.0f;
                    acc_69[31] = 0.0f;
                    float2 _fp8x2_decode_10;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_10.x), "=f"(_fp8x2_decode_10.y) : "h"((uint16_t)_fp4_wire_scales_0[4]));
                    float _mul_20;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(_fp8x2_decode_10.x), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                    float s0_70 = _mul_20;
                    float _mul_21;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(_fp8x2_decode_10.y), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                    float s1_71 = _mul_21;
                    float2 _f2_256 = make_float2(s0_70, s0_70);
                    float2 _f2_257 = make_float2(s1_71, s1_71);
                    float _unpack_fp4_16[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[16 + 0]));
                        (&_unpack_fp4_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_16[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_16[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_16[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_16[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_16[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_16[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_16[0]), "=f"(_unpack_fp4_16[1]), "=f"(_unpack_fp4_16[2]), "=f"(_unpack_fp4_16[3]), "=f"(_unpack_fp4_16[4]), "=f"(_unpack_fp4_16[5]), "=f"(_unpack_fp4_16[6]), "=f"(_unpack_fp4_16[7]) : "r"(_fp4_wire_values_0[16 + 0]));
                        #endif
                    }
                    float2 _f2_258 = make_float2(acc_69[0], acc_69[1]);
                    float2 _f2_259 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                    float2 r_72 = fma_f32x2_rn_noftz(_f2_259, _f2_256, _f2_258);
                    acc_69[0] = r_72.x;
                    acc_69[1] = r_72.y;
                    float2 _f2_260 = make_float2(acc_69[2], acc_69[3]);
                    float2 _f2_261 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                    float2 r_73 = fma_f32x2_rn_noftz(_f2_261, _f2_256, _f2_260);
                    acc_69[2] = r_73.x;
                    acc_69[3] = r_73.y;
                    float2 _f2_262 = make_float2(acc_69[4], acc_69[5]);
                    float2 _f2_263 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                    float2 r_74 = fma_f32x2_rn_noftz(_f2_263, _f2_256, _f2_262);
                    acc_69[4] = r_74.x;
                    acc_69[5] = r_74.y;
                    float2 _f2_264 = make_float2(acc_69[6], acc_69[7]);
                    float2 _f2_265 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                    float2 r_75 = fma_f32x2_rn_noftz(_f2_265, _f2_256, _f2_264);
                    acc_69[6] = r_75.x;
                    acc_69[7] = r_75.y;
                    float _unpack_fp4_17[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[17 + 0]));
                        (&_unpack_fp4_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_17[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_17[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_17[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_17[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_17[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_17[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_17[0]), "=f"(_unpack_fp4_17[1]), "=f"(_unpack_fp4_17[2]), "=f"(_unpack_fp4_17[3]), "=f"(_unpack_fp4_17[4]), "=f"(_unpack_fp4_17[5]), "=f"(_unpack_fp4_17[6]), "=f"(_unpack_fp4_17[7]) : "r"(_fp4_wire_values_0[17 + 0]));
                        #endif
                    }
                    float2 _f2_266 = make_float2(acc_69[8], acc_69[9]);
                    float2 _f2_267 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                    float2 r_76 = fma_f32x2_rn_noftz(_f2_267, _f2_256, _f2_266);
                    acc_69[8] = r_76.x;
                    acc_69[9] = r_76.y;
                    float2 _f2_268 = make_float2(acc_69[10], acc_69[11]);
                    float2 _f2_269 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                    float2 r_77 = fma_f32x2_rn_noftz(_f2_269, _f2_256, _f2_268);
                    acc_69[10] = r_77.x;
                    acc_69[11] = r_77.y;
                    float2 _f2_270 = make_float2(acc_69[12], acc_69[13]);
                    float2 _f2_271 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                    float2 r_78 = fma_f32x2_rn_noftz(_f2_271, _f2_256, _f2_270);
                    acc_69[12] = r_78.x;
                    acc_69[13] = r_78.y;
                    float2 _f2_272 = make_float2(acc_69[14], acc_69[15]);
                    float2 _f2_273 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                    float2 r_79 = fma_f32x2_rn_noftz(_f2_273, _f2_256, _f2_272);
                    acc_69[14] = r_79.x;
                    acc_69[15] = r_79.y;
                    float _unpack_fp4_18[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[18 + 0]));
                        (&_unpack_fp4_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_18[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_18[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_18[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_18[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_18[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_18[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_18[0]), "=f"(_unpack_fp4_18[1]), "=f"(_unpack_fp4_18[2]), "=f"(_unpack_fp4_18[3]), "=f"(_unpack_fp4_18[4]), "=f"(_unpack_fp4_18[5]), "=f"(_unpack_fp4_18[6]), "=f"(_unpack_fp4_18[7]) : "r"(_fp4_wire_values_0[18 + 0]));
                        #endif
                    }
                    float2 _f2_274 = make_float2(acc_69[16], acc_69[17]);
                    float2 _f2_275 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                    float2 r_80 = fma_f32x2_rn_noftz(_f2_275, _f2_257, _f2_274);
                    acc_69[16] = r_80.x;
                    acc_69[17] = r_80.y;
                    float2 _f2_276 = make_float2(acc_69[18], acc_69[19]);
                    float2 _f2_277 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                    float2 r_81 = fma_f32x2_rn_noftz(_f2_277, _f2_257, _f2_276);
                    acc_69[18] = r_81.x;
                    acc_69[19] = r_81.y;
                    float2 _f2_278 = make_float2(acc_69[20], acc_69[21]);
                    float2 _f2_279 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                    float2 r_82 = fma_f32x2_rn_noftz(_f2_279, _f2_257, _f2_278);
                    acc_69[20] = r_82.x;
                    acc_69[21] = r_82.y;
                    float2 _f2_280 = make_float2(acc_69[22], acc_69[23]);
                    float2 _f2_281 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                    float2 r_83 = fma_f32x2_rn_noftz(_f2_281, _f2_257, _f2_280);
                    acc_69[22] = r_83.x;
                    acc_69[23] = r_83.y;
                    float _unpack_fp4_19[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[19 + 0]));
                        (&_unpack_fp4_19[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_19[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_19[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_19[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_19[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_19[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_19[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_19[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_19[0]), "=f"(_unpack_fp4_19[1]), "=f"(_unpack_fp4_19[2]), "=f"(_unpack_fp4_19[3]), "=f"(_unpack_fp4_19[4]), "=f"(_unpack_fp4_19[5]), "=f"(_unpack_fp4_19[6]), "=f"(_unpack_fp4_19[7]) : "r"(_fp4_wire_values_0[19 + 0]));
                        #endif
                    }
                    float2 _f2_282 = make_float2(acc_69[24], acc_69[25]);
                    float2 _f2_283 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                    float2 r_84 = fma_f32x2_rn_noftz(_f2_283, _f2_257, _f2_282);
                    acc_69[24] = r_84.x;
                    acc_69[25] = r_84.y;
                    float2 _f2_284 = make_float2(acc_69[26], acc_69[27]);
                    float2 _f2_285 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                    float2 r_85 = fma_f32x2_rn_noftz(_f2_285, _f2_257, _f2_284);
                    acc_69[26] = r_85.x;
                    acc_69[27] = r_85.y;
                    float2 _f2_286 = make_float2(acc_69[28], acc_69[29]);
                    float2 _f2_287 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                    float2 r_86 = fma_f32x2_rn_noftz(_f2_287, _f2_257, _f2_286);
                    acc_69[28] = r_86.x;
                    acc_69[29] = r_86.y;
                    float2 _f2_288 = make_float2(acc_69[30], acc_69[31]);
                    float2 _f2_289 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                    float2 r_87 = fma_f32x2_rn_noftz(_f2_289, _f2_257, _f2_288);
                    acc_69[30] = r_87.x;
                    acc_69[31] = r_87.y;
                    float2 _fp8x2_decode_11;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_11.x), "=f"(_fp8x2_decode_11.y) : "h"((uint16_t)_fp4_wire_scales_0[5]));
                    float _mul_22;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(_fp8x2_decode_11.x), "f"(__uint_as_float(_sysv_poll32_group_1[0])));
                    float s0_88 = _mul_22;
                    float _mul_23;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_fp8x2_decode_11.y), "f"(__uint_as_float(_sysv_poll32_group_1[0])));
                    float s1_89 = _mul_23;
                    float2 _f2_290 = make_float2(s0_88, s0_88);
                    float2 _f2_291 = make_float2(s1_89, s1_89);
                    float _unpack_fp4_20[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[20 + 0]));
                        (&_unpack_fp4_20[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_20[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_20[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_20[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_20[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_20[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_20[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_20[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_20[0]), "=f"(_unpack_fp4_20[1]), "=f"(_unpack_fp4_20[2]), "=f"(_unpack_fp4_20[3]), "=f"(_unpack_fp4_20[4]), "=f"(_unpack_fp4_20[5]), "=f"(_unpack_fp4_20[6]), "=f"(_unpack_fp4_20[7]) : "r"(_fp4_wire_values_0[20 + 0]));
                        #endif
                    }
                    float2 _f2_292 = make_float2(acc_69[0], acc_69[1]);
                    float2 _f2_293 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                    float2 r_90 = fma_f32x2_rn_noftz(_f2_293, _f2_290, _f2_292);
                    acc_69[0] = r_90.x;
                    acc_69[1] = r_90.y;
                    float2 _f2_294 = make_float2(acc_69[2], acc_69[3]);
                    float2 _f2_295 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                    float2 r_91 = fma_f32x2_rn_noftz(_f2_295, _f2_290, _f2_294);
                    acc_69[2] = r_91.x;
                    acc_69[3] = r_91.y;
                    float2 _f2_296 = make_float2(acc_69[4], acc_69[5]);
                    float2 _f2_297 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                    float2 r_92 = fma_f32x2_rn_noftz(_f2_297, _f2_290, _f2_296);
                    acc_69[4] = r_92.x;
                    acc_69[5] = r_92.y;
                    float2 _f2_298 = make_float2(acc_69[6], acc_69[7]);
                    float2 _f2_299 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                    float2 r_93 = fma_f32x2_rn_noftz(_f2_299, _f2_290, _f2_298);
                    acc_69[6] = r_93.x;
                    acc_69[7] = r_93.y;
                    float _unpack_fp4_21[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[21 + 0]));
                        (&_unpack_fp4_21[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_21[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_21[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_21[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_21[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_21[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_21[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_21[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_21[0]), "=f"(_unpack_fp4_21[1]), "=f"(_unpack_fp4_21[2]), "=f"(_unpack_fp4_21[3]), "=f"(_unpack_fp4_21[4]), "=f"(_unpack_fp4_21[5]), "=f"(_unpack_fp4_21[6]), "=f"(_unpack_fp4_21[7]) : "r"(_fp4_wire_values_0[21 + 0]));
                        #endif
                    }
                    float2 _f2_300 = make_float2(acc_69[8], acc_69[9]);
                    float2 _f2_301 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                    float2 r_94 = fma_f32x2_rn_noftz(_f2_301, _f2_290, _f2_300);
                    acc_69[8] = r_94.x;
                    acc_69[9] = r_94.y;
                    float2 _f2_302 = make_float2(acc_69[10], acc_69[11]);
                    float2 _f2_303 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                    float2 r_95 = fma_f32x2_rn_noftz(_f2_303, _f2_290, _f2_302);
                    acc_69[10] = r_95.x;
                    acc_69[11] = r_95.y;
                    float2 _f2_304 = make_float2(acc_69[12], acc_69[13]);
                    float2 _f2_305 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                    float2 r_96 = fma_f32x2_rn_noftz(_f2_305, _f2_290, _f2_304);
                    acc_69[12] = r_96.x;
                    acc_69[13] = r_96.y;
                    float2 _f2_306 = make_float2(acc_69[14], acc_69[15]);
                    float2 _f2_307 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                    float2 r_97 = fma_f32x2_rn_noftz(_f2_307, _f2_290, _f2_306);
                    acc_69[14] = r_97.x;
                    acc_69[15] = r_97.y;
                    float _unpack_fp4_22[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[22 + 0]));
                        (&_unpack_fp4_22[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_22[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_22[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_22[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_22[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_22[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_22[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_22[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_22[0]), "=f"(_unpack_fp4_22[1]), "=f"(_unpack_fp4_22[2]), "=f"(_unpack_fp4_22[3]), "=f"(_unpack_fp4_22[4]), "=f"(_unpack_fp4_22[5]), "=f"(_unpack_fp4_22[6]), "=f"(_unpack_fp4_22[7]) : "r"(_fp4_wire_values_0[22 + 0]));
                        #endif
                    }
                    float2 _f2_308 = make_float2(acc_69[16], acc_69[17]);
                    float2 _f2_309 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                    float2 r_98 = fma_f32x2_rn_noftz(_f2_309, _f2_291, _f2_308);
                    acc_69[16] = r_98.x;
                    acc_69[17] = r_98.y;
                    float2 _f2_310 = make_float2(acc_69[18], acc_69[19]);
                    float2 _f2_311 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                    float2 r_99 = fma_f32x2_rn_noftz(_f2_311, _f2_291, _f2_310);
                    acc_69[18] = r_99.x;
                    acc_69[19] = r_99.y;
                    float2 _f2_312 = make_float2(acc_69[20], acc_69[21]);
                    float2 _f2_313 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                    float2 r_100 = fma_f32x2_rn_noftz(_f2_313, _f2_291, _f2_312);
                    acc_69[20] = r_100.x;
                    acc_69[21] = r_100.y;
                    float2 _f2_314 = make_float2(acc_69[22], acc_69[23]);
                    float2 _f2_315 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                    float2 r_101 = fma_f32x2_rn_noftz(_f2_315, _f2_291, _f2_314);
                    acc_69[22] = r_101.x;
                    acc_69[23] = r_101.y;
                    float _unpack_fp4_23[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[23 + 0]));
                        (&_unpack_fp4_23[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_23[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_23[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_23[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_23[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_23[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_23[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_23[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_23[0]), "=f"(_unpack_fp4_23[1]), "=f"(_unpack_fp4_23[2]), "=f"(_unpack_fp4_23[3]), "=f"(_unpack_fp4_23[4]), "=f"(_unpack_fp4_23[5]), "=f"(_unpack_fp4_23[6]), "=f"(_unpack_fp4_23[7]) : "r"(_fp4_wire_values_0[23 + 0]));
                        #endif
                    }
                    float2 _f2_316 = make_float2(acc_69[24], acc_69[25]);
                    float2 _f2_317 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                    float2 r_102 = fma_f32x2_rn_noftz(_f2_317, _f2_291, _f2_316);
                    acc_69[24] = r_102.x;
                    acc_69[25] = r_102.y;
                    float2 _f2_318 = make_float2(acc_69[26], acc_69[27]);
                    float2 _f2_319 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                    float2 r_103 = fma_f32x2_rn_noftz(_f2_319, _f2_291, _f2_318);
                    acc_69[26] = r_103.x;
                    acc_69[27] = r_103.y;
                    float2 _f2_320 = make_float2(acc_69[28], acc_69[29]);
                    float2 _f2_321 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                    float2 r_104 = fma_f32x2_rn_noftz(_f2_321, _f2_291, _f2_320);
                    acc_69[28] = r_104.x;
                    acc_69[29] = r_104.y;
                    float2 _f2_322 = make_float2(acc_69[30], acc_69[31]);
                    float2 _f2_323 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                    float2 r_105 = fma_f32x2_rn_noftz(_f2_323, _f2_291, _f2_322);
                    acc_69[30] = r_105.x;
                    acc_69[31] = r_105.y;
                    float2 _fp8x2_decode_12;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_12.x), "=f"(_fp8x2_decode_12.y) : "h"((uint16_t)_fp4_wire_scales_0[6]));
                    float _mul_24;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(_fp8x2_decode_12.x), "f"(__uint_as_float(_sysv_poll32_group_2[0])));
                    float s0_106 = _mul_24;
                    float _mul_25;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(_fp8x2_decode_12.y), "f"(__uint_as_float(_sysv_poll32_group_2[0])));
                    float s1_107 = _mul_25;
                    float2 _f2_324 = make_float2(s0_106, s0_106);
                    float2 _f2_325 = make_float2(s1_107, s1_107);
                    float _unpack_fp4_24[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[24 + 0]));
                        (&_unpack_fp4_24[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_24[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_24[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_24[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_24[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_24[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_24[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_24[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_24[0]), "=f"(_unpack_fp4_24[1]), "=f"(_unpack_fp4_24[2]), "=f"(_unpack_fp4_24[3]), "=f"(_unpack_fp4_24[4]), "=f"(_unpack_fp4_24[5]), "=f"(_unpack_fp4_24[6]), "=f"(_unpack_fp4_24[7]) : "r"(_fp4_wire_values_0[24 + 0]));
                        #endif
                    }
                    float2 _f2_326 = make_float2(acc_69[0], acc_69[1]);
                    float2 _f2_327 = make_float2(_unpack_fp4_24[0], _unpack_fp4_24[1]);
                    float2 r_108 = fma_f32x2_rn_noftz(_f2_327, _f2_324, _f2_326);
                    acc_69[0] = r_108.x;
                    acc_69[1] = r_108.y;
                    float2 _f2_328 = make_float2(acc_69[2], acc_69[3]);
                    float2 _f2_329 = make_float2(_unpack_fp4_24[2], _unpack_fp4_24[3]);
                    float2 r_109 = fma_f32x2_rn_noftz(_f2_329, _f2_324, _f2_328);
                    acc_69[2] = r_109.x;
                    acc_69[3] = r_109.y;
                    float2 _f2_330 = make_float2(acc_69[4], acc_69[5]);
                    float2 _f2_331 = make_float2(_unpack_fp4_24[4], _unpack_fp4_24[5]);
                    float2 r_110 = fma_f32x2_rn_noftz(_f2_331, _f2_324, _f2_330);
                    acc_69[4] = r_110.x;
                    acc_69[5] = r_110.y;
                    float2 _f2_332 = make_float2(acc_69[6], acc_69[7]);
                    float2 _f2_333 = make_float2(_unpack_fp4_24[6], _unpack_fp4_24[7]);
                    float2 r_111 = fma_f32x2_rn_noftz(_f2_333, _f2_324, _f2_332);
                    acc_69[6] = r_111.x;
                    acc_69[7] = r_111.y;
                    float _unpack_fp4_25[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[25 + 0]));
                        (&_unpack_fp4_25[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_25[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_25[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_25[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_25[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_25[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_25[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_25[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_25[0]), "=f"(_unpack_fp4_25[1]), "=f"(_unpack_fp4_25[2]), "=f"(_unpack_fp4_25[3]), "=f"(_unpack_fp4_25[4]), "=f"(_unpack_fp4_25[5]), "=f"(_unpack_fp4_25[6]), "=f"(_unpack_fp4_25[7]) : "r"(_fp4_wire_values_0[25 + 0]));
                        #endif
                    }
                    float2 _f2_334 = make_float2(acc_69[8], acc_69[9]);
                    float2 _f2_335 = make_float2(_unpack_fp4_25[0], _unpack_fp4_25[1]);
                    float2 r_112 = fma_f32x2_rn_noftz(_f2_335, _f2_324, _f2_334);
                    acc_69[8] = r_112.x;
                    acc_69[9] = r_112.y;
                    float2 _f2_336 = make_float2(acc_69[10], acc_69[11]);
                    float2 _f2_337 = make_float2(_unpack_fp4_25[2], _unpack_fp4_25[3]);
                    float2 r_113 = fma_f32x2_rn_noftz(_f2_337, _f2_324, _f2_336);
                    acc_69[10] = r_113.x;
                    acc_69[11] = r_113.y;
                    float2 _f2_338 = make_float2(acc_69[12], acc_69[13]);
                    float2 _f2_339 = make_float2(_unpack_fp4_25[4], _unpack_fp4_25[5]);
                    float2 r_114 = fma_f32x2_rn_noftz(_f2_339, _f2_324, _f2_338);
                    acc_69[12] = r_114.x;
                    acc_69[13] = r_114.y;
                    float2 _f2_340 = make_float2(acc_69[14], acc_69[15]);
                    float2 _f2_341 = make_float2(_unpack_fp4_25[6], _unpack_fp4_25[7]);
                    float2 r_115 = fma_f32x2_rn_noftz(_f2_341, _f2_324, _f2_340);
                    acc_69[14] = r_115.x;
                    acc_69[15] = r_115.y;
                    float _unpack_fp4_26[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[26 + 0]));
                        (&_unpack_fp4_26[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_26[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_26[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_26[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_26[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_26[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_26[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_26[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_26[0]), "=f"(_unpack_fp4_26[1]), "=f"(_unpack_fp4_26[2]), "=f"(_unpack_fp4_26[3]), "=f"(_unpack_fp4_26[4]), "=f"(_unpack_fp4_26[5]), "=f"(_unpack_fp4_26[6]), "=f"(_unpack_fp4_26[7]) : "r"(_fp4_wire_values_0[26 + 0]));
                        #endif
                    }
                    float2 _f2_342 = make_float2(acc_69[16], acc_69[17]);
                    float2 _f2_343 = make_float2(_unpack_fp4_26[0], _unpack_fp4_26[1]);
                    float2 r_116 = fma_f32x2_rn_noftz(_f2_343, _f2_325, _f2_342);
                    acc_69[16] = r_116.x;
                    acc_69[17] = r_116.y;
                    float2 _f2_344 = make_float2(acc_69[18], acc_69[19]);
                    float2 _f2_345 = make_float2(_unpack_fp4_26[2], _unpack_fp4_26[3]);
                    float2 r_117 = fma_f32x2_rn_noftz(_f2_345, _f2_325, _f2_344);
                    acc_69[18] = r_117.x;
                    acc_69[19] = r_117.y;
                    float2 _f2_346 = make_float2(acc_69[20], acc_69[21]);
                    float2 _f2_347 = make_float2(_unpack_fp4_26[4], _unpack_fp4_26[5]);
                    float2 r_118 = fma_f32x2_rn_noftz(_f2_347, _f2_325, _f2_346);
                    acc_69[20] = r_118.x;
                    acc_69[21] = r_118.y;
                    float2 _f2_348 = make_float2(acc_69[22], acc_69[23]);
                    float2 _f2_349 = make_float2(_unpack_fp4_26[6], _unpack_fp4_26[7]);
                    float2 r_119 = fma_f32x2_rn_noftz(_f2_349, _f2_325, _f2_348);
                    acc_69[22] = r_119.x;
                    acc_69[23] = r_119.y;
                    float _unpack_fp4_27[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[27 + 0]));
                        (&_unpack_fp4_27[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_27[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_27[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_27[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_27[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_27[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_27[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_27[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_27[0]), "=f"(_unpack_fp4_27[1]), "=f"(_unpack_fp4_27[2]), "=f"(_unpack_fp4_27[3]), "=f"(_unpack_fp4_27[4]), "=f"(_unpack_fp4_27[5]), "=f"(_unpack_fp4_27[6]), "=f"(_unpack_fp4_27[7]) : "r"(_fp4_wire_values_0[27 + 0]));
                        #endif
                    }
                    float2 _f2_350 = make_float2(acc_69[24], acc_69[25]);
                    float2 _f2_351 = make_float2(_unpack_fp4_27[0], _unpack_fp4_27[1]);
                    float2 r_120 = fma_f32x2_rn_noftz(_f2_351, _f2_325, _f2_350);
                    acc_69[24] = r_120.x;
                    acc_69[25] = r_120.y;
                    float2 _f2_352 = make_float2(acc_69[26], acc_69[27]);
                    float2 _f2_353 = make_float2(_unpack_fp4_27[2], _unpack_fp4_27[3]);
                    float2 r_121 = fma_f32x2_rn_noftz(_f2_353, _f2_325, _f2_352);
                    acc_69[26] = r_121.x;
                    acc_69[27] = r_121.y;
                    float2 _f2_354 = make_float2(acc_69[28], acc_69[29]);
                    float2 _f2_355 = make_float2(_unpack_fp4_27[4], _unpack_fp4_27[5]);
                    float2 r_122 = fma_f32x2_rn_noftz(_f2_355, _f2_325, _f2_354);
                    acc_69[28] = r_122.x;
                    acc_69[29] = r_122.y;
                    float2 _f2_356 = make_float2(acc_69[30], acc_69[31]);
                    float2 _f2_357 = make_float2(_unpack_fp4_27[6], _unpack_fp4_27[7]);
                    float2 r_123 = fma_f32x2_rn_noftz(_f2_357, _f2_325, _f2_356);
                    acc_69[30] = r_123.x;
                    acc_69[31] = r_123.y;
                    float2 _fp8x2_decode_13;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_13.x), "=f"(_fp8x2_decode_13.y) : "h"((uint16_t)_fp4_wire_scales_0[7]));
                    float _mul_26;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(_fp8x2_decode_13.x), "f"(__uint_as_float(_sysv_poll32_group_3[0])));
                    float s0_124 = _mul_26;
                    float _mul_27;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(_fp8x2_decode_13.y), "f"(__uint_as_float(_sysv_poll32_group_3[0])));
                    float s1_125 = _mul_27;
                    float2 _f2_358 = make_float2(s0_124, s0_124);
                    float2 _f2_359 = make_float2(s1_125, s1_125);
                    float _unpack_fp4_28[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[28 + 0]));
                        (&_unpack_fp4_28[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_28[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_28[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_28[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_28[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_28[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_28[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_28[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_28[0]), "=f"(_unpack_fp4_28[1]), "=f"(_unpack_fp4_28[2]), "=f"(_unpack_fp4_28[3]), "=f"(_unpack_fp4_28[4]), "=f"(_unpack_fp4_28[5]), "=f"(_unpack_fp4_28[6]), "=f"(_unpack_fp4_28[7]) : "r"(_fp4_wire_values_0[28 + 0]));
                        #endif
                    }
                    float2 _f2_360 = make_float2(acc_69[0], acc_69[1]);
                    float2 _f2_361 = make_float2(_unpack_fp4_28[0], _unpack_fp4_28[1]);
                    float2 r_126 = fma_f32x2_rn_noftz(_f2_361, _f2_358, _f2_360);
                    acc_69[0] = r_126.x;
                    acc_69[1] = r_126.y;
                    float2 _f2_362 = make_float2(acc_69[2], acc_69[3]);
                    float2 _f2_363 = make_float2(_unpack_fp4_28[2], _unpack_fp4_28[3]);
                    float2 r_127 = fma_f32x2_rn_noftz(_f2_363, _f2_358, _f2_362);
                    acc_69[2] = r_127.x;
                    acc_69[3] = r_127.y;
                    float2 _f2_364 = make_float2(acc_69[4], acc_69[5]);
                    float2 _f2_365 = make_float2(_unpack_fp4_28[4], _unpack_fp4_28[5]);
                    float2 r_128 = fma_f32x2_rn_noftz(_f2_365, _f2_358, _f2_364);
                    acc_69[4] = r_128.x;
                    acc_69[5] = r_128.y;
                    float2 _f2_366 = make_float2(acc_69[6], acc_69[7]);
                    float2 _f2_367 = make_float2(_unpack_fp4_28[6], _unpack_fp4_28[7]);
                    float2 r_129 = fma_f32x2_rn_noftz(_f2_367, _f2_358, _f2_366);
                    acc_69[6] = r_129.x;
                    acc_69[7] = r_129.y;
                    float _unpack_fp4_29[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[29 + 0]));
                        (&_unpack_fp4_29[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_29[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_29[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_29[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_29[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_29[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_29[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_29[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_29[0]), "=f"(_unpack_fp4_29[1]), "=f"(_unpack_fp4_29[2]), "=f"(_unpack_fp4_29[3]), "=f"(_unpack_fp4_29[4]), "=f"(_unpack_fp4_29[5]), "=f"(_unpack_fp4_29[6]), "=f"(_unpack_fp4_29[7]) : "r"(_fp4_wire_values_0[29 + 0]));
                        #endif
                    }
                    float2 _f2_368 = make_float2(acc_69[8], acc_69[9]);
                    float2 _f2_369 = make_float2(_unpack_fp4_29[0], _unpack_fp4_29[1]);
                    float2 r_130 = fma_f32x2_rn_noftz(_f2_369, _f2_358, _f2_368);
                    acc_69[8] = r_130.x;
                    acc_69[9] = r_130.y;
                    float2 _f2_370 = make_float2(acc_69[10], acc_69[11]);
                    float2 _f2_371 = make_float2(_unpack_fp4_29[2], _unpack_fp4_29[3]);
                    float2 r_131 = fma_f32x2_rn_noftz(_f2_371, _f2_358, _f2_370);
                    acc_69[10] = r_131.x;
                    acc_69[11] = r_131.y;
                    float2 _f2_372 = make_float2(acc_69[12], acc_69[13]);
                    float2 _f2_373 = make_float2(_unpack_fp4_29[4], _unpack_fp4_29[5]);
                    float2 r_132 = fma_f32x2_rn_noftz(_f2_373, _f2_358, _f2_372);
                    acc_69[12] = r_132.x;
                    acc_69[13] = r_132.y;
                    float2 _f2_374 = make_float2(acc_69[14], acc_69[15]);
                    float2 _f2_375 = make_float2(_unpack_fp4_29[6], _unpack_fp4_29[7]);
                    float2 r_133 = fma_f32x2_rn_noftz(_f2_375, _f2_358, _f2_374);
                    acc_69[14] = r_133.x;
                    acc_69[15] = r_133.y;
                    float _unpack_fp4_30[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[30 + 0]));
                        (&_unpack_fp4_30[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_30[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_30[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_30[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_30[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_30[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_30[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_30[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_30[0]), "=f"(_unpack_fp4_30[1]), "=f"(_unpack_fp4_30[2]), "=f"(_unpack_fp4_30[3]), "=f"(_unpack_fp4_30[4]), "=f"(_unpack_fp4_30[5]), "=f"(_unpack_fp4_30[6]), "=f"(_unpack_fp4_30[7]) : "r"(_fp4_wire_values_0[30 + 0]));
                        #endif
                    }
                    float2 _f2_376 = make_float2(acc_69[16], acc_69[17]);
                    float2 _f2_377 = make_float2(_unpack_fp4_30[0], _unpack_fp4_30[1]);
                    float2 r_134 = fma_f32x2_rn_noftz(_f2_377, _f2_359, _f2_376);
                    acc_69[16] = r_134.x;
                    acc_69[17] = r_134.y;
                    float2 _f2_378 = make_float2(acc_69[18], acc_69[19]);
                    float2 _f2_379 = make_float2(_unpack_fp4_30[2], _unpack_fp4_30[3]);
                    float2 r_135 = fma_f32x2_rn_noftz(_f2_379, _f2_359, _f2_378);
                    acc_69[18] = r_135.x;
                    acc_69[19] = r_135.y;
                    float2 _f2_380 = make_float2(acc_69[20], acc_69[21]);
                    float2 _f2_381 = make_float2(_unpack_fp4_30[4], _unpack_fp4_30[5]);
                    float2 r_136 = fma_f32x2_rn_noftz(_f2_381, _f2_359, _f2_380);
                    acc_69[20] = r_136.x;
                    acc_69[21] = r_136.y;
                    float2 _f2_382 = make_float2(acc_69[22], acc_69[23]);
                    float2 _f2_383 = make_float2(_unpack_fp4_30[6], _unpack_fp4_30[7]);
                    float2 r_137 = fma_f32x2_rn_noftz(_f2_383, _f2_359, _f2_382);
                    acc_69[22] = r_137.x;
                    acc_69[23] = r_137.y;
                    float _unpack_fp4_31[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[31 + 0]));
                        (&_unpack_fp4_31[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_31[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_31[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_31[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_31[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_31[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_31[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_31[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_31[0]), "=f"(_unpack_fp4_31[1]), "=f"(_unpack_fp4_31[2]), "=f"(_unpack_fp4_31[3]), "=f"(_unpack_fp4_31[4]), "=f"(_unpack_fp4_31[5]), "=f"(_unpack_fp4_31[6]), "=f"(_unpack_fp4_31[7]) : "r"(_fp4_wire_values_0[31 + 0]));
                        #endif
                    }
                    float2 _f2_384 = make_float2(acc_69[24], acc_69[25]);
                    float2 _f2_385 = make_float2(_unpack_fp4_31[0], _unpack_fp4_31[1]);
                    float2 r_138 = fma_f32x2_rn_noftz(_f2_385, _f2_359, _f2_384);
                    acc_69[24] = r_138.x;
                    acc_69[25] = r_138.y;
                    float2 _f2_386 = make_float2(acc_69[26], acc_69[27]);
                    float2 _f2_387 = make_float2(_unpack_fp4_31[2], _unpack_fp4_31[3]);
                    float2 r_139 = fma_f32x2_rn_noftz(_f2_387, _f2_359, _f2_386);
                    acc_69[26] = r_139.x;
                    acc_69[27] = r_139.y;
                    float2 _f2_388 = make_float2(acc_69[28], acc_69[29]);
                    float2 _f2_389 = make_float2(_unpack_fp4_31[4], _unpack_fp4_31[5]);
                    float2 r_140 = fma_f32x2_rn_noftz(_f2_389, _f2_359, _f2_388);
                    acc_69[28] = r_140.x;
                    acc_69[29] = r_140.y;
                    float2 _f2_390 = make_float2(acc_69[30], acc_69[31]);
                    float2 _f2_391 = make_float2(_unpack_fp4_31[6], _unpack_fp4_31[7]);
                    float2 r_141 = fma_f32x2_rn_noftz(_f2_391, _f2_359, _f2_390);
                    acc_69[30] = r_141.x;
                    acc_69[31] = r_141.y;
                    float cur_142 = am_1;
                    float _fabs_32 = fabsf(acc_69[0]);
                    float _fmax_38 = fmaxf(cur_142, _fabs_32);
                    cur_142 = _fmax_38;
                    float _fabs_33 = fabsf(acc_69[1]);
                    float _fmax_39 = fmaxf(cur_142, _fabs_33);
                    cur_142 = _fmax_39;
                    float _fabs_34 = fabsf(acc_69[2]);
                    float _fmax_40 = fmaxf(cur_142, _fabs_34);
                    cur_142 = _fmax_40;
                    float _fabs_35 = fabsf(acc_69[3]);
                    float _fmax_41 = fmaxf(cur_142, _fabs_35);
                    cur_142 = _fmax_41;
                    float _fabs_36 = fabsf(acc_69[4]);
                    float _fmax_42 = fmaxf(cur_142, _fabs_36);
                    cur_142 = _fmax_42;
                    float _fabs_37 = fabsf(acc_69[5]);
                    float _fmax_43 = fmaxf(cur_142, _fabs_37);
                    cur_142 = _fmax_43;
                    float _fabs_38 = fabsf(acc_69[6]);
                    float _fmax_44 = fmaxf(cur_142, _fabs_38);
                    cur_142 = _fmax_44;
                    float _fabs_39 = fabsf(acc_69[7]);
                    float _fmax_45 = fmaxf(cur_142, _fabs_39);
                    cur_142 = _fmax_45;
                    float _fabs_40 = fabsf(acc_69[8]);
                    float _fmax_46 = fmaxf(cur_142, _fabs_40);
                    cur_142 = _fmax_46;
                    float _fabs_41 = fabsf(acc_69[9]);
                    float _fmax_47 = fmaxf(cur_142, _fabs_41);
                    cur_142 = _fmax_47;
                    float _fabs_42 = fabsf(acc_69[10]);
                    float _fmax_48 = fmaxf(cur_142, _fabs_42);
                    cur_142 = _fmax_48;
                    float _fabs_43 = fabsf(acc_69[11]);
                    float _fmax_49 = fmaxf(cur_142, _fabs_43);
                    cur_142 = _fmax_49;
                    float _fabs_44 = fabsf(acc_69[12]);
                    float _fmax_50 = fmaxf(cur_142, _fabs_44);
                    cur_142 = _fmax_50;
                    float _fabs_45 = fabsf(acc_69[13]);
                    float _fmax_51 = fmaxf(cur_142, _fabs_45);
                    cur_142 = _fmax_51;
                    float _fabs_46 = fabsf(acc_69[14]);
                    float _fmax_52 = fmaxf(cur_142, _fabs_46);
                    cur_142 = _fmax_52;
                    float _fabs_47 = fabsf(acc_69[15]);
                    float _fmax_53 = fmaxf(cur_142, _fabs_47);
                    cur_142 = _fmax_53;
                    float _fabs_48 = fabsf(acc_69[16]);
                    float _fmax_54 = fmaxf(cur_142, _fabs_48);
                    cur_142 = _fmax_54;
                    float _fabs_49 = fabsf(acc_69[17]);
                    float _fmax_55 = fmaxf(cur_142, _fabs_49);
                    cur_142 = _fmax_55;
                    float _fabs_50 = fabsf(acc_69[18]);
                    float _fmax_56 = fmaxf(cur_142, _fabs_50);
                    cur_142 = _fmax_56;
                    float _fabs_51 = fabsf(acc_69[19]);
                    float _fmax_57 = fmaxf(cur_142, _fabs_51);
                    cur_142 = _fmax_57;
                    float _fabs_52 = fabsf(acc_69[20]);
                    float _fmax_58 = fmaxf(cur_142, _fabs_52);
                    cur_142 = _fmax_58;
                    float _fabs_53 = fabsf(acc_69[21]);
                    float _fmax_59 = fmaxf(cur_142, _fabs_53);
                    cur_142 = _fmax_59;
                    float _fabs_54 = fabsf(acc_69[22]);
                    float _fmax_60 = fmaxf(cur_142, _fabs_54);
                    cur_142 = _fmax_60;
                    float _fabs_55 = fabsf(acc_69[23]);
                    float _fmax_61 = fmaxf(cur_142, _fabs_55);
                    cur_142 = _fmax_61;
                    float _fabs_56 = fabsf(acc_69[24]);
                    float _fmax_62 = fmaxf(cur_142, _fabs_56);
                    cur_142 = _fmax_62;
                    float _fabs_57 = fabsf(acc_69[25]);
                    float _fmax_63 = fmaxf(cur_142, _fabs_57);
                    cur_142 = _fmax_63;
                    float _fabs_58 = fabsf(acc_69[26]);
                    float _fmax_64 = fmaxf(cur_142, _fabs_58);
                    cur_142 = _fmax_64;
                    float _fabs_59 = fabsf(acc_69[27]);
                    float _fmax_65 = fmaxf(cur_142, _fabs_59);
                    cur_142 = _fmax_65;
                    float _fabs_60 = fabsf(acc_69[28]);
                    float _fmax_66 = fmaxf(cur_142, _fabs_60);
                    cur_142 = _fmax_66;
                    float _fabs_61 = fabsf(acc_69[29]);
                    float _fmax_67 = fmaxf(cur_142, _fabs_61);
                    cur_142 = _fmax_67;
                    float _fabs_62 = fabsf(acc_69[30]);
                    float _fmax_68 = fmaxf(cur_142, _fabs_62);
                    cur_142 = _fmax_68;
                    float _fabs_63 = fabsf(acc_69[31]);
                    float _fmax_69 = fmaxf(cur_142, _fabs_63);
                    cur_142 = _fmax_69;
                    am_1 = cur_142;
                }
                float rs2 = 1.0f;
                float _warp_reduce_1 = am_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_1 = fmaxf(_warp_reduce_1, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset));
                am_1 = _warp_reduce_1;
                float zero_1 = 0.0f;
                float target_1 = 2688.0f;
                float _fdiv_rn_26 = __fdiv_rn(target_1, am_1);
                float inv_1 = ((am_1 > zero_1) ? _fdiv_rn_26 : zero_1);
                float _fdiv_rn_27 = __fdiv_rn(am_1, target_1);
                float rs_1 = _fdiv_rn_27;
                rs2 = rs_1;
                #pragma unroll 1
                for (int j = 0; j < 6; j++) {
                    int pp_1 = lane_1 + 32 * j;
                    unsigned int _vec_load_0[4];
                    {
                        uint32_t _uv4_137_0;
                        uint32_t _uv4_137_1;
                        uint32_t _uv4_137_2;
                        uint32_t _uv4_137_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_137_0), "=r"(_uv4_137_1), "=r"(_uv4_137_2), "=r"(_uv4_137_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + bix64 * 3072 + (long long)(16 * pp_1))) + (0))) : "memory");
                        _vec_load_0[0 + 0] = _uv4_137_0;
                        _vec_load_0[0 + 1] = _uv4_137_1;
                        _vec_load_0[0 + 2] = _uv4_137_2;
                        _vec_load_0[0 + 3] = _uv4_137_3;
                    }
                    unsigned int _vec_load_1[1];
                    {
                        uint16_t _scalar_bits_138;
                        asm volatile("ld.global.cg.u16 %0, [%1];"
                            : "=h"(_scalar_bits_138) : "l"((const void*)(reinterpret_cast<uint16_t*>(self_u8 + (rs_base + 50331648 + bix64 * 384 + (long long)(2 * pp_1))) + (0))) : "memory");
                        _vec_load_1[0] = (uint16_t)_scalar_bits_138;
                    }
                    unsigned int _vec_load_2[4];
                    {
                        uint32_t _uv4_139_0;
                        uint32_t _uv4_139_1;
                        uint32_t _uv4_139_2;
                        uint32_t _uv4_139_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_139_0), "=r"(_uv4_139_1), "=r"(_uv4_139_2), "=r"(_uv4_139_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + (4096 + bix64) * 3072 + (long long)(16 * pp_1))) + (0))) : "memory");
                        _vec_load_2[0 + 0] = _uv4_139_0;
                        _vec_load_2[0 + 1] = _uv4_139_1;
                        _vec_load_2[0 + 2] = _uv4_139_2;
                        _vec_load_2[0 + 3] = _uv4_139_3;
                    }
                    unsigned int _vec_load_3[1];
                    {
                        uint16_t _scalar_bits_140;
                        asm volatile("ld.global.cg.u16 %0, [%1];"
                            : "=h"(_scalar_bits_140) : "l"((const void*)(reinterpret_cast<uint16_t*>(self_u8 + (rs_base + 50331648 + (4096 + bix64) * 384 + (long long)(2 * pp_1))) + (0))) : "memory");
                        _vec_load_3[0] = (uint16_t)_scalar_bits_140;
                    }
                    unsigned int _vec_load_4[4];
                    {
                        uint32_t _uv4_141_0;
                        uint32_t _uv4_141_1;
                        uint32_t _uv4_141_2;
                        uint32_t _uv4_141_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_141_0), "=r"(_uv4_141_1), "=r"(_uv4_141_2), "=r"(_uv4_141_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + (8192 + bix64) * 3072 + (long long)(16 * pp_1))) + (0))) : "memory");
                        _vec_load_4[0 + 0] = _uv4_141_0;
                        _vec_load_4[0 + 1] = _uv4_141_1;
                        _vec_load_4[0 + 2] = _uv4_141_2;
                        _vec_load_4[0 + 3] = _uv4_141_3;
                    }
                    unsigned int _vec_load_5[1];
                    {
                        uint16_t _scalar_bits_142;
                        asm volatile("ld.global.cg.u16 %0, [%1];"
                            : "=h"(_scalar_bits_142) : "l"((const void*)(reinterpret_cast<uint16_t*>(self_u8 + (rs_base + 50331648 + (8192 + bix64) * 384 + (long long)(2 * pp_1))) + (0))) : "memory");
                        _vec_load_5[0] = (uint16_t)_scalar_bits_142;
                    }
                    unsigned int _vec_load_6[4];
                    {
                        uint32_t _uv4_143_0;
                        uint32_t _uv4_143_1;
                        uint32_t _uv4_143_2;
                        uint32_t _uv4_143_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_143_0), "=r"(_uv4_143_1), "=r"(_uv4_143_2), "=r"(_uv4_143_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(self_u8 + (rs_base + (12288 + bix64) * 3072 + (long long)(16 * pp_1))) + (0))) : "memory");
                        _vec_load_6[0 + 0] = _uv4_143_0;
                        _vec_load_6[0 + 1] = _uv4_143_1;
                        _vec_load_6[0 + 2] = _uv4_143_2;
                        _vec_load_6[0 + 3] = _uv4_143_3;
                    }
                    unsigned int _vec_load_7[1];
                    {
                        uint16_t _scalar_bits_144;
                        asm volatile("ld.global.cg.u16 %0, [%1];"
                            : "=h"(_scalar_bits_144) : "l"((const void*)(reinterpret_cast<uint16_t*>(self_u8 + (rs_base + 50331648 + (12288 + bix64) * 384 + (long long)(2 * pp_1))) + (0))) : "memory");
                        _vec_load_7[0] = (uint16_t)_scalar_bits_144;
                    }
                    float acc2[32];
                    acc2[0] = 0.0f;
                    acc2[1] = 0.0f;
                    acc2[2] = 0.0f;
                    acc2[3] = 0.0f;
                    acc2[4] = 0.0f;
                    acc2[5] = 0.0f;
                    acc2[6] = 0.0f;
                    acc2[7] = 0.0f;
                    acc2[8] = 0.0f;
                    acc2[9] = 0.0f;
                    acc2[10] = 0.0f;
                    acc2[11] = 0.0f;
                    acc2[12] = 0.0f;
                    acc2[13] = 0.0f;
                    acc2[14] = 0.0f;
                    acc2[15] = 0.0f;
                    acc2[16] = 0.0f;
                    acc2[17] = 0.0f;
                    acc2[18] = 0.0f;
                    acc2[19] = 0.0f;
                    acc2[20] = 0.0f;
                    acc2[21] = 0.0f;
                    acc2[22] = 0.0f;
                    acc2[23] = 0.0f;
                    acc2[24] = 0.0f;
                    acc2[25] = 0.0f;
                    acc2[26] = 0.0f;
                    acc2[27] = 0.0f;
                    acc2[28] = 0.0f;
                    acc2[29] = 0.0f;
                    acc2[30] = 0.0f;
                    acc2[31] = 0.0f;
                    float2 _fp8x2_decode_14;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_14.x), "=f"(_fp8x2_decode_14.y) : "h"((uint16_t)_vec_load_1[0]));
                    float _mul_28;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(_fp8x2_decode_14.x), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                    float s0_1 = _mul_28;
                    float _mul_29;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(_fp8x2_decode_14.y), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                    float s1_1 = _mul_29;
                    float2 _f2_392 = make_float2(s0_1, s0_1);
                    float2 _f2_393 = make_float2(s1_1, s1_1);
                    float _unpack_fp4_32[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_0[0]));
                        (&_unpack_fp4_32[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_32[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_32[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_32[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_32[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_32[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_32[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_32[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_32[0]), "=f"(_unpack_fp4_32[1]), "=f"(_unpack_fp4_32[2]), "=f"(_unpack_fp4_32[3]), "=f"(_unpack_fp4_32[4]), "=f"(_unpack_fp4_32[5]), "=f"(_unpack_fp4_32[6]), "=f"(_unpack_fp4_32[7]) : "r"(_vec_load_0[0]));
                        #endif
                    }
                    float2 _f2_394 = make_float2(acc2[0], acc2[1]);
                    float2 _f2_395 = make_float2(_unpack_fp4_32[0], _unpack_fp4_32[1]);
                    float2 r_15 = fma_f32x2_rn_noftz(_f2_395, _f2_392, _f2_394);
                    acc2[0] = r_15.x;
                    acc2[1] = r_15.y;
                    float2 _f2_396 = make_float2(acc2[2], acc2[3]);
                    float2 _f2_397 = make_float2(_unpack_fp4_32[2], _unpack_fp4_32[3]);
                    float2 r_0_1 = fma_f32x2_rn_noftz(_f2_397, _f2_392, _f2_396);
                    acc2[2] = r_0_1.x;
                    acc2[3] = r_0_1.y;
                    float2 _f2_398 = make_float2(acc2[4], acc2[5]);
                    float2 _f2_399 = make_float2(_unpack_fp4_32[4], _unpack_fp4_32[5]);
                    float2 r_1_1 = fma_f32x2_rn_noftz(_f2_399, _f2_392, _f2_398);
                    acc2[4] = r_1_1.x;
                    acc2[5] = r_1_1.y;
                    float2 _f2_400 = make_float2(acc2[6], acc2[7]);
                    float2 _f2_401 = make_float2(_unpack_fp4_32[6], _unpack_fp4_32[7]);
                    float2 r_2_1 = fma_f32x2_rn_noftz(_f2_401, _f2_392, _f2_400);
                    acc2[6] = r_2_1.x;
                    acc2[7] = r_2_1.y;
                    float _unpack_fp4_33[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_0[1 + 0]));
                        (&_unpack_fp4_33[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_33[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_33[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_33[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_33[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_33[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_33[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_33[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_33[0]), "=f"(_unpack_fp4_33[1]), "=f"(_unpack_fp4_33[2]), "=f"(_unpack_fp4_33[3]), "=f"(_unpack_fp4_33[4]), "=f"(_unpack_fp4_33[5]), "=f"(_unpack_fp4_33[6]), "=f"(_unpack_fp4_33[7]) : "r"(_vec_load_0[1 + 0]));
                        #endif
                    }
                    float2 _f2_402 = make_float2(acc2[8], acc2[9]);
                    float2 _f2_403 = make_float2(_unpack_fp4_33[0], _unpack_fp4_33[1]);
                    float2 r_3_1 = fma_f32x2_rn_noftz(_f2_403, _f2_392, _f2_402);
                    acc2[8] = r_3_1.x;
                    acc2[9] = r_3_1.y;
                    float2 _f2_404 = make_float2(acc2[10], acc2[11]);
                    float2 _f2_405 = make_float2(_unpack_fp4_33[2], _unpack_fp4_33[3]);
                    float2 r_4_1 = fma_f32x2_rn_noftz(_f2_405, _f2_392, _f2_404);
                    acc2[10] = r_4_1.x;
                    acc2[11] = r_4_1.y;
                    float2 _f2_406 = make_float2(acc2[12], acc2[13]);
                    float2 _f2_407 = make_float2(_unpack_fp4_33[4], _unpack_fp4_33[5]);
                    float2 r_5_1 = fma_f32x2_rn_noftz(_f2_407, _f2_392, _f2_406);
                    acc2[12] = r_5_1.x;
                    acc2[13] = r_5_1.y;
                    float2 _f2_408 = make_float2(acc2[14], acc2[15]);
                    float2 _f2_409 = make_float2(_unpack_fp4_33[6], _unpack_fp4_33[7]);
                    float2 r_6_1 = fma_f32x2_rn_noftz(_f2_409, _f2_392, _f2_408);
                    acc2[14] = r_6_1.x;
                    acc2[15] = r_6_1.y;
                    float _unpack_fp4_34[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_0[2 + 0]));
                        (&_unpack_fp4_34[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_34[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_34[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_34[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_34[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_34[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_34[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_34[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_34[0]), "=f"(_unpack_fp4_34[1]), "=f"(_unpack_fp4_34[2]), "=f"(_unpack_fp4_34[3]), "=f"(_unpack_fp4_34[4]), "=f"(_unpack_fp4_34[5]), "=f"(_unpack_fp4_34[6]), "=f"(_unpack_fp4_34[7]) : "r"(_vec_load_0[2 + 0]));
                        #endif
                    }
                    float2 _f2_410 = make_float2(acc2[16], acc2[17]);
                    float2 _f2_411 = make_float2(_unpack_fp4_34[0], _unpack_fp4_34[1]);
                    float2 r_7_1 = fma_f32x2_rn_noftz(_f2_411, _f2_393, _f2_410);
                    acc2[16] = r_7_1.x;
                    acc2[17] = r_7_1.y;
                    float2 _f2_412 = make_float2(acc2[18], acc2[19]);
                    float2 _f2_413 = make_float2(_unpack_fp4_34[2], _unpack_fp4_34[3]);
                    float2 r_8_1 = fma_f32x2_rn_noftz(_f2_413, _f2_393, _f2_412);
                    acc2[18] = r_8_1.x;
                    acc2[19] = r_8_1.y;
                    float2 _f2_414 = make_float2(acc2[20], acc2[21]);
                    float2 _f2_415 = make_float2(_unpack_fp4_34[4], _unpack_fp4_34[5]);
                    float2 r_9_1 = fma_f32x2_rn_noftz(_f2_415, _f2_393, _f2_414);
                    acc2[20] = r_9_1.x;
                    acc2[21] = r_9_1.y;
                    float2 _f2_416 = make_float2(acc2[22], acc2[23]);
                    float2 _f2_417 = make_float2(_unpack_fp4_34[6], _unpack_fp4_34[7]);
                    float2 r_10_1 = fma_f32x2_rn_noftz(_f2_417, _f2_393, _f2_416);
                    acc2[22] = r_10_1.x;
                    acc2[23] = r_10_1.y;
                    float _unpack_fp4_35[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_0[3 + 0]));
                        (&_unpack_fp4_35[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_35[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_35[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_35[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_35[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_35[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_35[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_35[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_35[0]), "=f"(_unpack_fp4_35[1]), "=f"(_unpack_fp4_35[2]), "=f"(_unpack_fp4_35[3]), "=f"(_unpack_fp4_35[4]), "=f"(_unpack_fp4_35[5]), "=f"(_unpack_fp4_35[6]), "=f"(_unpack_fp4_35[7]) : "r"(_vec_load_0[3 + 0]));
                        #endif
                    }
                    float2 _f2_418 = make_float2(acc2[24], acc2[25]);
                    float2 _f2_419 = make_float2(_unpack_fp4_35[0], _unpack_fp4_35[1]);
                    float2 r_11_1 = fma_f32x2_rn_noftz(_f2_419, _f2_393, _f2_418);
                    acc2[24] = r_11_1.x;
                    acc2[25] = r_11_1.y;
                    float2 _f2_420 = make_float2(acc2[26], acc2[27]);
                    float2 _f2_421 = make_float2(_unpack_fp4_35[2], _unpack_fp4_35[3]);
                    float2 r_12_1 = fma_f32x2_rn_noftz(_f2_421, _f2_393, _f2_420);
                    acc2[26] = r_12_1.x;
                    acc2[27] = r_12_1.y;
                    float2 _f2_422 = make_float2(acc2[28], acc2[29]);
                    float2 _f2_423 = make_float2(_unpack_fp4_35[4], _unpack_fp4_35[5]);
                    float2 r_13_1 = fma_f32x2_rn_noftz(_f2_423, _f2_393, _f2_422);
                    acc2[28] = r_13_1.x;
                    acc2[29] = r_13_1.y;
                    float2 _f2_424 = make_float2(acc2[30], acc2[31]);
                    float2 _f2_425 = make_float2(_unpack_fp4_35[6], _unpack_fp4_35[7]);
                    float2 r_14_1 = fma_f32x2_rn_noftz(_f2_425, _f2_393, _f2_424);
                    acc2[30] = r_14_1.x;
                    acc2[31] = r_14_1.y;
                    float2 _fp8x2_decode_15;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_15.x), "=f"(_fp8x2_decode_15.y) : "h"((uint16_t)_vec_load_3[0]));
                    float _mul_30;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(_fp8x2_decode_15.x), "f"(__uint_as_float(_sysv_poll32_group_1[0])));
                    float s0_15_1 = _mul_30;
                    float _mul_31;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(_fp8x2_decode_15.y), "f"(__uint_as_float(_sysv_poll32_group_1[0])));
                    float s1_16_1 = _mul_31;
                    float2 _f2_426 = make_float2(s0_15_1, s0_15_1);
                    float2 _f2_427 = make_float2(s1_16_1, s1_16_1);
                    float _unpack_fp4_36[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_2[0]));
                        (&_unpack_fp4_36[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_36[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_36[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_36[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_36[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_36[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_36[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_36[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_36[0]), "=f"(_unpack_fp4_36[1]), "=f"(_unpack_fp4_36[2]), "=f"(_unpack_fp4_36[3]), "=f"(_unpack_fp4_36[4]), "=f"(_unpack_fp4_36[5]), "=f"(_unpack_fp4_36[6]), "=f"(_unpack_fp4_36[7]) : "r"(_vec_load_2[0]));
                        #endif
                    }
                    float2 _f2_428 = make_float2(acc2[0], acc2[1]);
                    float2 _f2_429 = make_float2(_unpack_fp4_36[0], _unpack_fp4_36[1]);
                    float2 r_17_1 = fma_f32x2_rn_noftz(_f2_429, _f2_426, _f2_428);
                    acc2[0] = r_17_1.x;
                    acc2[1] = r_17_1.y;
                    float2 _f2_430 = make_float2(acc2[2], acc2[3]);
                    float2 _f2_431 = make_float2(_unpack_fp4_36[2], _unpack_fp4_36[3]);
                    float2 r_18_1 = fma_f32x2_rn_noftz(_f2_431, _f2_426, _f2_430);
                    acc2[2] = r_18_1.x;
                    acc2[3] = r_18_1.y;
                    float2 _f2_432 = make_float2(acc2[4], acc2[5]);
                    float2 _f2_433 = make_float2(_unpack_fp4_36[4], _unpack_fp4_36[5]);
                    float2 r_19_1 = fma_f32x2_rn_noftz(_f2_433, _f2_426, _f2_432);
                    acc2[4] = r_19_1.x;
                    acc2[5] = r_19_1.y;
                    float2 _f2_434 = make_float2(acc2[6], acc2[7]);
                    float2 _f2_435 = make_float2(_unpack_fp4_36[6], _unpack_fp4_36[7]);
                    float2 r_20_1 = fma_f32x2_rn_noftz(_f2_435, _f2_426, _f2_434);
                    acc2[6] = r_20_1.x;
                    acc2[7] = r_20_1.y;
                    float _unpack_fp4_37[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_2[1 + 0]));
                        (&_unpack_fp4_37[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_37[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_37[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_37[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_37[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_37[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_37[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_37[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_37[0]), "=f"(_unpack_fp4_37[1]), "=f"(_unpack_fp4_37[2]), "=f"(_unpack_fp4_37[3]), "=f"(_unpack_fp4_37[4]), "=f"(_unpack_fp4_37[5]), "=f"(_unpack_fp4_37[6]), "=f"(_unpack_fp4_37[7]) : "r"(_vec_load_2[1 + 0]));
                        #endif
                    }
                    float2 _f2_436 = make_float2(acc2[8], acc2[9]);
                    float2 _f2_437 = make_float2(_unpack_fp4_37[0], _unpack_fp4_37[1]);
                    float2 r_21_1 = fma_f32x2_rn_noftz(_f2_437, _f2_426, _f2_436);
                    acc2[8] = r_21_1.x;
                    acc2[9] = r_21_1.y;
                    float2 _f2_438 = make_float2(acc2[10], acc2[11]);
                    float2 _f2_439 = make_float2(_unpack_fp4_37[2], _unpack_fp4_37[3]);
                    float2 r_22_1 = fma_f32x2_rn_noftz(_f2_439, _f2_426, _f2_438);
                    acc2[10] = r_22_1.x;
                    acc2[11] = r_22_1.y;
                    float2 _f2_440 = make_float2(acc2[12], acc2[13]);
                    float2 _f2_441 = make_float2(_unpack_fp4_37[4], _unpack_fp4_37[5]);
                    float2 r_23_1 = fma_f32x2_rn_noftz(_f2_441, _f2_426, _f2_440);
                    acc2[12] = r_23_1.x;
                    acc2[13] = r_23_1.y;
                    float2 _f2_442 = make_float2(acc2[14], acc2[15]);
                    float2 _f2_443 = make_float2(_unpack_fp4_37[6], _unpack_fp4_37[7]);
                    float2 r_24_1 = fma_f32x2_rn_noftz(_f2_443, _f2_426, _f2_442);
                    acc2[14] = r_24_1.x;
                    acc2[15] = r_24_1.y;
                    float _unpack_fp4_38[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_2[2 + 0]));
                        (&_unpack_fp4_38[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_38[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_38[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_38[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_38[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_38[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_38[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_38[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_38[0]), "=f"(_unpack_fp4_38[1]), "=f"(_unpack_fp4_38[2]), "=f"(_unpack_fp4_38[3]), "=f"(_unpack_fp4_38[4]), "=f"(_unpack_fp4_38[5]), "=f"(_unpack_fp4_38[6]), "=f"(_unpack_fp4_38[7]) : "r"(_vec_load_2[2 + 0]));
                        #endif
                    }
                    float2 _f2_444 = make_float2(acc2[16], acc2[17]);
                    float2 _f2_445 = make_float2(_unpack_fp4_38[0], _unpack_fp4_38[1]);
                    float2 r_25_1 = fma_f32x2_rn_noftz(_f2_445, _f2_427, _f2_444);
                    acc2[16] = r_25_1.x;
                    acc2[17] = r_25_1.y;
                    float2 _f2_446 = make_float2(acc2[18], acc2[19]);
                    float2 _f2_447 = make_float2(_unpack_fp4_38[2], _unpack_fp4_38[3]);
                    float2 r_26_1 = fma_f32x2_rn_noftz(_f2_447, _f2_427, _f2_446);
                    acc2[18] = r_26_1.x;
                    acc2[19] = r_26_1.y;
                    float2 _f2_448 = make_float2(acc2[20], acc2[21]);
                    float2 _f2_449 = make_float2(_unpack_fp4_38[4], _unpack_fp4_38[5]);
                    float2 r_27_1 = fma_f32x2_rn_noftz(_f2_449, _f2_427, _f2_448);
                    acc2[20] = r_27_1.x;
                    acc2[21] = r_27_1.y;
                    float2 _f2_450 = make_float2(acc2[22], acc2[23]);
                    float2 _f2_451 = make_float2(_unpack_fp4_38[6], _unpack_fp4_38[7]);
                    float2 r_28_1 = fma_f32x2_rn_noftz(_f2_451, _f2_427, _f2_450);
                    acc2[22] = r_28_1.x;
                    acc2[23] = r_28_1.y;
                    float _unpack_fp4_39[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_2[3 + 0]));
                        (&_unpack_fp4_39[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_39[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_39[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_39[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_39[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_39[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_39[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_39[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_39[0]), "=f"(_unpack_fp4_39[1]), "=f"(_unpack_fp4_39[2]), "=f"(_unpack_fp4_39[3]), "=f"(_unpack_fp4_39[4]), "=f"(_unpack_fp4_39[5]), "=f"(_unpack_fp4_39[6]), "=f"(_unpack_fp4_39[7]) : "r"(_vec_load_2[3 + 0]));
                        #endif
                    }
                    float2 _f2_452 = make_float2(acc2[24], acc2[25]);
                    float2 _f2_453 = make_float2(_unpack_fp4_39[0], _unpack_fp4_39[1]);
                    float2 r_29_1 = fma_f32x2_rn_noftz(_f2_453, _f2_427, _f2_452);
                    acc2[24] = r_29_1.x;
                    acc2[25] = r_29_1.y;
                    float2 _f2_454 = make_float2(acc2[26], acc2[27]);
                    float2 _f2_455 = make_float2(_unpack_fp4_39[2], _unpack_fp4_39[3]);
                    float2 r_30_1 = fma_f32x2_rn_noftz(_f2_455, _f2_427, _f2_454);
                    acc2[26] = r_30_1.x;
                    acc2[27] = r_30_1.y;
                    float2 _f2_456 = make_float2(acc2[28], acc2[29]);
                    float2 _f2_457 = make_float2(_unpack_fp4_39[4], _unpack_fp4_39[5]);
                    float2 r_31_1 = fma_f32x2_rn_noftz(_f2_457, _f2_427, _f2_456);
                    acc2[28] = r_31_1.x;
                    acc2[29] = r_31_1.y;
                    float2 _f2_458 = make_float2(acc2[30], acc2[31]);
                    float2 _f2_459 = make_float2(_unpack_fp4_39[6], _unpack_fp4_39[7]);
                    float2 r_32_1 = fma_f32x2_rn_noftz(_f2_459, _f2_427, _f2_458);
                    acc2[30] = r_32_1.x;
                    acc2[31] = r_32_1.y;
                    float2 _fp8x2_decode_16;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_16.x), "=f"(_fp8x2_decode_16.y) : "h"((uint16_t)_vec_load_5[0]));
                    float _mul_32;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(_fp8x2_decode_16.x), "f"(__uint_as_float(_sysv_poll32_group_2[0])));
                    float s0_33_1 = _mul_32;
                    float _mul_33;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(_fp8x2_decode_16.y), "f"(__uint_as_float(_sysv_poll32_group_2[0])));
                    float s1_34_1 = _mul_33;
                    float2 _f2_460 = make_float2(s0_33_1, s0_33_1);
                    float2 _f2_461 = make_float2(s1_34_1, s1_34_1);
                    float _unpack_fp4_40[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_4[0]));
                        (&_unpack_fp4_40[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_40[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_40[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_40[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_40[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_40[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_40[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_40[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_40[0]), "=f"(_unpack_fp4_40[1]), "=f"(_unpack_fp4_40[2]), "=f"(_unpack_fp4_40[3]), "=f"(_unpack_fp4_40[4]), "=f"(_unpack_fp4_40[5]), "=f"(_unpack_fp4_40[6]), "=f"(_unpack_fp4_40[7]) : "r"(_vec_load_4[0]));
                        #endif
                    }
                    float2 _f2_462 = make_float2(acc2[0], acc2[1]);
                    float2 _f2_463 = make_float2(_unpack_fp4_40[0], _unpack_fp4_40[1]);
                    float2 r_35_1 = fma_f32x2_rn_noftz(_f2_463, _f2_460, _f2_462);
                    acc2[0] = r_35_1.x;
                    acc2[1] = r_35_1.y;
                    float2 _f2_464 = make_float2(acc2[2], acc2[3]);
                    float2 _f2_465 = make_float2(_unpack_fp4_40[2], _unpack_fp4_40[3]);
                    float2 r_36_1 = fma_f32x2_rn_noftz(_f2_465, _f2_460, _f2_464);
                    acc2[2] = r_36_1.x;
                    acc2[3] = r_36_1.y;
                    float2 _f2_466 = make_float2(acc2[4], acc2[5]);
                    float2 _f2_467 = make_float2(_unpack_fp4_40[4], _unpack_fp4_40[5]);
                    float2 r_37_1 = fma_f32x2_rn_noftz(_f2_467, _f2_460, _f2_466);
                    acc2[4] = r_37_1.x;
                    acc2[5] = r_37_1.y;
                    float2 _f2_468 = make_float2(acc2[6], acc2[7]);
                    float2 _f2_469 = make_float2(_unpack_fp4_40[6], _unpack_fp4_40[7]);
                    float2 r_38_1 = fma_f32x2_rn_noftz(_f2_469, _f2_460, _f2_468);
                    acc2[6] = r_38_1.x;
                    acc2[7] = r_38_1.y;
                    float _unpack_fp4_41[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_4[1 + 0]));
                        (&_unpack_fp4_41[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_41[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_41[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_41[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_41[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_41[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_41[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_41[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_41[0]), "=f"(_unpack_fp4_41[1]), "=f"(_unpack_fp4_41[2]), "=f"(_unpack_fp4_41[3]), "=f"(_unpack_fp4_41[4]), "=f"(_unpack_fp4_41[5]), "=f"(_unpack_fp4_41[6]), "=f"(_unpack_fp4_41[7]) : "r"(_vec_load_4[1 + 0]));
                        #endif
                    }
                    float2 _f2_470 = make_float2(acc2[8], acc2[9]);
                    float2 _f2_471 = make_float2(_unpack_fp4_41[0], _unpack_fp4_41[1]);
                    float2 r_39_1 = fma_f32x2_rn_noftz(_f2_471, _f2_460, _f2_470);
                    acc2[8] = r_39_1.x;
                    acc2[9] = r_39_1.y;
                    float2 _f2_472 = make_float2(acc2[10], acc2[11]);
                    float2 _f2_473 = make_float2(_unpack_fp4_41[2], _unpack_fp4_41[3]);
                    float2 r_40_1 = fma_f32x2_rn_noftz(_f2_473, _f2_460, _f2_472);
                    acc2[10] = r_40_1.x;
                    acc2[11] = r_40_1.y;
                    float2 _f2_474 = make_float2(acc2[12], acc2[13]);
                    float2 _f2_475 = make_float2(_unpack_fp4_41[4], _unpack_fp4_41[5]);
                    float2 r_41_1 = fma_f32x2_rn_noftz(_f2_475, _f2_460, _f2_474);
                    acc2[12] = r_41_1.x;
                    acc2[13] = r_41_1.y;
                    float2 _f2_476 = make_float2(acc2[14], acc2[15]);
                    float2 _f2_477 = make_float2(_unpack_fp4_41[6], _unpack_fp4_41[7]);
                    float2 r_42_1 = fma_f32x2_rn_noftz(_f2_477, _f2_460, _f2_476);
                    acc2[14] = r_42_1.x;
                    acc2[15] = r_42_1.y;
                    float _unpack_fp4_42[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_4[2 + 0]));
                        (&_unpack_fp4_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_42[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_42[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_42[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_42[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_42[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_42[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_42[0]), "=f"(_unpack_fp4_42[1]), "=f"(_unpack_fp4_42[2]), "=f"(_unpack_fp4_42[3]), "=f"(_unpack_fp4_42[4]), "=f"(_unpack_fp4_42[5]), "=f"(_unpack_fp4_42[6]), "=f"(_unpack_fp4_42[7]) : "r"(_vec_load_4[2 + 0]));
                        #endif
                    }
                    float2 _f2_478 = make_float2(acc2[16], acc2[17]);
                    float2 _f2_479 = make_float2(_unpack_fp4_42[0], _unpack_fp4_42[1]);
                    float2 r_43_1 = fma_f32x2_rn_noftz(_f2_479, _f2_461, _f2_478);
                    acc2[16] = r_43_1.x;
                    acc2[17] = r_43_1.y;
                    float2 _f2_480 = make_float2(acc2[18], acc2[19]);
                    float2 _f2_481 = make_float2(_unpack_fp4_42[2], _unpack_fp4_42[3]);
                    float2 r_44_1 = fma_f32x2_rn_noftz(_f2_481, _f2_461, _f2_480);
                    acc2[18] = r_44_1.x;
                    acc2[19] = r_44_1.y;
                    float2 _f2_482 = make_float2(acc2[20], acc2[21]);
                    float2 _f2_483 = make_float2(_unpack_fp4_42[4], _unpack_fp4_42[5]);
                    float2 r_45_1 = fma_f32x2_rn_noftz(_f2_483, _f2_461, _f2_482);
                    acc2[20] = r_45_1.x;
                    acc2[21] = r_45_1.y;
                    float2 _f2_484 = make_float2(acc2[22], acc2[23]);
                    float2 _f2_485 = make_float2(_unpack_fp4_42[6], _unpack_fp4_42[7]);
                    float2 r_46_1 = fma_f32x2_rn_noftz(_f2_485, _f2_461, _f2_484);
                    acc2[22] = r_46_1.x;
                    acc2[23] = r_46_1.y;
                    float _unpack_fp4_43[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_4[3 + 0]));
                        (&_unpack_fp4_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_43[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_43[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_43[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_43[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_43[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_43[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_43[0]), "=f"(_unpack_fp4_43[1]), "=f"(_unpack_fp4_43[2]), "=f"(_unpack_fp4_43[3]), "=f"(_unpack_fp4_43[4]), "=f"(_unpack_fp4_43[5]), "=f"(_unpack_fp4_43[6]), "=f"(_unpack_fp4_43[7]) : "r"(_vec_load_4[3 + 0]));
                        #endif
                    }
                    float2 _f2_486 = make_float2(acc2[24], acc2[25]);
                    float2 _f2_487 = make_float2(_unpack_fp4_43[0], _unpack_fp4_43[1]);
                    float2 r_47_1 = fma_f32x2_rn_noftz(_f2_487, _f2_461, _f2_486);
                    acc2[24] = r_47_1.x;
                    acc2[25] = r_47_1.y;
                    float2 _f2_488 = make_float2(acc2[26], acc2[27]);
                    float2 _f2_489 = make_float2(_unpack_fp4_43[2], _unpack_fp4_43[3]);
                    float2 r_48_1 = fma_f32x2_rn_noftz(_f2_489, _f2_461, _f2_488);
                    acc2[26] = r_48_1.x;
                    acc2[27] = r_48_1.y;
                    float2 _f2_490 = make_float2(acc2[28], acc2[29]);
                    float2 _f2_491 = make_float2(_unpack_fp4_43[4], _unpack_fp4_43[5]);
                    float2 r_49_1 = fma_f32x2_rn_noftz(_f2_491, _f2_461, _f2_490);
                    acc2[28] = r_49_1.x;
                    acc2[29] = r_49_1.y;
                    float2 _f2_492 = make_float2(acc2[30], acc2[31]);
                    float2 _f2_493 = make_float2(_unpack_fp4_43[6], _unpack_fp4_43[7]);
                    float2 r_50_1 = fma_f32x2_rn_noftz(_f2_493, _f2_461, _f2_492);
                    acc2[30] = r_50_1.x;
                    acc2[31] = r_50_1.y;
                    float2 _fp8x2_decode_17;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_17.x), "=f"(_fp8x2_decode_17.y) : "h"((uint16_t)_vec_load_7[0]));
                    float _mul_34;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(_fp8x2_decode_17.x), "f"(__uint_as_float(_sysv_poll32_group_3[0])));
                    float s0_51_1 = _mul_34;
                    float _mul_35;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(_fp8x2_decode_17.y), "f"(__uint_as_float(_sysv_poll32_group_3[0])));
                    float s1_52_1 = _mul_35;
                    float2 _f2_494 = make_float2(s0_51_1, s0_51_1);
                    float2 _f2_495 = make_float2(s1_52_1, s1_52_1);
                    float _unpack_fp4_44[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_6[0]));
                        (&_unpack_fp4_44[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_44[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_44[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_44[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_44[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_44[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_44[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_44[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_44[0]), "=f"(_unpack_fp4_44[1]), "=f"(_unpack_fp4_44[2]), "=f"(_unpack_fp4_44[3]), "=f"(_unpack_fp4_44[4]), "=f"(_unpack_fp4_44[5]), "=f"(_unpack_fp4_44[6]), "=f"(_unpack_fp4_44[7]) : "r"(_vec_load_6[0]));
                        #endif
                    }
                    float2 _f2_496 = make_float2(acc2[0], acc2[1]);
                    float2 _f2_497 = make_float2(_unpack_fp4_44[0], _unpack_fp4_44[1]);
                    float2 r_53_1 = fma_f32x2_rn_noftz(_f2_497, _f2_494, _f2_496);
                    acc2[0] = r_53_1.x;
                    acc2[1] = r_53_1.y;
                    float2 _f2_498 = make_float2(acc2[2], acc2[3]);
                    float2 _f2_499 = make_float2(_unpack_fp4_44[2], _unpack_fp4_44[3]);
                    float2 r_54_1 = fma_f32x2_rn_noftz(_f2_499, _f2_494, _f2_498);
                    acc2[2] = r_54_1.x;
                    acc2[3] = r_54_1.y;
                    float2 _f2_500 = make_float2(acc2[4], acc2[5]);
                    float2 _f2_501 = make_float2(_unpack_fp4_44[4], _unpack_fp4_44[5]);
                    float2 r_55_1 = fma_f32x2_rn_noftz(_f2_501, _f2_494, _f2_500);
                    acc2[4] = r_55_1.x;
                    acc2[5] = r_55_1.y;
                    float2 _f2_502 = make_float2(acc2[6], acc2[7]);
                    float2 _f2_503 = make_float2(_unpack_fp4_44[6], _unpack_fp4_44[7]);
                    float2 r_56_1 = fma_f32x2_rn_noftz(_f2_503, _f2_494, _f2_502);
                    acc2[6] = r_56_1.x;
                    acc2[7] = r_56_1.y;
                    float _unpack_fp4_45[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_6[1 + 0]));
                        (&_unpack_fp4_45[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_45[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_45[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_45[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_45[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_45[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_45[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_45[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_45[0]), "=f"(_unpack_fp4_45[1]), "=f"(_unpack_fp4_45[2]), "=f"(_unpack_fp4_45[3]), "=f"(_unpack_fp4_45[4]), "=f"(_unpack_fp4_45[5]), "=f"(_unpack_fp4_45[6]), "=f"(_unpack_fp4_45[7]) : "r"(_vec_load_6[1 + 0]));
                        #endif
                    }
                    float2 _f2_504 = make_float2(acc2[8], acc2[9]);
                    float2 _f2_505 = make_float2(_unpack_fp4_45[0], _unpack_fp4_45[1]);
                    float2 r_57_1 = fma_f32x2_rn_noftz(_f2_505, _f2_494, _f2_504);
                    acc2[8] = r_57_1.x;
                    acc2[9] = r_57_1.y;
                    float2 _f2_506 = make_float2(acc2[10], acc2[11]);
                    float2 _f2_507 = make_float2(_unpack_fp4_45[2], _unpack_fp4_45[3]);
                    float2 r_58_1 = fma_f32x2_rn_noftz(_f2_507, _f2_494, _f2_506);
                    acc2[10] = r_58_1.x;
                    acc2[11] = r_58_1.y;
                    float2 _f2_508 = make_float2(acc2[12], acc2[13]);
                    float2 _f2_509 = make_float2(_unpack_fp4_45[4], _unpack_fp4_45[5]);
                    float2 r_59_1 = fma_f32x2_rn_noftz(_f2_509, _f2_494, _f2_508);
                    acc2[12] = r_59_1.x;
                    acc2[13] = r_59_1.y;
                    float2 _f2_510 = make_float2(acc2[14], acc2[15]);
                    float2 _f2_511 = make_float2(_unpack_fp4_45[6], _unpack_fp4_45[7]);
                    float2 r_60_1 = fma_f32x2_rn_noftz(_f2_511, _f2_494, _f2_510);
                    acc2[14] = r_60_1.x;
                    acc2[15] = r_60_1.y;
                    float _unpack_fp4_46[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_6[2 + 0]));
                        (&_unpack_fp4_46[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_46[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_46[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_46[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_46[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_46[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_46[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_46[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_46[0]), "=f"(_unpack_fp4_46[1]), "=f"(_unpack_fp4_46[2]), "=f"(_unpack_fp4_46[3]), "=f"(_unpack_fp4_46[4]), "=f"(_unpack_fp4_46[5]), "=f"(_unpack_fp4_46[6]), "=f"(_unpack_fp4_46[7]) : "r"(_vec_load_6[2 + 0]));
                        #endif
                    }
                    float2 _f2_512 = make_float2(acc2[16], acc2[17]);
                    float2 _f2_513 = make_float2(_unpack_fp4_46[0], _unpack_fp4_46[1]);
                    float2 r_61_1 = fma_f32x2_rn_noftz(_f2_513, _f2_495, _f2_512);
                    acc2[16] = r_61_1.x;
                    acc2[17] = r_61_1.y;
                    float2 _f2_514 = make_float2(acc2[18], acc2[19]);
                    float2 _f2_515 = make_float2(_unpack_fp4_46[2], _unpack_fp4_46[3]);
                    float2 r_62_1 = fma_f32x2_rn_noftz(_f2_515, _f2_495, _f2_514);
                    acc2[18] = r_62_1.x;
                    acc2[19] = r_62_1.y;
                    float2 _f2_516 = make_float2(acc2[20], acc2[21]);
                    float2 _f2_517 = make_float2(_unpack_fp4_46[4], _unpack_fp4_46[5]);
                    float2 r_63_1 = fma_f32x2_rn_noftz(_f2_517, _f2_495, _f2_516);
                    acc2[20] = r_63_1.x;
                    acc2[21] = r_63_1.y;
                    float2 _f2_518 = make_float2(acc2[22], acc2[23]);
                    float2 _f2_519 = make_float2(_unpack_fp4_46[6], _unpack_fp4_46[7]);
                    float2 r_64_1 = fma_f32x2_rn_noftz(_f2_519, _f2_495, _f2_518);
                    acc2[22] = r_64_1.x;
                    acc2[23] = r_64_1.y;
                    float _unpack_fp4_47[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_6[3 + 0]));
                        (&_unpack_fp4_47[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_47[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_47[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_47[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_47[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_47[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_47[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_47[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_47[0]), "=f"(_unpack_fp4_47[1]), "=f"(_unpack_fp4_47[2]), "=f"(_unpack_fp4_47[3]), "=f"(_unpack_fp4_47[4]), "=f"(_unpack_fp4_47[5]), "=f"(_unpack_fp4_47[6]), "=f"(_unpack_fp4_47[7]) : "r"(_vec_load_6[3 + 0]));
                        #endif
                    }
                    float2 _f2_520 = make_float2(acc2[24], acc2[25]);
                    float2 _f2_521 = make_float2(_unpack_fp4_47[0], _unpack_fp4_47[1]);
                    float2 r_65_1 = fma_f32x2_rn_noftz(_f2_521, _f2_495, _f2_520);
                    acc2[24] = r_65_1.x;
                    acc2[25] = r_65_1.y;
                    float2 _f2_522 = make_float2(acc2[26], acc2[27]);
                    float2 _f2_523 = make_float2(_unpack_fp4_47[2], _unpack_fp4_47[3]);
                    float2 r_66_1 = fma_f32x2_rn_noftz(_f2_523, _f2_495, _f2_522);
                    acc2[26] = r_66_1.x;
                    acc2[27] = r_66_1.y;
                    float2 _f2_524 = make_float2(acc2[28], acc2[29]);
                    float2 _f2_525 = make_float2(_unpack_fp4_47[4], _unpack_fp4_47[5]);
                    float2 r_67_1 = fma_f32x2_rn_noftz(_f2_525, _f2_495, _f2_524);
                    acc2[28] = r_67_1.x;
                    acc2[29] = r_67_1.y;
                    float2 _f2_526 = make_float2(acc2[30], acc2[31]);
                    float2 _f2_527 = make_float2(_unpack_fp4_47[6], _unpack_fp4_47[7]);
                    float2 r_68_1 = fma_f32x2_rn_noftz(_f2_527, _f2_495, _f2_526);
                    acc2[30] = r_68_1.x;
                    acc2[31] = r_68_1.y;
                    unsigned int q[4];
                    float zero_69 = 0.0f;
                    float six_1 = 6.0f;
                    float bm0_1 = 0.0f;
                    float bm1_1 = 0.0f;
                    float _fabs_64 = fabsf(acc2[0]);
                    float _fmax_70 = fmaxf(bm0_1, _fabs_64);
                    bm0_1 = _fmax_70;
                    float _fabs_65 = fabsf(acc2[16]);
                    float _fmax_71 = fmaxf(bm1_1, _fabs_65);
                    bm1_1 = _fmax_71;
                    float _fabs_66 = fabsf(acc2[1]);
                    float _fmax_72 = fmaxf(bm0_1, _fabs_66);
                    bm0_1 = _fmax_72;
                    float _fabs_67 = fabsf(acc2[17]);
                    float _fmax_73 = fmaxf(bm1_1, _fabs_67);
                    bm1_1 = _fmax_73;
                    float _fabs_68 = fabsf(acc2[2]);
                    float _fmax_74 = fmaxf(bm0_1, _fabs_68);
                    bm0_1 = _fmax_74;
                    float _fabs_69 = fabsf(acc2[18]);
                    float _fmax_75 = fmaxf(bm1_1, _fabs_69);
                    bm1_1 = _fmax_75;
                    float _fabs_70 = fabsf(acc2[3]);
                    float _fmax_76 = fmaxf(bm0_1, _fabs_70);
                    bm0_1 = _fmax_76;
                    float _fabs_71 = fabsf(acc2[19]);
                    float _fmax_77 = fmaxf(bm1_1, _fabs_71);
                    bm1_1 = _fmax_77;
                    float _fabs_72 = fabsf(acc2[4]);
                    float _fmax_78 = fmaxf(bm0_1, _fabs_72);
                    bm0_1 = _fmax_78;
                    float _fabs_73 = fabsf(acc2[20]);
                    float _fmax_79 = fmaxf(bm1_1, _fabs_73);
                    bm1_1 = _fmax_79;
                    float _fabs_74 = fabsf(acc2[5]);
                    float _fmax_80 = fmaxf(bm0_1, _fabs_74);
                    bm0_1 = _fmax_80;
                    float _fabs_75 = fabsf(acc2[21]);
                    float _fmax_81 = fmaxf(bm1_1, _fabs_75);
                    bm1_1 = _fmax_81;
                    float _fabs_76 = fabsf(acc2[6]);
                    float _fmax_82 = fmaxf(bm0_1, _fabs_76);
                    bm0_1 = _fmax_82;
                    float _fabs_77 = fabsf(acc2[22]);
                    float _fmax_83 = fmaxf(bm1_1, _fabs_77);
                    bm1_1 = _fmax_83;
                    float _fabs_78 = fabsf(acc2[7]);
                    float _fmax_84 = fmaxf(bm0_1, _fabs_78);
                    bm0_1 = _fmax_84;
                    float _fabs_79 = fabsf(acc2[23]);
                    float _fmax_85 = fmaxf(bm1_1, _fabs_79);
                    bm1_1 = _fmax_85;
                    float _fabs_80 = fabsf(acc2[8]);
                    float _fmax_86 = fmaxf(bm0_1, _fabs_80);
                    bm0_1 = _fmax_86;
                    float _fabs_81 = fabsf(acc2[24]);
                    float _fmax_87 = fmaxf(bm1_1, _fabs_81);
                    bm1_1 = _fmax_87;
                    float _fabs_82 = fabsf(acc2[9]);
                    float _fmax_88 = fmaxf(bm0_1, _fabs_82);
                    bm0_1 = _fmax_88;
                    float _fabs_83 = fabsf(acc2[25]);
                    float _fmax_89 = fmaxf(bm1_1, _fabs_83);
                    bm1_1 = _fmax_89;
                    float _fabs_84 = fabsf(acc2[10]);
                    float _fmax_90 = fmaxf(bm0_1, _fabs_84);
                    bm0_1 = _fmax_90;
                    float _fabs_85 = fabsf(acc2[26]);
                    float _fmax_91 = fmaxf(bm1_1, _fabs_85);
                    bm1_1 = _fmax_91;
                    float _fabs_86 = fabsf(acc2[11]);
                    float _fmax_92 = fmaxf(bm0_1, _fabs_86);
                    bm0_1 = _fmax_92;
                    float _fabs_87 = fabsf(acc2[27]);
                    float _fmax_93 = fmaxf(bm1_1, _fabs_87);
                    bm1_1 = _fmax_93;
                    float _fabs_88 = fabsf(acc2[12]);
                    float _fmax_94 = fmaxf(bm0_1, _fabs_88);
                    bm0_1 = _fmax_94;
                    float _fabs_89 = fabsf(acc2[28]);
                    float _fmax_95 = fmaxf(bm1_1, _fabs_89);
                    bm1_1 = _fmax_95;
                    float _fabs_90 = fabsf(acc2[13]);
                    float _fmax_96 = fmaxf(bm0_1, _fabs_90);
                    bm0_1 = _fmax_96;
                    float _fabs_91 = fabsf(acc2[29]);
                    float _fmax_97 = fmaxf(bm1_1, _fabs_91);
                    bm1_1 = _fmax_97;
                    float _fabs_92 = fabsf(acc2[14]);
                    float _fmax_98 = fmaxf(bm0_1, _fabs_92);
                    bm0_1 = _fmax_98;
                    float _fabs_93 = fabsf(acc2[30]);
                    float _fmax_99 = fmaxf(bm1_1, _fabs_93);
                    bm1_1 = _fmax_99;
                    float _fabs_94 = fabsf(acc2[15]);
                    float _fmax_100 = fmaxf(bm0_1, _fabs_94);
                    bm0_1 = _fmax_100;
                    float _fabs_95 = fabsf(acc2[31]);
                    float _fmax_101 = fmaxf(bm1_1, _fabs_95);
                    bm1_1 = _fmax_101;
                    float _mul_36;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(bm0_1), "f"(inv_1));
                    float _fdiv_rn_28 = __fdiv_rn(_mul_36, six_1);
                    float _mul_37;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(bm1_1), "f"(inv_1));
                    float _fdiv_rn_29 = __fdiv_rn(_mul_37, six_1);
                    uint16_t _e4m3x2_f32_6;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_6) : "f"(_fdiv_rn_29), "f"(_fdiv_rn_28));
                    float2 _fp8x2_decode_18;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_18.x), "=f"(_fp8x2_decode_18.y) : "h"(_e4m3x2_f32_6));
                    float _fdiv_rn_30 = __fdiv_rn(inv_1, _fp8x2_decode_18.x);
                    float _fdiv_rn_31 = __fdiv_rn(inv_1, _fp8x2_decode_18.y);
                    float m0_1 = ((zero_69 < _fp8x2_decode_18.x) ? _fdiv_rn_30 : zero_69);
                    float m1_1 = ((zero_69 < _fp8x2_decode_18.y) ? _fdiv_rn_31 : zero_69);
                    float2 _f2_528 = make_float2(m0_1, m0_1);
                    float2 _f2_529 = make_float2(m1_1, m1_1);
                    float2 _f2_530 = make_float2(acc2[0], acc2[1]);
                    float2 _f32x2_mul_b_145 = ((1) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_96;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_96) : "l"(*(const unsigned long long*)&_f2_530), "l"(*(const unsigned long long*)&_f32x2_mul_b_145));
                    uint32_t _fp4_pair_96;
                    uint16_t _e2m1_pair_h_146;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_146) : "f"(_mul_f32x2_96.x), "f"(_mul_f32x2_96.y));
                    _fp4_pair_96 = (uint32_t)_e2m1_pair_h_146;
                    float2 _f2_531 = make_float2(acc2[2], acc2[3]);
                    float2 _f32x2_mul_b_147 = ((1) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_97;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_97) : "l"(*(const unsigned long long*)&_f2_531), "l"(*(const unsigned long long*)&_f32x2_mul_b_147));
                    uint32_t _fp4_pair_97;
                    uint16_t _e2m1_pair_h_148;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_148) : "f"(_mul_f32x2_97.x), "f"(_mul_f32x2_97.y));
                    _fp4_pair_97 = (uint32_t)_e2m1_pair_h_148;
                    float2 _f2_532 = make_float2(acc2[4], acc2[5]);
                    float2 _f32x2_mul_b_149 = ((1) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_98;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_98) : "l"(*(const unsigned long long*)&_f2_532), "l"(*(const unsigned long long*)&_f32x2_mul_b_149));
                    uint32_t _fp4_pair_98;
                    uint16_t _e2m1_pair_h_150;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_150) : "f"(_mul_f32x2_98.x), "f"(_mul_f32x2_98.y));
                    _fp4_pair_98 = (uint32_t)_e2m1_pair_h_150;
                    float2 _f2_533 = make_float2(acc2[6], acc2[7]);
                    float2 _f32x2_mul_b_151 = ((1) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_99;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_99) : "l"(*(const unsigned long long*)&_f2_533), "l"(*(const unsigned long long*)&_f32x2_mul_b_151));
                    uint32_t _fp4_pair_99;
                    uint16_t _e2m1_pair_h_152;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_152) : "f"(_mul_f32x2_99.x), "f"(_mul_f32x2_99.y));
                    _fp4_pair_99 = (uint32_t)_e2m1_pair_h_152;
                    unsigned int word_1 = _fp4_pair_96 | _fp4_pair_97 << 8 | _fp4_pair_98 << 16 | _fp4_pair_99 << 24;
                    uint32_t _e2m1_canon_24;
                    {
                        const uint32_t _canon_w_153 = word_1;
                        const uint32_t _canon_s_153 = (_canon_w_153 & 0x77777777u) + 0x77777777u;
                        _e2m1_canon_24 = _canon_w_153 & (_canon_s_153 | 0x77777777u);
                    }
                    q[0] = _e2m1_canon_24;
                    float2 _f2_534 = make_float2(acc2[8], acc2[9]);
                    float2 _f32x2_mul_b_154 = ((1) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_100;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_100) : "l"(*(const unsigned long long*)&_f2_534), "l"(*(const unsigned long long*)&_f32x2_mul_b_154));
                    uint32_t _fp4_pair_100;
                    uint16_t _e2m1_pair_h_155;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_155) : "f"(_mul_f32x2_100.x), "f"(_mul_f32x2_100.y));
                    _fp4_pair_100 = (uint32_t)_e2m1_pair_h_155;
                    float2 _f2_535 = make_float2(acc2[10], acc2[11]);
                    float2 _f32x2_mul_b_156 = ((1) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_101;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_101) : "l"(*(const unsigned long long*)&_f2_535), "l"(*(const unsigned long long*)&_f32x2_mul_b_156));
                    uint32_t _fp4_pair_101;
                    uint16_t _e2m1_pair_h_157;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_157) : "f"(_mul_f32x2_101.x), "f"(_mul_f32x2_101.y));
                    _fp4_pair_101 = (uint32_t)_e2m1_pair_h_157;
                    float2 _f2_536 = make_float2(acc2[12], acc2[13]);
                    float2 _f32x2_mul_b_158 = ((1) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_102;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_102) : "l"(*(const unsigned long long*)&_f2_536), "l"(*(const unsigned long long*)&_f32x2_mul_b_158));
                    uint32_t _fp4_pair_102;
                    uint16_t _e2m1_pair_h_159;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_159) : "f"(_mul_f32x2_102.x), "f"(_mul_f32x2_102.y));
                    _fp4_pair_102 = (uint32_t)_e2m1_pair_h_159;
                    float2 _f2_537 = make_float2(acc2[14], acc2[15]);
                    float2 _f32x2_mul_b_160 = ((1) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_103;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_103) : "l"(*(const unsigned long long*)&_f2_537), "l"(*(const unsigned long long*)&_f32x2_mul_b_160));
                    uint32_t _fp4_pair_103;
                    uint16_t _e2m1_pair_h_161;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_161) : "f"(_mul_f32x2_103.x), "f"(_mul_f32x2_103.y));
                    _fp4_pair_103 = (uint32_t)_e2m1_pair_h_161;
                    unsigned int word_70 = _fp4_pair_100 | _fp4_pair_101 << 8 | _fp4_pair_102 << 16 | _fp4_pair_103 << 24;
                    uint32_t _e2m1_canon_25;
                    {
                        const uint32_t _canon_w_162 = word_70;
                        const uint32_t _canon_s_162 = (_canon_w_162 & 0x77777777u) + 0x77777777u;
                        _e2m1_canon_25 = _canon_w_162 & (_canon_s_162 | 0x77777777u);
                    }
                    q[1] = _e2m1_canon_25;
                    float2 _f2_538 = make_float2(acc2[16], acc2[17]);
                    float2 _f32x2_mul_b_163 = ((0) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_104;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_104) : "l"(*(const unsigned long long*)&_f2_538), "l"(*(const unsigned long long*)&_f32x2_mul_b_163));
                    uint32_t _fp4_pair_104;
                    uint16_t _e2m1_pair_h_164;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_164) : "f"(_mul_f32x2_104.x), "f"(_mul_f32x2_104.y));
                    _fp4_pair_104 = (uint32_t)_e2m1_pair_h_164;
                    float2 _f2_539 = make_float2(acc2[18], acc2[19]);
                    float2 _f32x2_mul_b_165 = ((0) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_105;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_105) : "l"(*(const unsigned long long*)&_f2_539), "l"(*(const unsigned long long*)&_f32x2_mul_b_165));
                    uint32_t _fp4_pair_105;
                    uint16_t _e2m1_pair_h_166;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_166) : "f"(_mul_f32x2_105.x), "f"(_mul_f32x2_105.y));
                    _fp4_pair_105 = (uint32_t)_e2m1_pair_h_166;
                    float2 _f2_540 = make_float2(acc2[20], acc2[21]);
                    float2 _f32x2_mul_b_167 = ((0) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_106;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_106) : "l"(*(const unsigned long long*)&_f2_540), "l"(*(const unsigned long long*)&_f32x2_mul_b_167));
                    uint32_t _fp4_pair_106;
                    uint16_t _e2m1_pair_h_168;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_168) : "f"(_mul_f32x2_106.x), "f"(_mul_f32x2_106.y));
                    _fp4_pair_106 = (uint32_t)_e2m1_pair_h_168;
                    float2 _f2_541 = make_float2(acc2[22], acc2[23]);
                    float2 _f32x2_mul_b_169 = ((0) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_107;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_107) : "l"(*(const unsigned long long*)&_f2_541), "l"(*(const unsigned long long*)&_f32x2_mul_b_169));
                    uint32_t _fp4_pair_107;
                    uint16_t _e2m1_pair_h_170;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_170) : "f"(_mul_f32x2_107.x), "f"(_mul_f32x2_107.y));
                    _fp4_pair_107 = (uint32_t)_e2m1_pair_h_170;
                    unsigned int word_71 = _fp4_pair_104 | _fp4_pair_105 << 8 | _fp4_pair_106 << 16 | _fp4_pair_107 << 24;
                    uint32_t _e2m1_canon_26;
                    {
                        const uint32_t _canon_w_171 = word_71;
                        const uint32_t _canon_s_171 = (_canon_w_171 & 0x77777777u) + 0x77777777u;
                        _e2m1_canon_26 = _canon_w_171 & (_canon_s_171 | 0x77777777u);
                    }
                    q[2] = _e2m1_canon_26;
                    float2 _f2_542 = make_float2(acc2[24], acc2[25]);
                    float2 _f32x2_mul_b_172 = ((0) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_108;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_108) : "l"(*(const unsigned long long*)&_f2_542), "l"(*(const unsigned long long*)&_f32x2_mul_b_172));
                    uint32_t _fp4_pair_108;
                    uint16_t _e2m1_pair_h_173;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_173) : "f"(_mul_f32x2_108.x), "f"(_mul_f32x2_108.y));
                    _fp4_pair_108 = (uint32_t)_e2m1_pair_h_173;
                    float2 _f2_543 = make_float2(acc2[26], acc2[27]);
                    float2 _f32x2_mul_b_174 = ((0) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_109;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_109) : "l"(*(const unsigned long long*)&_f2_543), "l"(*(const unsigned long long*)&_f32x2_mul_b_174));
                    uint32_t _fp4_pair_109;
                    uint16_t _e2m1_pair_h_175;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_175) : "f"(_mul_f32x2_109.x), "f"(_mul_f32x2_109.y));
                    _fp4_pair_109 = (uint32_t)_e2m1_pair_h_175;
                    float2 _f2_544 = make_float2(acc2[28], acc2[29]);
                    float2 _f32x2_mul_b_176 = ((0) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_110;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_110) : "l"(*(const unsigned long long*)&_f2_544), "l"(*(const unsigned long long*)&_f32x2_mul_b_176));
                    uint32_t _fp4_pair_110;
                    uint16_t _e2m1_pair_h_177;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_177) : "f"(_mul_f32x2_110.x), "f"(_mul_f32x2_110.y));
                    _fp4_pair_110 = (uint32_t)_e2m1_pair_h_177;
                    float2 _f2_545 = make_float2(acc2[30], acc2[31]);
                    float2 _f32x2_mul_b_178 = ((0) ? _f2_528 : _f2_529);
                    float2 _mul_f32x2_111;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_111) : "l"(*(const unsigned long long*)&_f2_545), "l"(*(const unsigned long long*)&_f32x2_mul_b_178));
                    uint32_t _fp4_pair_111;
                    uint16_t _e2m1_pair_h_179;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_179) : "f"(_mul_f32x2_111.x), "f"(_mul_f32x2_111.y));
                    _fp4_pair_111 = (uint32_t)_e2m1_pair_h_179;
                    unsigned int word_72 = _fp4_pair_108 | _fp4_pair_109 << 8 | _fp4_pair_110 << 16 | _fp4_pair_111 << 24;
                    uint32_t _e2m1_canon_27;
                    {
                        const uint32_t _canon_w_180 = word_72;
                        const uint32_t _canon_s_180 = (_canon_w_180 & 0x77777777u) + 0x77777777u;
                        _e2m1_canon_27 = _canon_w_180 & (_canon_s_180 | 0x77777777u);
                    }
                    q[3] = _e2m1_canon_27;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(bob_off + 16 * pp_1)), "r"(q[0]), "r"(q[1]), "r"(q[2]), "r"(q[3]) : "memory");
                    wsm16[(bob_off + 3072 >> 1) + pp_1] = _e4m3x2_f32_6;
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
                    float2 _fp8x2_decode_19;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_19.x), "=f"(_fp8x2_decode_19.y) : "h"((uint16_t)_e4m3x2_f32_6));
                    float _mul_38;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(_fp8x2_decode_19.x), "f"(rs2));
                    float s0_73 = _mul_38;
                    float _mul_39;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(_fp8x2_decode_19.y), "f"(rs2));
                    float s1_74 = _mul_39;
                    float2 _f2_546 = make_float2(s0_73, s0_73);
                    float2 _f2_547 = make_float2(s1_74, s1_74);
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
                    float2 _f2_548 = make_float2(d[0], d[1]);
                    float2 _f2_549 = make_float2(_unpack_fp4_48[0], _unpack_fp4_48[1]);
                    float2 r_75_1 = fma_f32x2_rn_noftz(_f2_549, _f2_546, _f2_548);
                    d[0] = r_75_1.x;
                    d[1] = r_75_1.y;
                    float2 _f2_550 = make_float2(d[2], d[3]);
                    float2 _f2_551 = make_float2(_unpack_fp4_48[2], _unpack_fp4_48[3]);
                    float2 r_76_1 = fma_f32x2_rn_noftz(_f2_551, _f2_546, _f2_550);
                    d[2] = r_76_1.x;
                    d[3] = r_76_1.y;
                    float2 _f2_552 = make_float2(d[4], d[5]);
                    float2 _f2_553 = make_float2(_unpack_fp4_48[4], _unpack_fp4_48[5]);
                    float2 r_77_1 = fma_f32x2_rn_noftz(_f2_553, _f2_546, _f2_552);
                    d[4] = r_77_1.x;
                    d[5] = r_77_1.y;
                    float2 _f2_554 = make_float2(d[6], d[7]);
                    float2 _f2_555 = make_float2(_unpack_fp4_48[6], _unpack_fp4_48[7]);
                    float2 r_78_1 = fma_f32x2_rn_noftz(_f2_555, _f2_546, _f2_554);
                    d[6] = r_78_1.x;
                    d[7] = r_78_1.y;
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
                    float2 _f2_556 = make_float2(d[8], d[9]);
                    float2 _f2_557 = make_float2(_unpack_fp4_49[0], _unpack_fp4_49[1]);
                    float2 r_79_1 = fma_f32x2_rn_noftz(_f2_557, _f2_546, _f2_556);
                    d[8] = r_79_1.x;
                    d[9] = r_79_1.y;
                    float2 _f2_558 = make_float2(d[10], d[11]);
                    float2 _f2_559 = make_float2(_unpack_fp4_49[2], _unpack_fp4_49[3]);
                    float2 r_80_1 = fma_f32x2_rn_noftz(_f2_559, _f2_546, _f2_558);
                    d[10] = r_80_1.x;
                    d[11] = r_80_1.y;
                    float2 _f2_560 = make_float2(d[12], d[13]);
                    float2 _f2_561 = make_float2(_unpack_fp4_49[4], _unpack_fp4_49[5]);
                    float2 r_81_1 = fma_f32x2_rn_noftz(_f2_561, _f2_546, _f2_560);
                    d[12] = r_81_1.x;
                    d[13] = r_81_1.y;
                    float2 _f2_562 = make_float2(d[14], d[15]);
                    float2 _f2_563 = make_float2(_unpack_fp4_49[6], _unpack_fp4_49[7]);
                    float2 r_82_1 = fma_f32x2_rn_noftz(_f2_563, _f2_546, _f2_562);
                    d[14] = r_82_1.x;
                    d[15] = r_82_1.y;
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
                    float2 _f2_564 = make_float2(d[16], d[17]);
                    float2 _f2_565 = make_float2(_unpack_fp4_50[0], _unpack_fp4_50[1]);
                    float2 r_83_1 = fma_f32x2_rn_noftz(_f2_565, _f2_547, _f2_564);
                    d[16] = r_83_1.x;
                    d[17] = r_83_1.y;
                    float2 _f2_566 = make_float2(d[18], d[19]);
                    float2 _f2_567 = make_float2(_unpack_fp4_50[2], _unpack_fp4_50[3]);
                    float2 r_84_1 = fma_f32x2_rn_noftz(_f2_567, _f2_547, _f2_566);
                    d[18] = r_84_1.x;
                    d[19] = r_84_1.y;
                    float2 _f2_568 = make_float2(d[20], d[21]);
                    float2 _f2_569 = make_float2(_unpack_fp4_50[4], _unpack_fp4_50[5]);
                    float2 r_85_1 = fma_f32x2_rn_noftz(_f2_569, _f2_547, _f2_568);
                    d[20] = r_85_1.x;
                    d[21] = r_85_1.y;
                    float2 _f2_570 = make_float2(d[22], d[23]);
                    float2 _f2_571 = make_float2(_unpack_fp4_50[6], _unpack_fp4_50[7]);
                    float2 r_86_1 = fma_f32x2_rn_noftz(_f2_571, _f2_547, _f2_570);
                    d[22] = r_86_1.x;
                    d[23] = r_86_1.y;
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
                    float2 _f2_572 = make_float2(d[24], d[25]);
                    float2 _f2_573 = make_float2(_unpack_fp4_51[0], _unpack_fp4_51[1]);
                    float2 r_87_1 = fma_f32x2_rn_noftz(_f2_573, _f2_547, _f2_572);
                    d[24] = r_87_1.x;
                    d[25] = r_87_1.y;
                    float2 _f2_574 = make_float2(d[26], d[27]);
                    float2 _f2_575 = make_float2(_unpack_fp4_51[2], _unpack_fp4_51[3]);
                    float2 r_88 = fma_f32x2_rn_noftz(_f2_575, _f2_547, _f2_574);
                    d[26] = r_88.x;
                    d[27] = r_88.y;
                    float2 _f2_576 = make_float2(d[28], d[29]);
                    float2 _f2_577 = make_float2(_unpack_fp4_51[4], _unpack_fp4_51[5]);
                    float2 r_89 = fma_f32x2_rn_noftz(_f2_577, _f2_547, _f2_576);
                    d[28] = r_89.x;
                    d[29] = r_89.y;
                    float2 _f2_578 = make_float2(d[30], d[31]);
                    float2 _f2_579 = make_float2(_unpack_fp4_51[6], _unpack_fp4_51[7]);
                    float2 r_90_1 = fma_f32x2_rn_noftz(_f2_579, _f2_547, _f2_578);
                    d[30] = r_90_1.x;
                    d[31] = r_90_1.y;
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
                    long long idx = brow64 * 3072 + (long long)(4 * (128 * j + c));
                    {
                        int4 _iv4 = make_int4(w4[0 + 0], w4[0 + 1], w4[0 + 2], w4[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx)) = _iv4;
                    }
                    int c_91 = lane_1 + 32;
                    int l_92 = c_91 >> 2;
                    unsigned int w4_93[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_93[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_93[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_93[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_93[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_92 + (c_91 & 3 ^ l_92 >> 1 & 3)))));
                    long long idx_94 = brow64 * 3072 + (long long)(4 * (128 * j + c_91));
                    {
                        int4 _iv4 = make_int4(w4_93[0 + 0], w4_93[0 + 1], w4_93[0 + 2], w4_93[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_94)) = _iv4;
                    }
                    int c_95 = lane_1 + 64;
                    int l_96 = c_95 >> 2;
                    unsigned int w4_97[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_97[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_97[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_97[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_97[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_96 + (c_95 & 3 ^ l_96 >> 1 & 3)))));
                    long long idx_98 = brow64 * 3072 + (long long)(4 * (128 * j + c_95));
                    {
                        int4 _iv4 = make_int4(w4_97[0 + 0], w4_97[0 + 1], w4_97[0 + 2], w4_97[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_98)) = _iv4;
                    }
                    int c_99 = lane_1 + 96;
                    int l_100 = c_99 >> 2;
                    unsigned int w4_101[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_101[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_101[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_101[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_101[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_100 + (c_99 & 3 ^ l_100 >> 1 & 3)))));
                    long long idx_102 = brow64 * 3072 + (long long)(4 * (128 * j + c_99));
                    {
                        int4 _iv4 = make_int4(w4_101[0 + 0], w4_101[0 + 1], w4_101[0 + 2], w4_101[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_102)) = _iv4;
                    }
                    __syncwarp();
                }
                if (lane_1 < 4) {
                    wsm32[(bob_off + 3072 + 384 >> 2) + lane_1] = rs2;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane_1 == 0) {
                    {
                        void* _cpbulk_dst_181 = reinterpret_cast<void*>(peer_base_2 + (ag_base + bix64 * 3072));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_181), "r"(wsm_addr + (unsigned int)bob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_182 = reinterpret_cast<void*>(peer_base_2 + (ag_base + 50331648 + bix64 * 384));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_182), "r"(wsm_addr + (unsigned int)bob_off + 3072), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_183 = reinterpret_cast<void*>(peer_base_2 + (ag_base + 56623104 + bix64 * 16));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_183), "r"(wsm_addr + (unsigned int)bob_off + 3072 + 384), "r"((uint32_t)(16))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_184 = reinterpret_cast<void*>(peer_base_3 + (ag_base + bix64 * 3072));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_184), "r"(wsm_addr + (unsigned int)bob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_185 = reinterpret_cast<void*>(peer_base_3 + (ag_base + 50331648 + bix64 * 384));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_185), "r"(wsm_addr + (unsigned int)bob_off + 3072), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_186 = reinterpret_cast<void*>(peer_base_3 + (ag_base + 56623104 + bix64 * 16));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_186), "r"(wsm_addr + (unsigned int)bob_off + 3072 + 384), "r"((uint32_t)(16))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_187 = reinterpret_cast<void*>(peer_base_4 + (ag_base + bix64 * 3072));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_187), "r"(wsm_addr + (unsigned int)bob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_188 = reinterpret_cast<void*>(peer_base_4 + (ag_base + 50331648 + bix64 * 384));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_188), "r"(wsm_addr + (unsigned int)bob_off + 3072), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_189 = reinterpret_cast<void*>(peer_base_4 + (ag_base + 56623104 + bix64 * 16));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_189), "r"(wsm_addr + (unsigned int)bob_off + 3072 + 384), "r"((uint32_t)(16))
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
                int w_1 = c_1 % 3;
                int bump = ((w_1 >= fme) ? 1 : 0);
                int crow = r0 + blk * 4 + w_1 + bump;
                long long co64 = (long long)(crow % 4);
                long long cix64 = (long long)(crow / 4);
                long long crow64 = (long long)crow;
                long long csrc64 = co64;
                float rs_2 = 0.0f;
                uint32_t _sysv_poll32_group_4[1];
                do {
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_4[0]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (ag_base + 56623104 + (csrc64 * 4096 + cix64) * 16))) : "memory");
                } while (
                    (_sysv_poll32_group_4[0] == 0x80000000u)
                );
                rs_2 = __uint_as_float(_sysv_poll32_group_4[0]);
                #pragma unroll 1
                for (int jj_1 = 0; jj_1 < 2; jj_1++) {
                    int pp0_1 = lane_1 + 96 * jj_1;
                    int pp1_1 = pp0_1 + 32;
                    int pp2_1 = pp0_1 + 64;
                    uint32_t _fp4_wire_values_1[12];
                    uint32_t _fp4_wire_scales_1[3];
                    uint16_t _wire_h16_191_0;
                    uint16_t _wire_h16_191_1;
                    uint16_t _wire_h16_191_2;
                    uint32_t _wire_vpend_191 = 7u;
                    uint32_t _wire_spend_191 = 7u;
                    do {
                        if (_wire_vpend_191 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[0]), "=r"(_fp4_wire_values_1[1]), "=r"(_fp4_wire_values_1[2]), "=r"(_fp4_wire_values_1[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (ag_base + (csrc64 * 4096 + cix64) * 3072 + (long long)(16 * pp0_1)))) : "memory");
                        }
                        if (_wire_vpend_191 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[4]), "=r"(_fp4_wire_values_1[5]), "=r"(_fp4_wire_values_1[6]), "=r"(_fp4_wire_values_1[7]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (ag_base + (csrc64 * 4096 + cix64) * 3072 + (long long)(16 * pp1_1)))) : "memory");
                        }
                        if (_wire_vpend_191 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[8]), "=r"(_fp4_wire_values_1[9]), "=r"(_fp4_wire_values_1[10]), "=r"(_fp4_wire_values_1[11]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (ag_base + (csrc64 * 4096 + cix64) * 3072 + (long long)(16 * pp2_1)))) : "memory");
                        }
                        if (_wire_spend_191 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_191_0) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (ag_base + 50331648 + (csrc64 * 4096 + cix64) * 384 + (long long)(2 * pp0_1)))) : "memory");
                            _fp4_wire_scales_1[0] = (uint32_t)_wire_h16_191_0;
                        }
                        if (_wire_spend_191 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_191_1) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (ag_base + 50331648 + (csrc64 * 4096 + cix64) * 384 + (long long)(2 * pp1_1)))) : "memory");
                            _fp4_wire_scales_1[1] = (uint32_t)_wire_h16_191_1;
                        }
                        if (_wire_spend_191 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_191_2) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (ag_base + 50331648 + (csrc64 * 4096 + cix64) * 384 + (long long)(2 * pp2_1)))) : "memory");
                            _fp4_wire_scales_1[2] = (uint32_t)_wire_h16_191_2;
                        }
                        if (_wire_vpend_191 & 1u) {
                            if (!((((_fp4_wire_values_1[0] & ~((_fp4_wire_values_1[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[1] & ~((_fp4_wire_values_1[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[2] & ~((_fp4_wire_values_1[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[3] & ~((_fp4_wire_values_1[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_191 &= ~1u;
                        }
                        if (_wire_vpend_191 & 2u) {
                            if (!((((_fp4_wire_values_1[4] & ~((_fp4_wire_values_1[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[5] & ~((_fp4_wire_values_1[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[6] & ~((_fp4_wire_values_1[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[7] & ~((_fp4_wire_values_1[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_191 &= ~2u;
                        }
                        if (_wire_vpend_191 & 4u) {
                            if (!((((_fp4_wire_values_1[8] & ~((_fp4_wire_values_1[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[9] & ~((_fp4_wire_values_1[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[10] & ~((_fp4_wire_values_1[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_1[11] & ~((_fp4_wire_values_1[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_191 &= ~4u;
                        }
                        if (_wire_spend_191 & 1u) {
                            if (!(((_fp4_wire_scales_1[0] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_1[0] >> 8) == 0xFFu))) _wire_spend_191 &= ~1u;
                        }
                        if (_wire_spend_191 & 2u) {
                            if (!(((_fp4_wire_scales_1[1] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_1[1] >> 8) == 0xFFu))) _wire_spend_191 &= ~2u;
                        }
                        if (_wire_spend_191 & 4u) {
                            if (!(((_fp4_wire_scales_1[2] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_1[2] >> 8) == 0xFFu))) _wire_spend_191 &= ~4u;
                        }
                    } while ((_wire_vpend_191 | _wire_spend_191) != 0u);
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
                    float2 _fp8x2_decode_20;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_20.x), "=f"(_fp8x2_decode_20.y) : "h"((uint16_t)_fp4_wire_scales_1[0]));
                    float _mul_40;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_40) : "f"(_fp8x2_decode_20.x), "f"(rs_2));
                    float s0_2 = _mul_40;
                    float _mul_41;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_41) : "f"(_fp8x2_decode_20.y), "f"(rs_2));
                    float s1_2 = _mul_41;
                    float2 _f2_580 = make_float2(s0_2, s0_2);
                    float2 _f2_581 = make_float2(s1_2, s1_2);
                    float _unpack_fp4_52[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[0]));
                        (&_unpack_fp4_52[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_52[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_52[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_52[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_52[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_52[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_52[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_52[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_52[0]), "=f"(_unpack_fp4_52[1]), "=f"(_unpack_fp4_52[2]), "=f"(_unpack_fp4_52[3]), "=f"(_unpack_fp4_52[4]), "=f"(_unpack_fp4_52[5]), "=f"(_unpack_fp4_52[6]), "=f"(_unpack_fp4_52[7]) : "r"(_fp4_wire_values_1[0]));
                        #endif
                    }
                    float2 _f2_582 = make_float2(acc_1[0], acc_1[1]);
                    float2 _f2_583 = make_float2(_unpack_fp4_52[0], _unpack_fp4_52[1]);
                    float2 r_16 = fma_f32x2_rn_noftz(_f2_583, _f2_580, _f2_582);
                    acc_1[0] = r_16.x;
                    acc_1[1] = r_16.y;
                    float2 _f2_584 = make_float2(acc_1[2], acc_1[3]);
                    float2 _f2_585 = make_float2(_unpack_fp4_52[2], _unpack_fp4_52[3]);
                    float2 r_0_2 = fma_f32x2_rn_noftz(_f2_585, _f2_580, _f2_584);
                    acc_1[2] = r_0_2.x;
                    acc_1[3] = r_0_2.y;
                    float2 _f2_586 = make_float2(acc_1[4], acc_1[5]);
                    float2 _f2_587 = make_float2(_unpack_fp4_52[4], _unpack_fp4_52[5]);
                    float2 r_1_2 = fma_f32x2_rn_noftz(_f2_587, _f2_580, _f2_586);
                    acc_1[4] = r_1_2.x;
                    acc_1[5] = r_1_2.y;
                    float2 _f2_588 = make_float2(acc_1[6], acc_1[7]);
                    float2 _f2_589 = make_float2(_unpack_fp4_52[6], _unpack_fp4_52[7]);
                    float2 r_2_2 = fma_f32x2_rn_noftz(_f2_589, _f2_580, _f2_588);
                    acc_1[6] = r_2_2.x;
                    acc_1[7] = r_2_2.y;
                    float _unpack_fp4_53[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[1 + 0]));
                        (&_unpack_fp4_53[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_53[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_53[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_53[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_53[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_53[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_53[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_53[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_53[0]), "=f"(_unpack_fp4_53[1]), "=f"(_unpack_fp4_53[2]), "=f"(_unpack_fp4_53[3]), "=f"(_unpack_fp4_53[4]), "=f"(_unpack_fp4_53[5]), "=f"(_unpack_fp4_53[6]), "=f"(_unpack_fp4_53[7]) : "r"(_fp4_wire_values_1[1 + 0]));
                        #endif
                    }
                    float2 _f2_590 = make_float2(acc_1[8], acc_1[9]);
                    float2 _f2_591 = make_float2(_unpack_fp4_53[0], _unpack_fp4_53[1]);
                    float2 r_3_2 = fma_f32x2_rn_noftz(_f2_591, _f2_580, _f2_590);
                    acc_1[8] = r_3_2.x;
                    acc_1[9] = r_3_2.y;
                    float2 _f2_592 = make_float2(acc_1[10], acc_1[11]);
                    float2 _f2_593 = make_float2(_unpack_fp4_53[2], _unpack_fp4_53[3]);
                    float2 r_4_2 = fma_f32x2_rn_noftz(_f2_593, _f2_580, _f2_592);
                    acc_1[10] = r_4_2.x;
                    acc_1[11] = r_4_2.y;
                    float2 _f2_594 = make_float2(acc_1[12], acc_1[13]);
                    float2 _f2_595 = make_float2(_unpack_fp4_53[4], _unpack_fp4_53[5]);
                    float2 r_5_2 = fma_f32x2_rn_noftz(_f2_595, _f2_580, _f2_594);
                    acc_1[12] = r_5_2.x;
                    acc_1[13] = r_5_2.y;
                    float2 _f2_596 = make_float2(acc_1[14], acc_1[15]);
                    float2 _f2_597 = make_float2(_unpack_fp4_53[6], _unpack_fp4_53[7]);
                    float2 r_6_2 = fma_f32x2_rn_noftz(_f2_597, _f2_580, _f2_596);
                    acc_1[14] = r_6_2.x;
                    acc_1[15] = r_6_2.y;
                    float _unpack_fp4_54[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[2 + 0]));
                        (&_unpack_fp4_54[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_54[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_54[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_54[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_54[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_54[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_54[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_54[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_54[0]), "=f"(_unpack_fp4_54[1]), "=f"(_unpack_fp4_54[2]), "=f"(_unpack_fp4_54[3]), "=f"(_unpack_fp4_54[4]), "=f"(_unpack_fp4_54[5]), "=f"(_unpack_fp4_54[6]), "=f"(_unpack_fp4_54[7]) : "r"(_fp4_wire_values_1[2 + 0]));
                        #endif
                    }
                    float2 _f2_598 = make_float2(acc_1[16], acc_1[17]);
                    float2 _f2_599 = make_float2(_unpack_fp4_54[0], _unpack_fp4_54[1]);
                    float2 r_7_2 = fma_f32x2_rn_noftz(_f2_599, _f2_581, _f2_598);
                    acc_1[16] = r_7_2.x;
                    acc_1[17] = r_7_2.y;
                    float2 _f2_600 = make_float2(acc_1[18], acc_1[19]);
                    float2 _f2_601 = make_float2(_unpack_fp4_54[2], _unpack_fp4_54[3]);
                    float2 r_8_2 = fma_f32x2_rn_noftz(_f2_601, _f2_581, _f2_600);
                    acc_1[18] = r_8_2.x;
                    acc_1[19] = r_8_2.y;
                    float2 _f2_602 = make_float2(acc_1[20], acc_1[21]);
                    float2 _f2_603 = make_float2(_unpack_fp4_54[4], _unpack_fp4_54[5]);
                    float2 r_9_2 = fma_f32x2_rn_noftz(_f2_603, _f2_581, _f2_602);
                    acc_1[20] = r_9_2.x;
                    acc_1[21] = r_9_2.y;
                    float2 _f2_604 = make_float2(acc_1[22], acc_1[23]);
                    float2 _f2_605 = make_float2(_unpack_fp4_54[6], _unpack_fp4_54[7]);
                    float2 r_10_2 = fma_f32x2_rn_noftz(_f2_605, _f2_581, _f2_604);
                    acc_1[22] = r_10_2.x;
                    acc_1[23] = r_10_2.y;
                    float _unpack_fp4_55[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[3 + 0]));
                        (&_unpack_fp4_55[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_55[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_55[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_55[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_55[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_55[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_55[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_55[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_55[0]), "=f"(_unpack_fp4_55[1]), "=f"(_unpack_fp4_55[2]), "=f"(_unpack_fp4_55[3]), "=f"(_unpack_fp4_55[4]), "=f"(_unpack_fp4_55[5]), "=f"(_unpack_fp4_55[6]), "=f"(_unpack_fp4_55[7]) : "r"(_fp4_wire_values_1[3 + 0]));
                        #endif
                    }
                    float2 _f2_606 = make_float2(acc_1[24], acc_1[25]);
                    float2 _f2_607 = make_float2(_unpack_fp4_55[0], _unpack_fp4_55[1]);
                    float2 r_11_2 = fma_f32x2_rn_noftz(_f2_607, _f2_581, _f2_606);
                    acc_1[24] = r_11_2.x;
                    acc_1[25] = r_11_2.y;
                    float2 _f2_608 = make_float2(acc_1[26], acc_1[27]);
                    float2 _f2_609 = make_float2(_unpack_fp4_55[2], _unpack_fp4_55[3]);
                    float2 r_12_2 = fma_f32x2_rn_noftz(_f2_609, _f2_581, _f2_608);
                    acc_1[26] = r_12_2.x;
                    acc_1[27] = r_12_2.y;
                    float2 _f2_610 = make_float2(acc_1[28], acc_1[29]);
                    float2 _f2_611 = make_float2(_unpack_fp4_55[4], _unpack_fp4_55[5]);
                    float2 r_13_2 = fma_f32x2_rn_noftz(_f2_611, _f2_581, _f2_610);
                    acc_1[28] = r_13_2.x;
                    acc_1[29] = r_13_2.y;
                    float2 _f2_612 = make_float2(acc_1[30], acc_1[31]);
                    float2 _f2_613 = make_float2(_unpack_fp4_55[6], _unpack_fp4_55[7]);
                    float2 r_14_2 = fma_f32x2_rn_noftz(_f2_613, _f2_581, _f2_612);
                    acc_1[30] = r_14_2.x;
                    acc_1[31] = r_14_2.y;
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
                    long long idx_1 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj_1) + c_15));
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
                    long long idx_19 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj_1) + c_16));
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
                    long long idx_23 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj_1) + c_20));
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
                    long long idx_27 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj_1) + c_24));
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
                    float2 _fp8x2_decode_21;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_21.x), "=f"(_fp8x2_decode_21.y) : "h"((uint16_t)_fp4_wire_scales_1[1]));
                    float _mul_42;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_42) : "f"(_fp8x2_decode_21.x), "f"(rs_2));
                    float s0_29 = _mul_42;
                    float _mul_43;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_43) : "f"(_fp8x2_decode_21.y), "f"(rs_2));
                    float s1_30 = _mul_43;
                    float2 _f2_614 = make_float2(s0_29, s0_29);
                    float2 _f2_615 = make_float2(s1_30, s1_30);
                    float _unpack_fp4_56[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[4 + 0]));
                        (&_unpack_fp4_56[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_56[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_56[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_56[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_56[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_56[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_56[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_56[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_56[0]), "=f"(_unpack_fp4_56[1]), "=f"(_unpack_fp4_56[2]), "=f"(_unpack_fp4_56[3]), "=f"(_unpack_fp4_56[4]), "=f"(_unpack_fp4_56[5]), "=f"(_unpack_fp4_56[6]), "=f"(_unpack_fp4_56[7]) : "r"(_fp4_wire_values_1[4 + 0]));
                        #endif
                    }
                    float2 _f2_616 = make_float2(acc_28[0], acc_28[1]);
                    float2 _f2_617 = make_float2(_unpack_fp4_56[0], _unpack_fp4_56[1]);
                    float2 r_31_2 = fma_f32x2_rn_noftz(_f2_617, _f2_614, _f2_616);
                    acc_28[0] = r_31_2.x;
                    acc_28[1] = r_31_2.y;
                    float2 _f2_618 = make_float2(acc_28[2], acc_28[3]);
                    float2 _f2_619 = make_float2(_unpack_fp4_56[2], _unpack_fp4_56[3]);
                    float2 r_32_2 = fma_f32x2_rn_noftz(_f2_619, _f2_614, _f2_618);
                    acc_28[2] = r_32_2.x;
                    acc_28[3] = r_32_2.y;
                    float2 _f2_620 = make_float2(acc_28[4], acc_28[5]);
                    float2 _f2_621 = make_float2(_unpack_fp4_56[4], _unpack_fp4_56[5]);
                    float2 r_33 = fma_f32x2_rn_noftz(_f2_621, _f2_614, _f2_620);
                    acc_28[4] = r_33.x;
                    acc_28[5] = r_33.y;
                    float2 _f2_622 = make_float2(acc_28[6], acc_28[7]);
                    float2 _f2_623 = make_float2(_unpack_fp4_56[6], _unpack_fp4_56[7]);
                    float2 r_34 = fma_f32x2_rn_noftz(_f2_623, _f2_614, _f2_622);
                    acc_28[6] = r_34.x;
                    acc_28[7] = r_34.y;
                    float _unpack_fp4_57[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[5 + 0]));
                        (&_unpack_fp4_57[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_57[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_57[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_57[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_57[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_57[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_57[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_57[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_57[0]), "=f"(_unpack_fp4_57[1]), "=f"(_unpack_fp4_57[2]), "=f"(_unpack_fp4_57[3]), "=f"(_unpack_fp4_57[4]), "=f"(_unpack_fp4_57[5]), "=f"(_unpack_fp4_57[6]), "=f"(_unpack_fp4_57[7]) : "r"(_fp4_wire_values_1[5 + 0]));
                        #endif
                    }
                    float2 _f2_624 = make_float2(acc_28[8], acc_28[9]);
                    float2 _f2_625 = make_float2(_unpack_fp4_57[0], _unpack_fp4_57[1]);
                    float2 r_35_2 = fma_f32x2_rn_noftz(_f2_625, _f2_614, _f2_624);
                    acc_28[8] = r_35_2.x;
                    acc_28[9] = r_35_2.y;
                    float2 _f2_626 = make_float2(acc_28[10], acc_28[11]);
                    float2 _f2_627 = make_float2(_unpack_fp4_57[2], _unpack_fp4_57[3]);
                    float2 r_36_2 = fma_f32x2_rn_noftz(_f2_627, _f2_614, _f2_626);
                    acc_28[10] = r_36_2.x;
                    acc_28[11] = r_36_2.y;
                    float2 _f2_628 = make_float2(acc_28[12], acc_28[13]);
                    float2 _f2_629 = make_float2(_unpack_fp4_57[4], _unpack_fp4_57[5]);
                    float2 r_37_2 = fma_f32x2_rn_noftz(_f2_629, _f2_614, _f2_628);
                    acc_28[12] = r_37_2.x;
                    acc_28[13] = r_37_2.y;
                    float2 _f2_630 = make_float2(acc_28[14], acc_28[15]);
                    float2 _f2_631 = make_float2(_unpack_fp4_57[6], _unpack_fp4_57[7]);
                    float2 r_38_2 = fma_f32x2_rn_noftz(_f2_631, _f2_614, _f2_630);
                    acc_28[14] = r_38_2.x;
                    acc_28[15] = r_38_2.y;
                    float _unpack_fp4_58[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[6 + 0]));
                        (&_unpack_fp4_58[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_58[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_58[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_58[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_58[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_58[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_58[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_58[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_58[0]), "=f"(_unpack_fp4_58[1]), "=f"(_unpack_fp4_58[2]), "=f"(_unpack_fp4_58[3]), "=f"(_unpack_fp4_58[4]), "=f"(_unpack_fp4_58[5]), "=f"(_unpack_fp4_58[6]), "=f"(_unpack_fp4_58[7]) : "r"(_fp4_wire_values_1[6 + 0]));
                        #endif
                    }
                    float2 _f2_632 = make_float2(acc_28[16], acc_28[17]);
                    float2 _f2_633 = make_float2(_unpack_fp4_58[0], _unpack_fp4_58[1]);
                    float2 r_39_2 = fma_f32x2_rn_noftz(_f2_633, _f2_615, _f2_632);
                    acc_28[16] = r_39_2.x;
                    acc_28[17] = r_39_2.y;
                    float2 _f2_634 = make_float2(acc_28[18], acc_28[19]);
                    float2 _f2_635 = make_float2(_unpack_fp4_58[2], _unpack_fp4_58[3]);
                    float2 r_40_2 = fma_f32x2_rn_noftz(_f2_635, _f2_615, _f2_634);
                    acc_28[18] = r_40_2.x;
                    acc_28[19] = r_40_2.y;
                    float2 _f2_636 = make_float2(acc_28[20], acc_28[21]);
                    float2 _f2_637 = make_float2(_unpack_fp4_58[4], _unpack_fp4_58[5]);
                    float2 r_41_2 = fma_f32x2_rn_noftz(_f2_637, _f2_615, _f2_636);
                    acc_28[20] = r_41_2.x;
                    acc_28[21] = r_41_2.y;
                    float2 _f2_638 = make_float2(acc_28[22], acc_28[23]);
                    float2 _f2_639 = make_float2(_unpack_fp4_58[6], _unpack_fp4_58[7]);
                    float2 r_42_2 = fma_f32x2_rn_noftz(_f2_639, _f2_615, _f2_638);
                    acc_28[22] = r_42_2.x;
                    acc_28[23] = r_42_2.y;
                    float _unpack_fp4_59[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[7 + 0]));
                        (&_unpack_fp4_59[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_59[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_59[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_59[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_59[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_59[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_59[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_59[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_59[0]), "=f"(_unpack_fp4_59[1]), "=f"(_unpack_fp4_59[2]), "=f"(_unpack_fp4_59[3]), "=f"(_unpack_fp4_59[4]), "=f"(_unpack_fp4_59[5]), "=f"(_unpack_fp4_59[6]), "=f"(_unpack_fp4_59[7]) : "r"(_fp4_wire_values_1[7 + 0]));
                        #endif
                    }
                    float2 _f2_640 = make_float2(acc_28[24], acc_28[25]);
                    float2 _f2_641 = make_float2(_unpack_fp4_59[0], _unpack_fp4_59[1]);
                    float2 r_43_2 = fma_f32x2_rn_noftz(_f2_641, _f2_615, _f2_640);
                    acc_28[24] = r_43_2.x;
                    acc_28[25] = r_43_2.y;
                    float2 _f2_642 = make_float2(acc_28[26], acc_28[27]);
                    float2 _f2_643 = make_float2(_unpack_fp4_59[2], _unpack_fp4_59[3]);
                    float2 r_44_2 = fma_f32x2_rn_noftz(_f2_643, _f2_615, _f2_642);
                    acc_28[26] = r_44_2.x;
                    acc_28[27] = r_44_2.y;
                    float2 _f2_644 = make_float2(acc_28[28], acc_28[29]);
                    float2 _f2_645 = make_float2(_unpack_fp4_59[4], _unpack_fp4_59[5]);
                    float2 r_45_2 = fma_f32x2_rn_noftz(_f2_645, _f2_615, _f2_644);
                    acc_28[28] = r_45_2.x;
                    acc_28[29] = r_45_2.y;
                    float2 _f2_646 = make_float2(acc_28[30], acc_28[31]);
                    float2 _f2_647 = make_float2(_unpack_fp4_59[6], _unpack_fp4_59[7]);
                    float2 r_46_2 = fma_f32x2_rn_noftz(_f2_647, _f2_615, _f2_646);
                    acc_28[30] = r_46_2.x;
                    acc_28[31] = r_46_2.y;
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
                    long long idx_50 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 1) + c_47));
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
                    long long idx_54 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 1) + c_51));
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
                    long long idx_58 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 1) + c_55));
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
                    long long idx_62 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 1) + c_59));
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
                    float2 _fp8x2_decode_22;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_22.x), "=f"(_fp8x2_decode_22.y) : "h"((uint16_t)_fp4_wire_scales_1[2]));
                    float _mul_44;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_44) : "f"(_fp8x2_decode_22.x), "f"(rs_2));
                    float s0_64 = _mul_44;
                    float _mul_45;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_45) : "f"(_fp8x2_decode_22.y), "f"(rs_2));
                    float s1_65 = _mul_45;
                    float2 _f2_648 = make_float2(s0_64, s0_64);
                    float2 _f2_649 = make_float2(s1_65, s1_65);
                    float _unpack_fp4_60[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[8 + 0]));
                        (&_unpack_fp4_60[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_60[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_60[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_60[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_60[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_60[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_60[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_60[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_60[0]), "=f"(_unpack_fp4_60[1]), "=f"(_unpack_fp4_60[2]), "=f"(_unpack_fp4_60[3]), "=f"(_unpack_fp4_60[4]), "=f"(_unpack_fp4_60[5]), "=f"(_unpack_fp4_60[6]), "=f"(_unpack_fp4_60[7]) : "r"(_fp4_wire_values_1[8 + 0]));
                        #endif
                    }
                    float2 _f2_650 = make_float2(acc_63[0], acc_63[1]);
                    float2 _f2_651 = make_float2(_unpack_fp4_60[0], _unpack_fp4_60[1]);
                    float2 r_66_2 = fma_f32x2_rn_noftz(_f2_651, _f2_648, _f2_650);
                    acc_63[0] = r_66_2.x;
                    acc_63[1] = r_66_2.y;
                    float2 _f2_652 = make_float2(acc_63[2], acc_63[3]);
                    float2 _f2_653 = make_float2(_unpack_fp4_60[2], _unpack_fp4_60[3]);
                    float2 r_67_2 = fma_f32x2_rn_noftz(_f2_653, _f2_648, _f2_652);
                    acc_63[2] = r_67_2.x;
                    acc_63[3] = r_67_2.y;
                    float2 _f2_654 = make_float2(acc_63[4], acc_63[5]);
                    float2 _f2_655 = make_float2(_unpack_fp4_60[4], _unpack_fp4_60[5]);
                    float2 r_68_2 = fma_f32x2_rn_noftz(_f2_655, _f2_648, _f2_654);
                    acc_63[4] = r_68_2.x;
                    acc_63[5] = r_68_2.y;
                    float2 _f2_656 = make_float2(acc_63[6], acc_63[7]);
                    float2 _f2_657 = make_float2(_unpack_fp4_60[6], _unpack_fp4_60[7]);
                    float2 r_69 = fma_f32x2_rn_noftz(_f2_657, _f2_648, _f2_656);
                    acc_63[6] = r_69.x;
                    acc_63[7] = r_69.y;
                    float _unpack_fp4_61[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[9 + 0]));
                        (&_unpack_fp4_61[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_61[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_61[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_61[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_61[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_61[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_61[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_61[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_61[0]), "=f"(_unpack_fp4_61[1]), "=f"(_unpack_fp4_61[2]), "=f"(_unpack_fp4_61[3]), "=f"(_unpack_fp4_61[4]), "=f"(_unpack_fp4_61[5]), "=f"(_unpack_fp4_61[6]), "=f"(_unpack_fp4_61[7]) : "r"(_fp4_wire_values_1[9 + 0]));
                        #endif
                    }
                    float2 _f2_658 = make_float2(acc_63[8], acc_63[9]);
                    float2 _f2_659 = make_float2(_unpack_fp4_61[0], _unpack_fp4_61[1]);
                    float2 r_70 = fma_f32x2_rn_noftz(_f2_659, _f2_648, _f2_658);
                    acc_63[8] = r_70.x;
                    acc_63[9] = r_70.y;
                    float2 _f2_660 = make_float2(acc_63[10], acc_63[11]);
                    float2 _f2_661 = make_float2(_unpack_fp4_61[2], _unpack_fp4_61[3]);
                    float2 r_71 = fma_f32x2_rn_noftz(_f2_661, _f2_648, _f2_660);
                    acc_63[10] = r_71.x;
                    acc_63[11] = r_71.y;
                    float2 _f2_662 = make_float2(acc_63[12], acc_63[13]);
                    float2 _f2_663 = make_float2(_unpack_fp4_61[4], _unpack_fp4_61[5]);
                    float2 r_72_1 = fma_f32x2_rn_noftz(_f2_663, _f2_648, _f2_662);
                    acc_63[12] = r_72_1.x;
                    acc_63[13] = r_72_1.y;
                    float2 _f2_664 = make_float2(acc_63[14], acc_63[15]);
                    float2 _f2_665 = make_float2(_unpack_fp4_61[6], _unpack_fp4_61[7]);
                    float2 r_73_1 = fma_f32x2_rn_noftz(_f2_665, _f2_648, _f2_664);
                    acc_63[14] = r_73_1.x;
                    acc_63[15] = r_73_1.y;
                    float _unpack_fp4_62[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[10 + 0]));
                        (&_unpack_fp4_62[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_62[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_62[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_62[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_62[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_62[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_62[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_62[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_62[0]), "=f"(_unpack_fp4_62[1]), "=f"(_unpack_fp4_62[2]), "=f"(_unpack_fp4_62[3]), "=f"(_unpack_fp4_62[4]), "=f"(_unpack_fp4_62[5]), "=f"(_unpack_fp4_62[6]), "=f"(_unpack_fp4_62[7]) : "r"(_fp4_wire_values_1[10 + 0]));
                        #endif
                    }
                    float2 _f2_666 = make_float2(acc_63[16], acc_63[17]);
                    float2 _f2_667 = make_float2(_unpack_fp4_62[0], _unpack_fp4_62[1]);
                    float2 r_74_1 = fma_f32x2_rn_noftz(_f2_667, _f2_649, _f2_666);
                    acc_63[16] = r_74_1.x;
                    acc_63[17] = r_74_1.y;
                    float2 _f2_668 = make_float2(acc_63[18], acc_63[19]);
                    float2 _f2_669 = make_float2(_unpack_fp4_62[2], _unpack_fp4_62[3]);
                    float2 r_75_2 = fma_f32x2_rn_noftz(_f2_669, _f2_649, _f2_668);
                    acc_63[18] = r_75_2.x;
                    acc_63[19] = r_75_2.y;
                    float2 _f2_670 = make_float2(acc_63[20], acc_63[21]);
                    float2 _f2_671 = make_float2(_unpack_fp4_62[4], _unpack_fp4_62[5]);
                    float2 r_76_2 = fma_f32x2_rn_noftz(_f2_671, _f2_649, _f2_670);
                    acc_63[20] = r_76_2.x;
                    acc_63[21] = r_76_2.y;
                    float2 _f2_672 = make_float2(acc_63[22], acc_63[23]);
                    float2 _f2_673 = make_float2(_unpack_fp4_62[6], _unpack_fp4_62[7]);
                    float2 r_77_2 = fma_f32x2_rn_noftz(_f2_673, _f2_649, _f2_672);
                    acc_63[22] = r_77_2.x;
                    acc_63[23] = r_77_2.y;
                    float _unpack_fp4_63[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[11 + 0]));
                        (&_unpack_fp4_63[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_63[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_63[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_63[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_63[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_63[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_63[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_63[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_63[0]), "=f"(_unpack_fp4_63[1]), "=f"(_unpack_fp4_63[2]), "=f"(_unpack_fp4_63[3]), "=f"(_unpack_fp4_63[4]), "=f"(_unpack_fp4_63[5]), "=f"(_unpack_fp4_63[6]), "=f"(_unpack_fp4_63[7]) : "r"(_fp4_wire_values_1[11 + 0]));
                        #endif
                    }
                    float2 _f2_674 = make_float2(acc_63[24], acc_63[25]);
                    float2 _f2_675 = make_float2(_unpack_fp4_63[0], _unpack_fp4_63[1]);
                    float2 r_78_2 = fma_f32x2_rn_noftz(_f2_675, _f2_649, _f2_674);
                    acc_63[24] = r_78_2.x;
                    acc_63[25] = r_78_2.y;
                    float2 _f2_676 = make_float2(acc_63[26], acc_63[27]);
                    float2 _f2_677 = make_float2(_unpack_fp4_63[2], _unpack_fp4_63[3]);
                    float2 r_79_2 = fma_f32x2_rn_noftz(_f2_677, _f2_649, _f2_676);
                    acc_63[26] = r_79_2.x;
                    acc_63[27] = r_79_2.y;
                    float2 _f2_678 = make_float2(acc_63[28], acc_63[29]);
                    float2 _f2_679 = make_float2(_unpack_fp4_63[4], _unpack_fp4_63[5]);
                    float2 r_80_2 = fma_f32x2_rn_noftz(_f2_679, _f2_649, _f2_678);
                    acc_63[28] = r_80_2.x;
                    acc_63[29] = r_80_2.y;
                    float2 _f2_680 = make_float2(acc_63[30], acc_63[31]);
                    float2 _f2_681 = make_float2(_unpack_fp4_63[6], _unpack_fp4_63[7]);
                    float2 r_81_2 = fma_f32x2_rn_noftz(_f2_681, _f2_649, _f2_680);
                    acc_63[30] = r_81_2.x;
                    acc_63[31] = r_81_2.y;
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
                    long long idx_85 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 2) + c_82));
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
                    long long idx_89 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 2) + c_86));
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
                    long long idx_93 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 2) + c_90));
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
                    long long idx_97 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 2) + c_94));
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
                    unsigned int zero_2 = 0;
                    unsigned int rows_u = (unsigned int)rows;
                    unsigned int algo_u = 2;
                    ctrl[1] = zero_2;
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
                long long b64 = rs_base_1;
                long long g0 = lo + lane64;
#else
                long long lo = u * bid64_1 / g64_1 * 16;
                long long hi = u * (bid64_1 + 1) / g64_1 * 16;
                int npages = (int)((hi - lo + 4095) / 4096);
                int i0 = lane_0 - jbase & 31;
                long long pg = 4096;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1000
                for (long long g = g0; g < hi; g += 32) {
                    long long widx = b64 + 16 * g >> 2;
#else
                for (int i_2 = i0; i_2 < npages; i_2 += 32) {
                    long long o = lo + (long long)i_2 * 4096;
                    long long rem = hi - o;
                    unsigned int nb = (unsigned int)(((rem >= pg) ? pg : rem));
#endif
                    {
#if __CUDA_ARCH__ == 1000
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(self_i32 + (widx)) = _iv4;
#else
                        void* _cpbulk_dst_0 = reinterpret_cast<void*>(self_u8_1 + (rs_base_1 + o));
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
                long long b64_3 = rs_base_1 + 50331648;
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
                for (long long g_1 = g0_4; g_1 < hi_2; g_1 += 32) {
                    long long widx_1 = b64_3 + 16 * g_1 >> 2;
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
                        void* _cpbulk_dst_1 = reinterpret_cast<void*>(self_u8_1 + (rs_base_1 + 50331648 + o_1));
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
                long long b64_8 = rs_base_1 + 56623104;
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
                for (long long g_2 = g0_9; g_2 < hi_7; g_2 += 32) {
                    long long widx_2 = b64_8 + 16 * g_2 >> 2;
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
                        void* _cpbulk_dst_2 = reinterpret_cast<void*>(self_u8_1 + (rs_base_1 + 56623104 + o_2));
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
                long long b64_13 = rs_base_1 + 12582912;
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
                for (long long g_3 = g0_14; g_3 < hi_12; g_3 += 32) {
                    long long widx_3 = b64_13 + 16 * g_3 >> 2;
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
                        void* _cpbulk_dst_3 = reinterpret_cast<void*>(self_u8_1 + (rs_base_1 + 12582912 + o_3));
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
                long long b64_18 = rs_base_1 + 50331648 + 1572864;
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
                for (long long g_4 = g0_19; g_4 < hi_17; g_4 += 32) {
                    long long widx_4 = b64_18 + 16 * g_4 >> 2;
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
                        void* _cpbulk_dst_4 = reinterpret_cast<void*>(self_u8_1 + (rs_base_1 + 50331648 + 1572864 + o_4));
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
                long long b64_23 = rs_base_1 + 56623104 + 65536;
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
                for (long long g_5 = g0_24; g_5 < hi_22; g_5 += 32) {
                    long long widx_5 = b64_23 + 16 * g_5 >> 2;
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
                        void* _cpbulk_dst_5 = reinterpret_cast<void*>(self_u8_1 + (rs_base_1 + 56623104 + 65536 + o_5));
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
                long long b64_28 = rs_base_1 + 25165824;
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
                for (long long g_6 = g0_29; g_6 < hi_27; g_6 += 32) {
                    long long widx_6 = b64_28 + 16 * g_6 >> 2;
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
                        void* _cpbulk_dst_6 = reinterpret_cast<void*>(self_u8_1 + (rs_base_1 + 25165824 + o_6));
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
                long long b64_33 = rs_base_1 + 50331648 + 3145728;
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
                for (long long g_7 = g0_34; g_7 < hi_32; g_7 += 32) {
                    long long widx_7 = b64_33 + 16 * g_7 >> 2;
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
                        void* _cpbulk_dst_7 = reinterpret_cast<void*>(self_u8_1 + (rs_base_1 + 50331648 + 3145728 + o_7));
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
                long long b64_38 = rs_base_1 + 56623104 + 131072;
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
                for (long long g_8 = g0_39; g_8 < hi_37; g_8 += 32) {
                    long long widx_8 = b64_38 + 16 * g_8 >> 2;
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
                        void* _cpbulk_dst_8 = reinterpret_cast<void*>(self_u8_1 + (rs_base_1 + 56623104 + 131072 + o_8));
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
                long long b64_43 = rs_base_1 + 37748736;
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
                for (long long g_9 = g0_44; g_9 < hi_42; g_9 += 32) {
                    long long widx_9 = b64_43 + 16 * g_9 >> 2;
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
                        void* _cpbulk_dst_9 = reinterpret_cast<void*>(self_u8_1 + (rs_base_1 + 37748736 + o_9));
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
                long long b64_48 = rs_base_1 + 50331648 + 4718592;
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
                for (long long g_10 = g0_49; g_10 < hi_47; g_10 += 32) {
                    long long widx_10 = b64_48 + 16 * g_10 >> 2;
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
                        void* _cpbulk_dst_10 = reinterpret_cast<void*>(self_u8_1 + (rs_base_1 + 50331648 + 4718592 + o_10));
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
                long long b64_53 = rs_base_1 + 56623104 + 196608;
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
                for (long long g_11 = g0_54; g_11 < hi_52; g_11 += 32) {
                    long long widx_11 = b64_53 + 16 * g_11 >> 2;
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
                        void* _cpbulk_dst_11 = reinterpret_cast<void*>(self_u8_1 + (rs_base_1 + 56623104 + 196608 + o_11));
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
                long long b64_58 = ag_base_1 + 12582912;
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
                for (long long g_12 = g0_59; g_12 < hi_57; g_12 += 32) {
                    long long widx_12 = b64_58 + 16 * g_12 >> 2;
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
                        void* _cpbulk_dst_12 = reinterpret_cast<void*>(self_u8_1 + (ag_base_1 + 12582912 + o_12));
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
                long long b64_63 = ag_base_1 + 50331648 + 1572864;
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
                for (long long g_13 = g0_64; g_13 < hi_62; g_13 += 32) {
                    long long widx_13 = b64_63 + 16 * g_13 >> 2;
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
                        void* _cpbulk_dst_13 = reinterpret_cast<void*>(self_u8_1 + (ag_base_1 + 50331648 + 1572864 + o_13));
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
                long long b64_68 = ag_base_1 + 56623104 + 65536;
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
                for (long long g_14 = g0_69; g_14 < hi_67; g_14 += 32) {
                    long long widx_14 = b64_68 + 16 * g_14 >> 2;
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
                        void* _cpbulk_dst_14 = reinterpret_cast<void*>(self_u8_1 + (ag_base_1 + 56623104 + 65536 + o_14));
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
                long long b64_73 = ag_base_1 + 25165824;
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
                for (long long g_15 = g0_74; g_15 < hi_72; g_15 += 32) {
                    long long widx_15 = b64_73 + 16 * g_15 >> 2;
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
                        void* _cpbulk_dst_15 = reinterpret_cast<void*>(self_u8_1 + (ag_base_1 + 25165824 + o_15));
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
                long long b64_78 = ag_base_1 + 50331648 + 3145728;
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
                for (long long g_16 = g0_79; g_16 < hi_77; g_16 += 32) {
                    long long widx_16 = b64_78 + 16 * g_16 >> 2;
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
                        void* _cpbulk_dst_16 = reinterpret_cast<void*>(self_u8_1 + (ag_base_1 + 50331648 + 3145728 + o_16));
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
                long long b64_83 = ag_base_1 + 56623104 + 131072;
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
                for (long long g_17 = g0_84; g_17 < hi_82; g_17 += 32) {
                    long long widx_17 = b64_83 + 16 * g_17 >> 2;
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
                        void* _cpbulk_dst_17 = reinterpret_cast<void*>(self_u8_1 + (ag_base_1 + 56623104 + 131072 + o_17));
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
                long long b64_88 = ag_base_1 + 37748736;
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
                for (long long g_18 = g0_89; g_18 < hi_87; g_18 += 32) {
                    long long widx_18 = b64_88 + 16 * g_18 >> 2;
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
                        void* _cpbulk_dst_18 = reinterpret_cast<void*>(self_u8_1 + (ag_base_1 + 37748736 + o_18));
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
                long long b64_93 = ag_base_1 + 50331648 + 4718592;
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
                for (long long g_19 = g0_94; g_19 < hi_92; g_19 += 32) {
                    long long widx_19 = b64_93 + 16 * g_19 >> 2;
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
                        void* _cpbulk_dst_19 = reinterpret_cast<void*>(self_u8_1 + (ag_base_1 + 50331648 + 4718592 + o_19));
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
                long long b64_98 = ag_base_1 + 56623104 + 196608;
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
                for (long long g_20 = g0_99; g_20 < hi_97; g_20 += 32) {
                    long long widx_20 = b64_98 + 16 * g_20 >> 2;
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
                        void* _cpbulk_dst_20 = reinterpret_cast<void*>(self_u8_1 + (ag_base_1 + 56623104 + 196608 + o_20));
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
