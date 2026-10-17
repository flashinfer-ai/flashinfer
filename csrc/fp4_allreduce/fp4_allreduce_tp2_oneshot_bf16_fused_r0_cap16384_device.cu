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
#define SMEM_S_AM_OFF 13328
#define SMEM_S_AM_STAGE_BYTES 40
#define SMEM_S_AM_STRIDE 40
#define SMEM_WSM_OFF 13376
#define SMEM_WSM_STAGE_BYTES 212800
#define SMEM_WSM_STRIDE 212800
#define SMEM_WSM16_OFF 13376
#define SMEM_WSM16_STAGE_BYTES 212800
#define SMEM_WSM16_STRIDE 212800
#define SMEM_WSM32_OFF 13376
#define SMEM_WSM32_STAGE_BYTES 212800
#define SMEM_WSM32_STRIDE 212800
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
kernel_fp4_allreduce_tp2_oneshot_bf16_fused_r0_cap16384(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, unsigned int* __restrict__ ctrl, int rows, const unsigned long long* __restrict__ workspace)
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
            long long bid64 = (long long)bid;
            long long g64 = (long long)num_bids;
            long long rows64 = (long long)rows;
            int r0 = (int)(bid64 * rows64 / g64);
            int r1 = (int)((bid64 + 1) * rows64 / g64);
            int nrows = r1 - r0;
            int warp_0 = (int)warp;
            int lane_1 = (int)lane;
            int t = warp_0 >> 1;
            int h = warp_0 & 1;
            int leader = t * 2;
            if (lane_1 == 0 && h == 0 && nrows > t) {
                long long first_row = (long long)(r0 + t);
                mbarrier_arrive_expect_tx(xb_full_addr + (leader) * 8, 12288);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(leader * 21280), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(x) + ((unsigned long long)(first_row * 6144) * (unsigned long long)2)), 12288, xb_full_addr + (leader) * 8);
            }
            if (tid == 0) {
                int base = bid * 4;
                ctl[0] = ctrl[base];
                ctl[1] = ctrl[base + 2];
                ctl[2] = ctrl[base + 3];
            }
            asm volatile("barrier.sync 0, 352;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int call = ctl[0];
            unsigned int buf = call % 3;
            int* out_i32 = reinterpret_cast<int*>(out);
            int n = 0;
            if (nrows > t) {
                n = (nrows - t + 5 - 1) / 5;
            }
            int f = lane_1 >> 1 & 3;
            int xb_off = leader * 21280;
            int stg_off = warp_0 * 21280 + 12288 + 6944;
            uint8_t* self_u8 = reinterpret_cast<uint8_t*>(workspace[0]);
            uint8_t* peer_base = reinterpret_cast<uint8_t*>(workspace[1]);
            long long os_base = (long long)buf * 113770496;
            #pragma unroll 1
            for (int k = 0; k < n + 1; k++) {
                if (n > k) {
                    int slot = k % 2;
                    int ob_off = xb_off + 12288 + slot * 3472;
                    if (h == 0 && k > 1 && lane_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    float rs = 1.0f;
                    {
                        mbarrier_wait(xb_full_addr + (leader) * 8, k & 1);
                        unsigned int bmp[3];
                        float am = 0.0f;
                        int pp = lane_1 + 32 * (3 * h);
                        int base_1 = xb_off + 64 * pp;
                        unsigned int a = 0;
                        unsigned int b = 0;
                        unsigned int c0[4];
                        unsigned int c1[4];
                        unsigned int c2[4];
                        unsigned int c3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&c0[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base_1 + (unsigned int)(16 * f)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&c1[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base_1 + (unsigned int)(16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&c2[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base_1 + (unsigned int)(16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&c3[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base_1 + (unsigned int)(16 * (f ^ 3))));
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
                        int pp_0 = lane_1 + 32 * (3 * h + 1);
                        int base_1_1 = xb_off + 64 * pp_0;
                        unsigned int a_2 = 0;
                        unsigned int b_3 = 0;
                        unsigned int c0_4[4];
                        unsigned int c1_5[4];
                        unsigned int c2_6[4];
                        unsigned int c3_7[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&c0_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_4[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base_1_1 + (unsigned int)(16 * f)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&c1_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_5[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base_1_1 + (unsigned int)(16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&c2_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_6[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base_1_1 + (unsigned int)(16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&c3_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_7[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base_1_1 + (unsigned int)(16 * (f ^ 3))));
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
                        int pp_10 = lane_1 + 32 * (3 * h + 2);
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
                        float _warp_reduce_0 = am;
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            _warp_reduce_0 = fmaxf(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                        am = _warp_reduce_0;
                        if (lane_1 == 0) {
                            s_am[2 * t + h] = am;
                        }
                        asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                        float _fmax_3 = fmaxf(s_am[2 * t], s_am[2 * t + 1]);
                        am = _fmax_3;
                        float zero = 0.0f;
                        float six = 6.0f;
                        float target = 2688.0f;
                        float _fdiv_rn_0 = __fdiv_rn(target, am);
                        float inv = ((am > zero) ? _fdiv_rn_0 : zero);
                        float _fdiv_rn_1 = __fdiv_rn(am, target);
                        float rs_20 = _fdiv_rn_1;
                        int pp2 = lane_1 + 32 * (3 * h);
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
                        float m0 = ((zero < _fp8x2_decode_0.x) ? _fdiv_rn_4 : zero);
                        float m1 = ((zero < _fp8x2_decode_0.y) ? _fdiv_rn_5 : zero);
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
                        int cf_21 = f ^ 1;
                        unsigned int v4_22[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v4_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_22[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base2 + (unsigned int)(16 * cf_21)));
                        float m_23 = (((cf_21 & 2) != 0) ? m1 : m0);
                        float2 _f2_5 = make_float2(m_23, m_23);
                        float2 _f2_6 = make_float2(__uint_as_float(v4_22[0] << 16), __uint_as_float(v4_22[0] & 4294901760u));
                        float2 _mul_f32x2_4;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_6), "l"(*(const unsigned long long*)&_f2_5));
                        uint32_t _fp4_pair_4;
                        uint16_t _e2m1_pair_h_5;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_5) : "f"(_mul_f32x2_4.x), "f"(_mul_f32x2_4.y));
                        _fp4_pair_4 = (uint32_t)_e2m1_pair_h_5;
                        float2 _f2_7 = make_float2(__uint_as_float(v4_22[1] << 16), __uint_as_float(v4_22[1] & 4294901760u));
                        float2 _mul_f32x2_5;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_f2_7), "l"(*(const unsigned long long*)&_f2_5));
                        uint32_t _fp4_pair_5;
                        uint16_t _e2m1_pair_h_6;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_6) : "f"(_mul_f32x2_5.x), "f"(_mul_f32x2_5.y));
                        _fp4_pair_5 = (uint32_t)_e2m1_pair_h_6;
                        float2 _f2_8 = make_float2(__uint_as_float(v4_22[2] << 16), __uint_as_float(v4_22[2] & 4294901760u));
                        float2 _mul_f32x2_6;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&_f2_8), "l"(*(const unsigned long long*)&_f2_5));
                        uint32_t _fp4_pair_6;
                        uint16_t _e2m1_pair_h_7;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_7) : "f"(_mul_f32x2_6.x), "f"(_mul_f32x2_6.y));
                        _fp4_pair_6 = (uint32_t)_e2m1_pair_h_7;
                        float2 _f2_9 = make_float2(__uint_as_float(v4_22[3] << 16), __uint_as_float(v4_22[3] & 4294901760u));
                        float2 _mul_f32x2_7;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&_f2_9), "l"(*(const unsigned long long*)&_f2_5));
                        uint32_t _fp4_pair_7;
                        uint16_t _e2m1_pair_h_8;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_8) : "f"(_mul_f32x2_7.x), "f"(_mul_f32x2_7.y));
                        _fp4_pair_7 = (uint32_t)_e2m1_pair_h_8;
                        unsigned int word_24 = _fp4_pair_4 | _fp4_pair_5 << 8 | _fp4_pair_6 << 16 | _fp4_pair_7 << 24;
                        uint32_t _e2m1_canon_1;
                        {
                            const uint32_t _canon_w_9 = word_24;
                            const uint32_t _canon_s_9 = (_canon_w_9 & 0x77777777u) + 0x77777777u;
                            _e2m1_canon_1 = _canon_w_9 & (_canon_s_9 | 0x77777777u);
                        }
                        w[1] = _e2m1_canon_1;
                        int cf_25 = f ^ 2;
                        unsigned int v4_26[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v4_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_26[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base2 + (unsigned int)(16 * cf_25)));
                        float m_27 = (((cf_25 & 2) != 0) ? m1 : m0);
                        float2 _f2_10 = make_float2(m_27, m_27);
                        float2 _f2_11 = make_float2(__uint_as_float(v4_26[0] << 16), __uint_as_float(v4_26[0] & 4294901760u));
                        float2 _mul_f32x2_8;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_8) : "l"(*(const unsigned long long*)&_f2_11), "l"(*(const unsigned long long*)&_f2_10));
                        uint32_t _fp4_pair_8;
                        uint16_t _e2m1_pair_h_10;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_10) : "f"(_mul_f32x2_8.x), "f"(_mul_f32x2_8.y));
                        _fp4_pair_8 = (uint32_t)_e2m1_pair_h_10;
                        float2 _f2_12 = make_float2(__uint_as_float(v4_26[1] << 16), __uint_as_float(v4_26[1] & 4294901760u));
                        float2 _mul_f32x2_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&_f2_12), "l"(*(const unsigned long long*)&_f2_10));
                        uint32_t _fp4_pair_9;
                        uint16_t _e2m1_pair_h_11;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_11) : "f"(_mul_f32x2_9.x), "f"(_mul_f32x2_9.y));
                        _fp4_pair_9 = (uint32_t)_e2m1_pair_h_11;
                        float2 _f2_13 = make_float2(__uint_as_float(v4_26[2] << 16), __uint_as_float(v4_26[2] & 4294901760u));
                        float2 _mul_f32x2_10;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_10) : "l"(*(const unsigned long long*)&_f2_13), "l"(*(const unsigned long long*)&_f2_10));
                        uint32_t _fp4_pair_10;
                        uint16_t _e2m1_pair_h_12;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_12) : "f"(_mul_f32x2_10.x), "f"(_mul_f32x2_10.y));
                        _fp4_pair_10 = (uint32_t)_e2m1_pair_h_12;
                        float2 _f2_14 = make_float2(__uint_as_float(v4_26[3] << 16), __uint_as_float(v4_26[3] & 4294901760u));
                        float2 _mul_f32x2_11;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_11) : "l"(*(const unsigned long long*)&_f2_14), "l"(*(const unsigned long long*)&_f2_10));
                        uint32_t _fp4_pair_11;
                        uint16_t _e2m1_pair_h_13;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_13) : "f"(_mul_f32x2_11.x), "f"(_mul_f32x2_11.y));
                        _fp4_pair_11 = (uint32_t)_e2m1_pair_h_13;
                        unsigned int word_28 = _fp4_pair_8 | _fp4_pair_9 << 8 | _fp4_pair_10 << 16 | _fp4_pair_11 << 24;
                        uint32_t _e2m1_canon_2;
                        {
                            const uint32_t _canon_w_14 = word_28;
                            const uint32_t _canon_s_14 = (_canon_w_14 & 0x77777777u) + 0x77777777u;
                            _e2m1_canon_2 = _canon_w_14 & (_canon_s_14 | 0x77777777u);
                        }
                        w[2] = _e2m1_canon_2;
                        int cf_29 = f ^ 3;
                        unsigned int v4_30[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v4_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_30[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base2 + (unsigned int)(16 * cf_29)));
                        float m_31 = (((cf_29 & 2) != 0) ? m1 : m0);
                        float2 _f2_15 = make_float2(m_31, m_31);
                        float2 _f2_16 = make_float2(__uint_as_float(v4_30[0] << 16), __uint_as_float(v4_30[0] & 4294901760u));
                        float2 _mul_f32x2_12;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_12) : "l"(*(const unsigned long long*)&_f2_16), "l"(*(const unsigned long long*)&_f2_15));
                        uint32_t _fp4_pair_12;
                        uint16_t _e2m1_pair_h_15;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_15) : "f"(_mul_f32x2_12.x), "f"(_mul_f32x2_12.y));
                        _fp4_pair_12 = (uint32_t)_e2m1_pair_h_15;
                        float2 _f2_17 = make_float2(__uint_as_float(v4_30[1] << 16), __uint_as_float(v4_30[1] & 4294901760u));
                        float2 _mul_f32x2_13;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_13) : "l"(*(const unsigned long long*)&_f2_17), "l"(*(const unsigned long long*)&_f2_15));
                        uint32_t _fp4_pair_13;
                        uint16_t _e2m1_pair_h_16;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_16) : "f"(_mul_f32x2_13.x), "f"(_mul_f32x2_13.y));
                        _fp4_pair_13 = (uint32_t)_e2m1_pair_h_16;
                        float2 _f2_18 = make_float2(__uint_as_float(v4_30[2] << 16), __uint_as_float(v4_30[2] & 4294901760u));
                        float2 _mul_f32x2_14;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&_f2_18), "l"(*(const unsigned long long*)&_f2_15));
                        uint32_t _fp4_pair_14;
                        uint16_t _e2m1_pair_h_17;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_17) : "f"(_mul_f32x2_14.x), "f"(_mul_f32x2_14.y));
                        _fp4_pair_14 = (uint32_t)_e2m1_pair_h_17;
                        float2 _f2_19 = make_float2(__uint_as_float(v4_30[3] << 16), __uint_as_float(v4_30[3] & 4294901760u));
                        float2 _mul_f32x2_15;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&_f2_19), "l"(*(const unsigned long long*)&_f2_15));
                        uint32_t _fp4_pair_15;
                        uint16_t _e2m1_pair_h_18;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_18) : "f"(_mul_f32x2_15.x), "f"(_mul_f32x2_15.y));
                        _fp4_pair_15 = (uint32_t)_e2m1_pair_h_18;
                        unsigned int word_32 = _fp4_pair_12 | _fp4_pair_13 << 8 | _fp4_pair_14 << 16 | _fp4_pair_15 << 24;
                        uint32_t _e2m1_canon_3;
                        {
                            const uint32_t _canon_w_19 = word_32;
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
                        int pp2_33 = lane_1 + 32 * (3 * h + 1);
                        int base2_34 = xb_off + 64 * pp2_33;
                        float bm0_35 = __uint_as_float(bmp[1] << 16);
                        float bm1_36 = __uint_as_float(bmp[1] & 4294901760u);
                        float _mul_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(bm0_35), "f"(inv));
                        float _fdiv_rn_6 = __fdiv_rn(_mul_2, six);
                        float _mul_3;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(bm1_36), "f"(inv));
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
                        float m0_37 = ((zero < _fp8x2_decode_1.x) ? _fdiv_rn_8 : zero);
                        float m1_38 = ((zero < _fp8x2_decode_1.y) ? _fdiv_rn_9 : zero);
                        unsigned int w_39[4];
                        int cf_40 = f ^ 0;
                        unsigned int v4_41[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v4_41[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_41[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_41[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_41[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base2_34 + (unsigned int)(16 * cf_40)));
                        float m_42 = (((cf_40 & 2) != 0) ? m1_38 : m0_37);
                        float2 _f2_20 = make_float2(m_42, m_42);
                        float2 _f2_21 = make_float2(__uint_as_float(v4_41[0] << 16), __uint_as_float(v4_41[0] & 4294901760u));
                        float2 _mul_f32x2_16;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_16) : "l"(*(const unsigned long long*)&_f2_21), "l"(*(const unsigned long long*)&_f2_20));
                        uint32_t _fp4_pair_16;
                        uint16_t _e2m1_pair_h_20;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_20) : "f"(_mul_f32x2_16.x), "f"(_mul_f32x2_16.y));
                        _fp4_pair_16 = (uint32_t)_e2m1_pair_h_20;
                        float2 _f2_22 = make_float2(__uint_as_float(v4_41[1] << 16), __uint_as_float(v4_41[1] & 4294901760u));
                        float2 _mul_f32x2_17;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_17) : "l"(*(const unsigned long long*)&_f2_22), "l"(*(const unsigned long long*)&_f2_20));
                        uint32_t _fp4_pair_17;
                        uint16_t _e2m1_pair_h_21;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_21) : "f"(_mul_f32x2_17.x), "f"(_mul_f32x2_17.y));
                        _fp4_pair_17 = (uint32_t)_e2m1_pair_h_21;
                        float2 _f2_23 = make_float2(__uint_as_float(v4_41[2] << 16), __uint_as_float(v4_41[2] & 4294901760u));
                        float2 _mul_f32x2_18;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_18) : "l"(*(const unsigned long long*)&_f2_23), "l"(*(const unsigned long long*)&_f2_20));
                        uint32_t _fp4_pair_18;
                        uint16_t _e2m1_pair_h_22;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_22) : "f"(_mul_f32x2_18.x), "f"(_mul_f32x2_18.y));
                        _fp4_pair_18 = (uint32_t)_e2m1_pair_h_22;
                        float2 _f2_24 = make_float2(__uint_as_float(v4_41[3] << 16), __uint_as_float(v4_41[3] & 4294901760u));
                        float2 _mul_f32x2_19;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_19) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_20));
                        uint32_t _fp4_pair_19;
                        uint16_t _e2m1_pair_h_23;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_23) : "f"(_mul_f32x2_19.x), "f"(_mul_f32x2_19.y));
                        _fp4_pair_19 = (uint32_t)_e2m1_pair_h_23;
                        unsigned int word_43 = _fp4_pair_16 | _fp4_pair_17 << 8 | _fp4_pair_18 << 16 | _fp4_pair_19 << 24;
                        uint32_t _e2m1_canon_4;
                        {
                            const uint32_t _canon_w_24 = word_43;
                            const uint32_t _canon_s_24 = (_canon_w_24 & 0x77777777u) + 0x77777777u;
                            _e2m1_canon_4 = _canon_w_24 & (_canon_s_24 | 0x77777777u);
                        }
                        w_39[0] = _e2m1_canon_4;
                        int cf_44 = f ^ 1;
                        unsigned int v4_45[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v4_45[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_45[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_45[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_45[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base2_34 + (unsigned int)(16 * cf_44)));
                        float m_46 = (((cf_44 & 2) != 0) ? m1_38 : m0_37);
                        float2 _f2_25 = make_float2(m_46, m_46);
                        float2 _f2_26 = make_float2(__uint_as_float(v4_45[0] << 16), __uint_as_float(v4_45[0] & 4294901760u));
                        float2 _mul_f32x2_20;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_20) : "l"(*(const unsigned long long*)&_f2_26), "l"(*(const unsigned long long*)&_f2_25));
                        uint32_t _fp4_pair_20;
                        uint16_t _e2m1_pair_h_25;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_25) : "f"(_mul_f32x2_20.x), "f"(_mul_f32x2_20.y));
                        _fp4_pair_20 = (uint32_t)_e2m1_pair_h_25;
                        float2 _f2_27 = make_float2(__uint_as_float(v4_45[1] << 16), __uint_as_float(v4_45[1] & 4294901760u));
                        float2 _mul_f32x2_21;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_21) : "l"(*(const unsigned long long*)&_f2_27), "l"(*(const unsigned long long*)&_f2_25));
                        uint32_t _fp4_pair_21;
                        uint16_t _e2m1_pair_h_26;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_26) : "f"(_mul_f32x2_21.x), "f"(_mul_f32x2_21.y));
                        _fp4_pair_21 = (uint32_t)_e2m1_pair_h_26;
                        float2 _f2_28 = make_float2(__uint_as_float(v4_45[2] << 16), __uint_as_float(v4_45[2] & 4294901760u));
                        float2 _mul_f32x2_22;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_22) : "l"(*(const unsigned long long*)&_f2_28), "l"(*(const unsigned long long*)&_f2_25));
                        uint32_t _fp4_pair_22;
                        uint16_t _e2m1_pair_h_27;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_27) : "f"(_mul_f32x2_22.x), "f"(_mul_f32x2_22.y));
                        _fp4_pair_22 = (uint32_t)_e2m1_pair_h_27;
                        float2 _f2_29 = make_float2(__uint_as_float(v4_45[3] << 16), __uint_as_float(v4_45[3] & 4294901760u));
                        float2 _mul_f32x2_23;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_23) : "l"(*(const unsigned long long*)&_f2_29), "l"(*(const unsigned long long*)&_f2_25));
                        uint32_t _fp4_pair_23;
                        uint16_t _e2m1_pair_h_28;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_28) : "f"(_mul_f32x2_23.x), "f"(_mul_f32x2_23.y));
                        _fp4_pair_23 = (uint32_t)_e2m1_pair_h_28;
                        unsigned int word_47 = _fp4_pair_20 | _fp4_pair_21 << 8 | _fp4_pair_22 << 16 | _fp4_pair_23 << 24;
                        uint32_t _e2m1_canon_5;
                        {
                            const uint32_t _canon_w_29 = word_47;
                            const uint32_t _canon_s_29 = (_canon_w_29 & 0x77777777u) + 0x77777777u;
                            _e2m1_canon_5 = _canon_w_29 & (_canon_s_29 | 0x77777777u);
                        }
                        w_39[1] = _e2m1_canon_5;
                        int cf_48 = f ^ 2;
                        unsigned int v4_49[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v4_49[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_49[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_49[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_49[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base2_34 + (unsigned int)(16 * cf_48)));
                        float m_50 = (((cf_48 & 2) != 0) ? m1_38 : m0_37);
                        float2 _f2_30 = make_float2(m_50, m_50);
                        float2 _f2_31 = make_float2(__uint_as_float(v4_49[0] << 16), __uint_as_float(v4_49[0] & 4294901760u));
                        float2 _mul_f32x2_24;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_24) : "l"(*(const unsigned long long*)&_f2_31), "l"(*(const unsigned long long*)&_f2_30));
                        uint32_t _fp4_pair_24;
                        uint16_t _e2m1_pair_h_30;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_30) : "f"(_mul_f32x2_24.x), "f"(_mul_f32x2_24.y));
                        _fp4_pair_24 = (uint32_t)_e2m1_pair_h_30;
                        float2 _f2_32 = make_float2(__uint_as_float(v4_49[1] << 16), __uint_as_float(v4_49[1] & 4294901760u));
                        float2 _mul_f32x2_25;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_25) : "l"(*(const unsigned long long*)&_f2_32), "l"(*(const unsigned long long*)&_f2_30));
                        uint32_t _fp4_pair_25;
                        uint16_t _e2m1_pair_h_31;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_31) : "f"(_mul_f32x2_25.x), "f"(_mul_f32x2_25.y));
                        _fp4_pair_25 = (uint32_t)_e2m1_pair_h_31;
                        float2 _f2_33 = make_float2(__uint_as_float(v4_49[2] << 16), __uint_as_float(v4_49[2] & 4294901760u));
                        float2 _mul_f32x2_26;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_26) : "l"(*(const unsigned long long*)&_f2_33), "l"(*(const unsigned long long*)&_f2_30));
                        uint32_t _fp4_pair_26;
                        uint16_t _e2m1_pair_h_32;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_32) : "f"(_mul_f32x2_26.x), "f"(_mul_f32x2_26.y));
                        _fp4_pair_26 = (uint32_t)_e2m1_pair_h_32;
                        float2 _f2_34 = make_float2(__uint_as_float(v4_49[3] << 16), __uint_as_float(v4_49[3] & 4294901760u));
                        float2 _mul_f32x2_27;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_27) : "l"(*(const unsigned long long*)&_f2_34), "l"(*(const unsigned long long*)&_f2_30));
                        uint32_t _fp4_pair_27;
                        uint16_t _e2m1_pair_h_33;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_33) : "f"(_mul_f32x2_27.x), "f"(_mul_f32x2_27.y));
                        _fp4_pair_27 = (uint32_t)_e2m1_pair_h_33;
                        unsigned int word_51 = _fp4_pair_24 | _fp4_pair_25 << 8 | _fp4_pair_26 << 16 | _fp4_pair_27 << 24;
                        uint32_t _e2m1_canon_6;
                        {
                            const uint32_t _canon_w_34 = word_51;
                            const uint32_t _canon_s_34 = (_canon_w_34 & 0x77777777u) + 0x77777777u;
                            _e2m1_canon_6 = _canon_w_34 & (_canon_s_34 | 0x77777777u);
                        }
                        w_39[2] = _e2m1_canon_6;
                        int cf_52 = f ^ 3;
                        unsigned int v4_53[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v4_53[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_53[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_53[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_53[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base2_34 + (unsigned int)(16 * cf_52)));
                        float m_54 = (((cf_52 & 2) != 0) ? m1_38 : m0_37);
                        float2 _f2_35 = make_float2(m_54, m_54);
                        float2 _f2_36 = make_float2(__uint_as_float(v4_53[0] << 16), __uint_as_float(v4_53[0] & 4294901760u));
                        float2 _mul_f32x2_28;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_28) : "l"(*(const unsigned long long*)&_f2_36), "l"(*(const unsigned long long*)&_f2_35));
                        uint32_t _fp4_pair_28;
                        uint16_t _e2m1_pair_h_35;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_35) : "f"(_mul_f32x2_28.x), "f"(_mul_f32x2_28.y));
                        _fp4_pair_28 = (uint32_t)_e2m1_pair_h_35;
                        float2 _f2_37 = make_float2(__uint_as_float(v4_53[1] << 16), __uint_as_float(v4_53[1] & 4294901760u));
                        float2 _mul_f32x2_29;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_29) : "l"(*(const unsigned long long*)&_f2_37), "l"(*(const unsigned long long*)&_f2_35));
                        uint32_t _fp4_pair_29;
                        uint16_t _e2m1_pair_h_36;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_36) : "f"(_mul_f32x2_29.x), "f"(_mul_f32x2_29.y));
                        _fp4_pair_29 = (uint32_t)_e2m1_pair_h_36;
                        float2 _f2_38 = make_float2(__uint_as_float(v4_53[2] << 16), __uint_as_float(v4_53[2] & 4294901760u));
                        float2 _mul_f32x2_30;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_30) : "l"(*(const unsigned long long*)&_f2_38), "l"(*(const unsigned long long*)&_f2_35));
                        uint32_t _fp4_pair_30;
                        uint16_t _e2m1_pair_h_37;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_37) : "f"(_mul_f32x2_30.x), "f"(_mul_f32x2_30.y));
                        _fp4_pair_30 = (uint32_t)_e2m1_pair_h_37;
                        float2 _f2_39 = make_float2(__uint_as_float(v4_53[3] << 16), __uint_as_float(v4_53[3] & 4294901760u));
                        float2 _mul_f32x2_31;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_31) : "l"(*(const unsigned long long*)&_f2_39), "l"(*(const unsigned long long*)&_f2_35));
                        uint32_t _fp4_pair_31;
                        uint16_t _e2m1_pair_h_38;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_38) : "f"(_mul_f32x2_31.x), "f"(_mul_f32x2_31.y));
                        _fp4_pair_31 = (uint32_t)_e2m1_pair_h_38;
                        unsigned int word_55 = _fp4_pair_28 | _fp4_pair_29 << 8 | _fp4_pair_30 << 16 | _fp4_pair_31 << 24;
                        uint32_t _e2m1_canon_7;
                        {
                            const uint32_t _canon_w_39 = word_55;
                            const uint32_t _canon_s_39 = (_canon_w_39 & 0x77777777u) + 0x77777777u;
                            _e2m1_canon_7 = _canon_w_39 & (_canon_s_39 | 0x77777777u);
                        }
                        w_39[3] = _e2m1_canon_7;
                        unsigned int t0_56 = (((f & 1) != 0) ? w_39[1] : w_39[0]);
                        unsigned int t1_57 = (((f & 1) != 0) ? w_39[0] : w_39[1]);
                        unsigned int t2_58 = (((f & 1) != 0) ? w_39[3] : w_39[2]);
                        unsigned int t3_59 = (((f & 1) != 0) ? w_39[2] : w_39[3]);
                        unsigned int q0_60 = (((f & 2) != 0) ? t2_58 : t0_56);
                        unsigned int q1_61 = (((f & 2) != 0) ? t3_59 : t1_57);
                        unsigned int q2_62 = (((f & 2) != 0) ? t0_56 : t2_58);
                        unsigned int q3_63 = (((f & 2) != 0) ? t1_57 : t3_59);
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_33)), "r"(q0_60), "r"(q1_61), "r"(q2_62), "r"(q3_63) : "memory");
                        wsm16[(ob_off + 3072 >> 1) + pp2_33] = _e4m3x2_f32_1;
                        int pp2_64 = lane_1 + 32 * (3 * h + 2);
                        int base2_65 = xb_off + 64 * pp2_64;
                        float bm0_66 = __uint_as_float(bmp[2] << 16);
                        float bm1_67 = __uint_as_float(bmp[2] & 4294901760u);
                        float _mul_4;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(bm0_66), "f"(inv));
                        float _fdiv_rn_10 = __fdiv_rn(_mul_4, six);
                        float _mul_5;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(bm1_67), "f"(inv));
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
                        float m0_68 = ((zero < _fp8x2_decode_2.x) ? _fdiv_rn_12 : zero);
                        float m1_69 = ((zero < _fp8x2_decode_2.y) ? _fdiv_rn_13 : zero);
                        unsigned int w_70[4];
                        int cf_71 = f ^ 0;
                        unsigned int v4_72[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v4_72[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_72[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_72[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_72[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base2_65 + (unsigned int)(16 * cf_71)));
                        float m_73 = (((cf_71 & 2) != 0) ? m1_69 : m0_68);
                        float2 _f2_40 = make_float2(m_73, m_73);
                        float2 _f2_41 = make_float2(__uint_as_float(v4_72[0] << 16), __uint_as_float(v4_72[0] & 4294901760u));
                        float2 _mul_f32x2_32;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_32) : "l"(*(const unsigned long long*)&_f2_41), "l"(*(const unsigned long long*)&_f2_40));
                        uint32_t _fp4_pair_32;
                        uint16_t _e2m1_pair_h_40;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_40) : "f"(_mul_f32x2_32.x), "f"(_mul_f32x2_32.y));
                        _fp4_pair_32 = (uint32_t)_e2m1_pair_h_40;
                        float2 _f2_42 = make_float2(__uint_as_float(v4_72[1] << 16), __uint_as_float(v4_72[1] & 4294901760u));
                        float2 _mul_f32x2_33;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_33) : "l"(*(const unsigned long long*)&_f2_42), "l"(*(const unsigned long long*)&_f2_40));
                        uint32_t _fp4_pair_33;
                        uint16_t _e2m1_pair_h_41;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_41) : "f"(_mul_f32x2_33.x), "f"(_mul_f32x2_33.y));
                        _fp4_pair_33 = (uint32_t)_e2m1_pair_h_41;
                        float2 _f2_43 = make_float2(__uint_as_float(v4_72[2] << 16), __uint_as_float(v4_72[2] & 4294901760u));
                        float2 _mul_f32x2_34;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_34) : "l"(*(const unsigned long long*)&_f2_43), "l"(*(const unsigned long long*)&_f2_40));
                        uint32_t _fp4_pair_34;
                        uint16_t _e2m1_pair_h_42;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_42) : "f"(_mul_f32x2_34.x), "f"(_mul_f32x2_34.y));
                        _fp4_pair_34 = (uint32_t)_e2m1_pair_h_42;
                        float2 _f2_44 = make_float2(__uint_as_float(v4_72[3] << 16), __uint_as_float(v4_72[3] & 4294901760u));
                        float2 _mul_f32x2_35;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_35) : "l"(*(const unsigned long long*)&_f2_44), "l"(*(const unsigned long long*)&_f2_40));
                        uint32_t _fp4_pair_35;
                        uint16_t _e2m1_pair_h_43;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_43) : "f"(_mul_f32x2_35.x), "f"(_mul_f32x2_35.y));
                        _fp4_pair_35 = (uint32_t)_e2m1_pair_h_43;
                        unsigned int word_74 = _fp4_pair_32 | _fp4_pair_33 << 8 | _fp4_pair_34 << 16 | _fp4_pair_35 << 24;
                        uint32_t _e2m1_canon_8;
                        {
                            const uint32_t _canon_w_44 = word_74;
                            const uint32_t _canon_s_44 = (_canon_w_44 & 0x77777777u) + 0x77777777u;
                            _e2m1_canon_8 = _canon_w_44 & (_canon_s_44 | 0x77777777u);
                        }
                        w_70[0] = _e2m1_canon_8;
                        int cf_75 = f ^ 1;
                        unsigned int v4_76[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v4_76[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_76[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_76[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_76[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base2_65 + (unsigned int)(16 * cf_75)));
                        float m_77 = (((cf_75 & 2) != 0) ? m1_69 : m0_68);
                        float2 _f2_45 = make_float2(m_77, m_77);
                        float2 _f2_46 = make_float2(__uint_as_float(v4_76[0] << 16), __uint_as_float(v4_76[0] & 4294901760u));
                        float2 _mul_f32x2_36;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_36) : "l"(*(const unsigned long long*)&_f2_46), "l"(*(const unsigned long long*)&_f2_45));
                        uint32_t _fp4_pair_36;
                        uint16_t _e2m1_pair_h_45;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_45) : "f"(_mul_f32x2_36.x), "f"(_mul_f32x2_36.y));
                        _fp4_pair_36 = (uint32_t)_e2m1_pair_h_45;
                        float2 _f2_47 = make_float2(__uint_as_float(v4_76[1] << 16), __uint_as_float(v4_76[1] & 4294901760u));
                        float2 _mul_f32x2_37;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_37) : "l"(*(const unsigned long long*)&_f2_47), "l"(*(const unsigned long long*)&_f2_45));
                        uint32_t _fp4_pair_37;
                        uint16_t _e2m1_pair_h_46;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_46) : "f"(_mul_f32x2_37.x), "f"(_mul_f32x2_37.y));
                        _fp4_pair_37 = (uint32_t)_e2m1_pair_h_46;
                        float2 _f2_48 = make_float2(__uint_as_float(v4_76[2] << 16), __uint_as_float(v4_76[2] & 4294901760u));
                        float2 _mul_f32x2_38;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_38) : "l"(*(const unsigned long long*)&_f2_48), "l"(*(const unsigned long long*)&_f2_45));
                        uint32_t _fp4_pair_38;
                        uint16_t _e2m1_pair_h_47;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_47) : "f"(_mul_f32x2_38.x), "f"(_mul_f32x2_38.y));
                        _fp4_pair_38 = (uint32_t)_e2m1_pair_h_47;
                        float2 _f2_49 = make_float2(__uint_as_float(v4_76[3] << 16), __uint_as_float(v4_76[3] & 4294901760u));
                        float2 _mul_f32x2_39;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_39) : "l"(*(const unsigned long long*)&_f2_49), "l"(*(const unsigned long long*)&_f2_45));
                        uint32_t _fp4_pair_39;
                        uint16_t _e2m1_pair_h_48;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_48) : "f"(_mul_f32x2_39.x), "f"(_mul_f32x2_39.y));
                        _fp4_pair_39 = (uint32_t)_e2m1_pair_h_48;
                        unsigned int word_78 = _fp4_pair_36 | _fp4_pair_37 << 8 | _fp4_pair_38 << 16 | _fp4_pair_39 << 24;
                        uint32_t _e2m1_canon_9;
                        {
                            const uint32_t _canon_w_49 = word_78;
                            const uint32_t _canon_s_49 = (_canon_w_49 & 0x77777777u) + 0x77777777u;
                            _e2m1_canon_9 = _canon_w_49 & (_canon_s_49 | 0x77777777u);
                        }
                        w_70[1] = _e2m1_canon_9;
                        int cf_79 = f ^ 2;
                        unsigned int v4_80[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v4_80[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_80[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_80[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_80[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base2_65 + (unsigned int)(16 * cf_79)));
                        float m_81 = (((cf_79 & 2) != 0) ? m1_69 : m0_68);
                        float2 _f2_50 = make_float2(m_81, m_81);
                        float2 _f2_51 = make_float2(__uint_as_float(v4_80[0] << 16), __uint_as_float(v4_80[0] & 4294901760u));
                        float2 _mul_f32x2_40;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_40) : "l"(*(const unsigned long long*)&_f2_51), "l"(*(const unsigned long long*)&_f2_50));
                        uint32_t _fp4_pair_40;
                        uint16_t _e2m1_pair_h_50;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_50) : "f"(_mul_f32x2_40.x), "f"(_mul_f32x2_40.y));
                        _fp4_pair_40 = (uint32_t)_e2m1_pair_h_50;
                        float2 _f2_52 = make_float2(__uint_as_float(v4_80[1] << 16), __uint_as_float(v4_80[1] & 4294901760u));
                        float2 _mul_f32x2_41;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_41) : "l"(*(const unsigned long long*)&_f2_52), "l"(*(const unsigned long long*)&_f2_50));
                        uint32_t _fp4_pair_41;
                        uint16_t _e2m1_pair_h_51;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_51) : "f"(_mul_f32x2_41.x), "f"(_mul_f32x2_41.y));
                        _fp4_pair_41 = (uint32_t)_e2m1_pair_h_51;
                        float2 _f2_53 = make_float2(__uint_as_float(v4_80[2] << 16), __uint_as_float(v4_80[2] & 4294901760u));
                        float2 _mul_f32x2_42;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_42) : "l"(*(const unsigned long long*)&_f2_53), "l"(*(const unsigned long long*)&_f2_50));
                        uint32_t _fp4_pair_42;
                        uint16_t _e2m1_pair_h_52;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_52) : "f"(_mul_f32x2_42.x), "f"(_mul_f32x2_42.y));
                        _fp4_pair_42 = (uint32_t)_e2m1_pair_h_52;
                        float2 _f2_54 = make_float2(__uint_as_float(v4_80[3] << 16), __uint_as_float(v4_80[3] & 4294901760u));
                        float2 _mul_f32x2_43;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_43) : "l"(*(const unsigned long long*)&_f2_54), "l"(*(const unsigned long long*)&_f2_50));
                        uint32_t _fp4_pair_43;
                        uint16_t _e2m1_pair_h_53;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_53) : "f"(_mul_f32x2_43.x), "f"(_mul_f32x2_43.y));
                        _fp4_pair_43 = (uint32_t)_e2m1_pair_h_53;
                        unsigned int word_82 = _fp4_pair_40 | _fp4_pair_41 << 8 | _fp4_pair_42 << 16 | _fp4_pair_43 << 24;
                        uint32_t _e2m1_canon_10;
                        {
                            const uint32_t _canon_w_54 = word_82;
                            const uint32_t _canon_s_54 = (_canon_w_54 & 0x77777777u) + 0x77777777u;
                            _e2m1_canon_10 = _canon_w_54 & (_canon_s_54 | 0x77777777u);
                        }
                        w_70[2] = _e2m1_canon_10;
                        int cf_83 = f ^ 3;
                        unsigned int v4_84[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v4_84[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_84[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_84[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_84[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)base2_65 + (unsigned int)(16 * cf_83)));
                        float m_85 = (((cf_83 & 2) != 0) ? m1_69 : m0_68);
                        float2 _f2_55 = make_float2(m_85, m_85);
                        float2 _f2_56 = make_float2(__uint_as_float(v4_84[0] << 16), __uint_as_float(v4_84[0] & 4294901760u));
                        float2 _mul_f32x2_44;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_44) : "l"(*(const unsigned long long*)&_f2_56), "l"(*(const unsigned long long*)&_f2_55));
                        uint32_t _fp4_pair_44;
                        uint16_t _e2m1_pair_h_55;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_55) : "f"(_mul_f32x2_44.x), "f"(_mul_f32x2_44.y));
                        _fp4_pair_44 = (uint32_t)_e2m1_pair_h_55;
                        float2 _f2_57 = make_float2(__uint_as_float(v4_84[1] << 16), __uint_as_float(v4_84[1] & 4294901760u));
                        float2 _mul_f32x2_45;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_45) : "l"(*(const unsigned long long*)&_f2_57), "l"(*(const unsigned long long*)&_f2_55));
                        uint32_t _fp4_pair_45;
                        uint16_t _e2m1_pair_h_56;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_56) : "f"(_mul_f32x2_45.x), "f"(_mul_f32x2_45.y));
                        _fp4_pair_45 = (uint32_t)_e2m1_pair_h_56;
                        float2 _f2_58 = make_float2(__uint_as_float(v4_84[2] << 16), __uint_as_float(v4_84[2] & 4294901760u));
                        float2 _mul_f32x2_46;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_46) : "l"(*(const unsigned long long*)&_f2_58), "l"(*(const unsigned long long*)&_f2_55));
                        uint32_t _fp4_pair_46;
                        uint16_t _e2m1_pair_h_57;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_57) : "f"(_mul_f32x2_46.x), "f"(_mul_f32x2_46.y));
                        _fp4_pair_46 = (uint32_t)_e2m1_pair_h_57;
                        float2 _f2_59 = make_float2(__uint_as_float(v4_84[3] << 16), __uint_as_float(v4_84[3] & 4294901760u));
                        float2 _mul_f32x2_47;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_47) : "l"(*(const unsigned long long*)&_f2_59), "l"(*(const unsigned long long*)&_f2_55));
                        uint32_t _fp4_pair_47;
                        uint16_t _e2m1_pair_h_58;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_58) : "f"(_mul_f32x2_47.x), "f"(_mul_f32x2_47.y));
                        _fp4_pair_47 = (uint32_t)_e2m1_pair_h_58;
                        unsigned int word_86 = _fp4_pair_44 | _fp4_pair_45 << 8 | _fp4_pair_46 << 16 | _fp4_pair_47 << 24;
                        uint32_t _e2m1_canon_11;
                        {
                            const uint32_t _canon_w_59 = word_86;
                            const uint32_t _canon_s_59 = (_canon_w_59 & 0x77777777u) + 0x77777777u;
                            _e2m1_canon_11 = _canon_w_59 & (_canon_s_59 | 0x77777777u);
                        }
                        w_70[3] = _e2m1_canon_11;
                        unsigned int t0_87 = (((f & 1) != 0) ? w_70[1] : w_70[0]);
                        unsigned int t1_88 = (((f & 1) != 0) ? w_70[0] : w_70[1]);
                        unsigned int t2_89 = (((f & 1) != 0) ? w_70[3] : w_70[2]);
                        unsigned int t3_90 = (((f & 1) != 0) ? w_70[2] : w_70[3]);
                        unsigned int q0_91 = (((f & 2) != 0) ? t2_89 : t0_87);
                        unsigned int q1_92 = (((f & 2) != 0) ? t3_90 : t1_88);
                        unsigned int q2_93 = (((f & 2) != 0) ? t0_87 : t2_89);
                        unsigned int q3_94 = (((f & 2) != 0) ? t1_88 : t3_90);
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_64)), "r"(q0_91), "r"(q1_92), "r"(q2_93), "r"(q3_94) : "memory");
                        wsm16[(ob_off + 3072 >> 1) + pp2_64] = _e4m3x2_f32_2;
                        rs = rs_20;
                    }
                    if (h == 0 && lane_1 < 4) {
                        wsm32[(ob_off + 3072 + 384 >> 2) + lane_1] = rs;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                    if (h == 0 && lane_1 == 0) {
                        if (n > k + 1) {
                            long long next_row = (long long)(r0 + t + 5 * (k + 1));
                            mbarrier_arrive_expect_tx(xb_full_addr + (leader) * 8, 12288);
                            cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)xb_off, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(x) + ((unsigned long long)(next_row * 6144) * (unsigned long long)2)), 12288, xb_full_addr + (leader) * 8);
                        }
                        {
                            long long row64 = (long long)(r0 + t + 5 * k);
                            long long v_off = os_base + row64 * 3072;
                            long long sc_off = os_base + 100663296 + row64 * 384;
                            long long m_off = os_base + 113246208 + row64 * 16;
                            {
                                void* _cpbulk_dst_60 = reinterpret_cast<void*>(peer_base + v_off);
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_60), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_61 = reinterpret_cast<void*>(peer_base + sc_off);
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_61), "r"(wsm_addr + (unsigned int)ob_off + 3072), "r"((uint32_t)(384))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_62 = reinterpret_cast<void*>(peer_base + m_off);
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_62), "r"(wsm_addr + (unsigned int)ob_off + 3072 + 384), "r"((uint32_t)(16))
                                    : "memory");
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                int kd = k - 1;
                if (kd >= 0 && kd < n) {
                    long long drow = (long long)(r0 + t + 5 * kd);
                    int dslot = kd % 2;
                    int dob_off = xb_off + 12288 + dslot * 3472;
                    long long v_base = os_base + (16384 + drow) * 3072;
                    long long sc_base = os_base + 100663296 + (16384 + drow) * 384;
                    long long m_base = os_base + 113246208 + (16384 + drow) * 16;
                    uint32_t _sysv_poll32_group_0[1];
                    do {
                        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[0]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + m_base)) : "memory");
                    } while (
                        (_sysv_poll32_group_0[0] == 0x80000000u)
                    );
                    unsigned int own[12];
                    unsigned int own_sc[3];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[0])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * (lane_1 + 32 * (3 * h)))));
                    own_sc[0] = (unsigned int)wsm16[(dob_off + 3072 >> 1) + (lane_1 + 32 * (3 * h))];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[4])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * (lane_1 + 32 * (3 * h + 1)))));
                    own_sc[1] = (unsigned int)wsm16[(dob_off + 3072 >> 1) + (lane_1 + 32 * (3 * h + 1))];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[8])), "=r"(*reinterpret_cast<uint32_t*>(&own[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)dob_off + (unsigned int)(16 * (lane_1 + 32 * (3 * h + 2)))));
                    own_sc[2] = (unsigned int)wsm16[(dob_off + 3072 >> 1) + (lane_1 + 32 * (3 * h + 2))];
                    uint32_t _fp4_wire_values_0[12];
                    uint32_t _fp4_wire_scales_0[3];
                    uint16_t _wire_h16_64_0;
                    uint16_t _wire_h16_64_1;
                    uint16_t _wire_h16_64_2;
                    uint32_t _wire_vpend_64 = 7u;
                    uint32_t _wire_spend_64 = 7u;
                    do {
                        if (_wire_vpend_64 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_1 + 32 * (3 * h)))))) : "memory");
                        }
                        if (_wire_vpend_64 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_1 + 32 * (3 * h + 1)))))) : "memory");
                        }
                        if (_wire_vpend_64 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[8]), "=r"(_fp4_wire_values_0[9]), "=r"(_fp4_wire_values_0[10]), "=r"(_fp4_wire_values_0[11]) : "l"(reinterpret_cast<unsigned int*>(self_u8 + (v_base + (long long)(16 * (lane_1 + 32 * (3 * h + 2)))))) : "memory");
                        }
                        if (_wire_spend_64 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_64_0) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_1 + 32 * (3 * h)))))) : "memory");
                            _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_64_0;
                        }
                        if (_wire_spend_64 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_64_1) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_1 + 32 * (3 * h + 1)))))) : "memory");
                            _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_64_1;
                        }
                        if (_wire_spend_64 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_64_2) : "l"(reinterpret_cast<uint16_t*>(self_u8 + (sc_base + (long long)(2 * (lane_1 + 32 * (3 * h + 2)))))) : "memory");
                            _fp4_wire_scales_0[2] = (uint32_t)_wire_h16_64_2;
                        }
                        if (_wire_vpend_64 & 1u) {
                            if (!((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_64 &= ~1u;
                        }
                        if (_wire_vpend_64 & 2u) {
                            if (!((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_64 &= ~2u;
                        }
                        if (_wire_vpend_64 & 4u) {
                            if (!((((_fp4_wire_values_0[8] & ~((_fp4_wire_values_0[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[9] & ~((_fp4_wire_values_0[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[10] & ~((_fp4_wire_values_0[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u) || (((_fp4_wire_values_0[11] & ~((_fp4_wire_values_0[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u) != 0u))) _wire_vpend_64 &= ~4u;
                        }
                        if (_wire_spend_64 & 1u) {
                            if (!(((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[0] >> 8) == 0xFFu))) _wire_spend_64 &= ~1u;
                        }
                        if (_wire_spend_64 & 2u) {
                            if (!(((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[1] >> 8) == 0xFFu))) _wire_spend_64 &= ~2u;
                        }
                        if (_wire_spend_64 & 4u) {
                            if (!(((_fp4_wire_scales_0[2] & 0xFFu) == 0xFFu) || ((_fp4_wire_scales_0[2] >> 8) == 0xFFu))) _wire_spend_64 &= ~4u;
                        }
                    } while ((_wire_vpend_64 | _wire_spend_64) != 0u);
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
                    float2 _fp8x2_decode_3;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_3.x), "=f"(_fp8x2_decode_3.y) : "h"((uint16_t)own_sc[0]));
                    float _mul_6;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_fp8x2_decode_3.x), "f"(wsm32[dob_off + 3072 + 384 >> 2]));
                    float s0 = _mul_6;
                    float _mul_7;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_fp8x2_decode_3.y), "f"(wsm32[dob_off + 3072 + 384 >> 2]));
                    float s1 = _mul_7;
                    float2 _f2_60 = make_float2(s0, s0);
                    float2 _f2_61 = make_float2(s1, s1);
                    float _unpack_fp4_0[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[0]));
                        (&_unpack_fp4_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_0[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_0[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_0[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_0[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_0[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_0[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_0[0]), "=f"(_unpack_fp4_0[1]), "=f"(_unpack_fp4_0[2]), "=f"(_unpack_fp4_0[3]), "=f"(_unpack_fp4_0[4]), "=f"(_unpack_fp4_0[5]), "=f"(_unpack_fp4_0[6]), "=f"(_unpack_fp4_0[7]) : "r"(own[0]));
                        #endif
                    }
                    float2 _f2_62 = make_float2(acc[0], acc[1]);
                    float2 _f2_63 = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                    float2 r = fma_f32x2_rn_noftz(_f2_63, _f2_60, _f2_62);
                    acc[0] = r.x;
                    acc[1] = r.y;
                    float2 _f2_64 = make_float2(acc[2], acc[3]);
                    float2 _f2_65 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                    float2 r_0 = fma_f32x2_rn_noftz(_f2_65, _f2_60, _f2_64);
                    acc[2] = r_0.x;
                    acc[3] = r_0.y;
                    float2 _f2_66 = make_float2(acc[4], acc[5]);
                    float2 _f2_67 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                    float2 r_1 = fma_f32x2_rn_noftz(_f2_67, _f2_60, _f2_66);
                    acc[4] = r_1.x;
                    acc[5] = r_1.y;
                    float2 _f2_68 = make_float2(acc[6], acc[7]);
                    float2 _f2_69 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                    float2 r_2 = fma_f32x2_rn_noftz(_f2_69, _f2_60, _f2_68);
                    acc[6] = r_2.x;
                    acc[7] = r_2.y;
                    float _unpack_fp4_1[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[1 + 0]));
                        (&_unpack_fp4_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_1[0]), "=f"(_unpack_fp4_1[1]), "=f"(_unpack_fp4_1[2]), "=f"(_unpack_fp4_1[3]), "=f"(_unpack_fp4_1[4]), "=f"(_unpack_fp4_1[5]), "=f"(_unpack_fp4_1[6]), "=f"(_unpack_fp4_1[7]) : "r"(own[1 + 0]));
                        #endif
                    }
                    float2 _f2_70 = make_float2(acc[8], acc[9]);
                    float2 _f2_71 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                    float2 r_3 = fma_f32x2_rn_noftz(_f2_71, _f2_60, _f2_70);
                    acc[8] = r_3.x;
                    acc[9] = r_3.y;
                    float2 _f2_72 = make_float2(acc[10], acc[11]);
                    float2 _f2_73 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                    float2 r_4 = fma_f32x2_rn_noftz(_f2_73, _f2_60, _f2_72);
                    acc[10] = r_4.x;
                    acc[11] = r_4.y;
                    float2 _f2_74 = make_float2(acc[12], acc[13]);
                    float2 _f2_75 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                    float2 r_5 = fma_f32x2_rn_noftz(_f2_75, _f2_60, _f2_74);
                    acc[12] = r_5.x;
                    acc[13] = r_5.y;
                    float2 _f2_76 = make_float2(acc[14], acc[15]);
                    float2 _f2_77 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                    float2 r_6 = fma_f32x2_rn_noftz(_f2_77, _f2_60, _f2_76);
                    acc[14] = r_6.x;
                    acc[15] = r_6.y;
                    float _unpack_fp4_2[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[2 + 0]));
                        (&_unpack_fp4_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_2[0]), "=f"(_unpack_fp4_2[1]), "=f"(_unpack_fp4_2[2]), "=f"(_unpack_fp4_2[3]), "=f"(_unpack_fp4_2[4]), "=f"(_unpack_fp4_2[5]), "=f"(_unpack_fp4_2[6]), "=f"(_unpack_fp4_2[7]) : "r"(own[2 + 0]));
                        #endif
                    }
                    float2 _f2_78 = make_float2(acc[16], acc[17]);
                    float2 _f2_79 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                    float2 r_7 = fma_f32x2_rn_noftz(_f2_79, _f2_61, _f2_78);
                    acc[16] = r_7.x;
                    acc[17] = r_7.y;
                    float2 _f2_80 = make_float2(acc[18], acc[19]);
                    float2 _f2_81 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                    float2 r_8 = fma_f32x2_rn_noftz(_f2_81, _f2_61, _f2_80);
                    acc[18] = r_8.x;
                    acc[19] = r_8.y;
                    float2 _f2_82 = make_float2(acc[20], acc[21]);
                    float2 _f2_83 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                    float2 r_9 = fma_f32x2_rn_noftz(_f2_83, _f2_61, _f2_82);
                    acc[20] = r_9.x;
                    acc[21] = r_9.y;
                    float2 _f2_84 = make_float2(acc[22], acc[23]);
                    float2 _f2_85 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                    float2 r_10 = fma_f32x2_rn_noftz(_f2_85, _f2_61, _f2_84);
                    acc[22] = r_10.x;
                    acc[23] = r_10.y;
                    float _unpack_fp4_3[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[3 + 0]));
                        (&_unpack_fp4_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_3[0]), "=f"(_unpack_fp4_3[1]), "=f"(_unpack_fp4_3[2]), "=f"(_unpack_fp4_3[3]), "=f"(_unpack_fp4_3[4]), "=f"(_unpack_fp4_3[5]), "=f"(_unpack_fp4_3[6]), "=f"(_unpack_fp4_3[7]) : "r"(own[3 + 0]));
                        #endif
                    }
                    float2 _f2_86 = make_float2(acc[24], acc[25]);
                    float2 _f2_87 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                    float2 r_11 = fma_f32x2_rn_noftz(_f2_87, _f2_61, _f2_86);
                    acc[24] = r_11.x;
                    acc[25] = r_11.y;
                    float2 _f2_88 = make_float2(acc[26], acc[27]);
                    float2 _f2_89 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                    float2 r_12 = fma_f32x2_rn_noftz(_f2_89, _f2_61, _f2_88);
                    acc[26] = r_12.x;
                    acc[27] = r_12.y;
                    float2 _f2_90 = make_float2(acc[28], acc[29]);
                    float2 _f2_91 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                    float2 r_13 = fma_f32x2_rn_noftz(_f2_91, _f2_61, _f2_90);
                    acc[28] = r_13.x;
                    acc[29] = r_13.y;
                    float2 _f2_92 = make_float2(acc[30], acc[31]);
                    float2 _f2_93 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                    float2 r_14 = fma_f32x2_rn_noftz(_f2_93, _f2_61, _f2_92);
                    acc[30] = r_14.x;
                    acc[31] = r_14.y;
                    float2 _fp8x2_decode_4;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_4.x), "=f"(_fp8x2_decode_4.y) : "h"((uint16_t)_fp4_wire_scales_0[0]));
                    float _mul_8;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_fp8x2_decode_4.x), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                    float s0_15 = _mul_8;
                    float _mul_9;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_fp8x2_decode_4.y), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                    float s1_16 = _mul_9;
                    float2 _f2_94 = make_float2(s0_15, s0_15);
                    float2 _f2_95 = make_float2(s1_16, s1_16);
                    float _unpack_fp4_4[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[0]));
                        (&_unpack_fp4_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_4[0]), "=f"(_unpack_fp4_4[1]), "=f"(_unpack_fp4_4[2]), "=f"(_unpack_fp4_4[3]), "=f"(_unpack_fp4_4[4]), "=f"(_unpack_fp4_4[5]), "=f"(_unpack_fp4_4[6]), "=f"(_unpack_fp4_4[7]) : "r"(_fp4_wire_values_0[0]));
                        #endif
                    }
                    float2 _f2_96 = make_float2(acc[0], acc[1]);
                    float2 _f2_97 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                    float2 r_17 = fma_f32x2_rn_noftz(_f2_97, _f2_94, _f2_96);
                    acc[0] = r_17.x;
                    acc[1] = r_17.y;
                    float2 _f2_98 = make_float2(acc[2], acc[3]);
                    float2 _f2_99 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                    float2 r_18 = fma_f32x2_rn_noftz(_f2_99, _f2_94, _f2_98);
                    acc[2] = r_18.x;
                    acc[3] = r_18.y;
                    float2 _f2_100 = make_float2(acc[4], acc[5]);
                    float2 _f2_101 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                    float2 r_19 = fma_f32x2_rn_noftz(_f2_101, _f2_94, _f2_100);
                    acc[4] = r_19.x;
                    acc[5] = r_19.y;
                    float2 _f2_102 = make_float2(acc[6], acc[7]);
                    float2 _f2_103 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                    float2 r_20 = fma_f32x2_rn_noftz(_f2_103, _f2_94, _f2_102);
                    acc[6] = r_20.x;
                    acc[7] = r_20.y;
                    float _unpack_fp4_5[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[1 + 0]));
                        (&_unpack_fp4_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_5[0]), "=f"(_unpack_fp4_5[1]), "=f"(_unpack_fp4_5[2]), "=f"(_unpack_fp4_5[3]), "=f"(_unpack_fp4_5[4]), "=f"(_unpack_fp4_5[5]), "=f"(_unpack_fp4_5[6]), "=f"(_unpack_fp4_5[7]) : "r"(_fp4_wire_values_0[1 + 0]));
                        #endif
                    }
                    float2 _f2_104 = make_float2(acc[8], acc[9]);
                    float2 _f2_105 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                    float2 r_21 = fma_f32x2_rn_noftz(_f2_105, _f2_94, _f2_104);
                    acc[8] = r_21.x;
                    acc[9] = r_21.y;
                    float2 _f2_106 = make_float2(acc[10], acc[11]);
                    float2 _f2_107 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                    float2 r_22 = fma_f32x2_rn_noftz(_f2_107, _f2_94, _f2_106);
                    acc[10] = r_22.x;
                    acc[11] = r_22.y;
                    float2 _f2_108 = make_float2(acc[12], acc[13]);
                    float2 _f2_109 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                    float2 r_23 = fma_f32x2_rn_noftz(_f2_109, _f2_94, _f2_108);
                    acc[12] = r_23.x;
                    acc[13] = r_23.y;
                    float2 _f2_110 = make_float2(acc[14], acc[15]);
                    float2 _f2_111 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                    float2 r_24 = fma_f32x2_rn_noftz(_f2_111, _f2_94, _f2_110);
                    acc[14] = r_24.x;
                    acc[15] = r_24.y;
                    float _unpack_fp4_6[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[2 + 0]));
                        (&_unpack_fp4_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_6[0]), "=f"(_unpack_fp4_6[1]), "=f"(_unpack_fp4_6[2]), "=f"(_unpack_fp4_6[3]), "=f"(_unpack_fp4_6[4]), "=f"(_unpack_fp4_6[5]), "=f"(_unpack_fp4_6[6]), "=f"(_unpack_fp4_6[7]) : "r"(_fp4_wire_values_0[2 + 0]));
                        #endif
                    }
                    float2 _f2_112 = make_float2(acc[16], acc[17]);
                    float2 _f2_113 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                    float2 r_25 = fma_f32x2_rn_noftz(_f2_113, _f2_95, _f2_112);
                    acc[16] = r_25.x;
                    acc[17] = r_25.y;
                    float2 _f2_114 = make_float2(acc[18], acc[19]);
                    float2 _f2_115 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                    float2 r_26 = fma_f32x2_rn_noftz(_f2_115, _f2_95, _f2_114);
                    acc[18] = r_26.x;
                    acc[19] = r_26.y;
                    float2 _f2_116 = make_float2(acc[20], acc[21]);
                    float2 _f2_117 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                    float2 r_27 = fma_f32x2_rn_noftz(_f2_117, _f2_95, _f2_116);
                    acc[20] = r_27.x;
                    acc[21] = r_27.y;
                    float2 _f2_118 = make_float2(acc[22], acc[23]);
                    float2 _f2_119 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                    float2 r_28 = fma_f32x2_rn_noftz(_f2_119, _f2_95, _f2_118);
                    acc[22] = r_28.x;
                    acc[23] = r_28.y;
                    float _unpack_fp4_7[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[3 + 0]));
                        (&_unpack_fp4_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_7[0]), "=f"(_unpack_fp4_7[1]), "=f"(_unpack_fp4_7[2]), "=f"(_unpack_fp4_7[3]), "=f"(_unpack_fp4_7[4]), "=f"(_unpack_fp4_7[5]), "=f"(_unpack_fp4_7[6]), "=f"(_unpack_fp4_7[7]) : "r"(_fp4_wire_values_0[3 + 0]));
                        #endif
                    }
                    float2 _f2_120 = make_float2(acc[24], acc[25]);
                    float2 _f2_121 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                    float2 r_29 = fma_f32x2_rn_noftz(_f2_121, _f2_95, _f2_120);
                    acc[24] = r_29.x;
                    acc[25] = r_29.y;
                    float2 _f2_122 = make_float2(acc[26], acc[27]);
                    float2 _f2_123 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                    float2 r_30 = fma_f32x2_rn_noftz(_f2_123, _f2_95, _f2_122);
                    acc[26] = r_30.x;
                    acc[27] = r_30.y;
                    float2 _f2_124 = make_float2(acc[28], acc[29]);
                    float2 _f2_125 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                    float2 r_31 = fma_f32x2_rn_noftz(_f2_125, _f2_95, _f2_124);
                    acc[28] = r_31.x;
                    acc[29] = r_31.y;
                    float2 _f2_126 = make_float2(acc[30], acc[31]);
                    float2 _f2_127 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                    float2 r_32 = fma_f32x2_rn_noftz(_f2_127, _f2_95, _f2_126);
                    acc[30] = r_32.x;
                    acc[31] = r_32.y;
                    uint32_t acc_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                        acc_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_bf16[0]), "r"(acc_bf16[1]), "r"(acc_bf16[2]), "r"(acc_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_bf16[4]), "r"(acc_bf16[5]), "r"(acc_bf16[6]), "r"(acc_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_bf16[8]), "r"(acc_bf16[9]), "r"(acc_bf16[10]), "r"(acc_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_bf16[12]), "r"(acc_bf16[13]), "r"(acc_bf16[14]), "r"(acc_bf16[15]) : "memory");
                    __syncwarp();
                    int c = lane_1;
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
                    int c_33 = lane_1 + 32;
                    int l_34 = c_33 >> 2;
                    unsigned int w4_35[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_35[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_35[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_35[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_35[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_34 + (c_33 & 3 ^ l_34 >> 1 & 3)))));
                    long long idx_36 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_33));
                    {
                        int4 _iv4 = make_int4(w4_35[0 + 0], w4_35[0 + 1], w4_35[0 + 2], w4_35[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_36)) = _iv4;
                    }
                    int c_37 = lane_1 + 64;
                    int l_38 = c_37 >> 2;
                    unsigned int w4_39[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_39[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_38 + (c_37 & 3 ^ l_38 >> 1 & 3)))));
                    long long idx_40 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_37));
                    {
                        int4 _iv4 = make_int4(w4_39[0 + 0], w4_39[0 + 1], w4_39[0 + 2], w4_39[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_40)) = _iv4;
                    }
                    int c_41 = lane_1 + 96;
                    int l_42 = c_41 >> 2;
                    unsigned int w4_43[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_43[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_43[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_43[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_43[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_42 + (c_41 & 3 ^ l_42 >> 1 & 3)))));
                    long long idx_44 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_41));
                    {
                        int4 _iv4 = make_int4(w4_43[0 + 0], w4_43[0 + 1], w4_43[0 + 2], w4_43[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_44)) = _iv4;
                    }
                    __syncwarp();
                    float acc_45[32];
                    acc_45[0] = 0.0f;
                    acc_45[1] = 0.0f;
                    acc_45[2] = 0.0f;
                    acc_45[3] = 0.0f;
                    acc_45[4] = 0.0f;
                    acc_45[5] = 0.0f;
                    acc_45[6] = 0.0f;
                    acc_45[7] = 0.0f;
                    acc_45[8] = 0.0f;
                    acc_45[9] = 0.0f;
                    acc_45[10] = 0.0f;
                    acc_45[11] = 0.0f;
                    acc_45[12] = 0.0f;
                    acc_45[13] = 0.0f;
                    acc_45[14] = 0.0f;
                    acc_45[15] = 0.0f;
                    acc_45[16] = 0.0f;
                    acc_45[17] = 0.0f;
                    acc_45[18] = 0.0f;
                    acc_45[19] = 0.0f;
                    acc_45[20] = 0.0f;
                    acc_45[21] = 0.0f;
                    acc_45[22] = 0.0f;
                    acc_45[23] = 0.0f;
                    acc_45[24] = 0.0f;
                    acc_45[25] = 0.0f;
                    acc_45[26] = 0.0f;
                    acc_45[27] = 0.0f;
                    acc_45[28] = 0.0f;
                    acc_45[29] = 0.0f;
                    acc_45[30] = 0.0f;
                    acc_45[31] = 0.0f;
                    float2 _fp8x2_decode_5;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_5.x), "=f"(_fp8x2_decode_5.y) : "h"((uint16_t)own_sc[1]));
                    float _mul_10;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(_fp8x2_decode_5.x), "f"(wsm32[dob_off + 3072 + 384 >> 2]));
                    float s0_46 = _mul_10;
                    float _mul_11;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_fp8x2_decode_5.y), "f"(wsm32[dob_off + 3072 + 384 >> 2]));
                    float s1_47 = _mul_11;
                    float2 _f2_128 = make_float2(s0_46, s0_46);
                    float2 _f2_129 = make_float2(s1_47, s1_47);
                    float _unpack_fp4_8[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[4 + 0]));
                        (&_unpack_fp4_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_8[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_8[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_8[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_8[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_8[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_8[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_8[0]), "=f"(_unpack_fp4_8[1]), "=f"(_unpack_fp4_8[2]), "=f"(_unpack_fp4_8[3]), "=f"(_unpack_fp4_8[4]), "=f"(_unpack_fp4_8[5]), "=f"(_unpack_fp4_8[6]), "=f"(_unpack_fp4_8[7]) : "r"(own[4 + 0]));
                        #endif
                    }
                    float2 _f2_130 = make_float2(acc_45[0], acc_45[1]);
                    float2 _f2_131 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                    float2 r_48 = fma_f32x2_rn_noftz(_f2_131, _f2_128, _f2_130);
                    acc_45[0] = r_48.x;
                    acc_45[1] = r_48.y;
                    float2 _f2_132 = make_float2(acc_45[2], acc_45[3]);
                    float2 _f2_133 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                    float2 r_49 = fma_f32x2_rn_noftz(_f2_133, _f2_128, _f2_132);
                    acc_45[2] = r_49.x;
                    acc_45[3] = r_49.y;
                    float2 _f2_134 = make_float2(acc_45[4], acc_45[5]);
                    float2 _f2_135 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                    float2 r_50 = fma_f32x2_rn_noftz(_f2_135, _f2_128, _f2_134);
                    acc_45[4] = r_50.x;
                    acc_45[5] = r_50.y;
                    float2 _f2_136 = make_float2(acc_45[6], acc_45[7]);
                    float2 _f2_137 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                    float2 r_51 = fma_f32x2_rn_noftz(_f2_137, _f2_128, _f2_136);
                    acc_45[6] = r_51.x;
                    acc_45[7] = r_51.y;
                    float _unpack_fp4_9[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[5 + 0]));
                        (&_unpack_fp4_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_9[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_9[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_9[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_9[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_9[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_9[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_9[0]), "=f"(_unpack_fp4_9[1]), "=f"(_unpack_fp4_9[2]), "=f"(_unpack_fp4_9[3]), "=f"(_unpack_fp4_9[4]), "=f"(_unpack_fp4_9[5]), "=f"(_unpack_fp4_9[6]), "=f"(_unpack_fp4_9[7]) : "r"(own[5 + 0]));
                        #endif
                    }
                    float2 _f2_138 = make_float2(acc_45[8], acc_45[9]);
                    float2 _f2_139 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                    float2 r_52 = fma_f32x2_rn_noftz(_f2_139, _f2_128, _f2_138);
                    acc_45[8] = r_52.x;
                    acc_45[9] = r_52.y;
                    float2 _f2_140 = make_float2(acc_45[10], acc_45[11]);
                    float2 _f2_141 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                    float2 r_53 = fma_f32x2_rn_noftz(_f2_141, _f2_128, _f2_140);
                    acc_45[10] = r_53.x;
                    acc_45[11] = r_53.y;
                    float2 _f2_142 = make_float2(acc_45[12], acc_45[13]);
                    float2 _f2_143 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                    float2 r_54 = fma_f32x2_rn_noftz(_f2_143, _f2_128, _f2_142);
                    acc_45[12] = r_54.x;
                    acc_45[13] = r_54.y;
                    float2 _f2_144 = make_float2(acc_45[14], acc_45[15]);
                    float2 _f2_145 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                    float2 r_55 = fma_f32x2_rn_noftz(_f2_145, _f2_128, _f2_144);
                    acc_45[14] = r_55.x;
                    acc_45[15] = r_55.y;
                    float _unpack_fp4_10[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[6 + 0]));
                        (&_unpack_fp4_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_10[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_10[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_10[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_10[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_10[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_10[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_10[0]), "=f"(_unpack_fp4_10[1]), "=f"(_unpack_fp4_10[2]), "=f"(_unpack_fp4_10[3]), "=f"(_unpack_fp4_10[4]), "=f"(_unpack_fp4_10[5]), "=f"(_unpack_fp4_10[6]), "=f"(_unpack_fp4_10[7]) : "r"(own[6 + 0]));
                        #endif
                    }
                    float2 _f2_146 = make_float2(acc_45[16], acc_45[17]);
                    float2 _f2_147 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                    float2 r_56 = fma_f32x2_rn_noftz(_f2_147, _f2_129, _f2_146);
                    acc_45[16] = r_56.x;
                    acc_45[17] = r_56.y;
                    float2 _f2_148 = make_float2(acc_45[18], acc_45[19]);
                    float2 _f2_149 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                    float2 r_57 = fma_f32x2_rn_noftz(_f2_149, _f2_129, _f2_148);
                    acc_45[18] = r_57.x;
                    acc_45[19] = r_57.y;
                    float2 _f2_150 = make_float2(acc_45[20], acc_45[21]);
                    float2 _f2_151 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                    float2 r_58 = fma_f32x2_rn_noftz(_f2_151, _f2_129, _f2_150);
                    acc_45[20] = r_58.x;
                    acc_45[21] = r_58.y;
                    float2 _f2_152 = make_float2(acc_45[22], acc_45[23]);
                    float2 _f2_153 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                    float2 r_59 = fma_f32x2_rn_noftz(_f2_153, _f2_129, _f2_152);
                    acc_45[22] = r_59.x;
                    acc_45[23] = r_59.y;
                    float _unpack_fp4_11[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[7 + 0]));
                        (&_unpack_fp4_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_11[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_11[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_11[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_11[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_11[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_11[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_11[0]), "=f"(_unpack_fp4_11[1]), "=f"(_unpack_fp4_11[2]), "=f"(_unpack_fp4_11[3]), "=f"(_unpack_fp4_11[4]), "=f"(_unpack_fp4_11[5]), "=f"(_unpack_fp4_11[6]), "=f"(_unpack_fp4_11[7]) : "r"(own[7 + 0]));
                        #endif
                    }
                    float2 _f2_154 = make_float2(acc_45[24], acc_45[25]);
                    float2 _f2_155 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                    float2 r_60 = fma_f32x2_rn_noftz(_f2_155, _f2_129, _f2_154);
                    acc_45[24] = r_60.x;
                    acc_45[25] = r_60.y;
                    float2 _f2_156 = make_float2(acc_45[26], acc_45[27]);
                    float2 _f2_157 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                    float2 r_61 = fma_f32x2_rn_noftz(_f2_157, _f2_129, _f2_156);
                    acc_45[26] = r_61.x;
                    acc_45[27] = r_61.y;
                    float2 _f2_158 = make_float2(acc_45[28], acc_45[29]);
                    float2 _f2_159 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                    float2 r_62 = fma_f32x2_rn_noftz(_f2_159, _f2_129, _f2_158);
                    acc_45[28] = r_62.x;
                    acc_45[29] = r_62.y;
                    float2 _f2_160 = make_float2(acc_45[30], acc_45[31]);
                    float2 _f2_161 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                    float2 r_63 = fma_f32x2_rn_noftz(_f2_161, _f2_129, _f2_160);
                    acc_45[30] = r_63.x;
                    acc_45[31] = r_63.y;
                    float2 _fp8x2_decode_6;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_6.x), "=f"(_fp8x2_decode_6.y) : "h"((uint16_t)_fp4_wire_scales_0[1]));
                    float _mul_12;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(_fp8x2_decode_6.x), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                    float s0_64 = _mul_12;
                    float _mul_13;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(_fp8x2_decode_6.y), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                    float s1_65 = _mul_13;
                    float2 _f2_162 = make_float2(s0_64, s0_64);
                    float2 _f2_163 = make_float2(s1_65, s1_65);
                    float _unpack_fp4_12[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[4 + 0]));
                        (&_unpack_fp4_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_12[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_12[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_12[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_12[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_12[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_12[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_12[0]), "=f"(_unpack_fp4_12[1]), "=f"(_unpack_fp4_12[2]), "=f"(_unpack_fp4_12[3]), "=f"(_unpack_fp4_12[4]), "=f"(_unpack_fp4_12[5]), "=f"(_unpack_fp4_12[6]), "=f"(_unpack_fp4_12[7]) : "r"(_fp4_wire_values_0[4 + 0]));
                        #endif
                    }
                    float2 _f2_164 = make_float2(acc_45[0], acc_45[1]);
                    float2 _f2_165 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                    float2 r_66 = fma_f32x2_rn_noftz(_f2_165, _f2_162, _f2_164);
                    acc_45[0] = r_66.x;
                    acc_45[1] = r_66.y;
                    float2 _f2_166 = make_float2(acc_45[2], acc_45[3]);
                    float2 _f2_167 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                    float2 r_67 = fma_f32x2_rn_noftz(_f2_167, _f2_162, _f2_166);
                    acc_45[2] = r_67.x;
                    acc_45[3] = r_67.y;
                    float2 _f2_168 = make_float2(acc_45[4], acc_45[5]);
                    float2 _f2_169 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                    float2 r_68 = fma_f32x2_rn_noftz(_f2_169, _f2_162, _f2_168);
                    acc_45[4] = r_68.x;
                    acc_45[5] = r_68.y;
                    float2 _f2_170 = make_float2(acc_45[6], acc_45[7]);
                    float2 _f2_171 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                    float2 r_69 = fma_f32x2_rn_noftz(_f2_171, _f2_162, _f2_170);
                    acc_45[6] = r_69.x;
                    acc_45[7] = r_69.y;
                    float _unpack_fp4_13[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[5 + 0]));
                        (&_unpack_fp4_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_13[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_13[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_13[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_13[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_13[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_13[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_13[0]), "=f"(_unpack_fp4_13[1]), "=f"(_unpack_fp4_13[2]), "=f"(_unpack_fp4_13[3]), "=f"(_unpack_fp4_13[4]), "=f"(_unpack_fp4_13[5]), "=f"(_unpack_fp4_13[6]), "=f"(_unpack_fp4_13[7]) : "r"(_fp4_wire_values_0[5 + 0]));
                        #endif
                    }
                    float2 _f2_172 = make_float2(acc_45[8], acc_45[9]);
                    float2 _f2_173 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                    float2 r_70 = fma_f32x2_rn_noftz(_f2_173, _f2_162, _f2_172);
                    acc_45[8] = r_70.x;
                    acc_45[9] = r_70.y;
                    float2 _f2_174 = make_float2(acc_45[10], acc_45[11]);
                    float2 _f2_175 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                    float2 r_71 = fma_f32x2_rn_noftz(_f2_175, _f2_162, _f2_174);
                    acc_45[10] = r_71.x;
                    acc_45[11] = r_71.y;
                    float2 _f2_176 = make_float2(acc_45[12], acc_45[13]);
                    float2 _f2_177 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                    float2 r_72 = fma_f32x2_rn_noftz(_f2_177, _f2_162, _f2_176);
                    acc_45[12] = r_72.x;
                    acc_45[13] = r_72.y;
                    float2 _f2_178 = make_float2(acc_45[14], acc_45[15]);
                    float2 _f2_179 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                    float2 r_73 = fma_f32x2_rn_noftz(_f2_179, _f2_162, _f2_178);
                    acc_45[14] = r_73.x;
                    acc_45[15] = r_73.y;
                    float _unpack_fp4_14[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[6 + 0]));
                        (&_unpack_fp4_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_14[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_14[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_14[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_14[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_14[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_14[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_14[0]), "=f"(_unpack_fp4_14[1]), "=f"(_unpack_fp4_14[2]), "=f"(_unpack_fp4_14[3]), "=f"(_unpack_fp4_14[4]), "=f"(_unpack_fp4_14[5]), "=f"(_unpack_fp4_14[6]), "=f"(_unpack_fp4_14[7]) : "r"(_fp4_wire_values_0[6 + 0]));
                        #endif
                    }
                    float2 _f2_180 = make_float2(acc_45[16], acc_45[17]);
                    float2 _f2_181 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                    float2 r_74 = fma_f32x2_rn_noftz(_f2_181, _f2_163, _f2_180);
                    acc_45[16] = r_74.x;
                    acc_45[17] = r_74.y;
                    float2 _f2_182 = make_float2(acc_45[18], acc_45[19]);
                    float2 _f2_183 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                    float2 r_75 = fma_f32x2_rn_noftz(_f2_183, _f2_163, _f2_182);
                    acc_45[18] = r_75.x;
                    acc_45[19] = r_75.y;
                    float2 _f2_184 = make_float2(acc_45[20], acc_45[21]);
                    float2 _f2_185 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                    float2 r_76 = fma_f32x2_rn_noftz(_f2_185, _f2_163, _f2_184);
                    acc_45[20] = r_76.x;
                    acc_45[21] = r_76.y;
                    float2 _f2_186 = make_float2(acc_45[22], acc_45[23]);
                    float2 _f2_187 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                    float2 r_77 = fma_f32x2_rn_noftz(_f2_187, _f2_163, _f2_186);
                    acc_45[22] = r_77.x;
                    acc_45[23] = r_77.y;
                    float _unpack_fp4_15[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[7 + 0]));
                        (&_unpack_fp4_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_15[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_15[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_15[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_15[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_15[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_15[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_15[0]), "=f"(_unpack_fp4_15[1]), "=f"(_unpack_fp4_15[2]), "=f"(_unpack_fp4_15[3]), "=f"(_unpack_fp4_15[4]), "=f"(_unpack_fp4_15[5]), "=f"(_unpack_fp4_15[6]), "=f"(_unpack_fp4_15[7]) : "r"(_fp4_wire_values_0[7 + 0]));
                        #endif
                    }
                    float2 _f2_188 = make_float2(acc_45[24], acc_45[25]);
                    float2 _f2_189 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                    float2 r_78 = fma_f32x2_rn_noftz(_f2_189, _f2_163, _f2_188);
                    acc_45[24] = r_78.x;
                    acc_45[25] = r_78.y;
                    float2 _f2_190 = make_float2(acc_45[26], acc_45[27]);
                    float2 _f2_191 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                    float2 r_79 = fma_f32x2_rn_noftz(_f2_191, _f2_163, _f2_190);
                    acc_45[26] = r_79.x;
                    acc_45[27] = r_79.y;
                    float2 _f2_192 = make_float2(acc_45[28], acc_45[29]);
                    float2 _f2_193 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                    float2 r_80 = fma_f32x2_rn_noftz(_f2_193, _f2_163, _f2_192);
                    acc_45[28] = r_80.x;
                    acc_45[29] = r_80.y;
                    float2 _f2_194 = make_float2(acc_45[30], acc_45[31]);
                    float2 _f2_195 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                    float2 r_81 = fma_f32x2_rn_noftz(_f2_195, _f2_163, _f2_194);
                    acc_45[30] = r_81.x;
                    acc_45[31] = r_81.y;
                    uint32_t acc_45_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_45[_lp*2 + 0], acc_45[_lp*2+1 + 0]));
                        acc_45_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_45_bf16[0]), "r"(acc_45_bf16[1]), "r"(acc_45_bf16[2]), "r"(acc_45_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_45_bf16[4]), "r"(acc_45_bf16[5]), "r"(acc_45_bf16[6]), "r"(acc_45_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_45_bf16[8]), "r"(acc_45_bf16[9]), "r"(acc_45_bf16[10]), "r"(acc_45_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_45_bf16[12]), "r"(acc_45_bf16[13]), "r"(acc_45_bf16[14]), "r"(acc_45_bf16[15]) : "memory");
                    __syncwarp();
                    int c_82 = lane_1;
                    int l_83 = c_82 >> 2;
                    unsigned int w4_84[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_84[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_84[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_84[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_84[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_83 + (c_82 & 3 ^ l_83 >> 1 & 3)))));
                    long long idx_85 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_82));
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
                    long long idx_89 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_86));
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
                    long long idx_93 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_90));
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
                    long long idx_97 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_94));
                    {
                        int4 _iv4 = make_int4(w4_96[0 + 0], w4_96[0 + 1], w4_96[0 + 2], w4_96[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_97)) = _iv4;
                    }
                    __syncwarp();
                    float acc_98[32];
                    acc_98[0] = 0.0f;
                    acc_98[1] = 0.0f;
                    acc_98[2] = 0.0f;
                    acc_98[3] = 0.0f;
                    acc_98[4] = 0.0f;
                    acc_98[5] = 0.0f;
                    acc_98[6] = 0.0f;
                    acc_98[7] = 0.0f;
                    acc_98[8] = 0.0f;
                    acc_98[9] = 0.0f;
                    acc_98[10] = 0.0f;
                    acc_98[11] = 0.0f;
                    acc_98[12] = 0.0f;
                    acc_98[13] = 0.0f;
                    acc_98[14] = 0.0f;
                    acc_98[15] = 0.0f;
                    acc_98[16] = 0.0f;
                    acc_98[17] = 0.0f;
                    acc_98[18] = 0.0f;
                    acc_98[19] = 0.0f;
                    acc_98[20] = 0.0f;
                    acc_98[21] = 0.0f;
                    acc_98[22] = 0.0f;
                    acc_98[23] = 0.0f;
                    acc_98[24] = 0.0f;
                    acc_98[25] = 0.0f;
                    acc_98[26] = 0.0f;
                    acc_98[27] = 0.0f;
                    acc_98[28] = 0.0f;
                    acc_98[29] = 0.0f;
                    acc_98[30] = 0.0f;
                    acc_98[31] = 0.0f;
                    float2 _fp8x2_decode_7;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_7.x), "=f"(_fp8x2_decode_7.y) : "h"((uint16_t)own_sc[2]));
                    float _mul_14;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(_fp8x2_decode_7.x), "f"(wsm32[dob_off + 3072 + 384 >> 2]));
                    float s0_99 = _mul_14;
                    float _mul_15;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(_fp8x2_decode_7.y), "f"(wsm32[dob_off + 3072 + 384 >> 2]));
                    float s1_100 = _mul_15;
                    float2 _f2_196 = make_float2(s0_99, s0_99);
                    float2 _f2_197 = make_float2(s1_100, s1_100);
                    float _unpack_fp4_16[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[8 + 0]));
                        (&_unpack_fp4_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_16[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_16[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_16[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_16[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_16[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_16[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_16[0]), "=f"(_unpack_fp4_16[1]), "=f"(_unpack_fp4_16[2]), "=f"(_unpack_fp4_16[3]), "=f"(_unpack_fp4_16[4]), "=f"(_unpack_fp4_16[5]), "=f"(_unpack_fp4_16[6]), "=f"(_unpack_fp4_16[7]) : "r"(own[8 + 0]));
                        #endif
                    }
                    float2 _f2_198 = make_float2(acc_98[0], acc_98[1]);
                    float2 _f2_199 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                    float2 r_101 = fma_f32x2_rn_noftz(_f2_199, _f2_196, _f2_198);
                    acc_98[0] = r_101.x;
                    acc_98[1] = r_101.y;
                    float2 _f2_200 = make_float2(acc_98[2], acc_98[3]);
                    float2 _f2_201 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                    float2 r_102 = fma_f32x2_rn_noftz(_f2_201, _f2_196, _f2_200);
                    acc_98[2] = r_102.x;
                    acc_98[3] = r_102.y;
                    float2 _f2_202 = make_float2(acc_98[4], acc_98[5]);
                    float2 _f2_203 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                    float2 r_103 = fma_f32x2_rn_noftz(_f2_203, _f2_196, _f2_202);
                    acc_98[4] = r_103.x;
                    acc_98[5] = r_103.y;
                    float2 _f2_204 = make_float2(acc_98[6], acc_98[7]);
                    float2 _f2_205 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                    float2 r_104 = fma_f32x2_rn_noftz(_f2_205, _f2_196, _f2_204);
                    acc_98[6] = r_104.x;
                    acc_98[7] = r_104.y;
                    float _unpack_fp4_17[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[9 + 0]));
                        (&_unpack_fp4_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_17[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_17[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_17[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_17[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_17[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_17[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_17[0]), "=f"(_unpack_fp4_17[1]), "=f"(_unpack_fp4_17[2]), "=f"(_unpack_fp4_17[3]), "=f"(_unpack_fp4_17[4]), "=f"(_unpack_fp4_17[5]), "=f"(_unpack_fp4_17[6]), "=f"(_unpack_fp4_17[7]) : "r"(own[9 + 0]));
                        #endif
                    }
                    float2 _f2_206 = make_float2(acc_98[8], acc_98[9]);
                    float2 _f2_207 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                    float2 r_105 = fma_f32x2_rn_noftz(_f2_207, _f2_196, _f2_206);
                    acc_98[8] = r_105.x;
                    acc_98[9] = r_105.y;
                    float2 _f2_208 = make_float2(acc_98[10], acc_98[11]);
                    float2 _f2_209 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                    float2 r_106 = fma_f32x2_rn_noftz(_f2_209, _f2_196, _f2_208);
                    acc_98[10] = r_106.x;
                    acc_98[11] = r_106.y;
                    float2 _f2_210 = make_float2(acc_98[12], acc_98[13]);
                    float2 _f2_211 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                    float2 r_107 = fma_f32x2_rn_noftz(_f2_211, _f2_196, _f2_210);
                    acc_98[12] = r_107.x;
                    acc_98[13] = r_107.y;
                    float2 _f2_212 = make_float2(acc_98[14], acc_98[15]);
                    float2 _f2_213 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                    float2 r_108 = fma_f32x2_rn_noftz(_f2_213, _f2_196, _f2_212);
                    acc_98[14] = r_108.x;
                    acc_98[15] = r_108.y;
                    float _unpack_fp4_18[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[10 + 0]));
                        (&_unpack_fp4_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_18[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_18[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_18[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_18[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_18[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_18[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_18[0]), "=f"(_unpack_fp4_18[1]), "=f"(_unpack_fp4_18[2]), "=f"(_unpack_fp4_18[3]), "=f"(_unpack_fp4_18[4]), "=f"(_unpack_fp4_18[5]), "=f"(_unpack_fp4_18[6]), "=f"(_unpack_fp4_18[7]) : "r"(own[10 + 0]));
                        #endif
                    }
                    float2 _f2_214 = make_float2(acc_98[16], acc_98[17]);
                    float2 _f2_215 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                    float2 r_109 = fma_f32x2_rn_noftz(_f2_215, _f2_197, _f2_214);
                    acc_98[16] = r_109.x;
                    acc_98[17] = r_109.y;
                    float2 _f2_216 = make_float2(acc_98[18], acc_98[19]);
                    float2 _f2_217 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                    float2 r_110 = fma_f32x2_rn_noftz(_f2_217, _f2_197, _f2_216);
                    acc_98[18] = r_110.x;
                    acc_98[19] = r_110.y;
                    float2 _f2_218 = make_float2(acc_98[20], acc_98[21]);
                    float2 _f2_219 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                    float2 r_111 = fma_f32x2_rn_noftz(_f2_219, _f2_197, _f2_218);
                    acc_98[20] = r_111.x;
                    acc_98[21] = r_111.y;
                    float2 _f2_220 = make_float2(acc_98[22], acc_98[23]);
                    float2 _f2_221 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                    float2 r_112 = fma_f32x2_rn_noftz(_f2_221, _f2_197, _f2_220);
                    acc_98[22] = r_112.x;
                    acc_98[23] = r_112.y;
                    float _unpack_fp4_19[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(own[11 + 0]));
                        (&_unpack_fp4_19[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_19[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_19[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_19[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_19[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_19[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_19[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_19[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_19[0]), "=f"(_unpack_fp4_19[1]), "=f"(_unpack_fp4_19[2]), "=f"(_unpack_fp4_19[3]), "=f"(_unpack_fp4_19[4]), "=f"(_unpack_fp4_19[5]), "=f"(_unpack_fp4_19[6]), "=f"(_unpack_fp4_19[7]) : "r"(own[11 + 0]));
                        #endif
                    }
                    float2 _f2_222 = make_float2(acc_98[24], acc_98[25]);
                    float2 _f2_223 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                    float2 r_113 = fma_f32x2_rn_noftz(_f2_223, _f2_197, _f2_222);
                    acc_98[24] = r_113.x;
                    acc_98[25] = r_113.y;
                    float2 _f2_224 = make_float2(acc_98[26], acc_98[27]);
                    float2 _f2_225 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                    float2 r_114 = fma_f32x2_rn_noftz(_f2_225, _f2_197, _f2_224);
                    acc_98[26] = r_114.x;
                    acc_98[27] = r_114.y;
                    float2 _f2_226 = make_float2(acc_98[28], acc_98[29]);
                    float2 _f2_227 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                    float2 r_115 = fma_f32x2_rn_noftz(_f2_227, _f2_197, _f2_226);
                    acc_98[28] = r_115.x;
                    acc_98[29] = r_115.y;
                    float2 _f2_228 = make_float2(acc_98[30], acc_98[31]);
                    float2 _f2_229 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                    float2 r_116 = fma_f32x2_rn_noftz(_f2_229, _f2_197, _f2_228);
                    acc_98[30] = r_116.x;
                    acc_98[31] = r_116.y;
                    float2 _fp8x2_decode_8;
                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                        "mov.b32 {lo, hi}, pair;\n"
                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                        : "=f"(_fp8x2_decode_8.x), "=f"(_fp8x2_decode_8.y) : "h"((uint16_t)_fp4_wire_scales_0[2]));
                    float _mul_16;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_fp8x2_decode_8.x), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                    float s0_117 = _mul_16;
                    float _mul_17;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_fp8x2_decode_8.y), "f"(__uint_as_float(_sysv_poll32_group_0[0])));
                    float s1_118 = _mul_17;
                    float2 _f2_230 = make_float2(s0_117, s0_117);
                    float2 _f2_231 = make_float2(s1_118, s1_118);
                    float _unpack_fp4_20[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[8 + 0]));
                        (&_unpack_fp4_20[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_20[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_20[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_20[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_20[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_20[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_20[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_20[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_20[0]), "=f"(_unpack_fp4_20[1]), "=f"(_unpack_fp4_20[2]), "=f"(_unpack_fp4_20[3]), "=f"(_unpack_fp4_20[4]), "=f"(_unpack_fp4_20[5]), "=f"(_unpack_fp4_20[6]), "=f"(_unpack_fp4_20[7]) : "r"(_fp4_wire_values_0[8 + 0]));
                        #endif
                    }
                    float2 _f2_232 = make_float2(acc_98[0], acc_98[1]);
                    float2 _f2_233 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                    float2 r_119 = fma_f32x2_rn_noftz(_f2_233, _f2_230, _f2_232);
                    acc_98[0] = r_119.x;
                    acc_98[1] = r_119.y;
                    float2 _f2_234 = make_float2(acc_98[2], acc_98[3]);
                    float2 _f2_235 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                    float2 r_120 = fma_f32x2_rn_noftz(_f2_235, _f2_230, _f2_234);
                    acc_98[2] = r_120.x;
                    acc_98[3] = r_120.y;
                    float2 _f2_236 = make_float2(acc_98[4], acc_98[5]);
                    float2 _f2_237 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                    float2 r_121 = fma_f32x2_rn_noftz(_f2_237, _f2_230, _f2_236);
                    acc_98[4] = r_121.x;
                    acc_98[5] = r_121.y;
                    float2 _f2_238 = make_float2(acc_98[6], acc_98[7]);
                    float2 _f2_239 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                    float2 r_122 = fma_f32x2_rn_noftz(_f2_239, _f2_230, _f2_238);
                    acc_98[6] = r_122.x;
                    acc_98[7] = r_122.y;
                    float _unpack_fp4_21[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[9 + 0]));
                        (&_unpack_fp4_21[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_21[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_21[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_21[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_21[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_21[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_21[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_21[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_21[0]), "=f"(_unpack_fp4_21[1]), "=f"(_unpack_fp4_21[2]), "=f"(_unpack_fp4_21[3]), "=f"(_unpack_fp4_21[4]), "=f"(_unpack_fp4_21[5]), "=f"(_unpack_fp4_21[6]), "=f"(_unpack_fp4_21[7]) : "r"(_fp4_wire_values_0[9 + 0]));
                        #endif
                    }
                    float2 _f2_240 = make_float2(acc_98[8], acc_98[9]);
                    float2 _f2_241 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                    float2 r_123 = fma_f32x2_rn_noftz(_f2_241, _f2_230, _f2_240);
                    acc_98[8] = r_123.x;
                    acc_98[9] = r_123.y;
                    float2 _f2_242 = make_float2(acc_98[10], acc_98[11]);
                    float2 _f2_243 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                    float2 r_124 = fma_f32x2_rn_noftz(_f2_243, _f2_230, _f2_242);
                    acc_98[10] = r_124.x;
                    acc_98[11] = r_124.y;
                    float2 _f2_244 = make_float2(acc_98[12], acc_98[13]);
                    float2 _f2_245 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                    float2 r_125 = fma_f32x2_rn_noftz(_f2_245, _f2_230, _f2_244);
                    acc_98[12] = r_125.x;
                    acc_98[13] = r_125.y;
                    float2 _f2_246 = make_float2(acc_98[14], acc_98[15]);
                    float2 _f2_247 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                    float2 r_126 = fma_f32x2_rn_noftz(_f2_247, _f2_230, _f2_246);
                    acc_98[14] = r_126.x;
                    acc_98[15] = r_126.y;
                    float _unpack_fp4_22[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[10 + 0]));
                        (&_unpack_fp4_22[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_22[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_22[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_22[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_22[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_22[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_22[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_22[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_22[0]), "=f"(_unpack_fp4_22[1]), "=f"(_unpack_fp4_22[2]), "=f"(_unpack_fp4_22[3]), "=f"(_unpack_fp4_22[4]), "=f"(_unpack_fp4_22[5]), "=f"(_unpack_fp4_22[6]), "=f"(_unpack_fp4_22[7]) : "r"(_fp4_wire_values_0[10 + 0]));
                        #endif
                    }
                    float2 _f2_248 = make_float2(acc_98[16], acc_98[17]);
                    float2 _f2_249 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                    float2 r_127 = fma_f32x2_rn_noftz(_f2_249, _f2_231, _f2_248);
                    acc_98[16] = r_127.x;
                    acc_98[17] = r_127.y;
                    float2 _f2_250 = make_float2(acc_98[18], acc_98[19]);
                    float2 _f2_251 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                    float2 r_128 = fma_f32x2_rn_noftz(_f2_251, _f2_231, _f2_250);
                    acc_98[18] = r_128.x;
                    acc_98[19] = r_128.y;
                    float2 _f2_252 = make_float2(acc_98[20], acc_98[21]);
                    float2 _f2_253 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                    float2 r_129 = fma_f32x2_rn_noftz(_f2_253, _f2_231, _f2_252);
                    acc_98[20] = r_129.x;
                    acc_98[21] = r_129.y;
                    float2 _f2_254 = make_float2(acc_98[22], acc_98[23]);
                    float2 _f2_255 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                    float2 r_130 = fma_f32x2_rn_noftz(_f2_255, _f2_231, _f2_254);
                    acc_98[22] = r_130.x;
                    acc_98[23] = r_130.y;
                    float _unpack_fp4_23[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[11 + 0]));
                        (&_unpack_fp4_23[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_23[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_23[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_23[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_23[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_23[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_23[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_23[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_23[0]), "=f"(_unpack_fp4_23[1]), "=f"(_unpack_fp4_23[2]), "=f"(_unpack_fp4_23[3]), "=f"(_unpack_fp4_23[4]), "=f"(_unpack_fp4_23[5]), "=f"(_unpack_fp4_23[6]), "=f"(_unpack_fp4_23[7]) : "r"(_fp4_wire_values_0[11 + 0]));
                        #endif
                    }
                    float2 _f2_256 = make_float2(acc_98[24], acc_98[25]);
                    float2 _f2_257 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                    float2 r_131 = fma_f32x2_rn_noftz(_f2_257, _f2_231, _f2_256);
                    acc_98[24] = r_131.x;
                    acc_98[25] = r_131.y;
                    float2 _f2_258 = make_float2(acc_98[26], acc_98[27]);
                    float2 _f2_259 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                    float2 r_132 = fma_f32x2_rn_noftz(_f2_259, _f2_231, _f2_258);
                    acc_98[26] = r_132.x;
                    acc_98[27] = r_132.y;
                    float2 _f2_260 = make_float2(acc_98[28], acc_98[29]);
                    float2 _f2_261 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                    float2 r_133 = fma_f32x2_rn_noftz(_f2_261, _f2_231, _f2_260);
                    acc_98[28] = r_133.x;
                    acc_98[29] = r_133.y;
                    float2 _f2_262 = make_float2(acc_98[30], acc_98[31]);
                    float2 _f2_263 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                    float2 r_134 = fma_f32x2_rn_noftz(_f2_263, _f2_231, _f2_262);
                    acc_98[30] = r_134.x;
                    acc_98[31] = r_134.y;
                    uint32_t acc_98_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_98[_lp*2 + 0], acc_98[_lp*2+1 + 0]));
                        acc_98_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_98_bf16[0]), "r"(acc_98_bf16[1]), "r"(acc_98_bf16[2]), "r"(acc_98_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_98_bf16[4]), "r"(acc_98_bf16[5]), "r"(acc_98_bf16[6]), "r"(acc_98_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_98_bf16[8]), "r"(acc_98_bf16[9]), "r"(acc_98_bf16[10]), "r"(acc_98_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_98_bf16[12]), "r"(acc_98_bf16[13]), "r"(acc_98_bf16[14]), "r"(acc_98_bf16[15]) : "memory");
                    __syncwarp();
                    int c_135 = lane_1;
                    int l_136 = c_135 >> 2;
                    unsigned int w4_137[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_137[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_137[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_137[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_137[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_136 + (c_135 & 3 ^ l_136 >> 1 & 3)))));
                    long long idx_138 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_135));
                    {
                        int4 _iv4 = make_int4(w4_137[0 + 0], w4_137[0 + 1], w4_137[0 + 2], w4_137[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_138)) = _iv4;
                    }
                    int c_139 = lane_1 + 32;
                    int l_140 = c_139 >> 2;
                    unsigned int w4_141[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_141[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_141[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_141[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_141[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_140 + (c_139 & 3 ^ l_140 >> 1 & 3)))));
                    long long idx_142 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_139));
                    {
                        int4 _iv4 = make_int4(w4_141[0 + 0], w4_141[0 + 1], w4_141[0 + 2], w4_141[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_142)) = _iv4;
                    }
                    int c_143 = lane_1 + 64;
                    int l_144 = c_143 >> 2;
                    unsigned int w4_145[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_145[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_145[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_145[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_145[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_144 + (c_143 & 3 ^ l_144 >> 1 & 3)))));
                    long long idx_146 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_143));
                    {
                        int4 _iv4 = make_int4(w4_145[0 + 0], w4_145[0 + 1], w4_145[0 + 2], w4_145[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_146)) = _iv4;
                    }
                    int c_147 = lane_1 + 96;
                    int l_148 = c_147 >> 2;
                    unsigned int w4_149[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_149[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_149[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_149[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_149[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)stg_off + (unsigned int)(16 * (4 * l_148 + (c_147 & 3 ^ l_148 >> 1 & 3)))));
                    long long idx_150 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_147));
                    {
                        int4 _iv4 = make_int4(w4_149[0 + 0], w4_149[0 + 1], w4_149[0 + 2], w4_149[0 + 3]);
                        *reinterpret_cast<int4*>(out_i32 + (idx_150)) = _iv4;
                    }
                    __syncwarp();
                }
            }
            if (lane_1 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            if (tid == 0) {
                int base_2 = bid * 4;
                unsigned int rows_u = (unsigned int)rows;
                unsigned int algo_u = 1;
                ctrl[base_2 + 2] = rows_u;
                ctrl[base_2 + 3] = algo_u;
                ctrl[base_2] = call + 1;
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
            asm volatile("barrier.sync 0, 352;" ::: "memory");
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            unsigned int call_1 = ctl[0];
            unsigned int prev_rows = ctl[1];
            unsigned int prev_algo = ctl[2];
            unsigned int pbuf = (call_1 + 2) % 3;
            if (prev_algo != 0) {
                uint8_t* self_u8_1 = reinterpret_cast<uint8_t*>(workspace[0]);
                int* qself_i32 = reinterpret_cast<int*>(workspace[2]);
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
                    long long base_3 = pb64 * 113246208 + 56623104;
                    long long u = pr64 * 3456 / 16;
                    long long lo = u * bid64_1 / g64_1;
                    long long hi = u * (bid64_1 + 1) / g64_1;
                    long long g0 = lo + lane64;
                    int gi = (int)(g0 % 216);
                    #pragma unroll 1
                    for (long long g = g0; g < hi; g += 32) {
                        unsigned int w4_1[4];
                        w4_1[0] = ((gi < 192) ? sv4[0] : ss4[0]);
                        w4_1[1] = ((gi < 192) ? sv4[1] : ss4[1]);
                        w4_1[2] = ((gi < 192) ? sv4[2] : ss4[2]);
                        w4_1[3] = ((gi < 192) ? sv4[3] : ss4[3]);
                        long long widx = base_3 + 16 * g >> 2;
                        {
                            int4 _iv4 = make_int4(w4_1[0 + 0], w4_1[0 + 1], w4_1[0 + 2], w4_1[0 + 3]);
                            *reinterpret_cast<int4*>(qself_i32 + (widx)) = _iv4;
                        }
                        int gn = gi + 32;
                        gi = ((gn >= 216) ? gn - 216 : gn);
                    }
                } else if (prev_algo == 1) {
                    int jbase = 0;
                    long long v_base_1 = pb64 * 113770496 + 50331648;
                    long long sc_base_1 = pb64 * 113770496 + 100663296 + 6291456;
                    long long m_base_1 = pb64 * 113770496 + 113246208 + 262144;
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
                                void* _cpbulk_dst_0 = reinterpret_cast<void*>(self_u8_1 + (v_base_1 + o));
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
                                void* _cpbulk_dst_1 = reinterpret_cast<void*>(self_u8_1 + (sc_base_1 + o_1));
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
                                void* _cpbulk_dst_2 = reinterpret_cast<void*>(self_u8_1 + (m_base_1 + o_2));
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
