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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(352, LAUNCH_MIN_BLOCKS) void
cake_fp4_allreduce_tp2_oneshot_bf16_rso_el(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, uint8_t* __restrict__ os_wire, uint8_t* __restrict__ os_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap)
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
    unsigned int* pat = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int pat_addr = smem + 1024;
    unsigned int* ctl = reinterpret_cast<unsigned int*>(smem_raw + 13312);
    const int ctl_addr = smem + 13312;
    float* s_am = reinterpret_cast<float*>(smem_raw + 13328);
    const int s_am_addr = smem + 13328;
    uint8_t* wsm = reinterpret_cast<uint8_t*>(smem_raw + 13376);
    const int wsm_addr = smem + 13376;
    float* wsm32 = reinterpret_cast<float*>(smem_raw + 13376);
    const int wsm32_addr = smem + 13376;

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
            asm volatile("fence.proxy.async.global;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            long long bid64 = (long long)bid;
            long long g64 = (long long)gridDim.x;
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
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(leader * 21280), x + (first_row * 6144), 12288, xb_full_addr + (leader) * 8);
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
            int n = 0;
            if (nrows > t) {
                n = (nrows - t + 5 - 1) / 5;
            }
            int f = lane_1 >> 1 & 3;
            int xb_off = leader * 21280;
            int stg_off = warp_0 * 21280 + 12288 + 6944;
            int peer = 1 - rank;
            long long cap64 = (long long)cap;
            long long rank64 = (long long)rank;
            long long peer64 = (long long)peer;
            long long buf64 = (long long)buf;
            long long v_base = buf64 * (2 * cap64 * 3456);
            long long s_base = v_base + 2 * cap64 * 3072;
            long long m_base = buf64 * (2 * cap64 * 16);
            uint8_t* peer_wire = reinterpret_cast<uint8_t*>(peer_ptrs[peer]);
            uint8_t* peer_rs = reinterpret_cast<uint8_t*>(peer_ptrs[2 + peer]);
            unsigned int _phase_xb_full = 0;
            #pragma unroll 1
            for (int k = 0; k < n + 1; k++) {
                if (n > k) {
                    int slot = k % 2;
                    int ob_off = xb_off + 12288 + slot * 3472;
                    if (h == 0 && k > 1 && lane_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    float rs = 1.0f;
                    mbarrier_wait(xb_full_addr + (leader) * 8, _phase_xb_full);
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
                        : "r"(wsm_addr + (unsigned int)(base_1 + 16 * f)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c1[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_1 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c2[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_1 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c3[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_1 + 16 * (f ^ 3))));
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
                    unsigned int xa = ((a & 65535) > (a >> 16) ? (a & 65535) : (a >> 16));
                    unsigned int xb = ((b & 65535) > (b >> 16) ? (b & 65535) : (b >> 16));
                    if ((f & 2) != 0) {
                        bmp[0] = xb | xa << 16;
                    } else {
                        bmp[0] = xa | xb << 16;
                    }
                    unsigned int _max_0 = ((xa) > (xb) ? (xa) : (xb));
                    am = fmaxf(am, __uint_as_float(_max_0 << 16));
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
                        : "r"(wsm_addr + (unsigned int)(base_1_1 + 16 * f)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c1_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_5[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_1_1 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c2_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_6[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_1_1 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c3_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_7[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_1_1 + 16 * (f ^ 3))));
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
                    unsigned int xa_8 = ((a_2 & 65535) > (a_2 >> 16) ? (a_2 & 65535) : (a_2 >> 16));
                    unsigned int xb_9 = ((b_3 & 65535) > (b_3 >> 16) ? (b_3 & 65535) : (b_3 >> 16));
                    if ((f & 2) != 0) {
                        bmp[1] = xb_9 | xa_8 << 16;
                    } else {
                        bmp[1] = xa_8 | xb_9 << 16;
                    }
                    unsigned int _max_1 = ((xa_8) > (xb_9) ? (xa_8) : (xb_9));
                    am = fmaxf(am, __uint_as_float(_max_1 << 16));
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
                        : "r"(wsm_addr + (unsigned int)(base_11 + 16 * f)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c1_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_15[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_11 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c2_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_16[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_11 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c3_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_17[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_11 + 16 * (f ^ 3))));
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
                    unsigned int xa_18 = ((a_12 & 65535) > (a_12 >> 16) ? (a_12 & 65535) : (a_12 >> 16));
                    unsigned int xb_19 = ((b_13 & 65535) > (b_13 >> 16) ? (b_13 & 65535) : (b_13 >> 16));
                    if ((f & 2) != 0) {
                        bmp[2] = xb_19 | xa_18 << 16;
                    } else {
                        bmp[2] = xa_18 | xb_19 << 16;
                    }
                    unsigned int _max_2 = ((xa_18) > (xb_19) ? (xa_18) : (xb_19));
                    am = fmaxf(am, __uint_as_float(_max_2 << 16));
                    float _warp_reduce_3 = am;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_3 = max_noftz(_warp_reduce_3, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset));
                    am = _warp_reduce_3;
                    if (lane_1 == 0) {
                        s_am[2 * t + h] = am;
                    }
                    asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                    am = fmaxf(s_am[2 * t], s_am[2 * t + 1]);
                    float _fdiv_rn_0 = __fdiv_rn(2688.0f, am);
                    float inv = 0.0f;
                    if (am > 0.0f) {
                        inv = _fdiv_rn_0;
                    }
                    float rs_20 = __fdiv_rn(am, 2688.0f);
                    int pp2_1_0 = lane_1 + 32 * (3 * h);
                    int base2_1_0 = xb_off + 64 * pp2_1_0;
                    float bm0_1_0 = __uint_as_float(bmp[0] << 16);
                    float bm1_1_0 = __uint_as_float(bmp[0] & 4294901760u);
                    float _fdiv_4 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_0, inv), 6.0f);
                    float _fdiv_5 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_0, inv), 6.0f);
                    uint16_t _e4m3x2_f32_0;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_fdiv_4), "f"(_fdiv_5));
                    uint32_t e4m3x2_to_f16x2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2) : "h"((uint16_t)(_e4m3x2_f32_0)));
                    float cvt_f32_f16;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16) : "h"((uint16_t)(e4m3x2_to_f16x2)));
                    float cvt_f32_f16_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1) : "h"((uint16_t)(e4m3x2_to_f16x2 >> 16)));
                    float _fdiv_rn_1 = __fdiv_rn(inv, cvt_f32_f16);
                    float _fdiv_rn_7_0 = __fdiv_rn(inv, cvt_f32_f16_1);
                    float m0_1_0 = 0.0f;
                    float m1_1_0 = 0.0f;
                    if (cvt_f32_f16 > 0.0f) {
                        m0_1_0 = _fdiv_rn_1;
                    }
                    if (cvt_f32_f16_1 > 0.0f) {
                        m1_1_0 = _fdiv_rn_7_0;
                    }
                    unsigned int w_1_0[4];
                    int cf_1_0 = f ^ 0;
                    unsigned int v4_1_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_0[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_0 + 16 * cf_1_0)));
                    float m_1_0 = m0_1_0;
                    if ((cf_1_0 & 2) != 0) {
                        m_1_0 = m1_1_0;
                    }
                    float2 mm = make_float2(m_1_0, m_1_0);
                    float2 pair = make_float2(__uint_as_float(v4_1_0[0] << 16), __uint_as_float(v4_1_0[0] & 4294901760u));
                    float2 g;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g) : "l"(*(const unsigned long long*)&pair), "l"(*(const unsigned long long*)&mm));
                    uint32_t fp4_pair;
                    uint16_t _e2m1_pair_h_6;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_6) : "f"(g.x), "f"(g.y));
                    fp4_pair = (uint32_t)_e2m1_pair_h_6;
                    float2 pair_1 = make_float2(__uint_as_float(v4_1_0[1] << 16), __uint_as_float(v4_1_0[1] & 4294901760u));
                    float2 g_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1) : "l"(*(const unsigned long long*)&pair_1), "l"(*(const unsigned long long*)&mm));
                    uint32_t fp4_pair_1;
                    uint16_t _e2m1_pair_h_7;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_7) : "f"(g_1.x), "f"(g_1.y));
                    fp4_pair_1 = (uint32_t)_e2m1_pair_h_7;
                    float2 pair_2 = make_float2(__uint_as_float(v4_1_0[2] << 16), __uint_as_float(v4_1_0[2] & 4294901760u));
                    float2 g_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2) : "l"(*(const unsigned long long*)&pair_2), "l"(*(const unsigned long long*)&mm));
                    uint32_t fp4_pair_2;
                    uint16_t _e2m1_pair_h_8;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_8) : "f"(g_2.x), "f"(g_2.y));
                    fp4_pair_2 = (uint32_t)_e2m1_pair_h_8;
                    float2 pair_3 = make_float2(__uint_as_float(v4_1_0[3] << 16), __uint_as_float(v4_1_0[3] & 4294901760u));
                    float2 g_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3) : "l"(*(const unsigned long long*)&pair_3), "l"(*(const unsigned long long*)&mm));
                    uint32_t _fp4_e2m1_pack_pair_9;
                    uint16_t _e2m1_pair_h_10;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_10) : "f"(g_3.x), "f"(g_3.y));
                    _fp4_e2m1_pack_pair_9 = (uint32_t)_e2m1_pair_h_10;
                    unsigned int word = fp4_pair | fp4_pair_1 << 8 | fp4_pair_2 << 16 | _fp4_e2m1_pack_pair_9 << 24;
                    unsigned int carry = (word & 2004318071) + 2004318071;
                    w_1_0[0] = word & (carry | 2004318071);
                    int cf_1_1 = f ^ 1;
                    unsigned int v4_1_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_0 + 16 * cf_1_1)));
                    float m_1_1 = m0_1_0;
                    if ((cf_1_1 & 2) != 0) {
                        m_1_1 = m1_1_0;
                    }
                    float2 mm_0 = make_float2(m_1_1, m_1_1);
                    float2 pair_0 = make_float2(__uint_as_float(v4_1_1[0] << 16), __uint_as_float(v4_1_1[0] & 4294901760u));
                    float2 g_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_0) : "l"(*(const unsigned long long*)&pair_0), "l"(*(const unsigned long long*)&mm_0));
                    uint32_t fp4_pair_0;
                    uint16_t _e2m1_pair_h_11;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_11) : "f"(g_0.x), "f"(g_0.y));
                    fp4_pair_0 = (uint32_t)_e2m1_pair_h_11;
                    float2 pair_1_0 = make_float2(__uint_as_float(v4_1_1[1] << 16), __uint_as_float(v4_1_1[1] & 4294901760u));
                    float2 g_1_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_0) : "l"(*(const unsigned long long*)&pair_1_0), "l"(*(const unsigned long long*)&mm_0));
                    uint32_t fp4_pair_1_0;
                    uint16_t _e2m1_pair_h_12;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_12) : "f"(g_1_0.x), "f"(g_1_0.y));
                    fp4_pair_1_0 = (uint32_t)_e2m1_pair_h_12;
                    float2 pair_2_0 = make_float2(__uint_as_float(v4_1_1[2] << 16), __uint_as_float(v4_1_1[2] & 4294901760u));
                    float2 g_2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_0) : "l"(*(const unsigned long long*)&pair_2_0), "l"(*(const unsigned long long*)&mm_0));
                    uint32_t fp4_pair_2_0;
                    uint16_t _e2m1_pair_h_13;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_13) : "f"(g_2_0.x), "f"(g_2_0.y));
                    fp4_pair_2_0 = (uint32_t)_e2m1_pair_h_13;
                    float2 pair_3_0 = make_float2(__uint_as_float(v4_1_1[3] << 16), __uint_as_float(v4_1_1[3] & 4294901760u));
                    float2 g_3_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_0) : "l"(*(const unsigned long long*)&pair_3_0), "l"(*(const unsigned long long*)&mm_0));
                    uint32_t _fp4_e2m1_pack_pair_14;
                    uint16_t _e2m1_pair_h_15;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_15) : "f"(g_3_0.x), "f"(g_3_0.y));
                    _fp4_e2m1_pack_pair_14 = (uint32_t)_e2m1_pair_h_15;
                    unsigned int word_0 = fp4_pair_0 | fp4_pair_1_0 << 8 | fp4_pair_2_0 << 16 | _fp4_e2m1_pack_pair_14 << 24;
                    unsigned int carry_0 = (word_0 & 2004318071) + 2004318071;
                    w_1_0[1] = word_0 & (carry_0 | 2004318071);
                    int cf_1_2 = f ^ 2;
                    unsigned int v4_1_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_2[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_0 + 16 * cf_1_2)));
                    float m_1_2 = m0_1_0;
                    if ((cf_1_2 & 2) != 0) {
                        m_1_2 = m1_1_0;
                    }
                    float2 mm_1 = make_float2(m_1_2, m_1_2);
                    float2 pair_4 = make_float2(__uint_as_float(v4_1_2[0] << 16), __uint_as_float(v4_1_2[0] & 4294901760u));
                    float2 g_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_4) : "l"(*(const unsigned long long*)&pair_4), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t fp4_pair_3;
                    uint16_t _e2m1_pair_h_16;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_16) : "f"(g_4.x), "f"(g_4.y));
                    fp4_pair_3 = (uint32_t)_e2m1_pair_h_16;
                    float2 pair_1_1 = make_float2(__uint_as_float(v4_1_2[1] << 16), __uint_as_float(v4_1_2[1] & 4294901760u));
                    float2 g_1_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_1) : "l"(*(const unsigned long long*)&pair_1_1), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t fp4_pair_1_1;
                    uint16_t _e2m1_pair_h_17;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_17) : "f"(g_1_1.x), "f"(g_1_1.y));
                    fp4_pair_1_1 = (uint32_t)_e2m1_pair_h_17;
                    float2 pair_2_1 = make_float2(__uint_as_float(v4_1_2[2] << 16), __uint_as_float(v4_1_2[2] & 4294901760u));
                    float2 g_2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_1) : "l"(*(const unsigned long long*)&pair_2_1), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t fp4_pair_2_1;
                    uint16_t _e2m1_pair_h_18;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_18) : "f"(g_2_1.x), "f"(g_2_1.y));
                    fp4_pair_2_1 = (uint32_t)_e2m1_pair_h_18;
                    float2 pair_3_1 = make_float2(__uint_as_float(v4_1_2[3] << 16), __uint_as_float(v4_1_2[3] & 4294901760u));
                    float2 g_3_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_1) : "l"(*(const unsigned long long*)&pair_3_1), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t _fp4_e2m1_pack_pair_19;
                    uint16_t _e2m1_pair_h_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_20) : "f"(g_3_1.x), "f"(g_3_1.y));
                    _fp4_e2m1_pack_pair_19 = (uint32_t)_e2m1_pair_h_20;
                    unsigned int word_2 = fp4_pair_3 | fp4_pair_1_1 << 8 | fp4_pair_2_1 << 16 | _fp4_e2m1_pack_pair_19 << 24;
                    unsigned int carry_2 = (word_2 & 2004318071) + 2004318071;
                    w_1_0[2] = word_2 & (carry_2 | 2004318071);
                    int cf_1_3 = f ^ 3;
                    unsigned int v4_1_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_3[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_0 + 16 * cf_1_3)));
                    float m_1_3 = m0_1_0;
                    if ((cf_1_3 & 2) != 0) {
                        m_1_3 = m1_1_0;
                    }
                    float2 mm_2 = make_float2(m_1_3, m_1_3);
                    float2 pair_5 = make_float2(__uint_as_float(v4_1_3[0] << 16), __uint_as_float(v4_1_3[0] & 4294901760u));
                    float2 g_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_5) : "l"(*(const unsigned long long*)&pair_5), "l"(*(const unsigned long long*)&mm_2));
                    uint32_t fp4_pair_4;
                    uint16_t _e2m1_pair_h_21;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_21) : "f"(g_5.x), "f"(g_5.y));
                    fp4_pair_4 = (uint32_t)_e2m1_pair_h_21;
                    float2 pair_1_2 = make_float2(__uint_as_float(v4_1_3[1] << 16), __uint_as_float(v4_1_3[1] & 4294901760u));
                    float2 g_1_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_2) : "l"(*(const unsigned long long*)&pair_1_2), "l"(*(const unsigned long long*)&mm_2));
                    uint32_t fp4_pair_1_2;
                    uint16_t _e2m1_pair_h_22;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_22) : "f"(g_1_2.x), "f"(g_1_2.y));
                    fp4_pair_1_2 = (uint32_t)_e2m1_pair_h_22;
                    float2 pair_2_2 = make_float2(__uint_as_float(v4_1_3[2] << 16), __uint_as_float(v4_1_3[2] & 4294901760u));
                    float2 g_2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_2) : "l"(*(const unsigned long long*)&pair_2_2), "l"(*(const unsigned long long*)&mm_2));
                    uint32_t fp4_pair_2_2;
                    uint16_t _e2m1_pair_h_23;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_23) : "f"(g_2_2.x), "f"(g_2_2.y));
                    fp4_pair_2_2 = (uint32_t)_e2m1_pair_h_23;
                    float2 pair_3_2 = make_float2(__uint_as_float(v4_1_3[3] << 16), __uint_as_float(v4_1_3[3] & 4294901760u));
                    float2 g_3_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_2) : "l"(*(const unsigned long long*)&pair_3_2), "l"(*(const unsigned long long*)&mm_2));
                    uint32_t _fp4_e2m1_pack_pair_24;
                    uint16_t _e2m1_pair_h_25;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_25) : "f"(g_3_2.x), "f"(g_3_2.y));
                    _fp4_e2m1_pack_pair_24 = (uint32_t)_e2m1_pair_h_25;
                    unsigned int word_3 = fp4_pair_4 | fp4_pair_1_2 << 8 | fp4_pair_2_2 << 16 | _fp4_e2m1_pack_pair_24 << 24;
                    unsigned int carry_3 = (word_3 & 2004318071) + 2004318071;
                    w_1_0[3] = word_3 & (carry_3 | 2004318071);
                    bool fodd_1_0 = (f & 1) != 0;
                    bool fhi_1_0 = (f & 2) != 0;
                    unsigned int t0_1_0 = w_1_0[0];
                    unsigned int t1_1_0 = w_1_0[1];
                    unsigned int t2_1_0 = w_1_0[2];
                    unsigned int t3_1_0 = w_1_0[3];
                    if (fodd_1_0) {
                        t0_1_0 = w_1_0[1];
                        t1_1_0 = w_1_0[0];
                        t2_1_0 = w_1_0[3];
                        t3_1_0 = w_1_0[2];
                    }
                    unsigned int q0_1_0 = t0_1_0;
                    unsigned int q1_1_0 = t1_1_0;
                    unsigned int q2_1_0 = t2_1_0;
                    unsigned int q3_1_0 = t3_1_0;
                    if (fhi_1_0) {
                        q0_1_0 = t2_1_0;
                        q1_1_0 = t3_1_0;
                        q2_1_0 = t0_1_0;
                        q3_1_0 = t1_1_0;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_1_0)), "r"(q0_1_0), "r"(q1_1_0), "r"(q2_1_0), "r"(q3_1_0) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_0)), "h"((uint16_t)(_e4m3x2_f32_0 & 32639)) : "memory");
                    int pp2_1_1 = lane_1 + 32 * (3 * h + 1);
                    int base2_1_1 = xb_off + 64 * pp2_1_1;
                    float bm0_1_1 = __uint_as_float(bmp[1] << 16);
                    float bm1_1_1 = __uint_as_float(bmp[1] & 4294901760u);
                    float _fdiv_26 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_1, inv), 6.0f);
                    float _fdiv_27 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_1, inv), 6.0f);
                    uint16_t _e4m3x2_f32_1;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_fdiv_26), "f"(_fdiv_27));
                    uint32_t e4m3x2_to_f16x2_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_0) : "h"((uint16_t)(_e4m3x2_f32_1)));
                    float cvt_f32_f16_0;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_0) : "h"((uint16_t)(e4m3x2_to_f16x2_0)));
                    float cvt_f32_f16_1_0;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_0) : "h"((uint16_t)(e4m3x2_to_f16x2_0 >> 16)));
                    float _fdiv_rn_2 = __fdiv_rn(inv, cvt_f32_f16_0);
                    float _fdiv_rn_7_1 = __fdiv_rn(inv, cvt_f32_f16_1_0);
                    float m0_1_1 = 0.0f;
                    float m1_1_1 = 0.0f;
                    if (cvt_f32_f16_0 > 0.0f) {
                        m0_1_1 = _fdiv_rn_2;
                    }
                    if (cvt_f32_f16_1_0 > 0.0f) {
                        m1_1_1 = _fdiv_rn_7_1;
                    }
                    unsigned int w_1_1[4];
                    int cf_1_4 = f ^ 0;
                    unsigned int v4_1_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_1 + 16 * cf_1_4)));
                    float m_1_4 = m0_1_1;
                    if ((cf_1_4 & 2) != 0) {
                        m_1_4 = m1_1_1;
                    }
                    float2 mm_3 = make_float2(m_1_4, m_1_4);
                    float2 pair_6 = make_float2(__uint_as_float(v4_1_4[0] << 16), __uint_as_float(v4_1_4[0] & 4294901760u));
                    float2 g_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_6) : "l"(*(const unsigned long long*)&pair_6), "l"(*(const unsigned long long*)&mm_3));
                    uint32_t fp4_pair_5;
                    uint16_t _e2m1_pair_h_28;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_28) : "f"(g_6.x), "f"(g_6.y));
                    fp4_pair_5 = (uint32_t)_e2m1_pair_h_28;
                    float2 pair_1_3 = make_float2(__uint_as_float(v4_1_4[1] << 16), __uint_as_float(v4_1_4[1] & 4294901760u));
                    float2 g_1_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_3) : "l"(*(const unsigned long long*)&pair_1_3), "l"(*(const unsigned long long*)&mm_3));
                    uint32_t fp4_pair_1_3;
                    uint16_t _e2m1_pair_h_29;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_29) : "f"(g_1_3.x), "f"(g_1_3.y));
                    fp4_pair_1_3 = (uint32_t)_e2m1_pair_h_29;
                    float2 pair_2_3 = make_float2(__uint_as_float(v4_1_4[2] << 16), __uint_as_float(v4_1_4[2] & 4294901760u));
                    float2 g_2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_3) : "l"(*(const unsigned long long*)&pair_2_3), "l"(*(const unsigned long long*)&mm_3));
                    uint32_t fp4_pair_2_3;
                    uint16_t _e2m1_pair_h_30;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_30) : "f"(g_2_3.x), "f"(g_2_3.y));
                    fp4_pair_2_3 = (uint32_t)_e2m1_pair_h_30;
                    float2 pair_3_3 = make_float2(__uint_as_float(v4_1_4[3] << 16), __uint_as_float(v4_1_4[3] & 4294901760u));
                    float2 g_3_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_3) : "l"(*(const unsigned long long*)&pair_3_3), "l"(*(const unsigned long long*)&mm_3));
                    uint32_t _fp4_e2m1_pack_pair_31;
                    uint16_t _e2m1_pair_h_32;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_32) : "f"(g_3_3.x), "f"(g_3_3.y));
                    _fp4_e2m1_pack_pair_31 = (uint32_t)_e2m1_pair_h_32;
                    unsigned int word_4 = fp4_pair_5 | fp4_pair_1_3 << 8 | fp4_pair_2_3 << 16 | _fp4_e2m1_pack_pair_31 << 24;
                    unsigned int carry_4 = (word_4 & 2004318071) + 2004318071;
                    w_1_1[0] = word_4 & (carry_4 | 2004318071);
                    int cf_1_5 = f ^ 1;
                    unsigned int v4_1_5[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_5[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_1 + 16 * cf_1_5)));
                    float m_1_5 = m0_1_1;
                    if ((cf_1_5 & 2) != 0) {
                        m_1_5 = m1_1_1;
                    }
                    float2 mm_4 = make_float2(m_1_5, m_1_5);
                    float2 pair_7 = make_float2(__uint_as_float(v4_1_5[0] << 16), __uint_as_float(v4_1_5[0] & 4294901760u));
                    float2 g_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_7) : "l"(*(const unsigned long long*)&pair_7), "l"(*(const unsigned long long*)&mm_4));
                    uint32_t fp4_pair_6;
                    uint16_t _e2m1_pair_h_33;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_33) : "f"(g_7.x), "f"(g_7.y));
                    fp4_pair_6 = (uint32_t)_e2m1_pair_h_33;
                    float2 pair_1_4 = make_float2(__uint_as_float(v4_1_5[1] << 16), __uint_as_float(v4_1_5[1] & 4294901760u));
                    float2 g_1_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_4) : "l"(*(const unsigned long long*)&pair_1_4), "l"(*(const unsigned long long*)&mm_4));
                    uint32_t fp4_pair_1_4;
                    uint16_t _e2m1_pair_h_34;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_34) : "f"(g_1_4.x), "f"(g_1_4.y));
                    fp4_pair_1_4 = (uint32_t)_e2m1_pair_h_34;
                    float2 pair_2_4 = make_float2(__uint_as_float(v4_1_5[2] << 16), __uint_as_float(v4_1_5[2] & 4294901760u));
                    float2 g_2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_4) : "l"(*(const unsigned long long*)&pair_2_4), "l"(*(const unsigned long long*)&mm_4));
                    uint32_t fp4_pair_2_4;
                    uint16_t _e2m1_pair_h_35;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_35) : "f"(g_2_4.x), "f"(g_2_4.y));
                    fp4_pair_2_4 = (uint32_t)_e2m1_pair_h_35;
                    float2 pair_3_4 = make_float2(__uint_as_float(v4_1_5[3] << 16), __uint_as_float(v4_1_5[3] & 4294901760u));
                    float2 g_3_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_4) : "l"(*(const unsigned long long*)&pair_3_4), "l"(*(const unsigned long long*)&mm_4));
                    uint32_t _fp4_e2m1_pack_pair_36;
                    uint16_t _e2m1_pair_h_37;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_37) : "f"(g_3_4.x), "f"(g_3_4.y));
                    _fp4_e2m1_pack_pair_36 = (uint32_t)_e2m1_pair_h_37;
                    unsigned int word_5 = fp4_pair_6 | fp4_pair_1_4 << 8 | fp4_pair_2_4 << 16 | _fp4_e2m1_pack_pair_36 << 24;
                    unsigned int carry_5 = (word_5 & 2004318071) + 2004318071;
                    w_1_1[1] = word_5 & (carry_5 | 2004318071);
                    int cf_1_6 = f ^ 2;
                    unsigned int v4_1_6[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_6[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_1 + 16 * cf_1_6)));
                    float m_1_6 = m0_1_1;
                    if ((cf_1_6 & 2) != 0) {
                        m_1_6 = m1_1_1;
                    }
                    float2 mm_5 = make_float2(m_1_6, m_1_6);
                    float2 pair_8 = make_float2(__uint_as_float(v4_1_6[0] << 16), __uint_as_float(v4_1_6[0] & 4294901760u));
                    float2 g_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_8) : "l"(*(const unsigned long long*)&pair_8), "l"(*(const unsigned long long*)&mm_5));
                    uint32_t fp4_pair_7;
                    uint16_t _e2m1_pair_h_38;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_38) : "f"(g_8.x), "f"(g_8.y));
                    fp4_pair_7 = (uint32_t)_e2m1_pair_h_38;
                    float2 pair_1_5 = make_float2(__uint_as_float(v4_1_6[1] << 16), __uint_as_float(v4_1_6[1] & 4294901760u));
                    float2 g_1_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_5) : "l"(*(const unsigned long long*)&pair_1_5), "l"(*(const unsigned long long*)&mm_5));
                    uint32_t fp4_pair_1_5;
                    uint16_t _e2m1_pair_h_39;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_39) : "f"(g_1_5.x), "f"(g_1_5.y));
                    fp4_pair_1_5 = (uint32_t)_e2m1_pair_h_39;
                    float2 pair_2_5 = make_float2(__uint_as_float(v4_1_6[2] << 16), __uint_as_float(v4_1_6[2] & 4294901760u));
                    float2 g_2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_5) : "l"(*(const unsigned long long*)&pair_2_5), "l"(*(const unsigned long long*)&mm_5));
                    uint32_t fp4_pair_2_5;
                    uint16_t _e2m1_pair_h_40;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_40) : "f"(g_2_5.x), "f"(g_2_5.y));
                    fp4_pair_2_5 = (uint32_t)_e2m1_pair_h_40;
                    float2 pair_3_5 = make_float2(__uint_as_float(v4_1_6[3] << 16), __uint_as_float(v4_1_6[3] & 4294901760u));
                    float2 g_3_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_5) : "l"(*(const unsigned long long*)&pair_3_5), "l"(*(const unsigned long long*)&mm_5));
                    uint32_t _fp4_e2m1_pack_pair_41;
                    uint16_t _e2m1_pair_h_42;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_42) : "f"(g_3_5.x), "f"(g_3_5.y));
                    _fp4_e2m1_pack_pair_41 = (uint32_t)_e2m1_pair_h_42;
                    unsigned int word_6 = fp4_pair_7 | fp4_pair_1_5 << 8 | fp4_pair_2_5 << 16 | _fp4_e2m1_pack_pair_41 << 24;
                    unsigned int carry_6 = (word_6 & 2004318071) + 2004318071;
                    w_1_1[2] = word_6 & (carry_6 | 2004318071);
                    int cf_1_7 = f ^ 3;
                    unsigned int v4_1_7[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_7[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_1 + 16 * cf_1_7)));
                    float m_1_7 = m0_1_1;
                    if ((cf_1_7 & 2) != 0) {
                        m_1_7 = m1_1_1;
                    }
                    float2 mm_6 = make_float2(m_1_7, m_1_7);
                    float2 pair_9 = make_float2(__uint_as_float(v4_1_7[0] << 16), __uint_as_float(v4_1_7[0] & 4294901760u));
                    float2 g_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_9) : "l"(*(const unsigned long long*)&pair_9), "l"(*(const unsigned long long*)&mm_6));
                    uint32_t fp4_pair_8;
                    uint16_t _e2m1_pair_h_43;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_43) : "f"(g_9.x), "f"(g_9.y));
                    fp4_pair_8 = (uint32_t)_e2m1_pair_h_43;
                    float2 pair_1_6 = make_float2(__uint_as_float(v4_1_7[1] << 16), __uint_as_float(v4_1_7[1] & 4294901760u));
                    float2 g_1_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_6) : "l"(*(const unsigned long long*)&pair_1_6), "l"(*(const unsigned long long*)&mm_6));
                    uint32_t fp4_pair_1_6;
                    uint16_t _e2m1_pair_h_44;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_44) : "f"(g_1_6.x), "f"(g_1_6.y));
                    fp4_pair_1_6 = (uint32_t)_e2m1_pair_h_44;
                    float2 pair_2_6 = make_float2(__uint_as_float(v4_1_7[2] << 16), __uint_as_float(v4_1_7[2] & 4294901760u));
                    float2 g_2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_6) : "l"(*(const unsigned long long*)&pair_2_6), "l"(*(const unsigned long long*)&mm_6));
                    uint32_t fp4_pair_2_6;
                    uint16_t _e2m1_pair_h_45;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_45) : "f"(g_2_6.x), "f"(g_2_6.y));
                    fp4_pair_2_6 = (uint32_t)_e2m1_pair_h_45;
                    float2 pair_3_6 = make_float2(__uint_as_float(v4_1_7[3] << 16), __uint_as_float(v4_1_7[3] & 4294901760u));
                    float2 g_3_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_6) : "l"(*(const unsigned long long*)&pair_3_6), "l"(*(const unsigned long long*)&mm_6));
                    uint32_t _fp4_e2m1_pack_pair_46;
                    uint16_t _e2m1_pair_h_47;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_47) : "f"(g_3_6.x), "f"(g_3_6.y));
                    _fp4_e2m1_pack_pair_46 = (uint32_t)_e2m1_pair_h_47;
                    unsigned int word_7 = fp4_pair_8 | fp4_pair_1_6 << 8 | fp4_pair_2_6 << 16 | _fp4_e2m1_pack_pair_46 << 24;
                    unsigned int carry_7 = (word_7 & 2004318071) + 2004318071;
                    w_1_1[3] = word_7 & (carry_7 | 2004318071);
                    bool fodd_1_1 = (f & 1) != 0;
                    bool fhi_1_1 = (f & 2) != 0;
                    unsigned int t0_1_1 = w_1_1[0];
                    unsigned int t1_1_1 = w_1_1[1];
                    unsigned int t2_1_1 = w_1_1[2];
                    unsigned int t3_1_1 = w_1_1[3];
                    if (fodd_1_1) {
                        t0_1_1 = w_1_1[1];
                        t1_1_1 = w_1_1[0];
                        t2_1_1 = w_1_1[3];
                        t3_1_1 = w_1_1[2];
                    }
                    unsigned int q0_1_1 = t0_1_1;
                    unsigned int q1_1_1 = t1_1_1;
                    unsigned int q2_1_1 = t2_1_1;
                    unsigned int q3_1_1 = t3_1_1;
                    if (fhi_1_1) {
                        q0_1_1 = t2_1_1;
                        q1_1_1 = t3_1_1;
                        q2_1_1 = t0_1_1;
                        q3_1_1 = t1_1_1;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_1_1)), "r"(q0_1_1), "r"(q1_1_1), "r"(q2_1_1), "r"(q3_1_1) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_1)), "h"((uint16_t)(_e4m3x2_f32_1 & 32639)) : "memory");
                    int pp2_1_2 = lane_1 + 32 * (3 * h + 2);
                    int base2_1_2 = xb_off + 64 * pp2_1_2;
                    float bm0_1_2 = __uint_as_float(bmp[2] << 16);
                    float bm1_1_2 = __uint_as_float(bmp[2] & 4294901760u);
                    float _fdiv_48 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_2, inv), 6.0f);
                    float _fdiv_49 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_2, inv), 6.0f);
                    uint16_t _e4m3x2_f32_2;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(_fdiv_48), "f"(_fdiv_49));
                    uint32_t e4m3x2_to_f16x2_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_1) : "h"((uint16_t)(_e4m3x2_f32_2)));
                    float cvt_f32_f16_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_2) : "h"((uint16_t)(e4m3x2_to_f16x2_1)));
                    float cvt_f32_f16_1_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_1) : "h"((uint16_t)(e4m3x2_to_f16x2_1 >> 16)));
                    float _fdiv_rn_3 = __fdiv_rn(inv, cvt_f32_f16_2);
                    float _fdiv_rn_7_2 = __fdiv_rn(inv, cvt_f32_f16_1_1);
                    float m0_1_2 = 0.0f;
                    float m1_1_2 = 0.0f;
                    if (cvt_f32_f16_2 > 0.0f) {
                        m0_1_2 = _fdiv_rn_3;
                    }
                    if (cvt_f32_f16_1_1 > 0.0f) {
                        m1_1_2 = _fdiv_rn_7_2;
                    }
                    unsigned int w_1_2[4];
                    int cf_1_8 = f ^ 0;
                    unsigned int v4_1_8[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_8[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_2 + 16 * cf_1_8)));
                    float m_1_8 = m0_1_2;
                    if ((cf_1_8 & 2) != 0) {
                        m_1_8 = m1_1_2;
                    }
                    float2 mm_7 = make_float2(m_1_8, m_1_8);
                    float2 pair_10 = make_float2(__uint_as_float(v4_1_8[0] << 16), __uint_as_float(v4_1_8[0] & 4294901760u));
                    float2 g_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_10) : "l"(*(const unsigned long long*)&pair_10), "l"(*(const unsigned long long*)&mm_7));
                    uint32_t fp4_pair_9;
                    uint16_t _e2m1_pair_h_50;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_50) : "f"(g_10.x), "f"(g_10.y));
                    fp4_pair_9 = (uint32_t)_e2m1_pair_h_50;
                    float2 pair_1_7 = make_float2(__uint_as_float(v4_1_8[1] << 16), __uint_as_float(v4_1_8[1] & 4294901760u));
                    float2 g_1_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_7) : "l"(*(const unsigned long long*)&pair_1_7), "l"(*(const unsigned long long*)&mm_7));
                    uint32_t fp4_pair_1_7;
                    uint16_t _e2m1_pair_h_51;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_51) : "f"(g_1_7.x), "f"(g_1_7.y));
                    fp4_pair_1_7 = (uint32_t)_e2m1_pair_h_51;
                    float2 pair_2_7 = make_float2(__uint_as_float(v4_1_8[2] << 16), __uint_as_float(v4_1_8[2] & 4294901760u));
                    float2 g_2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_7) : "l"(*(const unsigned long long*)&pair_2_7), "l"(*(const unsigned long long*)&mm_7));
                    uint32_t fp4_pair_2_7;
                    uint16_t _e2m1_pair_h_52;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_52) : "f"(g_2_7.x), "f"(g_2_7.y));
                    fp4_pair_2_7 = (uint32_t)_e2m1_pair_h_52;
                    float2 pair_3_7 = make_float2(__uint_as_float(v4_1_8[3] << 16), __uint_as_float(v4_1_8[3] & 4294901760u));
                    float2 g_3_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_7) : "l"(*(const unsigned long long*)&pair_3_7), "l"(*(const unsigned long long*)&mm_7));
                    uint32_t _fp4_e2m1_pack_pair_53;
                    uint16_t _e2m1_pair_h_54;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_54) : "f"(g_3_7.x), "f"(g_3_7.y));
                    _fp4_e2m1_pack_pair_53 = (uint32_t)_e2m1_pair_h_54;
                    unsigned int word_8 = fp4_pair_9 | fp4_pair_1_7 << 8 | fp4_pair_2_7 << 16 | _fp4_e2m1_pack_pair_53 << 24;
                    unsigned int carry_8 = (word_8 & 2004318071) + 2004318071;
                    w_1_2[0] = word_8 & (carry_8 | 2004318071);
                    int cf_1_9 = f ^ 1;
                    unsigned int v4_1_9[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_9[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_2 + 16 * cf_1_9)));
                    float m_1_9 = m0_1_2;
                    if ((cf_1_9 & 2) != 0) {
                        m_1_9 = m1_1_2;
                    }
                    float2 mm_8 = make_float2(m_1_9, m_1_9);
                    float2 pair_11 = make_float2(__uint_as_float(v4_1_9[0] << 16), __uint_as_float(v4_1_9[0] & 4294901760u));
                    float2 g_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_11) : "l"(*(const unsigned long long*)&pair_11), "l"(*(const unsigned long long*)&mm_8));
                    uint32_t fp4_pair_10;
                    uint16_t _e2m1_pair_h_55;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_55) : "f"(g_11.x), "f"(g_11.y));
                    fp4_pair_10 = (uint32_t)_e2m1_pair_h_55;
                    float2 pair_1_8 = make_float2(__uint_as_float(v4_1_9[1] << 16), __uint_as_float(v4_1_9[1] & 4294901760u));
                    float2 g_1_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_8) : "l"(*(const unsigned long long*)&pair_1_8), "l"(*(const unsigned long long*)&mm_8));
                    uint32_t fp4_pair_1_8;
                    uint16_t _e2m1_pair_h_56;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_56) : "f"(g_1_8.x), "f"(g_1_8.y));
                    fp4_pair_1_8 = (uint32_t)_e2m1_pair_h_56;
                    float2 pair_2_8 = make_float2(__uint_as_float(v4_1_9[2] << 16), __uint_as_float(v4_1_9[2] & 4294901760u));
                    float2 g_2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_8) : "l"(*(const unsigned long long*)&pair_2_8), "l"(*(const unsigned long long*)&mm_8));
                    uint32_t fp4_pair_2_8;
                    uint16_t _e2m1_pair_h_57;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_57) : "f"(g_2_8.x), "f"(g_2_8.y));
                    fp4_pair_2_8 = (uint32_t)_e2m1_pair_h_57;
                    float2 pair_3_8 = make_float2(__uint_as_float(v4_1_9[3] << 16), __uint_as_float(v4_1_9[3] & 4294901760u));
                    float2 g_3_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_8) : "l"(*(const unsigned long long*)&pair_3_8), "l"(*(const unsigned long long*)&mm_8));
                    uint32_t _fp4_e2m1_pack_pair_58;
                    uint16_t _e2m1_pair_h_59;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_59) : "f"(g_3_8.x), "f"(g_3_8.y));
                    _fp4_e2m1_pack_pair_58 = (uint32_t)_e2m1_pair_h_59;
                    unsigned int word_9 = fp4_pair_10 | fp4_pair_1_8 << 8 | fp4_pair_2_8 << 16 | _fp4_e2m1_pack_pair_58 << 24;
                    unsigned int carry_9 = (word_9 & 2004318071) + 2004318071;
                    w_1_2[1] = word_9 & (carry_9 | 2004318071);
                    int cf_1_10 = f ^ 2;
                    unsigned int v4_1_10[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_10[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_2 + 16 * cf_1_10)));
                    float m_1_10 = m0_1_2;
                    if ((cf_1_10 & 2) != 0) {
                        m_1_10 = m1_1_2;
                    }
                    float2 mm_9 = make_float2(m_1_10, m_1_10);
                    float2 pair_12 = make_float2(__uint_as_float(v4_1_10[0] << 16), __uint_as_float(v4_1_10[0] & 4294901760u));
                    float2 g_12;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_12) : "l"(*(const unsigned long long*)&pair_12), "l"(*(const unsigned long long*)&mm_9));
                    uint32_t fp4_pair_11;
                    uint16_t _e2m1_pair_h_60;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_60) : "f"(g_12.x), "f"(g_12.y));
                    fp4_pair_11 = (uint32_t)_e2m1_pair_h_60;
                    float2 pair_1_9 = make_float2(__uint_as_float(v4_1_10[1] << 16), __uint_as_float(v4_1_10[1] & 4294901760u));
                    float2 g_1_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_9) : "l"(*(const unsigned long long*)&pair_1_9), "l"(*(const unsigned long long*)&mm_9));
                    uint32_t fp4_pair_1_9;
                    uint16_t _e2m1_pair_h_61;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_61) : "f"(g_1_9.x), "f"(g_1_9.y));
                    fp4_pair_1_9 = (uint32_t)_e2m1_pair_h_61;
                    float2 pair_2_9 = make_float2(__uint_as_float(v4_1_10[2] << 16), __uint_as_float(v4_1_10[2] & 4294901760u));
                    float2 g_2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_9) : "l"(*(const unsigned long long*)&pair_2_9), "l"(*(const unsigned long long*)&mm_9));
                    uint32_t fp4_pair_2_9;
                    uint16_t _e2m1_pair_h_62;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_62) : "f"(g_2_9.x), "f"(g_2_9.y));
                    fp4_pair_2_9 = (uint32_t)_e2m1_pair_h_62;
                    float2 pair_3_9 = make_float2(__uint_as_float(v4_1_10[3] << 16), __uint_as_float(v4_1_10[3] & 4294901760u));
                    float2 g_3_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_9) : "l"(*(const unsigned long long*)&pair_3_9), "l"(*(const unsigned long long*)&mm_9));
                    uint32_t _fp4_e2m1_pack_pair_63;
                    uint16_t _e2m1_pair_h_64;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_64) : "f"(g_3_9.x), "f"(g_3_9.y));
                    _fp4_e2m1_pack_pair_63 = (uint32_t)_e2m1_pair_h_64;
                    unsigned int word_10 = fp4_pair_11 | fp4_pair_1_9 << 8 | fp4_pair_2_9 << 16 | _fp4_e2m1_pack_pair_63 << 24;
                    unsigned int carry_10 = (word_10 & 2004318071) + 2004318071;
                    w_1_2[2] = word_10 & (carry_10 | 2004318071);
                    int cf_1_11 = f ^ 3;
                    unsigned int v4_1_11[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_11[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_2 + 16 * cf_1_11)));
                    float m_1_11 = m0_1_2;
                    if ((cf_1_11 & 2) != 0) {
                        m_1_11 = m1_1_2;
                    }
                    float2 mm_10 = make_float2(m_1_11, m_1_11);
                    float2 pair_13 = make_float2(__uint_as_float(v4_1_11[0] << 16), __uint_as_float(v4_1_11[0] & 4294901760u));
                    float2 g_13;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_13) : "l"(*(const unsigned long long*)&pair_13), "l"(*(const unsigned long long*)&mm_10));
                    uint32_t fp4_pair_12;
                    uint16_t _e2m1_pair_h_65;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_65) : "f"(g_13.x), "f"(g_13.y));
                    fp4_pair_12 = (uint32_t)_e2m1_pair_h_65;
                    float2 pair_1_10 = make_float2(__uint_as_float(v4_1_11[1] << 16), __uint_as_float(v4_1_11[1] & 4294901760u));
                    float2 g_1_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_10) : "l"(*(const unsigned long long*)&pair_1_10), "l"(*(const unsigned long long*)&mm_10));
                    uint32_t fp4_pair_1_10;
                    uint16_t _e2m1_pair_h_66;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_66) : "f"(g_1_10.x), "f"(g_1_10.y));
                    fp4_pair_1_10 = (uint32_t)_e2m1_pair_h_66;
                    float2 pair_2_10 = make_float2(__uint_as_float(v4_1_11[2] << 16), __uint_as_float(v4_1_11[2] & 4294901760u));
                    float2 g_2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_10) : "l"(*(const unsigned long long*)&pair_2_10), "l"(*(const unsigned long long*)&mm_10));
                    uint32_t fp4_pair_2_10;
                    uint16_t _e2m1_pair_h_67;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_67) : "f"(g_2_10.x), "f"(g_2_10.y));
                    fp4_pair_2_10 = (uint32_t)_e2m1_pair_h_67;
                    float2 pair_3_10 = make_float2(__uint_as_float(v4_1_11[3] << 16), __uint_as_float(v4_1_11[3] & 4294901760u));
                    float2 g_3_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_10) : "l"(*(const unsigned long long*)&pair_3_10), "l"(*(const unsigned long long*)&mm_10));
                    uint32_t _fp4_e2m1_pack_pair_68;
                    uint16_t _e2m1_pair_h_69;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_69) : "f"(g_3_10.x), "f"(g_3_10.y));
                    _fp4_e2m1_pack_pair_68 = (uint32_t)_e2m1_pair_h_69;
                    unsigned int word_11 = fp4_pair_12 | fp4_pair_1_10 << 8 | fp4_pair_2_10 << 16 | _fp4_e2m1_pack_pair_68 << 24;
                    unsigned int carry_11 = (word_11 & 2004318071) + 2004318071;
                    w_1_2[3] = word_11 & (carry_11 | 2004318071);
                    bool fodd_1_2 = (f & 1) != 0;
                    bool fhi_1_2 = (f & 2) != 0;
                    unsigned int t0_1_2 = w_1_2[0];
                    unsigned int t1_1_2 = w_1_2[1];
                    unsigned int t2_1_2 = w_1_2[2];
                    unsigned int t3_1_2 = w_1_2[3];
                    if (fodd_1_2) {
                        t0_1_2 = w_1_2[1];
                        t1_1_2 = w_1_2[0];
                        t2_1_2 = w_1_2[3];
                        t3_1_2 = w_1_2[2];
                    }
                    unsigned int q0_1_2 = t0_1_2;
                    unsigned int q1_1_2 = t1_1_2;
                    unsigned int q2_1_2 = t2_1_2;
                    unsigned int q3_1_2 = t3_1_2;
                    if (fhi_1_2) {
                        q0_1_2 = t2_1_2;
                        q1_1_2 = t3_1_2;
                        q2_1_2 = t0_1_2;
                        q3_1_2 = t1_1_2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_1_2)), "r"(q0_1_2), "r"(q1_1_2), "r"(q2_1_2), "r"(q3_1_2) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_2)), "h"((uint16_t)(_e4m3x2_f32_2 & 32639)) : "memory");
                    rs = rs_20;
                    if (h == 0 && lane_1 < 4) {
                        {
                            uint32_t _addr_70 = static_cast<uint32_t>(wsm_addr + (unsigned int)(ob_off + 3072 + 384 + 4 * lane_1));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_70), "f"(rs) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                    if (h == 0 && lane_1 == 0) {
                        if (n > k + 1) {
                            long long next_row = (long long)(r0 + t + 5 * (k + 1));
                            mbarrier_arrive_expect_tx(xb_full_addr + (leader) * 8, 12288);
                            cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)xb_off, x + (next_row * 6144), 12288, xb_full_addr + (leader) * 8);
                        }
                        long long row64 = (long long)(r0 + t + 5 * k);
                        {
                            void* _cpbulk_dst_71 = reinterpret_cast<void*>(peer_wire + (v_base + (rank64 * cap64 + row64) * 3072));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_71), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_72 = reinterpret_cast<void*>(peer_wire + (s_base + (rank64 * cap64 + row64) * 384));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_72), "r"(wsm_addr + (unsigned int)(ob_off + 3072)), "r"((uint32_t)(384))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_73 = reinterpret_cast<void*>(peer_rs + (m_base + (rank64 * cap64 + row64) * 16));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_73), "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 384)), "r"((uint32_t)(16))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                _phase_xb_full ^= 1;
                int kd = k - 1;
                if (kd >= 0 && kd < n) {
                    long long drow = (long long)(r0 + t + 5 * kd);
                    int dslot = kd % 2;
                    int dob_off = xb_off + 12288 + dslot * 3472;
                    long long src64 = peer64;
                    long long v_row = v_base + (src64 * cap64 + drow) * 3072;
                    long long s_row = s_base + (src64 * cap64 + drow) * 384;
                    long long m_row = m_base + (src64 * cap64 + drow) * 16;
                    float rs_own = wsm32[dob_off + 3072 + 384 >> 2];
                    unsigned int own[12];
                    unsigned int own_sc[3];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[0])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * (lane_1 + 32 * (3 * h)))));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * (lane_1 + 32 * (3 * h)))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[4])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * (lane_1 + 32 * (3 * h + 1)))));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc[1])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * (lane_1 + 32 * (3 * h + 1)))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[8])), "=r"(*reinterpret_cast<uint32_t*>(&own[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * (lane_1 + 32 * (3 * h + 2)))));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc[2])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * (lane_1 + 32 * (3 * h + 2)))));
                    unsigned int rsw[1];
                    uint32_t _sysv_u32_0;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_0) : "l"(reinterpret_cast<unsigned int*>(os_rs + m_row)) : "memory");
                    rsw[0] = _sysv_u32_0;
                    uint32_t _fp4_wire_values_0[12];
                    uint32_t _fp4_wire_scales_0[3];
                    uint16_t _wire_h16_74_0;
                    uint16_t _wire_h16_74_1;
                    uint16_t _wire_h16_74_2;
                    uint32_t _wire_vpend_74 = 7u;
                    uint32_t _wire_spend_74 = 7u;
                    do {
                        if (_wire_vpend_74 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * (lane_1 + 32 * (3 * h)))))) : "memory");
                        }
                        if (_wire_vpend_74 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * (lane_1 + 32 * (3 * h + 1)))))) : "memory");
                        }
                        if (_wire_vpend_74 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[8]), "=r"(_fp4_wire_values_0[9]), "=r"(_fp4_wire_values_0[10]), "=r"(_fp4_wire_values_0[11]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * (lane_1 + 32 * (3 * h + 2)))))) : "memory");
                        }
                        if (_wire_spend_74 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_74_0) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * (lane_1 + 32 * (3 * h)))))) : "memory");
                            _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_74_0;
                        }
                        if (_wire_spend_74 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_74_1) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * (lane_1 + 32 * (3 * h + 1)))))) : "memory");
                            _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_74_1;
                        }
                        if (_wire_spend_74 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_74_2) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * (lane_1 + 32 * (3 * h + 2)))))) : "memory");
                            _fp4_wire_scales_0[2] = (uint32_t)_wire_h16_74_2;
                        }
                        _wire_vpend_74 = ((uint32_t)((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((_fp4_wire_values_0[8] & ~((_fp4_wire_values_0[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[9] & ~((_fp4_wire_values_0[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[10] & ~((_fp4_wire_values_0[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[11] & ~((_fp4_wire_values_0[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2);
                        _wire_spend_74 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[2] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2);
                    } while ((_wire_vpend_74 | _wire_spend_74) != 0u);
                    int pending = __any_sync(0xFFFFFFFF, rsw[0] == 2147483648u);
                    if (pending != 0) {
                        uint32_t _sysv_poll32_group_0[1];
                        do {
                            asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[0]) : "l"(reinterpret_cast<unsigned int*>(os_rs + m_row)) : "memory");
                        } while (
                            (_sysv_poll32_group_0[0] == 0x80000000u)
                        );
                        rsw[0] = _sysv_poll32_group_0[0];
                    }
                    float rs_peer = __uint_as_float(rsw[0]);
                    bool first = rank == 0;
                    unsigned int a_1[4];
                    unsigned int b_1[4];
                    a_1[0] = ((first) ? own[0] : _fp4_wire_values_0[0]);
                    b_1[0] = ((first) ? _fp4_wire_values_0[0] : own[0]);
                    a_1[1] = ((first) ? own[1] : _fp4_wire_values_0[1]);
                    b_1[1] = ((first) ? _fp4_wire_values_0[1] : own[1]);
                    a_1[2] = ((first) ? own[2] : _fp4_wire_values_0[2]);
                    b_1[2] = ((first) ? _fp4_wire_values_0[2] : own[2]);
                    a_1[3] = ((first) ? own[3] : _fp4_wire_values_0[3]);
                    b_1[3] = ((first) ? _fp4_wire_values_0[3] : own[3]);
                    unsigned int a_sc = ((first) ? own_sc[0] : _fp4_wire_scales_0[0]);
                    unsigned int b_sc = ((first) ? _fp4_wire_scales_0[0] : own_sc[0]);
                    float a_rs = ((first) ? rs_own : rs_peer);
                    float b_rs = ((first) ? rs_peer : rs_own);
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
                    uint32_t e4m3x2_to_f16x2_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_2) : "h"((uint16_t)(a_sc)));
                    float cvt_f32_f16_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_3) : "h"((uint16_t)(e4m3x2_to_f16x2_2)));
                    float cvt_f32_f16_1_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_2) : "h"((uint16_t)(e4m3x2_to_f16x2_2 >> 16)));
                    float _mul_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(cvt_f32_f16_3), "f"(a_rs));
                    float _mul_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(cvt_f32_f16_1_2), "f"(a_rs));
                    float2 scale0 = make_float2(_mul_0, _mul_0);
                    float2 scale1 = make_float2(_mul_1, _mul_1);
                    float _unpack_fp4_0[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_1[0]));
                        (&_unpack_fp4_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_0[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_0[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_0[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_0[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_0[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_0[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_0[0]), "=f"(_unpack_fp4_0[1]), "=f"(_unpack_fp4_0[2]), "=f"(_unpack_fp4_0[3]), "=f"(_unpack_fp4_0[4]), "=f"(_unpack_fp4_0[5]), "=f"(_unpack_fp4_0[6]), "=f"(_unpack_fp4_0[7]) : "r"(a_1[0]));
                        #endif
                    }
                    float2 a_0 = make_float2(acc[0], acc[1]);
                    float2 q_0 = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                    float2 r_0 = fma_f32x2_rn_noftz(q_0, scale0, a_0);
                    acc[0] = r_0.x;
                    acc[1] = r_0.y;
                    float2 a_1_1 = make_float2(acc[2], acc[3]);
                    float2 q_1 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                    float2 r_1 = fma_f32x2_rn_noftz(q_1, scale0, a_1_1);
                    acc[2] = r_1.x;
                    acc[3] = r_1.y;
                    float2 a_3 = make_float2(acc[4], acc[5]);
                    float2 q_2 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                    float2 r_2 = fma_f32x2_rn_noftz(q_2, scale0, a_3);
                    acc[4] = r_2.x;
                    acc[5] = r_2.y;
                    float2 a_4 = make_float2(acc[6], acc[7]);
                    float2 q_3 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                    float2 r_3 = fma_f32x2_rn_noftz(q_3, scale0, a_4);
                    acc[6] = r_3.x;
                    acc[7] = r_3.y;
                    float _unpack_fp4_1[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_1[1 + 0]));
                        (&_unpack_fp4_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_1[0]), "=f"(_unpack_fp4_1[1]), "=f"(_unpack_fp4_1[2]), "=f"(_unpack_fp4_1[3]), "=f"(_unpack_fp4_1[4]), "=f"(_unpack_fp4_1[5]), "=f"(_unpack_fp4_1[6]), "=f"(_unpack_fp4_1[7]) : "r"(a_1[1 + 0]));
                        #endif
                    }
                    float2 a_5 = make_float2(acc[8], acc[9]);
                    float2 q_4 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                    float2 r_4 = fma_f32x2_rn_noftz(q_4, scale0, a_5);
                    acc[8] = r_4.x;
                    acc[9] = r_4.y;
                    float2 a_6 = make_float2(acc[10], acc[11]);
                    float2 q_5 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                    float2 r_5 = fma_f32x2_rn_noftz(q_5, scale0, a_6);
                    acc[10] = r_5.x;
                    acc[11] = r_5.y;
                    float2 a_7 = make_float2(acc[12], acc[13]);
                    float2 q_6 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                    float2 r_6 = fma_f32x2_rn_noftz(q_6, scale0, a_7);
                    acc[12] = r_6.x;
                    acc[13] = r_6.y;
                    float2 a_8 = make_float2(acc[14], acc[15]);
                    float2 q_7 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                    float2 r_7 = fma_f32x2_rn_noftz(q_7, scale0, a_8);
                    acc[14] = r_7.x;
                    acc[15] = r_7.y;
                    float _unpack_fp4_2[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_1[2 + 0]));
                        (&_unpack_fp4_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_2[0]), "=f"(_unpack_fp4_2[1]), "=f"(_unpack_fp4_2[2]), "=f"(_unpack_fp4_2[3]), "=f"(_unpack_fp4_2[4]), "=f"(_unpack_fp4_2[5]), "=f"(_unpack_fp4_2[6]), "=f"(_unpack_fp4_2[7]) : "r"(a_1[2 + 0]));
                        #endif
                    }
                    float2 a_9 = make_float2(acc[16], acc[17]);
                    float2 q_8 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                    float2 r_8 = fma_f32x2_rn_noftz(q_8, scale1, a_9);
                    acc[16] = r_8.x;
                    acc[17] = r_8.y;
                    float2 a_10 = make_float2(acc[18], acc[19]);
                    float2 q_9 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                    float2 r_9 = fma_f32x2_rn_noftz(q_9, scale1, a_10);
                    acc[18] = r_9.x;
                    acc[19] = r_9.y;
                    float2 a_11 = make_float2(acc[20], acc[21]);
                    float2 q_10 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                    float2 r_10 = fma_f32x2_rn_noftz(q_10, scale1, a_11);
                    acc[20] = r_10.x;
                    acc[21] = r_10.y;
                    float2 a_13 = make_float2(acc[22], acc[23]);
                    float2 q_11 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                    float2 r_11 = fma_f32x2_rn_noftz(q_11, scale1, a_13);
                    acc[22] = r_11.x;
                    acc[23] = r_11.y;
                    float _unpack_fp4_3[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_1[3 + 0]));
                        (&_unpack_fp4_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_3[0]), "=f"(_unpack_fp4_3[1]), "=f"(_unpack_fp4_3[2]), "=f"(_unpack_fp4_3[3]), "=f"(_unpack_fp4_3[4]), "=f"(_unpack_fp4_3[5]), "=f"(_unpack_fp4_3[6]), "=f"(_unpack_fp4_3[7]) : "r"(a_1[3 + 0]));
                        #endif
                    }
                    float2 a_14 = make_float2(acc[24], acc[25]);
                    float2 q_12 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                    float2 r_12 = fma_f32x2_rn_noftz(q_12, scale1, a_14);
                    acc[24] = r_12.x;
                    acc[25] = r_12.y;
                    float2 a_15 = make_float2(acc[26], acc[27]);
                    float2 q_13 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                    float2 r_13 = fma_f32x2_rn_noftz(q_13, scale1, a_15);
                    acc[26] = r_13.x;
                    acc[27] = r_13.y;
                    float2 a_16 = make_float2(acc[28], acc[29]);
                    float2 q_14 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                    float2 r_14 = fma_f32x2_rn_noftz(q_14, scale1, a_16);
                    acc[28] = r_14.x;
                    acc[29] = r_14.y;
                    float2 a_17 = make_float2(acc[30], acc[31]);
                    float2 q_15 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                    float2 r_15 = fma_f32x2_rn_noftz(q_15, scale1, a_17);
                    acc[30] = r_15.x;
                    acc[31] = r_15.y;
                    uint32_t e4m3x2_to_f16x2_2_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_2_1) : "h"((uint16_t)(b_sc)));
                    float cvt_f32_f16_3_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_3_1) : "h"((uint16_t)(e4m3x2_to_f16x2_2_1)));
                    float cvt_f32_f16_1_2_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_2_1) : "h"((uint16_t)(e4m3x2_to_f16x2_2_1 >> 16)));
                    float _mul_0_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0_0) : "f"(cvt_f32_f16_3_1), "f"(b_rs));
                    float _mul_1_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1_0) : "f"(cvt_f32_f16_1_2_1), "f"(b_rs));
                    float2 scale0_0 = make_float2(_mul_0_0, _mul_0_0);
                    float2 scale1_0 = make_float2(_mul_1_0, _mul_1_0);
                    float _unpack_fp4_4[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_1[0]));
                        (&_unpack_fp4_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_4[0]), "=f"(_unpack_fp4_4[1]), "=f"(_unpack_fp4_4[2]), "=f"(_unpack_fp4_4[3]), "=f"(_unpack_fp4_4[4]), "=f"(_unpack_fp4_4[5]), "=f"(_unpack_fp4_4[6]), "=f"(_unpack_fp4_4[7]) : "r"(b_1[0]));
                        #endif
                    }
                    float2 a_18 = make_float2(acc[0], acc[1]);
                    float2 q_16 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                    float2 r_16 = fma_f32x2_rn_noftz(q_16, scale0_0, a_18);
                    acc[0] = r_16.x;
                    acc[1] = r_16.y;
                    float2 a_19 = make_float2(acc[2], acc[3]);
                    float2 q_17 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                    float2 r_17 = fma_f32x2_rn_noftz(q_17, scale0_0, a_19);
                    acc[2] = r_17.x;
                    acc[3] = r_17.y;
                    float2 a_20 = make_float2(acc[4], acc[5]);
                    float2 q_18 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                    float2 r_18 = fma_f32x2_rn_noftz(q_18, scale0_0, a_20);
                    acc[4] = r_18.x;
                    acc[5] = r_18.y;
                    float2 a_21 = make_float2(acc[6], acc[7]);
                    float2 q_19 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                    float2 r_19 = fma_f32x2_rn_noftz(q_19, scale0_0, a_21);
                    acc[6] = r_19.x;
                    acc[7] = r_19.y;
                    float _unpack_fp4_5[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_1[1 + 0]));
                        (&_unpack_fp4_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_5[0]), "=f"(_unpack_fp4_5[1]), "=f"(_unpack_fp4_5[2]), "=f"(_unpack_fp4_5[3]), "=f"(_unpack_fp4_5[4]), "=f"(_unpack_fp4_5[5]), "=f"(_unpack_fp4_5[6]), "=f"(_unpack_fp4_5[7]) : "r"(b_1[1 + 0]));
                        #endif
                    }
                    float2 a_22 = make_float2(acc[8], acc[9]);
                    float2 q_20 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                    float2 r_20 = fma_f32x2_rn_noftz(q_20, scale0_0, a_22);
                    acc[8] = r_20.x;
                    acc[9] = r_20.y;
                    float2 a_23 = make_float2(acc[10], acc[11]);
                    float2 q_21 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                    float2 r_21 = fma_f32x2_rn_noftz(q_21, scale0_0, a_23);
                    acc[10] = r_21.x;
                    acc[11] = r_21.y;
                    float2 a_24 = make_float2(acc[12], acc[13]);
                    float2 q_22 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                    float2 r_22 = fma_f32x2_rn_noftz(q_22, scale0_0, a_24);
                    acc[12] = r_22.x;
                    acc[13] = r_22.y;
                    float2 a_25 = make_float2(acc[14], acc[15]);
                    float2 q_23 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                    float2 r_23 = fma_f32x2_rn_noftz(q_23, scale0_0, a_25);
                    acc[14] = r_23.x;
                    acc[15] = r_23.y;
                    float _unpack_fp4_6[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_1[2 + 0]));
                        (&_unpack_fp4_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_6[0]), "=f"(_unpack_fp4_6[1]), "=f"(_unpack_fp4_6[2]), "=f"(_unpack_fp4_6[3]), "=f"(_unpack_fp4_6[4]), "=f"(_unpack_fp4_6[5]), "=f"(_unpack_fp4_6[6]), "=f"(_unpack_fp4_6[7]) : "r"(b_1[2 + 0]));
                        #endif
                    }
                    float2 a_26 = make_float2(acc[16], acc[17]);
                    float2 q_24 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                    float2 r_24 = fma_f32x2_rn_noftz(q_24, scale1_0, a_26);
                    acc[16] = r_24.x;
                    acc[17] = r_24.y;
                    float2 a_27 = make_float2(acc[18], acc[19]);
                    float2 q_25 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                    float2 r_25 = fma_f32x2_rn_noftz(q_25, scale1_0, a_27);
                    acc[18] = r_25.x;
                    acc[19] = r_25.y;
                    float2 a_28 = make_float2(acc[20], acc[21]);
                    float2 q_26 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                    float2 r_26 = fma_f32x2_rn_noftz(q_26, scale1_0, a_28);
                    acc[20] = r_26.x;
                    acc[21] = r_26.y;
                    float2 a_29 = make_float2(acc[22], acc[23]);
                    float2 q_27 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                    float2 r_27 = fma_f32x2_rn_noftz(q_27, scale1_0, a_29);
                    acc[22] = r_27.x;
                    acc[23] = r_27.y;
                    float _unpack_fp4_7[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_1[3 + 0]));
                        (&_unpack_fp4_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_7[0]), "=f"(_unpack_fp4_7[1]), "=f"(_unpack_fp4_7[2]), "=f"(_unpack_fp4_7[3]), "=f"(_unpack_fp4_7[4]), "=f"(_unpack_fp4_7[5]), "=f"(_unpack_fp4_7[6]), "=f"(_unpack_fp4_7[7]) : "r"(b_1[3 + 0]));
                        #endif
                    }
                    float2 a_30 = make_float2(acc[24], acc[25]);
                    float2 q_28 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                    float2 r_28 = fma_f32x2_rn_noftz(q_28, scale1_0, a_30);
                    acc[24] = r_28.x;
                    acc[25] = r_28.y;
                    float2 a_31 = make_float2(acc[26], acc[27]);
                    float2 q_29 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                    float2 r_29 = fma_f32x2_rn_noftz(q_29, scale1_0, a_31);
                    acc[26] = r_29.x;
                    acc[27] = r_29.y;
                    float2 a_32 = make_float2(acc[28], acc[29]);
                    float2 q_30 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                    float2 r_30 = fma_f32x2_rn_noftz(q_30, scale1_0, a_32);
                    acc[28] = r_30.x;
                    acc[29] = r_30.y;
                    float2 a_33 = make_float2(acc[30], acc[31]);
                    float2 q_31 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                    float2 r_31 = fma_f32x2_rn_noftz(q_31, scale1_0, a_33);
                    acc[30] = r_31.x;
                    acc[31] = r_31.y;
                    unsigned int acc_bf16[16];
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
                    int c_1_0 = lane_1;
                    int l_1_0 = c_1_0 >> 2;
                    unsigned int w4_1_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_0[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1_0 + (c_1_0 & 3 ^ l_1_0 >> 1 & 3)))));
                    long long idx_1_0 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_1_0));
                    {
                        int4 _iv4 = make_int4(w4_1_0[0 + 0], w4_1_0[0 + 1], w4_1_0[0 + 2], w4_1_0[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1_0) = _iv4;
                    }
                    int c_1_1 = lane_1 + 32;
                    int l_1_1 = c_1_1 >> 2;
                    unsigned int w4_1_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1_1 + (c_1_1 & 3 ^ l_1_1 >> 1 & 3)))));
                    long long idx_1_1 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_1_1));
                    {
                        int4 _iv4 = make_int4(w4_1_1[0 + 0], w4_1_1[0 + 1], w4_1_1[0 + 2], w4_1_1[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1_1) = _iv4;
                    }
                    int c_1_2 = lane_1 + 64;
                    int l_1_2 = c_1_2 >> 2;
                    unsigned int w4_1_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_2[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1_2 + (c_1_2 & 3 ^ l_1_2 >> 1 & 3)))));
                    long long idx_1_2 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_1_2));
                    {
                        int4 _iv4 = make_int4(w4_1_2[0 + 0], w4_1_2[0 + 1], w4_1_2[0 + 2], w4_1_2[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1_2) = _iv4;
                    }
                    int c_1_3 = lane_1 + 96;
                    int l_1_3 = c_1_3 >> 2;
                    unsigned int w4_1_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_3[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1_3 + (c_1_3 & 3 ^ l_1_3 >> 1 & 3)))));
                    long long idx_1_3 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_1_3));
                    {
                        int4 _iv4 = make_int4(w4_1_3[0 + 0], w4_1_3[0 + 1], w4_1_3[0 + 2], w4_1_3[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1_3) = _iv4;
                    }
                    __syncwarp();
                    unsigned int a_108[4];
                    unsigned int b_109[4];
                    a_108[0] = ((first) ? own[4] : _fp4_wire_values_0[4]);
                    b_109[0] = ((first) ? _fp4_wire_values_0[4] : own[4]);
                    a_108[1] = ((first) ? own[5] : _fp4_wire_values_0[5]);
                    b_109[1] = ((first) ? _fp4_wire_values_0[5] : own[5]);
                    a_108[2] = ((first) ? own[6] : _fp4_wire_values_0[6]);
                    b_109[2] = ((first) ? _fp4_wire_values_0[6] : own[6]);
                    a_108[3] = ((first) ? own[7] : _fp4_wire_values_0[7]);
                    b_109[3] = ((first) ? _fp4_wire_values_0[7] : own[7]);
                    unsigned int a_sc_110 = ((first) ? own_sc[1] : _fp4_wire_scales_0[1]);
                    unsigned int b_sc_111 = ((first) ? _fp4_wire_scales_0[1] : own_sc[1]);
                    float a_rs_112 = ((first) ? rs_own : rs_peer);
                    float b_rs_113 = ((first) ? rs_peer : rs_own);
                    float acc_114[32];
                    acc_114[0] = 0.0f;
                    acc_114[1] = 0.0f;
                    acc_114[2] = 0.0f;
                    acc_114[3] = 0.0f;
                    acc_114[4] = 0.0f;
                    acc_114[5] = 0.0f;
                    acc_114[6] = 0.0f;
                    acc_114[7] = 0.0f;
                    acc_114[8] = 0.0f;
                    acc_114[9] = 0.0f;
                    acc_114[10] = 0.0f;
                    acc_114[11] = 0.0f;
                    acc_114[12] = 0.0f;
                    acc_114[13] = 0.0f;
                    acc_114[14] = 0.0f;
                    acc_114[15] = 0.0f;
                    acc_114[16] = 0.0f;
                    acc_114[17] = 0.0f;
                    acc_114[18] = 0.0f;
                    acc_114[19] = 0.0f;
                    acc_114[20] = 0.0f;
                    acc_114[21] = 0.0f;
                    acc_114[22] = 0.0f;
                    acc_114[23] = 0.0f;
                    acc_114[24] = 0.0f;
                    acc_114[25] = 0.0f;
                    acc_114[26] = 0.0f;
                    acc_114[27] = 0.0f;
                    acc_114[28] = 0.0f;
                    acc_114[29] = 0.0f;
                    acc_114[30] = 0.0f;
                    acc_114[31] = 0.0f;
                    uint32_t e4m3x2_to_f16x2_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_3) : "h"((uint16_t)(a_sc_110)));
                    float cvt_f32_f16_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_4) : "h"((uint16_t)(e4m3x2_to_f16x2_3)));
                    float cvt_f32_f16_1_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_3) : "h"((uint16_t)(e4m3x2_to_f16x2_3 >> 16)));
                    float _mul_0_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0_1) : "f"(cvt_f32_f16_4), "f"(a_rs_112));
                    float _mul_1_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1_1) : "f"(cvt_f32_f16_1_3), "f"(a_rs_112));
                    float2 scale0_1 = make_float2(_mul_0_1, _mul_0_1);
                    float2 scale1_1 = make_float2(_mul_1_1, _mul_1_1);
                    float _unpack_fp4_8[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_108[0]));
                        (&_unpack_fp4_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_8[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_8[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_8[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_8[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_8[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_8[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_8[0]), "=f"(_unpack_fp4_8[1]), "=f"(_unpack_fp4_8[2]), "=f"(_unpack_fp4_8[3]), "=f"(_unpack_fp4_8[4]), "=f"(_unpack_fp4_8[5]), "=f"(_unpack_fp4_8[6]), "=f"(_unpack_fp4_8[7]) : "r"(a_108[0]));
                        #endif
                    }
                    float2 a_34 = make_float2(acc_114[0], acc_114[1]);
                    float2 q_32 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                    float2 r_32 = fma_f32x2_rn_noftz(q_32, scale0_1, a_34);
                    acc_114[0] = r_32.x;
                    acc_114[1] = r_32.y;
                    float2 a_35 = make_float2(acc_114[2], acc_114[3]);
                    float2 q_33 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                    float2 r_33 = fma_f32x2_rn_noftz(q_33, scale0_1, a_35);
                    acc_114[2] = r_33.x;
                    acc_114[3] = r_33.y;
                    float2 a_36 = make_float2(acc_114[4], acc_114[5]);
                    float2 q_34 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                    float2 r_34 = fma_f32x2_rn_noftz(q_34, scale0_1, a_36);
                    acc_114[4] = r_34.x;
                    acc_114[5] = r_34.y;
                    float2 a_37 = make_float2(acc_114[6], acc_114[7]);
                    float2 q_35 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                    float2 r_35 = fma_f32x2_rn_noftz(q_35, scale0_1, a_37);
                    acc_114[6] = r_35.x;
                    acc_114[7] = r_35.y;
                    float _unpack_fp4_9[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_108[1 + 0]));
                        (&_unpack_fp4_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_9[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_9[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_9[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_9[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_9[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_9[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_9[0]), "=f"(_unpack_fp4_9[1]), "=f"(_unpack_fp4_9[2]), "=f"(_unpack_fp4_9[3]), "=f"(_unpack_fp4_9[4]), "=f"(_unpack_fp4_9[5]), "=f"(_unpack_fp4_9[6]), "=f"(_unpack_fp4_9[7]) : "r"(a_108[1 + 0]));
                        #endif
                    }
                    float2 a_38 = make_float2(acc_114[8], acc_114[9]);
                    float2 q_36 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                    float2 r_36 = fma_f32x2_rn_noftz(q_36, scale0_1, a_38);
                    acc_114[8] = r_36.x;
                    acc_114[9] = r_36.y;
                    float2 a_39 = make_float2(acc_114[10], acc_114[11]);
                    float2 q_37 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                    float2 r_37 = fma_f32x2_rn_noftz(q_37, scale0_1, a_39);
                    acc_114[10] = r_37.x;
                    acc_114[11] = r_37.y;
                    float2 a_40 = make_float2(acc_114[12], acc_114[13]);
                    float2 q_38 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                    float2 r_38 = fma_f32x2_rn_noftz(q_38, scale0_1, a_40);
                    acc_114[12] = r_38.x;
                    acc_114[13] = r_38.y;
                    float2 a_41 = make_float2(acc_114[14], acc_114[15]);
                    float2 q_39 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                    float2 r_39 = fma_f32x2_rn_noftz(q_39, scale0_1, a_41);
                    acc_114[14] = r_39.x;
                    acc_114[15] = r_39.y;
                    float _unpack_fp4_10[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_108[2 + 0]));
                        (&_unpack_fp4_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_10[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_10[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_10[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_10[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_10[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_10[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_10[0]), "=f"(_unpack_fp4_10[1]), "=f"(_unpack_fp4_10[2]), "=f"(_unpack_fp4_10[3]), "=f"(_unpack_fp4_10[4]), "=f"(_unpack_fp4_10[5]), "=f"(_unpack_fp4_10[6]), "=f"(_unpack_fp4_10[7]) : "r"(a_108[2 + 0]));
                        #endif
                    }
                    float2 a_42 = make_float2(acc_114[16], acc_114[17]);
                    float2 q_40 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                    float2 r_40 = fma_f32x2_rn_noftz(q_40, scale1_1, a_42);
                    acc_114[16] = r_40.x;
                    acc_114[17] = r_40.y;
                    float2 a_43 = make_float2(acc_114[18], acc_114[19]);
                    float2 q_41 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                    float2 r_41 = fma_f32x2_rn_noftz(q_41, scale1_1, a_43);
                    acc_114[18] = r_41.x;
                    acc_114[19] = r_41.y;
                    float2 a_44 = make_float2(acc_114[20], acc_114[21]);
                    float2 q_42 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                    float2 r_42 = fma_f32x2_rn_noftz(q_42, scale1_1, a_44);
                    acc_114[20] = r_42.x;
                    acc_114[21] = r_42.y;
                    float2 a_45 = make_float2(acc_114[22], acc_114[23]);
                    float2 q_43 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                    float2 r_43 = fma_f32x2_rn_noftz(q_43, scale1_1, a_45);
                    acc_114[22] = r_43.x;
                    acc_114[23] = r_43.y;
                    float _unpack_fp4_11[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_108[3 + 0]));
                        (&_unpack_fp4_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_11[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_11[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_11[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_11[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_11[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_11[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_11[0]), "=f"(_unpack_fp4_11[1]), "=f"(_unpack_fp4_11[2]), "=f"(_unpack_fp4_11[3]), "=f"(_unpack_fp4_11[4]), "=f"(_unpack_fp4_11[5]), "=f"(_unpack_fp4_11[6]), "=f"(_unpack_fp4_11[7]) : "r"(a_108[3 + 0]));
                        #endif
                    }
                    float2 a_46 = make_float2(acc_114[24], acc_114[25]);
                    float2 q_44 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                    float2 r_44 = fma_f32x2_rn_noftz(q_44, scale1_1, a_46);
                    acc_114[24] = r_44.x;
                    acc_114[25] = r_44.y;
                    float2 a_47 = make_float2(acc_114[26], acc_114[27]);
                    float2 q_45 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                    float2 r_45 = fma_f32x2_rn_noftz(q_45, scale1_1, a_47);
                    acc_114[26] = r_45.x;
                    acc_114[27] = r_45.y;
                    float2 a_48 = make_float2(acc_114[28], acc_114[29]);
                    float2 q_46 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                    float2 r_46 = fma_f32x2_rn_noftz(q_46, scale1_1, a_48);
                    acc_114[28] = r_46.x;
                    acc_114[29] = r_46.y;
                    float2 a_49 = make_float2(acc_114[30], acc_114[31]);
                    float2 q_47 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                    float2 r_47 = fma_f32x2_rn_noftz(q_47, scale1_1, a_49);
                    acc_114[30] = r_47.x;
                    acc_114[31] = r_47.y;
                    uint32_t e4m3x2_to_f16x2_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_4) : "h"((uint16_t)(b_sc_111)));
                    float cvt_f32_f16_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_5) : "h"((uint16_t)(e4m3x2_to_f16x2_4)));
                    float cvt_f32_f16_1_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_4) : "h"((uint16_t)(e4m3x2_to_f16x2_4 >> 16)));
                    float _mul_0_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0_2) : "f"(cvt_f32_f16_5), "f"(b_rs_113));
                    float _mul_1_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1_2) : "f"(cvt_f32_f16_1_4), "f"(b_rs_113));
                    float2 scale0_2 = make_float2(_mul_0_2, _mul_0_2);
                    float2 scale1_2 = make_float2(_mul_1_2, _mul_1_2);
                    float _unpack_fp4_12[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_109[0]));
                        (&_unpack_fp4_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_12[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_12[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_12[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_12[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_12[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_12[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_12[0]), "=f"(_unpack_fp4_12[1]), "=f"(_unpack_fp4_12[2]), "=f"(_unpack_fp4_12[3]), "=f"(_unpack_fp4_12[4]), "=f"(_unpack_fp4_12[5]), "=f"(_unpack_fp4_12[6]), "=f"(_unpack_fp4_12[7]) : "r"(b_109[0]));
                        #endif
                    }
                    float2 a_50 = make_float2(acc_114[0], acc_114[1]);
                    float2 q_48 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                    float2 r_48 = fma_f32x2_rn_noftz(q_48, scale0_2, a_50);
                    acc_114[0] = r_48.x;
                    acc_114[1] = r_48.y;
                    float2 a_51 = make_float2(acc_114[2], acc_114[3]);
                    float2 q_49 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                    float2 r_49 = fma_f32x2_rn_noftz(q_49, scale0_2, a_51);
                    acc_114[2] = r_49.x;
                    acc_114[3] = r_49.y;
                    float2 a_52 = make_float2(acc_114[4], acc_114[5]);
                    float2 q_50 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                    float2 r_50 = fma_f32x2_rn_noftz(q_50, scale0_2, a_52);
                    acc_114[4] = r_50.x;
                    acc_114[5] = r_50.y;
                    float2 a_53 = make_float2(acc_114[6], acc_114[7]);
                    float2 q_51 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                    float2 r_51 = fma_f32x2_rn_noftz(q_51, scale0_2, a_53);
                    acc_114[6] = r_51.x;
                    acc_114[7] = r_51.y;
                    float _unpack_fp4_13[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_109[1 + 0]));
                        (&_unpack_fp4_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_13[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_13[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_13[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_13[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_13[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_13[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_13[0]), "=f"(_unpack_fp4_13[1]), "=f"(_unpack_fp4_13[2]), "=f"(_unpack_fp4_13[3]), "=f"(_unpack_fp4_13[4]), "=f"(_unpack_fp4_13[5]), "=f"(_unpack_fp4_13[6]), "=f"(_unpack_fp4_13[7]) : "r"(b_109[1 + 0]));
                        #endif
                    }
                    float2 a_54 = make_float2(acc_114[8], acc_114[9]);
                    float2 q_52 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                    float2 r_52 = fma_f32x2_rn_noftz(q_52, scale0_2, a_54);
                    acc_114[8] = r_52.x;
                    acc_114[9] = r_52.y;
                    float2 a_55 = make_float2(acc_114[10], acc_114[11]);
                    float2 q_53 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                    float2 r_53 = fma_f32x2_rn_noftz(q_53, scale0_2, a_55);
                    acc_114[10] = r_53.x;
                    acc_114[11] = r_53.y;
                    float2 a_56 = make_float2(acc_114[12], acc_114[13]);
                    float2 q_54 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                    float2 r_54 = fma_f32x2_rn_noftz(q_54, scale0_2, a_56);
                    acc_114[12] = r_54.x;
                    acc_114[13] = r_54.y;
                    float2 a_57 = make_float2(acc_114[14], acc_114[15]);
                    float2 q_55 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                    float2 r_55 = fma_f32x2_rn_noftz(q_55, scale0_2, a_57);
                    acc_114[14] = r_55.x;
                    acc_114[15] = r_55.y;
                    float _unpack_fp4_14[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_109[2 + 0]));
                        (&_unpack_fp4_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_14[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_14[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_14[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_14[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_14[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_14[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_14[0]), "=f"(_unpack_fp4_14[1]), "=f"(_unpack_fp4_14[2]), "=f"(_unpack_fp4_14[3]), "=f"(_unpack_fp4_14[4]), "=f"(_unpack_fp4_14[5]), "=f"(_unpack_fp4_14[6]), "=f"(_unpack_fp4_14[7]) : "r"(b_109[2 + 0]));
                        #endif
                    }
                    float2 a_58 = make_float2(acc_114[16], acc_114[17]);
                    float2 q_56 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                    float2 r_56 = fma_f32x2_rn_noftz(q_56, scale1_2, a_58);
                    acc_114[16] = r_56.x;
                    acc_114[17] = r_56.y;
                    float2 a_59 = make_float2(acc_114[18], acc_114[19]);
                    float2 q_57 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                    float2 r_57 = fma_f32x2_rn_noftz(q_57, scale1_2, a_59);
                    acc_114[18] = r_57.x;
                    acc_114[19] = r_57.y;
                    float2 a_60 = make_float2(acc_114[20], acc_114[21]);
                    float2 q_58 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                    float2 r_58 = fma_f32x2_rn_noftz(q_58, scale1_2, a_60);
                    acc_114[20] = r_58.x;
                    acc_114[21] = r_58.y;
                    float2 a_61 = make_float2(acc_114[22], acc_114[23]);
                    float2 q_59 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                    float2 r_59 = fma_f32x2_rn_noftz(q_59, scale1_2, a_61);
                    acc_114[22] = r_59.x;
                    acc_114[23] = r_59.y;
                    float _unpack_fp4_15[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_109[3 + 0]));
                        (&_unpack_fp4_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_15[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_15[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_15[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_15[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_15[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_15[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_15[0]), "=f"(_unpack_fp4_15[1]), "=f"(_unpack_fp4_15[2]), "=f"(_unpack_fp4_15[3]), "=f"(_unpack_fp4_15[4]), "=f"(_unpack_fp4_15[5]), "=f"(_unpack_fp4_15[6]), "=f"(_unpack_fp4_15[7]) : "r"(b_109[3 + 0]));
                        #endif
                    }
                    float2 a_62 = make_float2(acc_114[24], acc_114[25]);
                    float2 q_60 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                    float2 r_60 = fma_f32x2_rn_noftz(q_60, scale1_2, a_62);
                    acc_114[24] = r_60.x;
                    acc_114[25] = r_60.y;
                    float2 a_63 = make_float2(acc_114[26], acc_114[27]);
                    float2 q_61 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                    float2 r_61 = fma_f32x2_rn_noftz(q_61, scale1_2, a_63);
                    acc_114[26] = r_61.x;
                    acc_114[27] = r_61.y;
                    float2 a_64 = make_float2(acc_114[28], acc_114[29]);
                    float2 q_62 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                    float2 r_62 = fma_f32x2_rn_noftz(q_62, scale1_2, a_64);
                    acc_114[28] = r_62.x;
                    acc_114[29] = r_62.y;
                    float2 a_65 = make_float2(acc_114[30], acc_114[31]);
                    float2 q_63 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                    float2 r_63 = fma_f32x2_rn_noftz(q_63, scale1_2, a_65);
                    acc_114[30] = r_63.x;
                    acc_114[31] = r_63.y;
                    unsigned int acc_114_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_114[_lp*2 + 0], acc_114[_lp*2+1 + 0]));
                        acc_114_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_114_bf16[0]), "r"(acc_114_bf16[1]), "r"(acc_114_bf16[2]), "r"(acc_114_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_114_bf16[4]), "r"(acc_114_bf16[5]), "r"(acc_114_bf16[6]), "r"(acc_114_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_114_bf16[8]), "r"(acc_114_bf16[9]), "r"(acc_114_bf16[10]), "r"(acc_114_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_114_bf16[12]), "r"(acc_114_bf16[13]), "r"(acc_114_bf16[14]), "r"(acc_114_bf16[15]) : "memory");
                    __syncwarp();
                    int c_0 = lane_1;
                    int l_0 = c_0 >> 2;
                    unsigned int w4_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_0 + (c_0 & 3 ^ l_0 >> 1 & 3)))));
                    long long idx_0 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_0));
                    {
                        int4 _iv4 = make_int4(w4_0[0 + 0], w4_0[0 + 1], w4_0[0 + 2], w4_0[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_0) = _iv4;
                    }
                    int c_1 = lane_1 + 32;
                    int l_1 = c_1 >> 2;
                    unsigned int w4_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1 + (c_1 & 3 ^ l_1 >> 1 & 3)))));
                    long long idx_1 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_1));
                    {
                        int4 _iv4 = make_int4(w4_1[0 + 0], w4_1[0 + 1], w4_1[0 + 2], w4_1[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1) = _iv4;
                    }
                    int c_2 = lane_1 + 64;
                    int l_2 = c_2 >> 2;
                    unsigned int w4_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_2 + (c_2 & 3 ^ l_2 >> 1 & 3)))));
                    long long idx_2 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_2));
                    {
                        int4 _iv4 = make_int4(w4_2[0 + 0], w4_2[0 + 1], w4_2[0 + 2], w4_2[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_2) = _iv4;
                    }
                    int c_3 = lane_1 + 96;
                    int l_3 = c_3 >> 2;
                    unsigned int w4_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_3 + (c_3 & 3 ^ l_3 >> 1 & 3)))));
                    long long idx_3 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_3));
                    {
                        int4 _iv4 = make_int4(w4_3[0 + 0], w4_3[0 + 1], w4_3[0 + 2], w4_3[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_3) = _iv4;
                    }
                    __syncwarp();
                    unsigned int a_231[4];
                    unsigned int b_232[4];
                    a_231[0] = ((first) ? own[8] : _fp4_wire_values_0[8]);
                    b_232[0] = ((first) ? _fp4_wire_values_0[8] : own[8]);
                    a_231[1] = ((first) ? own[9] : _fp4_wire_values_0[9]);
                    b_232[1] = ((first) ? _fp4_wire_values_0[9] : own[9]);
                    a_231[2] = ((first) ? own[10] : _fp4_wire_values_0[10]);
                    b_232[2] = ((first) ? _fp4_wire_values_0[10] : own[10]);
                    a_231[3] = ((first) ? own[11] : _fp4_wire_values_0[11]);
                    b_232[3] = ((first) ? _fp4_wire_values_0[11] : own[11]);
                    unsigned int a_sc_233 = ((first) ? own_sc[2] : _fp4_wire_scales_0[2]);
                    unsigned int b_sc_234 = ((first) ? _fp4_wire_scales_0[2] : own_sc[2]);
                    float a_rs_235 = ((first) ? rs_own : rs_peer);
                    float b_rs_236 = ((first) ? rs_peer : rs_own);
                    float acc_237[32];
                    acc_237[0] = 0.0f;
                    acc_237[1] = 0.0f;
                    acc_237[2] = 0.0f;
                    acc_237[3] = 0.0f;
                    acc_237[4] = 0.0f;
                    acc_237[5] = 0.0f;
                    acc_237[6] = 0.0f;
                    acc_237[7] = 0.0f;
                    acc_237[8] = 0.0f;
                    acc_237[9] = 0.0f;
                    acc_237[10] = 0.0f;
                    acc_237[11] = 0.0f;
                    acc_237[12] = 0.0f;
                    acc_237[13] = 0.0f;
                    acc_237[14] = 0.0f;
                    acc_237[15] = 0.0f;
                    acc_237[16] = 0.0f;
                    acc_237[17] = 0.0f;
                    acc_237[18] = 0.0f;
                    acc_237[19] = 0.0f;
                    acc_237[20] = 0.0f;
                    acc_237[21] = 0.0f;
                    acc_237[22] = 0.0f;
                    acc_237[23] = 0.0f;
                    acc_237[24] = 0.0f;
                    acc_237[25] = 0.0f;
                    acc_237[26] = 0.0f;
                    acc_237[27] = 0.0f;
                    acc_237[28] = 0.0f;
                    acc_237[29] = 0.0f;
                    acc_237[30] = 0.0f;
                    acc_237[31] = 0.0f;
                    uint32_t e4m3x2_to_f16x2_5;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_5) : "h"((uint16_t)(a_sc_233)));
                    float cvt_f32_f16_6;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_6) : "h"((uint16_t)(e4m3x2_to_f16x2_5)));
                    float cvt_f32_f16_1_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_5) : "h"((uint16_t)(e4m3x2_to_f16x2_5 >> 16)));
                    float _mul_0_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0_3) : "f"(cvt_f32_f16_6), "f"(a_rs_235));
                    float _mul_1_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1_3) : "f"(cvt_f32_f16_1_5), "f"(a_rs_235));
                    float2 scale0_3 = make_float2(_mul_0_3, _mul_0_3);
                    float2 scale1_3 = make_float2(_mul_1_3, _mul_1_3);
                    float _unpack_fp4_16[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_231[0]));
                        (&_unpack_fp4_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_16[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_16[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_16[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_16[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_16[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_16[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_16[0]), "=f"(_unpack_fp4_16[1]), "=f"(_unpack_fp4_16[2]), "=f"(_unpack_fp4_16[3]), "=f"(_unpack_fp4_16[4]), "=f"(_unpack_fp4_16[5]), "=f"(_unpack_fp4_16[6]), "=f"(_unpack_fp4_16[7]) : "r"(a_231[0]));
                        #endif
                    }
                    float2 a_66 = make_float2(acc_237[0], acc_237[1]);
                    float2 q_64 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                    float2 r_64 = fma_f32x2_rn_noftz(q_64, scale0_3, a_66);
                    acc_237[0] = r_64.x;
                    acc_237[1] = r_64.y;
                    float2 a_67 = make_float2(acc_237[2], acc_237[3]);
                    float2 q_65 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                    float2 r_65 = fma_f32x2_rn_noftz(q_65, scale0_3, a_67);
                    acc_237[2] = r_65.x;
                    acc_237[3] = r_65.y;
                    float2 a_68 = make_float2(acc_237[4], acc_237[5]);
                    float2 q_66 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                    float2 r_66 = fma_f32x2_rn_noftz(q_66, scale0_3, a_68);
                    acc_237[4] = r_66.x;
                    acc_237[5] = r_66.y;
                    float2 a_69 = make_float2(acc_237[6], acc_237[7]);
                    float2 q_67 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                    float2 r_67 = fma_f32x2_rn_noftz(q_67, scale0_3, a_69);
                    acc_237[6] = r_67.x;
                    acc_237[7] = r_67.y;
                    float _unpack_fp4_17[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_231[1 + 0]));
                        (&_unpack_fp4_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_17[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_17[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_17[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_17[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_17[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_17[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_17[0]), "=f"(_unpack_fp4_17[1]), "=f"(_unpack_fp4_17[2]), "=f"(_unpack_fp4_17[3]), "=f"(_unpack_fp4_17[4]), "=f"(_unpack_fp4_17[5]), "=f"(_unpack_fp4_17[6]), "=f"(_unpack_fp4_17[7]) : "r"(a_231[1 + 0]));
                        #endif
                    }
                    float2 a_70 = make_float2(acc_237[8], acc_237[9]);
                    float2 q_68 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                    float2 r_68 = fma_f32x2_rn_noftz(q_68, scale0_3, a_70);
                    acc_237[8] = r_68.x;
                    acc_237[9] = r_68.y;
                    float2 a_71 = make_float2(acc_237[10], acc_237[11]);
                    float2 q_69 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                    float2 r_69 = fma_f32x2_rn_noftz(q_69, scale0_3, a_71);
                    acc_237[10] = r_69.x;
                    acc_237[11] = r_69.y;
                    float2 a_72 = make_float2(acc_237[12], acc_237[13]);
                    float2 q_70 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                    float2 r_70 = fma_f32x2_rn_noftz(q_70, scale0_3, a_72);
                    acc_237[12] = r_70.x;
                    acc_237[13] = r_70.y;
                    float2 a_73 = make_float2(acc_237[14], acc_237[15]);
                    float2 q_71 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                    float2 r_71 = fma_f32x2_rn_noftz(q_71, scale0_3, a_73);
                    acc_237[14] = r_71.x;
                    acc_237[15] = r_71.y;
                    float _unpack_fp4_18[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_231[2 + 0]));
                        (&_unpack_fp4_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_18[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_18[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_18[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_18[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_18[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_18[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_18[0]), "=f"(_unpack_fp4_18[1]), "=f"(_unpack_fp4_18[2]), "=f"(_unpack_fp4_18[3]), "=f"(_unpack_fp4_18[4]), "=f"(_unpack_fp4_18[5]), "=f"(_unpack_fp4_18[6]), "=f"(_unpack_fp4_18[7]) : "r"(a_231[2 + 0]));
                        #endif
                    }
                    float2 a_74 = make_float2(acc_237[16], acc_237[17]);
                    float2 q_72 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                    float2 r_72 = fma_f32x2_rn_noftz(q_72, scale1_3, a_74);
                    acc_237[16] = r_72.x;
                    acc_237[17] = r_72.y;
                    float2 a_75 = make_float2(acc_237[18], acc_237[19]);
                    float2 q_73 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                    float2 r_73 = fma_f32x2_rn_noftz(q_73, scale1_3, a_75);
                    acc_237[18] = r_73.x;
                    acc_237[19] = r_73.y;
                    float2 a_76 = make_float2(acc_237[20], acc_237[21]);
                    float2 q_74 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                    float2 r_74 = fma_f32x2_rn_noftz(q_74, scale1_3, a_76);
                    acc_237[20] = r_74.x;
                    acc_237[21] = r_74.y;
                    float2 a_77 = make_float2(acc_237[22], acc_237[23]);
                    float2 q_75 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                    float2 r_75 = fma_f32x2_rn_noftz(q_75, scale1_3, a_77);
                    acc_237[22] = r_75.x;
                    acc_237[23] = r_75.y;
                    float _unpack_fp4_19[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_231[3 + 0]));
                        (&_unpack_fp4_19[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_19[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_19[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_19[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_19[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_19[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_19[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_19[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_19[0]), "=f"(_unpack_fp4_19[1]), "=f"(_unpack_fp4_19[2]), "=f"(_unpack_fp4_19[3]), "=f"(_unpack_fp4_19[4]), "=f"(_unpack_fp4_19[5]), "=f"(_unpack_fp4_19[6]), "=f"(_unpack_fp4_19[7]) : "r"(a_231[3 + 0]));
                        #endif
                    }
                    float2 a_78 = make_float2(acc_237[24], acc_237[25]);
                    float2 q_76 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                    float2 r_76 = fma_f32x2_rn_noftz(q_76, scale1_3, a_78);
                    acc_237[24] = r_76.x;
                    acc_237[25] = r_76.y;
                    float2 a_79 = make_float2(acc_237[26], acc_237[27]);
                    float2 q_77 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                    float2 r_77 = fma_f32x2_rn_noftz(q_77, scale1_3, a_79);
                    acc_237[26] = r_77.x;
                    acc_237[27] = r_77.y;
                    float2 a_80 = make_float2(acc_237[28], acc_237[29]);
                    float2 q_78 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                    float2 r_78 = fma_f32x2_rn_noftz(q_78, scale1_3, a_80);
                    acc_237[28] = r_78.x;
                    acc_237[29] = r_78.y;
                    float2 a_81 = make_float2(acc_237[30], acc_237[31]);
                    float2 q_79 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                    float2 r_79 = fma_f32x2_rn_noftz(q_79, scale1_3, a_81);
                    acc_237[30] = r_79.x;
                    acc_237[31] = r_79.y;
                    uint32_t e4m3x2_to_f16x2_6;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_6) : "h"((uint16_t)(b_sc_234)));
                    float cvt_f32_f16_7;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_7) : "h"((uint16_t)(e4m3x2_to_f16x2_6)));
                    float cvt_f32_f16_1_6;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_6) : "h"((uint16_t)(e4m3x2_to_f16x2_6 >> 16)));
                    float _mul_0_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0_4) : "f"(cvt_f32_f16_7), "f"(b_rs_236));
                    float _mul_1_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1_4) : "f"(cvt_f32_f16_1_6), "f"(b_rs_236));
                    float2 scale0_4 = make_float2(_mul_0_4, _mul_0_4);
                    float2 scale1_4 = make_float2(_mul_1_4, _mul_1_4);
                    float _unpack_fp4_20[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_232[0]));
                        (&_unpack_fp4_20[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_20[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_20[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_20[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_20[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_20[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_20[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_20[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_20[0]), "=f"(_unpack_fp4_20[1]), "=f"(_unpack_fp4_20[2]), "=f"(_unpack_fp4_20[3]), "=f"(_unpack_fp4_20[4]), "=f"(_unpack_fp4_20[5]), "=f"(_unpack_fp4_20[6]), "=f"(_unpack_fp4_20[7]) : "r"(b_232[0]));
                        #endif
                    }
                    float2 a_82 = make_float2(acc_237[0], acc_237[1]);
                    float2 q_80 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                    float2 r_80 = fma_f32x2_rn_noftz(q_80, scale0_4, a_82);
                    acc_237[0] = r_80.x;
                    acc_237[1] = r_80.y;
                    float2 a_83 = make_float2(acc_237[2], acc_237[3]);
                    float2 q_81 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                    float2 r_81 = fma_f32x2_rn_noftz(q_81, scale0_4, a_83);
                    acc_237[2] = r_81.x;
                    acc_237[3] = r_81.y;
                    float2 a_84 = make_float2(acc_237[4], acc_237[5]);
                    float2 q_82 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                    float2 r_82 = fma_f32x2_rn_noftz(q_82, scale0_4, a_84);
                    acc_237[4] = r_82.x;
                    acc_237[5] = r_82.y;
                    float2 a_85 = make_float2(acc_237[6], acc_237[7]);
                    float2 q_83 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                    float2 r_83 = fma_f32x2_rn_noftz(q_83, scale0_4, a_85);
                    acc_237[6] = r_83.x;
                    acc_237[7] = r_83.y;
                    float _unpack_fp4_21[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_232[1 + 0]));
                        (&_unpack_fp4_21[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_21[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_21[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_21[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_21[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_21[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_21[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_21[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_21[0]), "=f"(_unpack_fp4_21[1]), "=f"(_unpack_fp4_21[2]), "=f"(_unpack_fp4_21[3]), "=f"(_unpack_fp4_21[4]), "=f"(_unpack_fp4_21[5]), "=f"(_unpack_fp4_21[6]), "=f"(_unpack_fp4_21[7]) : "r"(b_232[1 + 0]));
                        #endif
                    }
                    float2 a_86 = make_float2(acc_237[8], acc_237[9]);
                    float2 q_84 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                    float2 r_84 = fma_f32x2_rn_noftz(q_84, scale0_4, a_86);
                    acc_237[8] = r_84.x;
                    acc_237[9] = r_84.y;
                    float2 a_87 = make_float2(acc_237[10], acc_237[11]);
                    float2 q_85 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                    float2 r_85 = fma_f32x2_rn_noftz(q_85, scale0_4, a_87);
                    acc_237[10] = r_85.x;
                    acc_237[11] = r_85.y;
                    float2 a_88 = make_float2(acc_237[12], acc_237[13]);
                    float2 q_86 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                    float2 r_86 = fma_f32x2_rn_noftz(q_86, scale0_4, a_88);
                    acc_237[12] = r_86.x;
                    acc_237[13] = r_86.y;
                    float2 a_89 = make_float2(acc_237[14], acc_237[15]);
                    float2 q_87 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                    float2 r_87 = fma_f32x2_rn_noftz(q_87, scale0_4, a_89);
                    acc_237[14] = r_87.x;
                    acc_237[15] = r_87.y;
                    float _unpack_fp4_22[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_232[2 + 0]));
                        (&_unpack_fp4_22[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_22[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_22[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_22[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_22[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_22[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_22[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_22[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_22[0]), "=f"(_unpack_fp4_22[1]), "=f"(_unpack_fp4_22[2]), "=f"(_unpack_fp4_22[3]), "=f"(_unpack_fp4_22[4]), "=f"(_unpack_fp4_22[5]), "=f"(_unpack_fp4_22[6]), "=f"(_unpack_fp4_22[7]) : "r"(b_232[2 + 0]));
                        #endif
                    }
                    float2 a_90 = make_float2(acc_237[16], acc_237[17]);
                    float2 q_88 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                    float2 r_88 = fma_f32x2_rn_noftz(q_88, scale1_4, a_90);
                    acc_237[16] = r_88.x;
                    acc_237[17] = r_88.y;
                    float2 a_91 = make_float2(acc_237[18], acc_237[19]);
                    float2 q_89 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                    float2 r_89 = fma_f32x2_rn_noftz(q_89, scale1_4, a_91);
                    acc_237[18] = r_89.x;
                    acc_237[19] = r_89.y;
                    float2 a_92 = make_float2(acc_237[20], acc_237[21]);
                    float2 q_90 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                    float2 r_90 = fma_f32x2_rn_noftz(q_90, scale1_4, a_92);
                    acc_237[20] = r_90.x;
                    acc_237[21] = r_90.y;
                    float2 a_93 = make_float2(acc_237[22], acc_237[23]);
                    float2 q_91 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                    float2 r_91 = fma_f32x2_rn_noftz(q_91, scale1_4, a_93);
                    acc_237[22] = r_91.x;
                    acc_237[23] = r_91.y;
                    float _unpack_fp4_23[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_232[3 + 0]));
                        (&_unpack_fp4_23[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_23[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_23[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_23[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_23[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_23[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_23[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_23[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_23[0]), "=f"(_unpack_fp4_23[1]), "=f"(_unpack_fp4_23[2]), "=f"(_unpack_fp4_23[3]), "=f"(_unpack_fp4_23[4]), "=f"(_unpack_fp4_23[5]), "=f"(_unpack_fp4_23[6]), "=f"(_unpack_fp4_23[7]) : "r"(b_232[3 + 0]));
                        #endif
                    }
                    float2 a_94 = make_float2(acc_237[24], acc_237[25]);
                    float2 q_92 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                    float2 r_92 = fma_f32x2_rn_noftz(q_92, scale1_4, a_94);
                    acc_237[24] = r_92.x;
                    acc_237[25] = r_92.y;
                    float2 a_95 = make_float2(acc_237[26], acc_237[27]);
                    float2 q_93 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                    float2 r_93 = fma_f32x2_rn_noftz(q_93, scale1_4, a_95);
                    acc_237[26] = r_93.x;
                    acc_237[27] = r_93.y;
                    float2 a_96 = make_float2(acc_237[28], acc_237[29]);
                    float2 q_94 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                    float2 r_94 = fma_f32x2_rn_noftz(q_94, scale1_4, a_96);
                    acc_237[28] = r_94.x;
                    acc_237[29] = r_94.y;
                    float2 a_97 = make_float2(acc_237[30], acc_237[31]);
                    float2 q_95 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                    float2 r_95 = fma_f32x2_rn_noftz(q_95, scale1_4, a_97);
                    acc_237[30] = r_95.x;
                    acc_237[31] = r_95.y;
                    unsigned int acc_237_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_237[_lp*2 + 0], acc_237[_lp*2+1 + 0]));
                        acc_237_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_237_bf16[0]), "r"(acc_237_bf16[1]), "r"(acc_237_bf16[2]), "r"(acc_237_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_237_bf16[4]), "r"(acc_237_bf16[5]), "r"(acc_237_bf16[6]), "r"(acc_237_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_237_bf16[8]), "r"(acc_237_bf16[9]), "r"(acc_237_bf16[10]), "r"(acc_237_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_237_bf16[12]), "r"(acc_237_bf16[13]), "r"(acc_237_bf16[14]), "r"(acc_237_bf16[15]) : "memory");
                    __syncwarp();
                    int c_4 = lane_1;
                    int l_4 = c_4 >> 2;
                    unsigned int w4_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_4 + (c_4 & 3 ^ l_4 >> 1 & 3)))));
                    long long idx_4 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_4));
                    {
                        int4 _iv4 = make_int4(w4_4[0 + 0], w4_4[0 + 1], w4_4[0 + 2], w4_4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_4) = _iv4;
                    }
                    int c_5 = lane_1 + 32;
                    int l_5 = c_5 >> 2;
                    unsigned int w4_5[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_5 + (c_5 & 3 ^ l_5 >> 1 & 3)))));
                    long long idx_5 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_5));
                    {
                        int4 _iv4 = make_int4(w4_5[0 + 0], w4_5[0 + 1], w4_5[0 + 2], w4_5[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_5) = _iv4;
                    }
                    int c_6 = lane_1 + 64;
                    int l_6 = c_6 >> 2;
                    unsigned int w4_6[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_6 + (c_6 & 3 ^ l_6 >> 1 & 3)))));
                    long long idx_6 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_6));
                    {
                        int4 _iv4 = make_int4(w4_6[0 + 0], w4_6[0 + 1], w4_6[0 + 2], w4_6[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_6) = _iv4;
                    }
                    int c_7 = lane_1 + 96;
                    int l_7 = c_7 >> 2;
                    unsigned int w4_7[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_7 + (c_7 & 3 ^ l_7 >> 1 & 3)))));
                    long long idx_7 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_7));
                    {
                        int4 _iv4 = make_int4(w4_7[0 + 0], w4_7[0 + 1], w4_7[0 + 2], w4_7[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_7) = _iv4;
                    }
                    __syncwarp();
                }
            }
            if (lane_1 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            if (tid == 0) {
                int base_2 = bid * 4;
                ctrl[base_2 + 2] = (unsigned int)rows;
                ctrl[base_2 + 3] = 1;
                ctrl[base_2] = (call + 1) % 3;
            }
        }
    }
    // ---- Role: reset_warp ----
    if (warp == 10) {
        { // reset_warp_main
            int lane_0 = (int)lane;
            unsigned int sv = 2290649224;
            unsigned int ss = 4294967295;
            unsigned int sr = 2147483648;
            #pragma unroll 1
            for (int i = lane_0; i < 1024; i += 32) {
                pat[i] = sv;
                pat[1024 + i] = ss;
                pat[2048 + i] = sr;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("fence.proxy.async.global;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            asm volatile("barrier.sync 0, 352;" ::: "memory");
            unsigned int prev_rows = ctl[1];
            unsigned int prev_algo = ctl[2];
            unsigned int pbuf = (ctl[0] + 2) % 3;
            if (prev_algo != 0) {
                long long bid64_1 = (long long)bid;
                long long g64_1 = (long long)gridDim.x;
                long long pr64 = (long long)prev_rows;
                long long pb64 = (long long)pbuf;
                long long cap64_1 = (long long)cap;
                long long peer64_1 = (long long)(1 - rank);
                int jbase = 0;
                long long v_base_1 = pb64 * (2 * cap64_1 * 3456) + peer64_1 * cap64_1 * 3072;
                long long s_base_1 = pb64 * (2 * cap64_1 * 3456) + 2 * cap64_1 * 3072 + peer64_1 * cap64_1 * 384;
                long long m_base_1 = pb64 * (2 * cap64_1 * 16) + peer64_1 * cap64_1 * 16;
                long long u = pr64 * 3072 / 16;
                long long lo = u * bid64_1 / g64_1 * 16;
                long long hi = u * (bid64_1 + 1) / g64_1 * 16;
                int npages = (int)((hi - lo + 4095) / 4096);
                #pragma unroll 1
                for (int i_1 = 0; i_1 < npages; i_1++) {
                    if ((jbase + i_1 & 31) == lane_0) {
                        long long o = lo + (long long)i_1 * 4096;
                        long long rem = hi - o;
                        unsigned int nb = 4096;
                        if (rem < 4096) {
                            nb = (unsigned int)rem;
                        }
                        {
                            void* _cpbulk_dst_0 = reinterpret_cast<void*>(os_wire + (v_base_1 + o));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_0), "r"(pat_addr), "r"((uint32_t)(nb))
                                : "memory");
                        }
                    }
                }
                jbase = jbase + npages;
                long long u_0 = pr64 * 384 / 16;
                long long lo_0 = u_0 * bid64_1 / g64_1 * 16;
                long long hi_0 = u_0 * (bid64_1 + 1) / g64_1 * 16;
                int npages_0 = (int)((hi_0 - lo_0 + 4095) / 4096);
                #pragma unroll 1
                for (int i_2 = 0; i_2 < npages_0; i_2++) {
                    if ((jbase + i_2 & 31) == lane_0) {
                        long long o_1 = lo_0 + (long long)i_2 * 4096;
                        long long rem_1 = hi_0 - o_1;
                        unsigned int nb_1 = 4096;
                        if (rem_1 < 4096) {
                            nb_1 = (unsigned int)rem_1;
                        }
                        {
                            void* _cpbulk_dst_1 = reinterpret_cast<void*>(os_wire + (s_base_1 + o_1));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_1), "r"(pat_addr + 4096), "r"((uint32_t)(nb_1))
                                : "memory");
                        }
                    }
                }
                jbase = jbase + npages_0;
                long long u_1 = pr64 * 16 / 16;
                long long lo_1 = u_1 * bid64_1 / g64_1 * 16;
                long long hi_1 = u_1 * (bid64_1 + 1) / g64_1 * 16;
                int npages_1 = (int)((hi_1 - lo_1 + 4095) / 4096);
                #pragma unroll 1
                for (int i_3 = 0; i_3 < npages_1; i_3++) {
                    if ((jbase + i_3 & 31) == lane_0) {
                        long long o_2 = lo_1 + (long long)i_3 * 4096;
                        long long rem_2 = hi_1 - o_2;
                        unsigned int nb_2 = 4096;
                        if (rem_2 < 4096) {
                            nb_2 = (unsigned int)rem_2;
                        }
                        {
                            void* _cpbulk_dst_2 = reinterpret_cast<void*>(os_rs + (m_base_1 + o_2));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_2), "r"(pat_addr + 8192), "r"((uint32_t)(nb_2))
                                : "memory");
                        }
                    }
                }
                jbase = jbase + npages_1;
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }

    // Cleanup
}

} // extern "C"
