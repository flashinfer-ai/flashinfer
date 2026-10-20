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

extern "C" {

__global__ __launch_bounds__(352, LAUNCH_MIN_BLOCKS) void
cake_fp4_allreduce_tp2_oneshot_bf16_rso(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, uint8_t* __restrict__ os_wire, uint8_t* __restrict__ os_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap)
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
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
            #pragma unroll 1
            for (int k = 0; k < n + 1; k++) {
                if (n > k) {
                    int slot = k % 2;
                    int ob_off = xb_off + 12288 + slot * 3472;
                    if (h == 0 && k > 1 && lane_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    float rs = 1.0f;
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
                    unsigned int _max_0 = ((a & 65535) > (a >> 16) ? (a & 65535) : (a >> 16));
                    unsigned int xa = _max_0;
                    unsigned int _max_1 = ((b & 65535) > (b >> 16) ? (b & 65535) : (b >> 16));
                    unsigned int xb = _max_1;
                    if ((f & 2) != 0) {
                        bmp[0] = xb | xa << 16;
                    } else {
                        bmp[0] = xa | xb << 16;
                    }
                    unsigned int _max_2 = ((xa) > (xb) ? (xa) : (xb));
                    float _max_3 = max_noftz(am, __uint_as_float(_max_2 << 16));
                    am = _max_3;
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
                    unsigned int _max_4 = ((a_2 & 65535) > (a_2 >> 16) ? (a_2 & 65535) : (a_2 >> 16));
                    unsigned int xa_8 = _max_4;
                    unsigned int _max_5 = ((b_3 & 65535) > (b_3 >> 16) ? (b_3 & 65535) : (b_3 >> 16));
                    unsigned int xb_9 = _max_5;
                    if ((f & 2) != 0) {
                        bmp[1] = xb_9 | xa_8 << 16;
                    } else {
                        bmp[1] = xa_8 | xb_9 << 16;
                    }
                    unsigned int _max_6 = ((xa_8) > (xb_9) ? (xa_8) : (xb_9));
                    float _max_7 = max_noftz(am, __uint_as_float(_max_6 << 16));
                    am = _max_7;
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
                    unsigned int _max_8 = ((a_12 & 65535) > (a_12 >> 16) ? (a_12 & 65535) : (a_12 >> 16));
                    unsigned int xa_18 = _max_8;
                    unsigned int _max_9 = ((b_13 & 65535) > (b_13 >> 16) ? (b_13 & 65535) : (b_13 >> 16));
                    unsigned int xb_19 = _max_9;
                    if ((f & 2) != 0) {
                        bmp[2] = xb_19 | xa_18 << 16;
                    } else {
                        bmp[2] = xa_18 | xb_19 << 16;
                    }
                    unsigned int _max_10 = ((xa_18) > (xb_19) ? (xa_18) : (xb_19));
                    float _max_11 = max_noftz(am, __uint_as_float(_max_10 << 16));
                    am = _max_11;
                    float _warp_reduce_0 = am;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                    am = _warp_reduce_0;
                    if (lane_1 == 0) {
                        s_am[2 * t + h] = am;
                    }
                    asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                    float _max_12 = max_noftz(s_am[2 * t], s_am[2 * t + 1]);
                    am = _max_12;
                    float _fdiv_rn_0 = __fdiv_rn(2688.0f, am);
                    float inv = 0.0f;
                    if (am > 0.0f) {
                        inv = _fdiv_rn_0;
                    }
                    float _fdiv_rn_1 = __fdiv_rn(am, 2688.0f);
                    float rs_20 = _fdiv_rn_1;
                    int pp2 = lane_1 + 32 * (3 * h);
                    int base2 = xb_off + 64 * pp2;
                    float bm0 = __uint_as_float(bmp[0] << 16);
                    float bm1 = __uint_as_float(bmp[0] & 4294901760u);
                    float _mul_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(bm0), "f"(inv));
                    float _fdiv_rn_2 = __fdiv_rn(_mul_0, 6.0f);
                    float _mul_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(bm1), "f"(inv));
                    float _fdiv_rn_3 = __fdiv_rn(_mul_1, 6.0f);
                    uint16_t _e4m3x2_f32_0;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_fdiv_rn_3), "f"(_fdiv_rn_2));
                    uint32_t _e4m3x2_to_f16x2_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_0) : "h"((uint16_t)(_e4m3x2_f32_0)));
                    float _cvt_f32_f16_0;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_0) : "h"((uint16_t)(_e4m3x2_to_f16x2_0)));
                    float _cvt_f32_f16_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1) : "h"((uint16_t)(_e4m3x2_to_f16x2_0 >> 16)));
                    float _fdiv_rn_4 = __fdiv_rn(inv, _cvt_f32_f16_0);
                    float _fdiv_rn_5 = __fdiv_rn(inv, _cvt_f32_f16_1);
                    float m0 = 0.0f;
                    float m1 = 0.0f;
                    if (_cvt_f32_f16_0 > 0.0f) {
                        m0 = _fdiv_rn_4;
                    }
                    if (_cvt_f32_f16_1 > 0.0f) {
                        m1 = _fdiv_rn_5;
                    }
                    unsigned int w[4];
                    int cf = f ^ 0;
                    unsigned int v4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2 + 16 * cf)));
                    float m = m0;
                    if ((cf & 2) != 0) {
                        m = m1;
                    }
                    float2 _f2_0 = make_float2(m, m);
                    float2 mm = _f2_0;
                    float2 _f2_1 = make_float2(__uint_as_float(v4[0] << 16), __uint_as_float(v4[0] & 4294901760u));
                    float2 pair = _f2_1;
                    float2 g;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g) : "l"(*(const unsigned long long*)&pair), "l"(*(const unsigned long long*)&mm));
                    uint32_t _fp4_pair_0;
                    uint16_t _e2m1_pair_h_0;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_0) : "f"(g.x), "f"(g.y));
                    _fp4_pair_0 = (uint32_t)_e2m1_pair_h_0;
                    float2 _f2_2 = make_float2(__uint_as_float(v4[1] << 16), __uint_as_float(v4[1] & 4294901760u));
                    float2 pair_21 = _f2_2;
                    float2 g_22;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_22) : "l"(*(const unsigned long long*)&pair_21), "l"(*(const unsigned long long*)&mm));
                    uint32_t _fp4_pair_1;
                    uint16_t _e2m1_pair_h_1;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_1) : "f"(g_22.x), "f"(g_22.y));
                    _fp4_pair_1 = (uint32_t)_e2m1_pair_h_1;
                    float2 _f2_3 = make_float2(__uint_as_float(v4[2] << 16), __uint_as_float(v4[2] & 4294901760u));
                    float2 pair_23 = _f2_3;
                    float2 g_24;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_24) : "l"(*(const unsigned long long*)&pair_23), "l"(*(const unsigned long long*)&mm));
                    uint32_t _fp4_pair_2;
                    uint16_t _e2m1_pair_h_2;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_2) : "f"(g_24.x), "f"(g_24.y));
                    _fp4_pair_2 = (uint32_t)_e2m1_pair_h_2;
                    float2 _f2_4 = make_float2(__uint_as_float(v4[3] << 16), __uint_as_float(v4[3] & 4294901760u));
                    float2 pair_25 = _f2_4;
                    float2 g_26;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_26) : "l"(*(const unsigned long long*)&pair_25), "l"(*(const unsigned long long*)&mm));
                    uint32_t _fp4_pair_3;
                    uint16_t _e2m1_pair_h_3;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_3) : "f"(g_26.x), "f"(g_26.y));
                    _fp4_pair_3 = (uint32_t)_e2m1_pair_h_3;
                    unsigned int word = _fp4_pair_0 | _fp4_pair_1 << 8 | _fp4_pair_2 << 16 | _fp4_pair_3 << 24;
                    unsigned int carry = (word & 2004318071) + 2004318071;
                    w[0] = word & (carry | 2004318071);
                    int cf_27 = f ^ 1;
                    unsigned int v4_28[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_28[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_28[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_28[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2 + 16 * cf_27)));
                    float m_29 = m0;
                    if ((cf_27 & 2) != 0) {
                        m_29 = m1;
                    }
                    float2 _f2_5 = make_float2(m_29, m_29);
                    float2 mm_30 = _f2_5;
                    float2 _f2_6 = make_float2(__uint_as_float(v4_28[0] << 16), __uint_as_float(v4_28[0] & 4294901760u));
                    float2 pair_31 = _f2_6;
                    float2 g_32;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_32) : "l"(*(const unsigned long long*)&pair_31), "l"(*(const unsigned long long*)&mm_30));
                    uint32_t _fp4_pair_4;
                    uint16_t _e2m1_pair_h_4;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_4) : "f"(g_32.x), "f"(g_32.y));
                    _fp4_pair_4 = (uint32_t)_e2m1_pair_h_4;
                    float2 _f2_7 = make_float2(__uint_as_float(v4_28[1] << 16), __uint_as_float(v4_28[1] & 4294901760u));
                    float2 pair_33 = _f2_7;
                    float2 g_34;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_34) : "l"(*(const unsigned long long*)&pair_33), "l"(*(const unsigned long long*)&mm_30));
                    uint32_t _fp4_pair_5;
                    uint16_t _e2m1_pair_h_5;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_5) : "f"(g_34.x), "f"(g_34.y));
                    _fp4_pair_5 = (uint32_t)_e2m1_pair_h_5;
                    float2 _f2_8 = make_float2(__uint_as_float(v4_28[2] << 16), __uint_as_float(v4_28[2] & 4294901760u));
                    float2 pair_35 = _f2_8;
                    float2 g_36;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_36) : "l"(*(const unsigned long long*)&pair_35), "l"(*(const unsigned long long*)&mm_30));
                    uint32_t _fp4_pair_6;
                    uint16_t _e2m1_pair_h_6;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_6) : "f"(g_36.x), "f"(g_36.y));
                    _fp4_pair_6 = (uint32_t)_e2m1_pair_h_6;
                    float2 _f2_9 = make_float2(__uint_as_float(v4_28[3] << 16), __uint_as_float(v4_28[3] & 4294901760u));
                    float2 pair_37 = _f2_9;
                    float2 g_38;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_38) : "l"(*(const unsigned long long*)&pair_37), "l"(*(const unsigned long long*)&mm_30));
                    uint32_t _fp4_pair_7;
                    uint16_t _e2m1_pair_h_7;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_7) : "f"(g_38.x), "f"(g_38.y));
                    _fp4_pair_7 = (uint32_t)_e2m1_pair_h_7;
                    unsigned int word_39 = _fp4_pair_4 | _fp4_pair_5 << 8 | _fp4_pair_6 << 16 | _fp4_pair_7 << 24;
                    unsigned int carry_40 = (word_39 & 2004318071) + 2004318071;
                    w[1] = word_39 & (carry_40 | 2004318071);
                    int cf_41 = f ^ 2;
                    unsigned int v4_42[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_42[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_42[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_42[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_42[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2 + 16 * cf_41)));
                    float m_43 = m0;
                    if ((cf_41 & 2) != 0) {
                        m_43 = m1;
                    }
                    float2 _f2_10 = make_float2(m_43, m_43);
                    float2 mm_44 = _f2_10;
                    float2 _f2_11 = make_float2(__uint_as_float(v4_42[0] << 16), __uint_as_float(v4_42[0] & 4294901760u));
                    float2 pair_45 = _f2_11;
                    float2 g_46;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_46) : "l"(*(const unsigned long long*)&pair_45), "l"(*(const unsigned long long*)&mm_44));
                    uint32_t _fp4_pair_8;
                    uint16_t _e2m1_pair_h_8;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_8) : "f"(g_46.x), "f"(g_46.y));
                    _fp4_pair_8 = (uint32_t)_e2m1_pair_h_8;
                    float2 _f2_12 = make_float2(__uint_as_float(v4_42[1] << 16), __uint_as_float(v4_42[1] & 4294901760u));
                    float2 pair_47 = _f2_12;
                    float2 g_48;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_48) : "l"(*(const unsigned long long*)&pair_47), "l"(*(const unsigned long long*)&mm_44));
                    uint32_t _fp4_pair_9;
                    uint16_t _e2m1_pair_h_9;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_9) : "f"(g_48.x), "f"(g_48.y));
                    _fp4_pair_9 = (uint32_t)_e2m1_pair_h_9;
                    float2 _f2_13 = make_float2(__uint_as_float(v4_42[2] << 16), __uint_as_float(v4_42[2] & 4294901760u));
                    float2 pair_49 = _f2_13;
                    float2 g_50;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_50) : "l"(*(const unsigned long long*)&pair_49), "l"(*(const unsigned long long*)&mm_44));
                    uint32_t _fp4_pair_10;
                    uint16_t _e2m1_pair_h_10;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_10) : "f"(g_50.x), "f"(g_50.y));
                    _fp4_pair_10 = (uint32_t)_e2m1_pair_h_10;
                    float2 _f2_14 = make_float2(__uint_as_float(v4_42[3] << 16), __uint_as_float(v4_42[3] & 4294901760u));
                    float2 pair_51 = _f2_14;
                    float2 g_52;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_52) : "l"(*(const unsigned long long*)&pair_51), "l"(*(const unsigned long long*)&mm_44));
                    uint32_t _fp4_pair_11;
                    uint16_t _e2m1_pair_h_11;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_11) : "f"(g_52.x), "f"(g_52.y));
                    _fp4_pair_11 = (uint32_t)_e2m1_pair_h_11;
                    unsigned int word_53 = _fp4_pair_8 | _fp4_pair_9 << 8 | _fp4_pair_10 << 16 | _fp4_pair_11 << 24;
                    unsigned int carry_54 = (word_53 & 2004318071) + 2004318071;
                    w[2] = word_53 & (carry_54 | 2004318071);
                    int cf_55 = f ^ 3;
                    unsigned int v4_56[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_56[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_56[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_56[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_56[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2 + 16 * cf_55)));
                    float m_57 = m0;
                    if ((cf_55 & 2) != 0) {
                        m_57 = m1;
                    }
                    float2 _f2_15 = make_float2(m_57, m_57);
                    float2 mm_58 = _f2_15;
                    float2 _f2_16 = make_float2(__uint_as_float(v4_56[0] << 16), __uint_as_float(v4_56[0] & 4294901760u));
                    float2 pair_59 = _f2_16;
                    float2 g_60;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_60) : "l"(*(const unsigned long long*)&pair_59), "l"(*(const unsigned long long*)&mm_58));
                    uint32_t _fp4_pair_12;
                    uint16_t _e2m1_pair_h_12;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_12) : "f"(g_60.x), "f"(g_60.y));
                    _fp4_pair_12 = (uint32_t)_e2m1_pair_h_12;
                    float2 _f2_17 = make_float2(__uint_as_float(v4_56[1] << 16), __uint_as_float(v4_56[1] & 4294901760u));
                    float2 pair_61 = _f2_17;
                    float2 g_62;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_62) : "l"(*(const unsigned long long*)&pair_61), "l"(*(const unsigned long long*)&mm_58));
                    uint32_t _fp4_pair_13;
                    uint16_t _e2m1_pair_h_13;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_13) : "f"(g_62.x), "f"(g_62.y));
                    _fp4_pair_13 = (uint32_t)_e2m1_pair_h_13;
                    float2 _f2_18 = make_float2(__uint_as_float(v4_56[2] << 16), __uint_as_float(v4_56[2] & 4294901760u));
                    float2 pair_63 = _f2_18;
                    float2 g_64;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_64) : "l"(*(const unsigned long long*)&pair_63), "l"(*(const unsigned long long*)&mm_58));
                    uint32_t _fp4_pair_14;
                    uint16_t _e2m1_pair_h_14;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_14) : "f"(g_64.x), "f"(g_64.y));
                    _fp4_pair_14 = (uint32_t)_e2m1_pair_h_14;
                    float2 _f2_19 = make_float2(__uint_as_float(v4_56[3] << 16), __uint_as_float(v4_56[3] & 4294901760u));
                    float2 pair_65 = _f2_19;
                    float2 g_66;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_66) : "l"(*(const unsigned long long*)&pair_65), "l"(*(const unsigned long long*)&mm_58));
                    uint32_t _fp4_pair_15;
                    uint16_t _e2m1_pair_h_15;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_15) : "f"(g_66.x), "f"(g_66.y));
                    _fp4_pair_15 = (uint32_t)_e2m1_pair_h_15;
                    unsigned int word_67 = _fp4_pair_12 | _fp4_pair_13 << 8 | _fp4_pair_14 << 16 | _fp4_pair_15 << 24;
                    unsigned int carry_68 = (word_67 & 2004318071) + 2004318071;
                    w[3] = word_67 & (carry_68 | 2004318071);
                    bool fodd = (f & 1) != 0;
                    bool fhi = (f & 2) != 0;
                    unsigned int t0 = w[0];
                    unsigned int t1 = w[1];
                    unsigned int t2 = w[2];
                    unsigned int t3 = w[3];
                    if (fodd) {
                        t0 = w[1];
                        t1 = w[0];
                        t2 = w[3];
                        t3 = w[2];
                    }
                    unsigned int q0 = t0;
                    unsigned int q1 = t1;
                    unsigned int q2 = t2;
                    unsigned int q3 = t3;
                    if (fhi) {
                        q0 = t2;
                        q1 = t3;
                        q2 = t0;
                        q3 = t1;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2)), "r"(q0), "r"(q1), "r"(q2), "r"(q3) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2)), "h"((uint16_t)(_e4m3x2_f32_0)) : "memory");
                    int pp2_69 = lane_1 + 32 * (3 * h + 1);
                    int base2_70 = xb_off + 64 * pp2_69;
                    float bm0_71 = __uint_as_float(bmp[1] << 16);
                    float bm1_72 = __uint_as_float(bmp[1] & 4294901760u);
                    float _mul_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(bm0_71), "f"(inv));
                    float _fdiv_rn_6 = __fdiv_rn(_mul_2, 6.0f);
                    float _mul_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(bm1_72), "f"(inv));
                    float _fdiv_rn_7 = __fdiv_rn(_mul_3, 6.0f);
                    uint16_t _e4m3x2_f32_1;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_fdiv_rn_7), "f"(_fdiv_rn_6));
                    uint32_t _e4m3x2_to_f16x2_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_1) : "h"((uint16_t)(_e4m3x2_f32_1)));
                    float _cvt_f32_f16_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_2) : "h"((uint16_t)(_e4m3x2_to_f16x2_1)));
                    float _cvt_f32_f16_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_3) : "h"((uint16_t)(_e4m3x2_to_f16x2_1 >> 16)));
                    float _fdiv_rn_8 = __fdiv_rn(inv, _cvt_f32_f16_2);
                    float _fdiv_rn_9 = __fdiv_rn(inv, _cvt_f32_f16_3);
                    float m0_73 = 0.0f;
                    float m1_74 = 0.0f;
                    if (_cvt_f32_f16_2 > 0.0f) {
                        m0_73 = _fdiv_rn_8;
                    }
                    if (_cvt_f32_f16_3 > 0.0f) {
                        m1_74 = _fdiv_rn_9;
                    }
                    unsigned int w_75[4];
                    int cf_76 = f ^ 0;
                    unsigned int v4_77[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_77[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_77[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_77[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_77[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_70 + 16 * cf_76)));
                    float m_78 = m0_73;
                    if ((cf_76 & 2) != 0) {
                        m_78 = m1_74;
                    }
                    float2 _f2_20 = make_float2(m_78, m_78);
                    float2 mm_79 = _f2_20;
                    float2 _f2_21 = make_float2(__uint_as_float(v4_77[0] << 16), __uint_as_float(v4_77[0] & 4294901760u));
                    float2 pair_80 = _f2_21;
                    float2 g_81;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_81) : "l"(*(const unsigned long long*)&pair_80), "l"(*(const unsigned long long*)&mm_79));
                    uint32_t _fp4_pair_16;
                    uint16_t _e2m1_pair_h_16;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_16) : "f"(g_81.x), "f"(g_81.y));
                    _fp4_pair_16 = (uint32_t)_e2m1_pair_h_16;
                    float2 _f2_22 = make_float2(__uint_as_float(v4_77[1] << 16), __uint_as_float(v4_77[1] & 4294901760u));
                    float2 pair_82 = _f2_22;
                    float2 g_83;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_83) : "l"(*(const unsigned long long*)&pair_82), "l"(*(const unsigned long long*)&mm_79));
                    uint32_t _fp4_pair_17;
                    uint16_t _e2m1_pair_h_17;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_17) : "f"(g_83.x), "f"(g_83.y));
                    _fp4_pair_17 = (uint32_t)_e2m1_pair_h_17;
                    float2 _f2_23 = make_float2(__uint_as_float(v4_77[2] << 16), __uint_as_float(v4_77[2] & 4294901760u));
                    float2 pair_84 = _f2_23;
                    float2 g_85;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_85) : "l"(*(const unsigned long long*)&pair_84), "l"(*(const unsigned long long*)&mm_79));
                    uint32_t _fp4_pair_18;
                    uint16_t _e2m1_pair_h_18;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_18) : "f"(g_85.x), "f"(g_85.y));
                    _fp4_pair_18 = (uint32_t)_e2m1_pair_h_18;
                    float2 _f2_24 = make_float2(__uint_as_float(v4_77[3] << 16), __uint_as_float(v4_77[3] & 4294901760u));
                    float2 pair_86 = _f2_24;
                    float2 g_87;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_87) : "l"(*(const unsigned long long*)&pair_86), "l"(*(const unsigned long long*)&mm_79));
                    uint32_t _fp4_pair_19;
                    uint16_t _e2m1_pair_h_19;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_19) : "f"(g_87.x), "f"(g_87.y));
                    _fp4_pair_19 = (uint32_t)_e2m1_pair_h_19;
                    unsigned int word_88 = _fp4_pair_16 | _fp4_pair_17 << 8 | _fp4_pair_18 << 16 | _fp4_pair_19 << 24;
                    unsigned int carry_89 = (word_88 & 2004318071) + 2004318071;
                    w_75[0] = word_88 & (carry_89 | 2004318071);
                    int cf_90 = f ^ 1;
                    unsigned int v4_91[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_91[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_91[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_91[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_91[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_70 + 16 * cf_90)));
                    float m_92 = m0_73;
                    if ((cf_90 & 2) != 0) {
                        m_92 = m1_74;
                    }
                    float2 _f2_25 = make_float2(m_92, m_92);
                    float2 mm_93 = _f2_25;
                    float2 _f2_26 = make_float2(__uint_as_float(v4_91[0] << 16), __uint_as_float(v4_91[0] & 4294901760u));
                    float2 pair_94 = _f2_26;
                    float2 g_95;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_95) : "l"(*(const unsigned long long*)&pair_94), "l"(*(const unsigned long long*)&mm_93));
                    uint32_t _fp4_pair_20;
                    uint16_t _e2m1_pair_h_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_20) : "f"(g_95.x), "f"(g_95.y));
                    _fp4_pair_20 = (uint32_t)_e2m1_pair_h_20;
                    float2 _f2_27 = make_float2(__uint_as_float(v4_91[1] << 16), __uint_as_float(v4_91[1] & 4294901760u));
                    float2 pair_96 = _f2_27;
                    float2 g_97;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_97) : "l"(*(const unsigned long long*)&pair_96), "l"(*(const unsigned long long*)&mm_93));
                    uint32_t _fp4_pair_21;
                    uint16_t _e2m1_pair_h_21;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_21) : "f"(g_97.x), "f"(g_97.y));
                    _fp4_pair_21 = (uint32_t)_e2m1_pair_h_21;
                    float2 _f2_28 = make_float2(__uint_as_float(v4_91[2] << 16), __uint_as_float(v4_91[2] & 4294901760u));
                    float2 pair_98 = _f2_28;
                    float2 g_99;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_99) : "l"(*(const unsigned long long*)&pair_98), "l"(*(const unsigned long long*)&mm_93));
                    uint32_t _fp4_pair_22;
                    uint16_t _e2m1_pair_h_22;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_22) : "f"(g_99.x), "f"(g_99.y));
                    _fp4_pair_22 = (uint32_t)_e2m1_pair_h_22;
                    float2 _f2_29 = make_float2(__uint_as_float(v4_91[3] << 16), __uint_as_float(v4_91[3] & 4294901760u));
                    float2 pair_100 = _f2_29;
                    float2 g_101;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_101) : "l"(*(const unsigned long long*)&pair_100), "l"(*(const unsigned long long*)&mm_93));
                    uint32_t _fp4_pair_23;
                    uint16_t _e2m1_pair_h_23;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_23) : "f"(g_101.x), "f"(g_101.y));
                    _fp4_pair_23 = (uint32_t)_e2m1_pair_h_23;
                    unsigned int word_102 = _fp4_pair_20 | _fp4_pair_21 << 8 | _fp4_pair_22 << 16 | _fp4_pair_23 << 24;
                    unsigned int carry_103 = (word_102 & 2004318071) + 2004318071;
                    w_75[1] = word_102 & (carry_103 | 2004318071);
                    int cf_104 = f ^ 2;
                    unsigned int v4_105[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_105[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_105[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_105[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_105[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_70 + 16 * cf_104)));
                    float m_106 = m0_73;
                    if ((cf_104 & 2) != 0) {
                        m_106 = m1_74;
                    }
                    float2 _f2_30 = make_float2(m_106, m_106);
                    float2 mm_107 = _f2_30;
                    float2 _f2_31 = make_float2(__uint_as_float(v4_105[0] << 16), __uint_as_float(v4_105[0] & 4294901760u));
                    float2 pair_108 = _f2_31;
                    float2 g_109;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_109) : "l"(*(const unsigned long long*)&pair_108), "l"(*(const unsigned long long*)&mm_107));
                    uint32_t _fp4_pair_24;
                    uint16_t _e2m1_pair_h_24;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_24) : "f"(g_109.x), "f"(g_109.y));
                    _fp4_pair_24 = (uint32_t)_e2m1_pair_h_24;
                    float2 _f2_32 = make_float2(__uint_as_float(v4_105[1] << 16), __uint_as_float(v4_105[1] & 4294901760u));
                    float2 pair_110 = _f2_32;
                    float2 g_111;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_111) : "l"(*(const unsigned long long*)&pair_110), "l"(*(const unsigned long long*)&mm_107));
                    uint32_t _fp4_pair_25;
                    uint16_t _e2m1_pair_h_25;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_25) : "f"(g_111.x), "f"(g_111.y));
                    _fp4_pair_25 = (uint32_t)_e2m1_pair_h_25;
                    float2 _f2_33 = make_float2(__uint_as_float(v4_105[2] << 16), __uint_as_float(v4_105[2] & 4294901760u));
                    float2 pair_112 = _f2_33;
                    float2 g_113;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_113) : "l"(*(const unsigned long long*)&pair_112), "l"(*(const unsigned long long*)&mm_107));
                    uint32_t _fp4_pair_26;
                    uint16_t _e2m1_pair_h_26;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_26) : "f"(g_113.x), "f"(g_113.y));
                    _fp4_pair_26 = (uint32_t)_e2m1_pair_h_26;
                    float2 _f2_34 = make_float2(__uint_as_float(v4_105[3] << 16), __uint_as_float(v4_105[3] & 4294901760u));
                    float2 pair_114 = _f2_34;
                    float2 g_115;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_115) : "l"(*(const unsigned long long*)&pair_114), "l"(*(const unsigned long long*)&mm_107));
                    uint32_t _fp4_pair_27;
                    uint16_t _e2m1_pair_h_27;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_27) : "f"(g_115.x), "f"(g_115.y));
                    _fp4_pair_27 = (uint32_t)_e2m1_pair_h_27;
                    unsigned int word_116 = _fp4_pair_24 | _fp4_pair_25 << 8 | _fp4_pair_26 << 16 | _fp4_pair_27 << 24;
                    unsigned int carry_117 = (word_116 & 2004318071) + 2004318071;
                    w_75[2] = word_116 & (carry_117 | 2004318071);
                    int cf_118 = f ^ 3;
                    unsigned int v4_119[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_119[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_119[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_119[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_119[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_70 + 16 * cf_118)));
                    float m_120 = m0_73;
                    if ((cf_118 & 2) != 0) {
                        m_120 = m1_74;
                    }
                    float2 _f2_35 = make_float2(m_120, m_120);
                    float2 mm_121 = _f2_35;
                    float2 _f2_36 = make_float2(__uint_as_float(v4_119[0] << 16), __uint_as_float(v4_119[0] & 4294901760u));
                    float2 pair_122 = _f2_36;
                    float2 g_123;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_123) : "l"(*(const unsigned long long*)&pair_122), "l"(*(const unsigned long long*)&mm_121));
                    uint32_t _fp4_pair_28;
                    uint16_t _e2m1_pair_h_28;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_28) : "f"(g_123.x), "f"(g_123.y));
                    _fp4_pair_28 = (uint32_t)_e2m1_pair_h_28;
                    float2 _f2_37 = make_float2(__uint_as_float(v4_119[1] << 16), __uint_as_float(v4_119[1] & 4294901760u));
                    float2 pair_124 = _f2_37;
                    float2 g_125;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_125) : "l"(*(const unsigned long long*)&pair_124), "l"(*(const unsigned long long*)&mm_121));
                    uint32_t _fp4_pair_29;
                    uint16_t _e2m1_pair_h_29;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_29) : "f"(g_125.x), "f"(g_125.y));
                    _fp4_pair_29 = (uint32_t)_e2m1_pair_h_29;
                    float2 _f2_38 = make_float2(__uint_as_float(v4_119[2] << 16), __uint_as_float(v4_119[2] & 4294901760u));
                    float2 pair_126 = _f2_38;
                    float2 g_127;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_127) : "l"(*(const unsigned long long*)&pair_126), "l"(*(const unsigned long long*)&mm_121));
                    uint32_t _fp4_pair_30;
                    uint16_t _e2m1_pair_h_30;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_30) : "f"(g_127.x), "f"(g_127.y));
                    _fp4_pair_30 = (uint32_t)_e2m1_pair_h_30;
                    float2 _f2_39 = make_float2(__uint_as_float(v4_119[3] << 16), __uint_as_float(v4_119[3] & 4294901760u));
                    float2 pair_128 = _f2_39;
                    float2 g_129;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_129) : "l"(*(const unsigned long long*)&pair_128), "l"(*(const unsigned long long*)&mm_121));
                    uint32_t _fp4_pair_31;
                    uint16_t _e2m1_pair_h_31;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_31) : "f"(g_129.x), "f"(g_129.y));
                    _fp4_pair_31 = (uint32_t)_e2m1_pair_h_31;
                    unsigned int word_130 = _fp4_pair_28 | _fp4_pair_29 << 8 | _fp4_pair_30 << 16 | _fp4_pair_31 << 24;
                    unsigned int carry_131 = (word_130 & 2004318071) + 2004318071;
                    w_75[3] = word_130 & (carry_131 | 2004318071);
                    bool fodd_132 = (f & 1) != 0;
                    bool fhi_133 = (f & 2) != 0;
                    unsigned int t0_134 = w_75[0];
                    unsigned int t1_135 = w_75[1];
                    unsigned int t2_136 = w_75[2];
                    unsigned int t3_137 = w_75[3];
                    if (fodd_132) {
                        t0_134 = w_75[1];
                        t1_135 = w_75[0];
                        t2_136 = w_75[3];
                        t3_137 = w_75[2];
                    }
                    unsigned int q0_138 = t0_134;
                    unsigned int q1_139 = t1_135;
                    unsigned int q2_140 = t2_136;
                    unsigned int q3_141 = t3_137;
                    if (fhi_133) {
                        q0_138 = t2_136;
                        q1_139 = t3_137;
                        q2_140 = t0_134;
                        q3_141 = t1_135;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_69)), "r"(q0_138), "r"(q1_139), "r"(q2_140), "r"(q3_141) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_69)), "h"((uint16_t)(_e4m3x2_f32_1)) : "memory");
                    int pp2_142 = lane_1 + 32 * (3 * h + 2);
                    int base2_143 = xb_off + 64 * pp2_142;
                    float bm0_144 = __uint_as_float(bmp[2] << 16);
                    float bm1_145 = __uint_as_float(bmp[2] & 4294901760u);
                    float _mul_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(bm0_144), "f"(inv));
                    float _fdiv_rn_10 = __fdiv_rn(_mul_4, 6.0f);
                    float _mul_5;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(bm1_145), "f"(inv));
                    float _fdiv_rn_11 = __fdiv_rn(_mul_5, 6.0f);
                    uint16_t _e4m3x2_f32_2;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(_fdiv_rn_11), "f"(_fdiv_rn_10));
                    uint32_t _e4m3x2_to_f16x2_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_2) : "h"((uint16_t)(_e4m3x2_f32_2)));
                    float _cvt_f32_f16_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_4) : "h"((uint16_t)(_e4m3x2_to_f16x2_2)));
                    float _cvt_f32_f16_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_5) : "h"((uint16_t)(_e4m3x2_to_f16x2_2 >> 16)));
                    float _fdiv_rn_12 = __fdiv_rn(inv, _cvt_f32_f16_4);
                    float _fdiv_rn_13 = __fdiv_rn(inv, _cvt_f32_f16_5);
                    float m0_146 = 0.0f;
                    float m1_147 = 0.0f;
                    if (_cvt_f32_f16_4 > 0.0f) {
                        m0_146 = _fdiv_rn_12;
                    }
                    if (_cvt_f32_f16_5 > 0.0f) {
                        m1_147 = _fdiv_rn_13;
                    }
                    unsigned int w_148[4];
                    int cf_149 = f ^ 0;
                    unsigned int v4_150[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_150[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_150[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_150[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_150[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_143 + 16 * cf_149)));
                    float m_151 = m0_146;
                    if ((cf_149 & 2) != 0) {
                        m_151 = m1_147;
                    }
                    float2 _f2_40 = make_float2(m_151, m_151);
                    float2 mm_152 = _f2_40;
                    float2 _f2_41 = make_float2(__uint_as_float(v4_150[0] << 16), __uint_as_float(v4_150[0] & 4294901760u));
                    float2 pair_153 = _f2_41;
                    float2 g_154;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_154) : "l"(*(const unsigned long long*)&pair_153), "l"(*(const unsigned long long*)&mm_152));
                    uint32_t _fp4_pair_32;
                    uint16_t _e2m1_pair_h_32;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_32) : "f"(g_154.x), "f"(g_154.y));
                    _fp4_pair_32 = (uint32_t)_e2m1_pair_h_32;
                    float2 _f2_42 = make_float2(__uint_as_float(v4_150[1] << 16), __uint_as_float(v4_150[1] & 4294901760u));
                    float2 pair_155 = _f2_42;
                    float2 g_156;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_156) : "l"(*(const unsigned long long*)&pair_155), "l"(*(const unsigned long long*)&mm_152));
                    uint32_t _fp4_pair_33;
                    uint16_t _e2m1_pair_h_33;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_33) : "f"(g_156.x), "f"(g_156.y));
                    _fp4_pair_33 = (uint32_t)_e2m1_pair_h_33;
                    float2 _f2_43 = make_float2(__uint_as_float(v4_150[2] << 16), __uint_as_float(v4_150[2] & 4294901760u));
                    float2 pair_157 = _f2_43;
                    float2 g_158;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_158) : "l"(*(const unsigned long long*)&pair_157), "l"(*(const unsigned long long*)&mm_152));
                    uint32_t _fp4_pair_34;
                    uint16_t _e2m1_pair_h_34;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_34) : "f"(g_158.x), "f"(g_158.y));
                    _fp4_pair_34 = (uint32_t)_e2m1_pair_h_34;
                    float2 _f2_44 = make_float2(__uint_as_float(v4_150[3] << 16), __uint_as_float(v4_150[3] & 4294901760u));
                    float2 pair_159 = _f2_44;
                    float2 g_160;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_160) : "l"(*(const unsigned long long*)&pair_159), "l"(*(const unsigned long long*)&mm_152));
                    uint32_t _fp4_pair_35;
                    uint16_t _e2m1_pair_h_35;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_35) : "f"(g_160.x), "f"(g_160.y));
                    _fp4_pair_35 = (uint32_t)_e2m1_pair_h_35;
                    unsigned int word_161 = _fp4_pair_32 | _fp4_pair_33 << 8 | _fp4_pair_34 << 16 | _fp4_pair_35 << 24;
                    unsigned int carry_162 = (word_161 & 2004318071) + 2004318071;
                    w_148[0] = word_161 & (carry_162 | 2004318071);
                    int cf_163 = f ^ 1;
                    unsigned int v4_164[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_164[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_164[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_164[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_164[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_143 + 16 * cf_163)));
                    float m_165 = m0_146;
                    if ((cf_163 & 2) != 0) {
                        m_165 = m1_147;
                    }
                    float2 _f2_45 = make_float2(m_165, m_165);
                    float2 mm_166 = _f2_45;
                    float2 _f2_46 = make_float2(__uint_as_float(v4_164[0] << 16), __uint_as_float(v4_164[0] & 4294901760u));
                    float2 pair_167 = _f2_46;
                    float2 g_168;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_168) : "l"(*(const unsigned long long*)&pair_167), "l"(*(const unsigned long long*)&mm_166));
                    uint32_t _fp4_pair_36;
                    uint16_t _e2m1_pair_h_36;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_36) : "f"(g_168.x), "f"(g_168.y));
                    _fp4_pair_36 = (uint32_t)_e2m1_pair_h_36;
                    float2 _f2_47 = make_float2(__uint_as_float(v4_164[1] << 16), __uint_as_float(v4_164[1] & 4294901760u));
                    float2 pair_169 = _f2_47;
                    float2 g_170;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_170) : "l"(*(const unsigned long long*)&pair_169), "l"(*(const unsigned long long*)&mm_166));
                    uint32_t _fp4_pair_37;
                    uint16_t _e2m1_pair_h_37;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_37) : "f"(g_170.x), "f"(g_170.y));
                    _fp4_pair_37 = (uint32_t)_e2m1_pair_h_37;
                    float2 _f2_48 = make_float2(__uint_as_float(v4_164[2] << 16), __uint_as_float(v4_164[2] & 4294901760u));
                    float2 pair_171 = _f2_48;
                    float2 g_172;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_172) : "l"(*(const unsigned long long*)&pair_171), "l"(*(const unsigned long long*)&mm_166));
                    uint32_t _fp4_pair_38;
                    uint16_t _e2m1_pair_h_38;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_38) : "f"(g_172.x), "f"(g_172.y));
                    _fp4_pair_38 = (uint32_t)_e2m1_pair_h_38;
                    float2 _f2_49 = make_float2(__uint_as_float(v4_164[3] << 16), __uint_as_float(v4_164[3] & 4294901760u));
                    float2 pair_173 = _f2_49;
                    float2 g_174;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_174) : "l"(*(const unsigned long long*)&pair_173), "l"(*(const unsigned long long*)&mm_166));
                    uint32_t _fp4_pair_39;
                    uint16_t _e2m1_pair_h_39;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_39) : "f"(g_174.x), "f"(g_174.y));
                    _fp4_pair_39 = (uint32_t)_e2m1_pair_h_39;
                    unsigned int word_175 = _fp4_pair_36 | _fp4_pair_37 << 8 | _fp4_pair_38 << 16 | _fp4_pair_39 << 24;
                    unsigned int carry_176 = (word_175 & 2004318071) + 2004318071;
                    w_148[1] = word_175 & (carry_176 | 2004318071);
                    int cf_177 = f ^ 2;
                    unsigned int v4_178[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_178[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_178[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_178[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_178[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_143 + 16 * cf_177)));
                    float m_179 = m0_146;
                    if ((cf_177 & 2) != 0) {
                        m_179 = m1_147;
                    }
                    float2 _f2_50 = make_float2(m_179, m_179);
                    float2 mm_180 = _f2_50;
                    float2 _f2_51 = make_float2(__uint_as_float(v4_178[0] << 16), __uint_as_float(v4_178[0] & 4294901760u));
                    float2 pair_181 = _f2_51;
                    float2 g_182;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_182) : "l"(*(const unsigned long long*)&pair_181), "l"(*(const unsigned long long*)&mm_180));
                    uint32_t _fp4_pair_40;
                    uint16_t _e2m1_pair_h_40;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_40) : "f"(g_182.x), "f"(g_182.y));
                    _fp4_pair_40 = (uint32_t)_e2m1_pair_h_40;
                    float2 _f2_52 = make_float2(__uint_as_float(v4_178[1] << 16), __uint_as_float(v4_178[1] & 4294901760u));
                    float2 pair_183 = _f2_52;
                    float2 g_184;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_184) : "l"(*(const unsigned long long*)&pair_183), "l"(*(const unsigned long long*)&mm_180));
                    uint32_t _fp4_pair_41;
                    uint16_t _e2m1_pair_h_41;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_41) : "f"(g_184.x), "f"(g_184.y));
                    _fp4_pair_41 = (uint32_t)_e2m1_pair_h_41;
                    float2 _f2_53 = make_float2(__uint_as_float(v4_178[2] << 16), __uint_as_float(v4_178[2] & 4294901760u));
                    float2 pair_185 = _f2_53;
                    float2 g_186;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_186) : "l"(*(const unsigned long long*)&pair_185), "l"(*(const unsigned long long*)&mm_180));
                    uint32_t _fp4_pair_42;
                    uint16_t _e2m1_pair_h_42;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_42) : "f"(g_186.x), "f"(g_186.y));
                    _fp4_pair_42 = (uint32_t)_e2m1_pair_h_42;
                    float2 _f2_54 = make_float2(__uint_as_float(v4_178[3] << 16), __uint_as_float(v4_178[3] & 4294901760u));
                    float2 pair_187 = _f2_54;
                    float2 g_188;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_188) : "l"(*(const unsigned long long*)&pair_187), "l"(*(const unsigned long long*)&mm_180));
                    uint32_t _fp4_pair_43;
                    uint16_t _e2m1_pair_h_43;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_43) : "f"(g_188.x), "f"(g_188.y));
                    _fp4_pair_43 = (uint32_t)_e2m1_pair_h_43;
                    unsigned int word_189 = _fp4_pair_40 | _fp4_pair_41 << 8 | _fp4_pair_42 << 16 | _fp4_pair_43 << 24;
                    unsigned int carry_190 = (word_189 & 2004318071) + 2004318071;
                    w_148[2] = word_189 & (carry_190 | 2004318071);
                    int cf_191 = f ^ 3;
                    unsigned int v4_192[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_192[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_192[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_192[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_192[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_143 + 16 * cf_191)));
                    float m_193 = m0_146;
                    if ((cf_191 & 2) != 0) {
                        m_193 = m1_147;
                    }
                    float2 _f2_55 = make_float2(m_193, m_193);
                    float2 mm_194 = _f2_55;
                    float2 _f2_56 = make_float2(__uint_as_float(v4_192[0] << 16), __uint_as_float(v4_192[0] & 4294901760u));
                    float2 pair_195 = _f2_56;
                    float2 g_196;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_196) : "l"(*(const unsigned long long*)&pair_195), "l"(*(const unsigned long long*)&mm_194));
                    uint32_t _fp4_pair_44;
                    uint16_t _e2m1_pair_h_44;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_44) : "f"(g_196.x), "f"(g_196.y));
                    _fp4_pair_44 = (uint32_t)_e2m1_pair_h_44;
                    float2 _f2_57 = make_float2(__uint_as_float(v4_192[1] << 16), __uint_as_float(v4_192[1] & 4294901760u));
                    float2 pair_197 = _f2_57;
                    float2 g_198;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_198) : "l"(*(const unsigned long long*)&pair_197), "l"(*(const unsigned long long*)&mm_194));
                    uint32_t _fp4_pair_45;
                    uint16_t _e2m1_pair_h_45;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_45) : "f"(g_198.x), "f"(g_198.y));
                    _fp4_pair_45 = (uint32_t)_e2m1_pair_h_45;
                    float2 _f2_58 = make_float2(__uint_as_float(v4_192[2] << 16), __uint_as_float(v4_192[2] & 4294901760u));
                    float2 pair_199 = _f2_58;
                    float2 g_200;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_200) : "l"(*(const unsigned long long*)&pair_199), "l"(*(const unsigned long long*)&mm_194));
                    uint32_t _fp4_pair_46;
                    uint16_t _e2m1_pair_h_46;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_46) : "f"(g_200.x), "f"(g_200.y));
                    _fp4_pair_46 = (uint32_t)_e2m1_pair_h_46;
                    float2 _f2_59 = make_float2(__uint_as_float(v4_192[3] << 16), __uint_as_float(v4_192[3] & 4294901760u));
                    float2 pair_201 = _f2_59;
                    float2 g_202;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_202) : "l"(*(const unsigned long long*)&pair_201), "l"(*(const unsigned long long*)&mm_194));
                    uint32_t _fp4_pair_47;
                    uint16_t _e2m1_pair_h_47;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_47) : "f"(g_202.x), "f"(g_202.y));
                    _fp4_pair_47 = (uint32_t)_e2m1_pair_h_47;
                    unsigned int word_203 = _fp4_pair_44 | _fp4_pair_45 << 8 | _fp4_pair_46 << 16 | _fp4_pair_47 << 24;
                    unsigned int carry_204 = (word_203 & 2004318071) + 2004318071;
                    w_148[3] = word_203 & (carry_204 | 2004318071);
                    bool fodd_205 = (f & 1) != 0;
                    bool fhi_206 = (f & 2) != 0;
                    unsigned int t0_207 = w_148[0];
                    unsigned int t1_208 = w_148[1];
                    unsigned int t2_209 = w_148[2];
                    unsigned int t3_210 = w_148[3];
                    if (fodd_205) {
                        t0_207 = w_148[1];
                        t1_208 = w_148[0];
                        t2_209 = w_148[3];
                        t3_210 = w_148[2];
                    }
                    unsigned int q0_211 = t0_207;
                    unsigned int q1_212 = t1_208;
                    unsigned int q2_213 = t2_209;
                    unsigned int q3_214 = t3_210;
                    if (fhi_206) {
                        q0_211 = t2_209;
                        q1_212 = t3_210;
                        q2_213 = t0_207;
                        q3_214 = t1_208;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_142)), "r"(q0_211), "r"(q1_212), "r"(q2_213), "r"(q3_214) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_142)), "h"((uint16_t)(_e4m3x2_f32_2)) : "memory");
                    rs = rs_20;
                    if (h == 0 && lane_1 < 4) {
                        {
                            uint32_t _addr_48 = static_cast<uint32_t>(wsm_addr + (unsigned int)(ob_off + 3072 + 384 + 4 * lane_1));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_48), "f"(rs) : "memory");
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
                            void* _cpbulk_dst_49 = reinterpret_cast<void*>(peer_wire + (v_base + (rank64 * cap64 + row64) * 3072));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_49), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_50 = reinterpret_cast<void*>(peer_wire + (s_base + (rank64 * cap64 + row64) * 384));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_50), "r"(wsm_addr + (unsigned int)(ob_off + 3072)), "r"((uint32_t)(384))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_51 = reinterpret_cast<void*>(peer_rs + (m_base + (rank64 * cap64 + row64) * 16));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_51), "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 384)), "r"((uint32_t)(16))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
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
                    uint16_t _wire_h16_52_0;
                    uint16_t _wire_h16_52_1;
                    uint16_t _wire_h16_52_2;
                    uint32_t _wire_vpend_52 = 7u;
                    uint32_t _wire_spend_52 = 7u;
                    do {
                        if (_wire_vpend_52 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * (lane_1 + 32 * (3 * h)))))) : "memory");
                        }
                        if (_wire_vpend_52 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * (lane_1 + 32 * (3 * h + 1)))))) : "memory");
                        }
                        if (_wire_vpend_52 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[8]), "=r"(_fp4_wire_values_0[9]), "=r"(_fp4_wire_values_0[10]), "=r"(_fp4_wire_values_0[11]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * (lane_1 + 32 * (3 * h + 2)))))) : "memory");
                        }
                        if (_wire_spend_52 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_52_0) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * (lane_1 + 32 * (3 * h)))))) : "memory");
                            _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_52_0;
                        }
                        if (_wire_spend_52 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_52_1) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * (lane_1 + 32 * (3 * h + 1)))))) : "memory");
                            _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_52_1;
                        }
                        if (_wire_spend_52 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_52_2) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * (lane_1 + 32 * (3 * h + 2)))))) : "memory");
                            _fp4_wire_scales_0[2] = (uint32_t)_wire_h16_52_2;
                        }
                        _wire_vpend_52 = ((uint32_t)((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((_fp4_wire_values_0[8] & ~((_fp4_wire_values_0[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[9] & ~((_fp4_wire_values_0[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[10] & ~((_fp4_wire_values_0[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[11] & ~((_fp4_wire_values_0[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2);
                        _wire_spend_52 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[2] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2);
                    } while ((_wire_vpend_52 | _wire_spend_52) != 0u);
                    int _vote_0 = __any_sync(0xFFFFFFFF, rsw[0] == 2147483648u);
                    int pending = _vote_0;
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
                    uint32_t _e4m3x2_to_f16x2_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_3) : "h"((uint16_t)(a_sc)));
                    float _cvt_f32_f16_6;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_6) : "h"((uint16_t)(_e4m3x2_to_f16x2_3)));
                    float _cvt_f32_f16_7;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(_e4m3x2_to_f16x2_3 >> 16)));
                    float _mul_6;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_cvt_f32_f16_6), "f"(a_rs));
                    float _mul_7;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_cvt_f32_f16_7), "f"(a_rs));
                    float2 _f2_60 = make_float2(_mul_6, _mul_6);
                    float2 scale0 = _f2_60;
                    float2 _f2_61 = make_float2(_mul_7, _mul_7);
                    float2 scale1 = _f2_61;
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
                    float2 _f2_62 = make_float2(acc[0], acc[1]);
                    float2 a_0 = _f2_62;
                    float2 _f2_63 = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                    float2 q = _f2_63;
                    float2 r = fma_f32x2_rn_noftz(q, scale0, a_0);
                    acc[0] = r.x;
                    acc[1] = r.y;
                    float2 _f2_64 = make_float2(acc[2], acc[3]);
                    float2 a_1_1 = _f2_64;
                    float2 _f2_65 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                    float2 q_2 = _f2_65;
                    float2 r_3 = fma_f32x2_rn_noftz(q_2, scale0, a_1_1);
                    acc[2] = r_3.x;
                    acc[3] = r_3.y;
                    float2 _f2_66 = make_float2(acc[4], acc[5]);
                    float2 a_4 = _f2_66;
                    float2 _f2_67 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                    float2 q_5 = _f2_67;
                    float2 r_6 = fma_f32x2_rn_noftz(q_5, scale0, a_4);
                    acc[4] = r_6.x;
                    acc[5] = r_6.y;
                    float2 _f2_68 = make_float2(acc[6], acc[7]);
                    float2 a_7 = _f2_68;
                    float2 _f2_69 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                    float2 q_8 = _f2_69;
                    float2 r_9 = fma_f32x2_rn_noftz(q_8, scale0, a_7);
                    acc[6] = r_9.x;
                    acc[7] = r_9.y;
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
                    float2 _f2_70 = make_float2(acc[8], acc[9]);
                    float2 a_10 = _f2_70;
                    float2 _f2_71 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                    float2 q_11 = _f2_71;
                    float2 r_12 = fma_f32x2_rn_noftz(q_11, scale0, a_10);
                    acc[8] = r_12.x;
                    acc[9] = r_12.y;
                    float2 _f2_72 = make_float2(acc[10], acc[11]);
                    float2 a_13 = _f2_72;
                    float2 _f2_73 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                    float2 q_14 = _f2_73;
                    float2 r_15 = fma_f32x2_rn_noftz(q_14, scale0, a_13);
                    acc[10] = r_15.x;
                    acc[11] = r_15.y;
                    float2 _f2_74 = make_float2(acc[12], acc[13]);
                    float2 a_16 = _f2_74;
                    float2 _f2_75 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                    float2 q_17 = _f2_75;
                    float2 r_18 = fma_f32x2_rn_noftz(q_17, scale0, a_16);
                    acc[12] = r_18.x;
                    acc[13] = r_18.y;
                    float2 _f2_76 = make_float2(acc[14], acc[15]);
                    float2 a_19 = _f2_76;
                    float2 _f2_77 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                    float2 q_20 = _f2_77;
                    float2 r_21 = fma_f32x2_rn_noftz(q_20, scale0, a_19);
                    acc[14] = r_21.x;
                    acc[15] = r_21.y;
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
                    float2 _f2_78 = make_float2(acc[16], acc[17]);
                    float2 a_22 = _f2_78;
                    float2 _f2_79 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                    float2 q_23 = _f2_79;
                    float2 r_24 = fma_f32x2_rn_noftz(q_23, scale1, a_22);
                    acc[16] = r_24.x;
                    acc[17] = r_24.y;
                    float2 _f2_80 = make_float2(acc[18], acc[19]);
                    float2 a_25 = _f2_80;
                    float2 _f2_81 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                    float2 q_26 = _f2_81;
                    float2 r_27 = fma_f32x2_rn_noftz(q_26, scale1, a_25);
                    acc[18] = r_27.x;
                    acc[19] = r_27.y;
                    float2 _f2_82 = make_float2(acc[20], acc[21]);
                    float2 a_28 = _f2_82;
                    float2 _f2_83 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                    float2 q_29 = _f2_83;
                    float2 r_30 = fma_f32x2_rn_noftz(q_29, scale1, a_28);
                    acc[20] = r_30.x;
                    acc[21] = r_30.y;
                    float2 _f2_84 = make_float2(acc[22], acc[23]);
                    float2 a_31 = _f2_84;
                    float2 _f2_85 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                    float2 q_32 = _f2_85;
                    float2 r_33 = fma_f32x2_rn_noftz(q_32, scale1, a_31);
                    acc[22] = r_33.x;
                    acc[23] = r_33.y;
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
                    float2 _f2_86 = make_float2(acc[24], acc[25]);
                    float2 a_34 = _f2_86;
                    float2 _f2_87 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                    float2 q_35 = _f2_87;
                    float2 r_36 = fma_f32x2_rn_noftz(q_35, scale1, a_34);
                    acc[24] = r_36.x;
                    acc[25] = r_36.y;
                    float2 _f2_88 = make_float2(acc[26], acc[27]);
                    float2 a_37 = _f2_88;
                    float2 _f2_89 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                    float2 q_38 = _f2_89;
                    float2 r_39 = fma_f32x2_rn_noftz(q_38, scale1, a_37);
                    acc[26] = r_39.x;
                    acc[27] = r_39.y;
                    float2 _f2_90 = make_float2(acc[28], acc[29]);
                    float2 a_40 = _f2_90;
                    float2 _f2_91 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                    float2 q_41 = _f2_91;
                    float2 r_42 = fma_f32x2_rn_noftz(q_41, scale1, a_40);
                    acc[28] = r_42.x;
                    acc[29] = r_42.y;
                    float2 _f2_92 = make_float2(acc[30], acc[31]);
                    float2 a_43 = _f2_92;
                    float2 _f2_93 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                    float2 q_44 = _f2_93;
                    float2 r_45 = fma_f32x2_rn_noftz(q_44, scale1, a_43);
                    acc[30] = r_45.x;
                    acc[31] = r_45.y;
                    uint32_t _e4m3x2_to_f16x2_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_4) : "h"((uint16_t)(b_sc)));
                    float _cvt_f32_f16_8;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_8) : "h"((uint16_t)(_e4m3x2_to_f16x2_4)));
                    float _cvt_f32_f16_9;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_9) : "h"((uint16_t)(_e4m3x2_to_f16x2_4 >> 16)));
                    float _mul_8;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_cvt_f32_f16_8), "f"(b_rs));
                    float _mul_9;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_cvt_f32_f16_9), "f"(b_rs));
                    float2 _f2_94 = make_float2(_mul_8, _mul_8);
                    float2 scale0_46 = _f2_94;
                    float2 _f2_95 = make_float2(_mul_9, _mul_9);
                    float2 scale1_47 = _f2_95;
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
                    float2 _f2_96 = make_float2(acc[0], acc[1]);
                    float2 a_48 = _f2_96;
                    float2 _f2_97 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                    float2 q_49 = _f2_97;
                    float2 r_50 = fma_f32x2_rn_noftz(q_49, scale0_46, a_48);
                    acc[0] = r_50.x;
                    acc[1] = r_50.y;
                    float2 _f2_98 = make_float2(acc[2], acc[3]);
                    float2 a_51 = _f2_98;
                    float2 _f2_99 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                    float2 q_52 = _f2_99;
                    float2 r_53 = fma_f32x2_rn_noftz(q_52, scale0_46, a_51);
                    acc[2] = r_53.x;
                    acc[3] = r_53.y;
                    float2 _f2_100 = make_float2(acc[4], acc[5]);
                    float2 a_54 = _f2_100;
                    float2 _f2_101 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                    float2 q_55 = _f2_101;
                    float2 r_56 = fma_f32x2_rn_noftz(q_55, scale0_46, a_54);
                    acc[4] = r_56.x;
                    acc[5] = r_56.y;
                    float2 _f2_102 = make_float2(acc[6], acc[7]);
                    float2 a_57 = _f2_102;
                    float2 _f2_103 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                    float2 q_58 = _f2_103;
                    float2 r_59 = fma_f32x2_rn_noftz(q_58, scale0_46, a_57);
                    acc[6] = r_59.x;
                    acc[7] = r_59.y;
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
                    float2 _f2_104 = make_float2(acc[8], acc[9]);
                    float2 a_60 = _f2_104;
                    float2 _f2_105 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                    float2 q_61 = _f2_105;
                    float2 r_62 = fma_f32x2_rn_noftz(q_61, scale0_46, a_60);
                    acc[8] = r_62.x;
                    acc[9] = r_62.y;
                    float2 _f2_106 = make_float2(acc[10], acc[11]);
                    float2 a_63 = _f2_106;
                    float2 _f2_107 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                    float2 q_64 = _f2_107;
                    float2 r_65 = fma_f32x2_rn_noftz(q_64, scale0_46, a_63);
                    acc[10] = r_65.x;
                    acc[11] = r_65.y;
                    float2 _f2_108 = make_float2(acc[12], acc[13]);
                    float2 a_66 = _f2_108;
                    float2 _f2_109 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                    float2 q_67 = _f2_109;
                    float2 r_68 = fma_f32x2_rn_noftz(q_67, scale0_46, a_66);
                    acc[12] = r_68.x;
                    acc[13] = r_68.y;
                    float2 _f2_110 = make_float2(acc[14], acc[15]);
                    float2 a_69 = _f2_110;
                    float2 _f2_111 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                    float2 q_70 = _f2_111;
                    float2 r_71 = fma_f32x2_rn_noftz(q_70, scale0_46, a_69);
                    acc[14] = r_71.x;
                    acc[15] = r_71.y;
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
                    float2 _f2_112 = make_float2(acc[16], acc[17]);
                    float2 a_72 = _f2_112;
                    float2 _f2_113 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                    float2 q_73 = _f2_113;
                    float2 r_74 = fma_f32x2_rn_noftz(q_73, scale1_47, a_72);
                    acc[16] = r_74.x;
                    acc[17] = r_74.y;
                    float2 _f2_114 = make_float2(acc[18], acc[19]);
                    float2 a_75 = _f2_114;
                    float2 _f2_115 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                    float2 q_76 = _f2_115;
                    float2 r_77 = fma_f32x2_rn_noftz(q_76, scale1_47, a_75);
                    acc[18] = r_77.x;
                    acc[19] = r_77.y;
                    float2 _f2_116 = make_float2(acc[20], acc[21]);
                    float2 a_78 = _f2_116;
                    float2 _f2_117 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                    float2 q_79 = _f2_117;
                    float2 r_80 = fma_f32x2_rn_noftz(q_79, scale1_47, a_78);
                    acc[20] = r_80.x;
                    acc[21] = r_80.y;
                    float2 _f2_118 = make_float2(acc[22], acc[23]);
                    float2 a_81 = _f2_118;
                    float2 _f2_119 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                    float2 q_82 = _f2_119;
                    float2 r_83 = fma_f32x2_rn_noftz(q_82, scale1_47, a_81);
                    acc[22] = r_83.x;
                    acc[23] = r_83.y;
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
                    float2 _f2_120 = make_float2(acc[24], acc[25]);
                    float2 a_84 = _f2_120;
                    float2 _f2_121 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                    float2 q_85 = _f2_121;
                    float2 r_86 = fma_f32x2_rn_noftz(q_85, scale1_47, a_84);
                    acc[24] = r_86.x;
                    acc[25] = r_86.y;
                    float2 _f2_122 = make_float2(acc[26], acc[27]);
                    float2 a_87 = _f2_122;
                    float2 _f2_123 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                    float2 q_88 = _f2_123;
                    float2 r_89 = fma_f32x2_rn_noftz(q_88, scale1_47, a_87);
                    acc[26] = r_89.x;
                    acc[27] = r_89.y;
                    float2 _f2_124 = make_float2(acc[28], acc[29]);
                    float2 a_90 = _f2_124;
                    float2 _f2_125 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                    float2 q_91 = _f2_125;
                    float2 r_92 = fma_f32x2_rn_noftz(q_91, scale1_47, a_90);
                    acc[28] = r_92.x;
                    acc[29] = r_92.y;
                    float2 _f2_126 = make_float2(acc[30], acc[31]);
                    float2 a_93 = _f2_126;
                    float2 _f2_127 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                    float2 q_94 = _f2_127;
                    float2 r_95 = fma_f32x2_rn_noftz(q_94, scale1_47, a_93);
                    acc[30] = r_95.x;
                    acc[31] = r_95.y;
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
                    int c = lane_1;
                    int l = c >> 2;
                    unsigned int w4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l + (c & 3 ^ l >> 1 & 3)))));
                    long long idx = drow * 3072 + (long long)(4 * (128 * (3 * h) + c));
                    {
                        int4 _iv4 = make_int4(w4[0 + 0], w4[0 + 1], w4[0 + 2], w4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx) = _iv4;
                    }
                    int c_96 = lane_1 + 32;
                    int l_97 = c_96 >> 2;
                    unsigned int w4_98[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_98[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_98[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_98[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_98[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_97 + (c_96 & 3 ^ l_97 >> 1 & 3)))));
                    long long idx_99 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_96));
                    {
                        int4 _iv4 = make_int4(w4_98[0 + 0], w4_98[0 + 1], w4_98[0 + 2], w4_98[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_99) = _iv4;
                    }
                    int c_100 = lane_1 + 64;
                    int l_101 = c_100 >> 2;
                    unsigned int w4_102[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_102[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_102[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_102[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_102[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_101 + (c_100 & 3 ^ l_101 >> 1 & 3)))));
                    long long idx_103 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_100));
                    {
                        int4 _iv4 = make_int4(w4_102[0 + 0], w4_102[0 + 1], w4_102[0 + 2], w4_102[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_103) = _iv4;
                    }
                    int c_104 = lane_1 + 96;
                    int l_105 = c_104 >> 2;
                    unsigned int w4_106[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_106[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_106[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_106[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_106[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_105 + (c_104 & 3 ^ l_105 >> 1 & 3)))));
                    long long idx_107 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_104));
                    {
                        int4 _iv4 = make_int4(w4_106[0 + 0], w4_106[0 + 1], w4_106[0 + 2], w4_106[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_107) = _iv4;
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
                    uint32_t _e4m3x2_to_f16x2_5;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_5) : "h"((uint16_t)(a_sc_110)));
                    float _cvt_f32_f16_10;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_10) : "h"((uint16_t)(_e4m3x2_to_f16x2_5)));
                    float _cvt_f32_f16_11;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_11) : "h"((uint16_t)(_e4m3x2_to_f16x2_5 >> 16)));
                    float _mul_10;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(_cvt_f32_f16_10), "f"(a_rs_112));
                    float _mul_11;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_cvt_f32_f16_11), "f"(a_rs_112));
                    float2 _f2_128 = make_float2(_mul_10, _mul_10);
                    float2 scale0_115 = _f2_128;
                    float2 _f2_129 = make_float2(_mul_11, _mul_11);
                    float2 scale1_116 = _f2_129;
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
                    float2 _f2_130 = make_float2(acc_114[0], acc_114[1]);
                    float2 a_117 = _f2_130;
                    float2 _f2_131 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                    float2 q_118 = _f2_131;
                    float2 r_119 = fma_f32x2_rn_noftz(q_118, scale0_115, a_117);
                    acc_114[0] = r_119.x;
                    acc_114[1] = r_119.y;
                    float2 _f2_132 = make_float2(acc_114[2], acc_114[3]);
                    float2 a_120 = _f2_132;
                    float2 _f2_133 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                    float2 q_121 = _f2_133;
                    float2 r_122 = fma_f32x2_rn_noftz(q_121, scale0_115, a_120);
                    acc_114[2] = r_122.x;
                    acc_114[3] = r_122.y;
                    float2 _f2_134 = make_float2(acc_114[4], acc_114[5]);
                    float2 a_123 = _f2_134;
                    float2 _f2_135 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                    float2 q_124 = _f2_135;
                    float2 r_125 = fma_f32x2_rn_noftz(q_124, scale0_115, a_123);
                    acc_114[4] = r_125.x;
                    acc_114[5] = r_125.y;
                    float2 _f2_136 = make_float2(acc_114[6], acc_114[7]);
                    float2 a_126 = _f2_136;
                    float2 _f2_137 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                    float2 q_127 = _f2_137;
                    float2 r_128 = fma_f32x2_rn_noftz(q_127, scale0_115, a_126);
                    acc_114[6] = r_128.x;
                    acc_114[7] = r_128.y;
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
                    float2 _f2_138 = make_float2(acc_114[8], acc_114[9]);
                    float2 a_129 = _f2_138;
                    float2 _f2_139 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                    float2 q_130 = _f2_139;
                    float2 r_131 = fma_f32x2_rn_noftz(q_130, scale0_115, a_129);
                    acc_114[8] = r_131.x;
                    acc_114[9] = r_131.y;
                    float2 _f2_140 = make_float2(acc_114[10], acc_114[11]);
                    float2 a_132 = _f2_140;
                    float2 _f2_141 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                    float2 q_133 = _f2_141;
                    float2 r_134 = fma_f32x2_rn_noftz(q_133, scale0_115, a_132);
                    acc_114[10] = r_134.x;
                    acc_114[11] = r_134.y;
                    float2 _f2_142 = make_float2(acc_114[12], acc_114[13]);
                    float2 a_135 = _f2_142;
                    float2 _f2_143 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                    float2 q_136 = _f2_143;
                    float2 r_137 = fma_f32x2_rn_noftz(q_136, scale0_115, a_135);
                    acc_114[12] = r_137.x;
                    acc_114[13] = r_137.y;
                    float2 _f2_144 = make_float2(acc_114[14], acc_114[15]);
                    float2 a_138 = _f2_144;
                    float2 _f2_145 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                    float2 q_139 = _f2_145;
                    float2 r_140 = fma_f32x2_rn_noftz(q_139, scale0_115, a_138);
                    acc_114[14] = r_140.x;
                    acc_114[15] = r_140.y;
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
                    float2 _f2_146 = make_float2(acc_114[16], acc_114[17]);
                    float2 a_141 = _f2_146;
                    float2 _f2_147 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                    float2 q_142 = _f2_147;
                    float2 r_143 = fma_f32x2_rn_noftz(q_142, scale1_116, a_141);
                    acc_114[16] = r_143.x;
                    acc_114[17] = r_143.y;
                    float2 _f2_148 = make_float2(acc_114[18], acc_114[19]);
                    float2 a_144 = _f2_148;
                    float2 _f2_149 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                    float2 q_145 = _f2_149;
                    float2 r_146 = fma_f32x2_rn_noftz(q_145, scale1_116, a_144);
                    acc_114[18] = r_146.x;
                    acc_114[19] = r_146.y;
                    float2 _f2_150 = make_float2(acc_114[20], acc_114[21]);
                    float2 a_147 = _f2_150;
                    float2 _f2_151 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                    float2 q_148 = _f2_151;
                    float2 r_149 = fma_f32x2_rn_noftz(q_148, scale1_116, a_147);
                    acc_114[20] = r_149.x;
                    acc_114[21] = r_149.y;
                    float2 _f2_152 = make_float2(acc_114[22], acc_114[23]);
                    float2 a_150 = _f2_152;
                    float2 _f2_153 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                    float2 q_151 = _f2_153;
                    float2 r_152 = fma_f32x2_rn_noftz(q_151, scale1_116, a_150);
                    acc_114[22] = r_152.x;
                    acc_114[23] = r_152.y;
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
                    float2 _f2_154 = make_float2(acc_114[24], acc_114[25]);
                    float2 a_153 = _f2_154;
                    float2 _f2_155 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                    float2 q_154 = _f2_155;
                    float2 r_155 = fma_f32x2_rn_noftz(q_154, scale1_116, a_153);
                    acc_114[24] = r_155.x;
                    acc_114[25] = r_155.y;
                    float2 _f2_156 = make_float2(acc_114[26], acc_114[27]);
                    float2 a_156 = _f2_156;
                    float2 _f2_157 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                    float2 q_157 = _f2_157;
                    float2 r_158 = fma_f32x2_rn_noftz(q_157, scale1_116, a_156);
                    acc_114[26] = r_158.x;
                    acc_114[27] = r_158.y;
                    float2 _f2_158 = make_float2(acc_114[28], acc_114[29]);
                    float2 a_159 = _f2_158;
                    float2 _f2_159 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                    float2 q_160 = _f2_159;
                    float2 r_161 = fma_f32x2_rn_noftz(q_160, scale1_116, a_159);
                    acc_114[28] = r_161.x;
                    acc_114[29] = r_161.y;
                    float2 _f2_160 = make_float2(acc_114[30], acc_114[31]);
                    float2 a_162 = _f2_160;
                    float2 _f2_161 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                    float2 q_163 = _f2_161;
                    float2 r_164 = fma_f32x2_rn_noftz(q_163, scale1_116, a_162);
                    acc_114[30] = r_164.x;
                    acc_114[31] = r_164.y;
                    uint32_t _e4m3x2_to_f16x2_6;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_6) : "h"((uint16_t)(b_sc_111)));
                    float _cvt_f32_f16_12;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_12) : "h"((uint16_t)(_e4m3x2_to_f16x2_6)));
                    float _cvt_f32_f16_13;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_13) : "h"((uint16_t)(_e4m3x2_to_f16x2_6 >> 16)));
                    float _mul_12;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(_cvt_f32_f16_12), "f"(b_rs_113));
                    float _mul_13;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(_cvt_f32_f16_13), "f"(b_rs_113));
                    float2 _f2_162 = make_float2(_mul_12, _mul_12);
                    float2 scale0_165 = _f2_162;
                    float2 _f2_163 = make_float2(_mul_13, _mul_13);
                    float2 scale1_166 = _f2_163;
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
                    float2 _f2_164 = make_float2(acc_114[0], acc_114[1]);
                    float2 a_167 = _f2_164;
                    float2 _f2_165 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                    float2 q_168 = _f2_165;
                    float2 r_169 = fma_f32x2_rn_noftz(q_168, scale0_165, a_167);
                    acc_114[0] = r_169.x;
                    acc_114[1] = r_169.y;
                    float2 _f2_166 = make_float2(acc_114[2], acc_114[3]);
                    float2 a_170 = _f2_166;
                    float2 _f2_167 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                    float2 q_171 = _f2_167;
                    float2 r_172 = fma_f32x2_rn_noftz(q_171, scale0_165, a_170);
                    acc_114[2] = r_172.x;
                    acc_114[3] = r_172.y;
                    float2 _f2_168 = make_float2(acc_114[4], acc_114[5]);
                    float2 a_173 = _f2_168;
                    float2 _f2_169 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                    float2 q_174 = _f2_169;
                    float2 r_175 = fma_f32x2_rn_noftz(q_174, scale0_165, a_173);
                    acc_114[4] = r_175.x;
                    acc_114[5] = r_175.y;
                    float2 _f2_170 = make_float2(acc_114[6], acc_114[7]);
                    float2 a_176 = _f2_170;
                    float2 _f2_171 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                    float2 q_177 = _f2_171;
                    float2 r_178 = fma_f32x2_rn_noftz(q_177, scale0_165, a_176);
                    acc_114[6] = r_178.x;
                    acc_114[7] = r_178.y;
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
                    float2 _f2_172 = make_float2(acc_114[8], acc_114[9]);
                    float2 a_179 = _f2_172;
                    float2 _f2_173 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                    float2 q_180 = _f2_173;
                    float2 r_181 = fma_f32x2_rn_noftz(q_180, scale0_165, a_179);
                    acc_114[8] = r_181.x;
                    acc_114[9] = r_181.y;
                    float2 _f2_174 = make_float2(acc_114[10], acc_114[11]);
                    float2 a_182 = _f2_174;
                    float2 _f2_175 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                    float2 q_183 = _f2_175;
                    float2 r_184 = fma_f32x2_rn_noftz(q_183, scale0_165, a_182);
                    acc_114[10] = r_184.x;
                    acc_114[11] = r_184.y;
                    float2 _f2_176 = make_float2(acc_114[12], acc_114[13]);
                    float2 a_185 = _f2_176;
                    float2 _f2_177 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                    float2 q_186 = _f2_177;
                    float2 r_187 = fma_f32x2_rn_noftz(q_186, scale0_165, a_185);
                    acc_114[12] = r_187.x;
                    acc_114[13] = r_187.y;
                    float2 _f2_178 = make_float2(acc_114[14], acc_114[15]);
                    float2 a_188 = _f2_178;
                    float2 _f2_179 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                    float2 q_189 = _f2_179;
                    float2 r_190 = fma_f32x2_rn_noftz(q_189, scale0_165, a_188);
                    acc_114[14] = r_190.x;
                    acc_114[15] = r_190.y;
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
                    float2 _f2_180 = make_float2(acc_114[16], acc_114[17]);
                    float2 a_191 = _f2_180;
                    float2 _f2_181 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                    float2 q_192 = _f2_181;
                    float2 r_193 = fma_f32x2_rn_noftz(q_192, scale1_166, a_191);
                    acc_114[16] = r_193.x;
                    acc_114[17] = r_193.y;
                    float2 _f2_182 = make_float2(acc_114[18], acc_114[19]);
                    float2 a_194 = _f2_182;
                    float2 _f2_183 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                    float2 q_195 = _f2_183;
                    float2 r_196 = fma_f32x2_rn_noftz(q_195, scale1_166, a_194);
                    acc_114[18] = r_196.x;
                    acc_114[19] = r_196.y;
                    float2 _f2_184 = make_float2(acc_114[20], acc_114[21]);
                    float2 a_197 = _f2_184;
                    float2 _f2_185 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                    float2 q_198 = _f2_185;
                    float2 r_199 = fma_f32x2_rn_noftz(q_198, scale1_166, a_197);
                    acc_114[20] = r_199.x;
                    acc_114[21] = r_199.y;
                    float2 _f2_186 = make_float2(acc_114[22], acc_114[23]);
                    float2 a_200 = _f2_186;
                    float2 _f2_187 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                    float2 q_201 = _f2_187;
                    float2 r_202 = fma_f32x2_rn_noftz(q_201, scale1_166, a_200);
                    acc_114[22] = r_202.x;
                    acc_114[23] = r_202.y;
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
                    float2 _f2_188 = make_float2(acc_114[24], acc_114[25]);
                    float2 a_203 = _f2_188;
                    float2 _f2_189 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                    float2 q_204 = _f2_189;
                    float2 r_205 = fma_f32x2_rn_noftz(q_204, scale1_166, a_203);
                    acc_114[24] = r_205.x;
                    acc_114[25] = r_205.y;
                    float2 _f2_190 = make_float2(acc_114[26], acc_114[27]);
                    float2 a_206 = _f2_190;
                    float2 _f2_191 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                    float2 q_207 = _f2_191;
                    float2 r_208 = fma_f32x2_rn_noftz(q_207, scale1_166, a_206);
                    acc_114[26] = r_208.x;
                    acc_114[27] = r_208.y;
                    float2 _f2_192 = make_float2(acc_114[28], acc_114[29]);
                    float2 a_209 = _f2_192;
                    float2 _f2_193 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                    float2 q_210 = _f2_193;
                    float2 r_211 = fma_f32x2_rn_noftz(q_210, scale1_166, a_209);
                    acc_114[28] = r_211.x;
                    acc_114[29] = r_211.y;
                    float2 _f2_194 = make_float2(acc_114[30], acc_114[31]);
                    float2 a_212 = _f2_194;
                    float2 _f2_195 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                    float2 q_213 = _f2_195;
                    float2 r_214 = fma_f32x2_rn_noftz(q_213, scale1_166, a_212);
                    acc_114[30] = r_214.x;
                    acc_114[31] = r_214.y;
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
                    int c_215 = lane_1;
                    int l_216 = c_215 >> 2;
                    unsigned int w4_217[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_217[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_217[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_217[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_217[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_216 + (c_215 & 3 ^ l_216 >> 1 & 3)))));
                    long long idx_218 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_215));
                    {
                        int4 _iv4 = make_int4(w4_217[0 + 0], w4_217[0 + 1], w4_217[0 + 2], w4_217[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_218) = _iv4;
                    }
                    int c_219 = lane_1 + 32;
                    int l_220 = c_219 >> 2;
                    unsigned int w4_221[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_221[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_221[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_221[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_221[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_220 + (c_219 & 3 ^ l_220 >> 1 & 3)))));
                    long long idx_222 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_219));
                    {
                        int4 _iv4 = make_int4(w4_221[0 + 0], w4_221[0 + 1], w4_221[0 + 2], w4_221[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_222) = _iv4;
                    }
                    int c_223 = lane_1 + 64;
                    int l_224 = c_223 >> 2;
                    unsigned int w4_225[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_225[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_225[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_225[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_225[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_224 + (c_223 & 3 ^ l_224 >> 1 & 3)))));
                    long long idx_226 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_223));
                    {
                        int4 _iv4 = make_int4(w4_225[0 + 0], w4_225[0 + 1], w4_225[0 + 2], w4_225[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_226) = _iv4;
                    }
                    int c_227 = lane_1 + 96;
                    int l_228 = c_227 >> 2;
                    unsigned int w4_229[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_229[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_229[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_229[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_229[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_228 + (c_227 & 3 ^ l_228 >> 1 & 3)))));
                    long long idx_230 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_227));
                    {
                        int4 _iv4 = make_int4(w4_229[0 + 0], w4_229[0 + 1], w4_229[0 + 2], w4_229[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_230) = _iv4;
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
                    uint32_t _e4m3x2_to_f16x2_7;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_7) : "h"((uint16_t)(a_sc_233)));
                    float _cvt_f32_f16_14;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_14) : "h"((uint16_t)(_e4m3x2_to_f16x2_7)));
                    float _cvt_f32_f16_15;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_15) : "h"((uint16_t)(_e4m3x2_to_f16x2_7 >> 16)));
                    float _mul_14;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(_cvt_f32_f16_14), "f"(a_rs_235));
                    float _mul_15;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(_cvt_f32_f16_15), "f"(a_rs_235));
                    float2 _f2_196 = make_float2(_mul_14, _mul_14);
                    float2 scale0_238 = _f2_196;
                    float2 _f2_197 = make_float2(_mul_15, _mul_15);
                    float2 scale1_239 = _f2_197;
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
                    float2 _f2_198 = make_float2(acc_237[0], acc_237[1]);
                    float2 a_240 = _f2_198;
                    float2 _f2_199 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                    float2 q_241 = _f2_199;
                    float2 r_242 = fma_f32x2_rn_noftz(q_241, scale0_238, a_240);
                    acc_237[0] = r_242.x;
                    acc_237[1] = r_242.y;
                    float2 _f2_200 = make_float2(acc_237[2], acc_237[3]);
                    float2 a_243 = _f2_200;
                    float2 _f2_201 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                    float2 q_244 = _f2_201;
                    float2 r_245 = fma_f32x2_rn_noftz(q_244, scale0_238, a_243);
                    acc_237[2] = r_245.x;
                    acc_237[3] = r_245.y;
                    float2 _f2_202 = make_float2(acc_237[4], acc_237[5]);
                    float2 a_246 = _f2_202;
                    float2 _f2_203 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                    float2 q_247 = _f2_203;
                    float2 r_248 = fma_f32x2_rn_noftz(q_247, scale0_238, a_246);
                    acc_237[4] = r_248.x;
                    acc_237[5] = r_248.y;
                    float2 _f2_204 = make_float2(acc_237[6], acc_237[7]);
                    float2 a_249 = _f2_204;
                    float2 _f2_205 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                    float2 q_250 = _f2_205;
                    float2 r_251 = fma_f32x2_rn_noftz(q_250, scale0_238, a_249);
                    acc_237[6] = r_251.x;
                    acc_237[7] = r_251.y;
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
                    float2 _f2_206 = make_float2(acc_237[8], acc_237[9]);
                    float2 a_252 = _f2_206;
                    float2 _f2_207 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                    float2 q_253 = _f2_207;
                    float2 r_254 = fma_f32x2_rn_noftz(q_253, scale0_238, a_252);
                    acc_237[8] = r_254.x;
                    acc_237[9] = r_254.y;
                    float2 _f2_208 = make_float2(acc_237[10], acc_237[11]);
                    float2 a_255 = _f2_208;
                    float2 _f2_209 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                    float2 q_256 = _f2_209;
                    float2 r_257 = fma_f32x2_rn_noftz(q_256, scale0_238, a_255);
                    acc_237[10] = r_257.x;
                    acc_237[11] = r_257.y;
                    float2 _f2_210 = make_float2(acc_237[12], acc_237[13]);
                    float2 a_258 = _f2_210;
                    float2 _f2_211 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                    float2 q_259 = _f2_211;
                    float2 r_260 = fma_f32x2_rn_noftz(q_259, scale0_238, a_258);
                    acc_237[12] = r_260.x;
                    acc_237[13] = r_260.y;
                    float2 _f2_212 = make_float2(acc_237[14], acc_237[15]);
                    float2 a_261 = _f2_212;
                    float2 _f2_213 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                    float2 q_262 = _f2_213;
                    float2 r_263 = fma_f32x2_rn_noftz(q_262, scale0_238, a_261);
                    acc_237[14] = r_263.x;
                    acc_237[15] = r_263.y;
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
                    float2 _f2_214 = make_float2(acc_237[16], acc_237[17]);
                    float2 a_264 = _f2_214;
                    float2 _f2_215 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                    float2 q_265 = _f2_215;
                    float2 r_266 = fma_f32x2_rn_noftz(q_265, scale1_239, a_264);
                    acc_237[16] = r_266.x;
                    acc_237[17] = r_266.y;
                    float2 _f2_216 = make_float2(acc_237[18], acc_237[19]);
                    float2 a_267 = _f2_216;
                    float2 _f2_217 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                    float2 q_268 = _f2_217;
                    float2 r_269 = fma_f32x2_rn_noftz(q_268, scale1_239, a_267);
                    acc_237[18] = r_269.x;
                    acc_237[19] = r_269.y;
                    float2 _f2_218 = make_float2(acc_237[20], acc_237[21]);
                    float2 a_270 = _f2_218;
                    float2 _f2_219 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                    float2 q_271 = _f2_219;
                    float2 r_272 = fma_f32x2_rn_noftz(q_271, scale1_239, a_270);
                    acc_237[20] = r_272.x;
                    acc_237[21] = r_272.y;
                    float2 _f2_220 = make_float2(acc_237[22], acc_237[23]);
                    float2 a_273 = _f2_220;
                    float2 _f2_221 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                    float2 q_274 = _f2_221;
                    float2 r_275 = fma_f32x2_rn_noftz(q_274, scale1_239, a_273);
                    acc_237[22] = r_275.x;
                    acc_237[23] = r_275.y;
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
                    float2 _f2_222 = make_float2(acc_237[24], acc_237[25]);
                    float2 a_276 = _f2_222;
                    float2 _f2_223 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                    float2 q_277 = _f2_223;
                    float2 r_278 = fma_f32x2_rn_noftz(q_277, scale1_239, a_276);
                    acc_237[24] = r_278.x;
                    acc_237[25] = r_278.y;
                    float2 _f2_224 = make_float2(acc_237[26], acc_237[27]);
                    float2 a_279 = _f2_224;
                    float2 _f2_225 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                    float2 q_280 = _f2_225;
                    float2 r_281 = fma_f32x2_rn_noftz(q_280, scale1_239, a_279);
                    acc_237[26] = r_281.x;
                    acc_237[27] = r_281.y;
                    float2 _f2_226 = make_float2(acc_237[28], acc_237[29]);
                    float2 a_282 = _f2_226;
                    float2 _f2_227 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                    float2 q_283 = _f2_227;
                    float2 r_284 = fma_f32x2_rn_noftz(q_283, scale1_239, a_282);
                    acc_237[28] = r_284.x;
                    acc_237[29] = r_284.y;
                    float2 _f2_228 = make_float2(acc_237[30], acc_237[31]);
                    float2 a_285 = _f2_228;
                    float2 _f2_229 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                    float2 q_286 = _f2_229;
                    float2 r_287 = fma_f32x2_rn_noftz(q_286, scale1_239, a_285);
                    acc_237[30] = r_287.x;
                    acc_237[31] = r_287.y;
                    uint32_t _e4m3x2_to_f16x2_8;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_8) : "h"((uint16_t)(b_sc_234)));
                    float _cvt_f32_f16_16;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_16) : "h"((uint16_t)(_e4m3x2_to_f16x2_8)));
                    float _cvt_f32_f16_17;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_17) : "h"((uint16_t)(_e4m3x2_to_f16x2_8 >> 16)));
                    float _mul_16;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_cvt_f32_f16_16), "f"(b_rs_236));
                    float _mul_17;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_cvt_f32_f16_17), "f"(b_rs_236));
                    float2 _f2_230 = make_float2(_mul_16, _mul_16);
                    float2 scale0_288 = _f2_230;
                    float2 _f2_231 = make_float2(_mul_17, _mul_17);
                    float2 scale1_289 = _f2_231;
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
                    float2 _f2_232 = make_float2(acc_237[0], acc_237[1]);
                    float2 a_290 = _f2_232;
                    float2 _f2_233 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                    float2 q_291 = _f2_233;
                    float2 r_292 = fma_f32x2_rn_noftz(q_291, scale0_288, a_290);
                    acc_237[0] = r_292.x;
                    acc_237[1] = r_292.y;
                    float2 _f2_234 = make_float2(acc_237[2], acc_237[3]);
                    float2 a_293 = _f2_234;
                    float2 _f2_235 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                    float2 q_294 = _f2_235;
                    float2 r_295 = fma_f32x2_rn_noftz(q_294, scale0_288, a_293);
                    acc_237[2] = r_295.x;
                    acc_237[3] = r_295.y;
                    float2 _f2_236 = make_float2(acc_237[4], acc_237[5]);
                    float2 a_296 = _f2_236;
                    float2 _f2_237 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                    float2 q_297 = _f2_237;
                    float2 r_298 = fma_f32x2_rn_noftz(q_297, scale0_288, a_296);
                    acc_237[4] = r_298.x;
                    acc_237[5] = r_298.y;
                    float2 _f2_238 = make_float2(acc_237[6], acc_237[7]);
                    float2 a_299 = _f2_238;
                    float2 _f2_239 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                    float2 q_300 = _f2_239;
                    float2 r_301 = fma_f32x2_rn_noftz(q_300, scale0_288, a_299);
                    acc_237[6] = r_301.x;
                    acc_237[7] = r_301.y;
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
                    float2 _f2_240 = make_float2(acc_237[8], acc_237[9]);
                    float2 a_302 = _f2_240;
                    float2 _f2_241 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                    float2 q_303 = _f2_241;
                    float2 r_304 = fma_f32x2_rn_noftz(q_303, scale0_288, a_302);
                    acc_237[8] = r_304.x;
                    acc_237[9] = r_304.y;
                    float2 _f2_242 = make_float2(acc_237[10], acc_237[11]);
                    float2 a_305 = _f2_242;
                    float2 _f2_243 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                    float2 q_306 = _f2_243;
                    float2 r_307 = fma_f32x2_rn_noftz(q_306, scale0_288, a_305);
                    acc_237[10] = r_307.x;
                    acc_237[11] = r_307.y;
                    float2 _f2_244 = make_float2(acc_237[12], acc_237[13]);
                    float2 a_308 = _f2_244;
                    float2 _f2_245 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                    float2 q_309 = _f2_245;
                    float2 r_310 = fma_f32x2_rn_noftz(q_309, scale0_288, a_308);
                    acc_237[12] = r_310.x;
                    acc_237[13] = r_310.y;
                    float2 _f2_246 = make_float2(acc_237[14], acc_237[15]);
                    float2 a_311 = _f2_246;
                    float2 _f2_247 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                    float2 q_312 = _f2_247;
                    float2 r_313 = fma_f32x2_rn_noftz(q_312, scale0_288, a_311);
                    acc_237[14] = r_313.x;
                    acc_237[15] = r_313.y;
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
                    float2 _f2_248 = make_float2(acc_237[16], acc_237[17]);
                    float2 a_314 = _f2_248;
                    float2 _f2_249 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                    float2 q_315 = _f2_249;
                    float2 r_316 = fma_f32x2_rn_noftz(q_315, scale1_289, a_314);
                    acc_237[16] = r_316.x;
                    acc_237[17] = r_316.y;
                    float2 _f2_250 = make_float2(acc_237[18], acc_237[19]);
                    float2 a_317 = _f2_250;
                    float2 _f2_251 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                    float2 q_318 = _f2_251;
                    float2 r_319 = fma_f32x2_rn_noftz(q_318, scale1_289, a_317);
                    acc_237[18] = r_319.x;
                    acc_237[19] = r_319.y;
                    float2 _f2_252 = make_float2(acc_237[20], acc_237[21]);
                    float2 a_320 = _f2_252;
                    float2 _f2_253 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                    float2 q_321 = _f2_253;
                    float2 r_322 = fma_f32x2_rn_noftz(q_321, scale1_289, a_320);
                    acc_237[20] = r_322.x;
                    acc_237[21] = r_322.y;
                    float2 _f2_254 = make_float2(acc_237[22], acc_237[23]);
                    float2 a_323 = _f2_254;
                    float2 _f2_255 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                    float2 q_324 = _f2_255;
                    float2 r_325 = fma_f32x2_rn_noftz(q_324, scale1_289, a_323);
                    acc_237[22] = r_325.x;
                    acc_237[23] = r_325.y;
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
                    float2 _f2_256 = make_float2(acc_237[24], acc_237[25]);
                    float2 a_326 = _f2_256;
                    float2 _f2_257 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                    float2 q_327 = _f2_257;
                    float2 r_328 = fma_f32x2_rn_noftz(q_327, scale1_289, a_326);
                    acc_237[24] = r_328.x;
                    acc_237[25] = r_328.y;
                    float2 _f2_258 = make_float2(acc_237[26], acc_237[27]);
                    float2 a_329 = _f2_258;
                    float2 _f2_259 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                    float2 q_330 = _f2_259;
                    float2 r_331 = fma_f32x2_rn_noftz(q_330, scale1_289, a_329);
                    acc_237[26] = r_331.x;
                    acc_237[27] = r_331.y;
                    float2 _f2_260 = make_float2(acc_237[28], acc_237[29]);
                    float2 a_332 = _f2_260;
                    float2 _f2_261 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                    float2 q_333 = _f2_261;
                    float2 r_334 = fma_f32x2_rn_noftz(q_333, scale1_289, a_332);
                    acc_237[28] = r_334.x;
                    acc_237[29] = r_334.y;
                    float2 _f2_262 = make_float2(acc_237[30], acc_237[31]);
                    float2 a_335 = _f2_262;
                    float2 _f2_263 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                    float2 q_336 = _f2_263;
                    float2 r_337 = fma_f32x2_rn_noftz(q_336, scale1_289, a_335);
                    acc_237[30] = r_337.x;
                    acc_237[31] = r_337.y;
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
                    int c_338 = lane_1;
                    int l_339 = c_338 >> 2;
                    unsigned int w4_340[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_340[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_340[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_340[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_340[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_339 + (c_338 & 3 ^ l_339 >> 1 & 3)))));
                    long long idx_341 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_338));
                    {
                        int4 _iv4 = make_int4(w4_340[0 + 0], w4_340[0 + 1], w4_340[0 + 2], w4_340[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_341) = _iv4;
                    }
                    int c_342 = lane_1 + 32;
                    int l_343 = c_342 >> 2;
                    unsigned int w4_344[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_344[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_344[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_344[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_344[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_343 + (c_342 & 3 ^ l_343 >> 1 & 3)))));
                    long long idx_345 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_342));
                    {
                        int4 _iv4 = make_int4(w4_344[0 + 0], w4_344[0 + 1], w4_344[0 + 2], w4_344[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_345) = _iv4;
                    }
                    int c_346 = lane_1 + 64;
                    int l_347 = c_346 >> 2;
                    unsigned int w4_348[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_348[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_348[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_348[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_348[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_347 + (c_346 & 3 ^ l_347 >> 1 & 3)))));
                    long long idx_349 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_346));
                    {
                        int4 _iv4 = make_int4(w4_348[0 + 0], w4_348[0 + 1], w4_348[0 + 2], w4_348[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_349) = _iv4;
                    }
                    int c_350 = lane_1 + 96;
                    int l_351 = c_350 >> 2;
                    unsigned int w4_352[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_352[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_352[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_352[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_352[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_351 + (c_350 & 3 ^ l_351 >> 1 & 3)))));
                    long long idx_353 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_350));
                    {
                        int4 _iv4 = make_int4(w4_352[0 + 0], w4_352[0 + 1], w4_352[0 + 2], w4_352[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_353) = _iv4;
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
                ctrl[base_2] = call + 1;
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
            asm volatile("barrier.sync 0, 352;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
                long long lo_1 = u_0 * bid64_1 / g64_1 * 16;
                long long hi_2 = u_0 * (bid64_1 + 1) / g64_1 * 16;
                int npages_3 = (int)((hi_2 - lo_1 + 4095) / 4096);
                #pragma unroll 1
                for (int i_2 = 0; i_2 < npages_3; i_2++) {
                    if ((jbase + i_2 & 31) == lane_0) {
                        long long o_1 = lo_1 + (long long)i_2 * 4096;
                        long long rem_1 = hi_2 - o_1;
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
                jbase = jbase + npages_3;
                long long u_4 = pr64 * 16 / 16;
                long long lo_5 = u_4 * bid64_1 / g64_1 * 16;
                long long hi_6 = u_4 * (bid64_1 + 1) / g64_1 * 16;
                int npages_7 = (int)((hi_6 - lo_5 + 4095) / 4096);
                #pragma unroll 1
                for (int i_3 = 0; i_3 < npages_7; i_3++) {
                    if ((jbase + i_3 & 31) == lane_0) {
                        long long o_2 = lo_5 + (long long)i_3 * 4096;
                        long long rem_2 = hi_6 - o_2;
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
                jbase = jbase + npages_7;
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }

    // Cleanup
}

} // extern "C"
