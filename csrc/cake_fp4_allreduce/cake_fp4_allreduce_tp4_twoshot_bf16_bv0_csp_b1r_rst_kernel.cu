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
#define SMEM_S_AM_OFF 226176
#define SMEM_S_AM_STAGE_BYTES 80
#define SMEM_S_AM_STRIDE 80
#define SMEM_TAB_OFF 226304
#define SMEM_TAB_STAGE_BYTES 64
#define SMEM_TAB_STRIDE 64
#define SMEM_WSM_OFF 13376
#define SMEM_WSM_STAGE_BYTES 212800
#define SMEM_WSM_STRIDE 212800
#define SMEM_TOTAL 226432
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
cake_fp4_allreduce_tp4_twoshot_bf16_bv0_csp_b1r_rst(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, uint8_t* __restrict__ ts_wire, uint8_t* __restrict__ ts_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap)
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
    float* s_am = reinterpret_cast<float*>(smem_raw + 226176);
    const int s_am_addr = smem + 226176;
    long long* tab = reinterpret_cast<long long*>(smem_raw + 226304);
    const int tab_addr = smem + 226304;
    uint8_t* wsm = reinterpret_cast<uint8_t*>(smem_raw + 13376);
    const int wsm_addr = smem + 13376;

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
            long long g64 = (long long)gridDim.x;
            long long quads64 = (long long)(rows / 4);
            int r0 = (int)(4 * (bid64 * quads64 / g64));
            int r1 = (int)(4 * ((bid64 + 1) * quads64 / g64));
            int nrows = r1 - r0;
            int warp_0 = (int)warp;
            int lane_1 = (int)lane;
            if (tid == 0) {
                int base = bid * 4;
                ctl[0] = ctrl[base];
                ctl[1] = ctrl[base + 2];
                ctl[2] = ctrl[base + 3];
            }
            if (tid < 8) {
                tab[tid] = peer_ptrs[tid];
            }
            asm volatile("barrier.sync 0, 352;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int call = ctl[0];
            unsigned int buf = call % 3;
            int f = lane_1 >> 1 & 3;
            int xb_off = warp_0 * 21280;
            int own_off = xb_off + 12288;
            int stg_off = own_off + 6944;
            long long rank64 = (long long)rank;
            long long capo64 = (long long)((cap + 4 - 1) / 4);
            long long buf64 = (long long)buf;
            long long sa_w = 4 * capo64 * 3456;
            long long sa_r = 4 * capo64 * 16;
            long long rs_w = buf64 * (2 * sa_w);
            long long ag_w = rs_w + sa_w;
            long long rs_r = buf64 * (2 * sa_r);
            long long ag_r = rs_r + sa_r;
            int n = 0;
            if (nrows > warp_0) {
                n = (nrows - warp_0 + 10 - 1) / 10;
            }
            if (n > 0 && lane_1 == 0) {
                long long first_row = (long long)(r0 + warp_0);
                int stage = 2 * warp_0;
                mbarrier_arrive_expect_tx(xb_full_addr + (stage) * 8, 12288);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)xb_off, x + (first_row * 6144), 12288, xb_full_addr + (stage) * 8);
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
                int pp_0 = lane_1 + 32;
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
                    : "r"(wsm_addr + (unsigned int)(base_21 + 16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_25[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_25[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_25[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_21 + 16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_26[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_21 + 16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_27[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_21 + 16 * (f ^ 3))));
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
                unsigned int xa_28 = ((a_22 & 65535) > (a_22 >> 16) ? (a_22 & 65535) : (a_22 >> 16));
                unsigned int xb_29 = ((b_23 & 65535) > (b_23 >> 16) ? (b_23 & 65535) : (b_23 >> 16));
                if ((f & 2) != 0) {
                    bmp[3] = xb_29 | xa_28 << 16;
                } else {
                    bmp[3] = xa_28 | xb_29 << 16;
                }
                unsigned int _max_3 = ((xa_28) > (xb_29) ? (xa_28) : (xb_29));
                am = fmaxf(am, __uint_as_float(_max_3 << 16));
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
                    : "r"(wsm_addr + (unsigned int)(base_31 + 16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1_35[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_35[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_35[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_35[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_31 + 16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_36[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_36[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_36[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_31 + 16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_37[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_37[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_37[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_31 + 16 * (f ^ 3))));
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
                unsigned int xa_38 = ((a_32 & 65535) > (a_32 >> 16) ? (a_32 & 65535) : (a_32 >> 16));
                unsigned int xb_39 = ((b_33 & 65535) > (b_33 >> 16) ? (b_33 & 65535) : (b_33 >> 16));
                if ((f & 2) != 0) {
                    bmp[4] = xb_39 | xa_38 << 16;
                } else {
                    bmp[4] = xa_38 | xb_39 << 16;
                }
                unsigned int _max_4 = ((xa_38) > (xb_39) ? (xa_38) : (xb_39));
                am = fmaxf(am, __uint_as_float(_max_4 << 16));
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
                    : "r"(wsm_addr + (unsigned int)(base_41 + 16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1_45[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_45[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_45[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_45[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_41 + 16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2_46[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_46[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_46[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_46[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_41 + 16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3_47[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_47[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_47[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_47[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_41 + 16 * (f ^ 3))));
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
                unsigned int xa_48 = ((a_42 & 65535) > (a_42 >> 16) ? (a_42 & 65535) : (a_42 >> 16));
                unsigned int xb_49 = ((b_43 & 65535) > (b_43 >> 16) ? (b_43 & 65535) : (b_43 >> 16));
                if ((f & 2) != 0) {
                    bmp[5] = xb_49 | xa_48 << 16;
                } else {
                    bmp[5] = xa_48 | xb_49 << 16;
                }
                unsigned int _max_5 = ((xa_48) > (xb_49) ? (xa_48) : (xb_49));
                am = fmaxf(am, __uint_as_float(_max_5 << 16));
                float _warp_reduce_6 = am;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_6 = max_noftz(_warp_reduce_6, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_6, offset));
                am = _warp_reduce_6;
                float _fdiv_rn_0 = __fdiv_rn(2688.0f, am);
                float inv = 0.0f;
                if (am > 0.0f) {
                    inv = _fdiv_rn_0;
                }
                float rs_50 = __fdiv_rn(am, 2688.0f);
                int pp2 = lane_1;
                int base2 = xb_off + 64 * pp2;
                float bm0 = __uint_as_float(bmp[0] << 16);
                float bm1 = __uint_as_float(bmp[0] & 4294901760u);
                float _fdiv_7 = __fdiv_rn(mul_f32_rn_noftz(bm1, inv), 6.0f);
                float _fdiv_8 = __fdiv_rn(mul_f32_rn_noftz(bm0, inv), 6.0f);
                uint16_t _e4m3x2_f32_0;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_fdiv_7), "f"(_fdiv_8));
                uint32_t _e4m3x2_to_f16x2_0;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_0) : "h"((uint16_t)(_e4m3x2_f32_0)));
                float _cvt_f32_f16_0;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_0) : "h"((uint16_t)(_e4m3x2_to_f16x2_0)));
                float _cvt_f32_f16_1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1) : "h"((uint16_t)(_e4m3x2_to_f16x2_0 >> 16)));
                float _fdiv_rn_1 = __fdiv_rn(inv, _cvt_f32_f16_0);
                float _fdiv_rn_2 = __fdiv_rn(inv, _cvt_f32_f16_1);
                float m0 = 0.0f;
                float m1 = 0.0f;
                if (_cvt_f32_f16_0 > 0.0f) {
                    m0 = _fdiv_rn_1;
                }
                if (_cvt_f32_f16_1 > 0.0f) {
                    m1 = _fdiv_rn_2;
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
                float2 mm = make_float2(m, m);
                float2 pair = make_float2(__uint_as_float(v4[0] << 16), __uint_as_float(v4[0] & 4294901760u));
                float2 g;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g) : "l"(*(const unsigned long long*)&pair), "l"(*(const unsigned long long*)&mm));
                uint32_t _fp4_pair_0;
                uint16_t _e2m1_pair_h_9;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_9) : "f"(g.x), "f"(g.y));
                _fp4_pair_0 = (uint32_t)_e2m1_pair_h_9;
                float2 pair_51 = make_float2(__uint_as_float(v4[1] << 16), __uint_as_float(v4[1] & 4294901760u));
                float2 g_52;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_52) : "l"(*(const unsigned long long*)&pair_51), "l"(*(const unsigned long long*)&mm));
                uint32_t _fp4_pair_1;
                uint16_t _e2m1_pair_h_10;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_10) : "f"(g_52.x), "f"(g_52.y));
                _fp4_pair_1 = (uint32_t)_e2m1_pair_h_10;
                float2 pair_53 = make_float2(__uint_as_float(v4[2] << 16), __uint_as_float(v4[2] & 4294901760u));
                float2 g_54;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_54) : "l"(*(const unsigned long long*)&pair_53), "l"(*(const unsigned long long*)&mm));
                uint32_t _fp4_pair_2;
                uint16_t _e2m1_pair_h_11;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_11) : "f"(g_54.x), "f"(g_54.y));
                _fp4_pair_2 = (uint32_t)_e2m1_pair_h_11;
                float2 pair_55 = make_float2(__uint_as_float(v4[3] << 16), __uint_as_float(v4[3] & 4294901760u));
                float2 g_56;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_56) : "l"(*(const unsigned long long*)&pair_55), "l"(*(const unsigned long long*)&mm));
                uint32_t _fp4_e2m1_pack_pair_12;
                uint16_t _e2m1_pair_h_13;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_13) : "f"(g_56.x), "f"(g_56.y));
                _fp4_e2m1_pack_pair_12 = (uint32_t)_e2m1_pair_h_13;
                unsigned int word = _fp4_pair_0 | _fp4_pair_1 << 8 | _fp4_pair_2 << 16 | _fp4_e2m1_pack_pair_12 << 24;
                unsigned int carry = (word & 2004318071) + 2004318071;
                w[0] = word & (carry | 2004318071);
                int cf_57 = f ^ 1;
                unsigned int v4_58[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_58[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_58[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_58[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_58[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2 + 16 * cf_57)));
                float m_59 = m0;
                if ((cf_57 & 2) != 0) {
                    m_59 = m1;
                }
                float2 mm_60 = make_float2(m_59, m_59);
                float2 pair_61 = make_float2(__uint_as_float(v4_58[0] << 16), __uint_as_float(v4_58[0] & 4294901760u));
                float2 g_62;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_62) : "l"(*(const unsigned long long*)&pair_61), "l"(*(const unsigned long long*)&mm_60));
                uint32_t _fp4_pair_3;
                uint16_t _e2m1_pair_h_14;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_14) : "f"(g_62.x), "f"(g_62.y));
                _fp4_pair_3 = (uint32_t)_e2m1_pair_h_14;
                float2 pair_63 = make_float2(__uint_as_float(v4_58[1] << 16), __uint_as_float(v4_58[1] & 4294901760u));
                float2 g_64;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_64) : "l"(*(const unsigned long long*)&pair_63), "l"(*(const unsigned long long*)&mm_60));
                uint32_t _fp4_pair_4;
                uint16_t _e2m1_pair_h_15;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_15) : "f"(g_64.x), "f"(g_64.y));
                _fp4_pair_4 = (uint32_t)_e2m1_pair_h_15;
                float2 pair_65 = make_float2(__uint_as_float(v4_58[2] << 16), __uint_as_float(v4_58[2] & 4294901760u));
                float2 g_66;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_66) : "l"(*(const unsigned long long*)&pair_65), "l"(*(const unsigned long long*)&mm_60));
                uint32_t _fp4_pair_5;
                uint16_t _e2m1_pair_h_16;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_16) : "f"(g_66.x), "f"(g_66.y));
                _fp4_pair_5 = (uint32_t)_e2m1_pair_h_16;
                float2 pair_67 = make_float2(__uint_as_float(v4_58[3] << 16), __uint_as_float(v4_58[3] & 4294901760u));
                float2 g_68;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_68) : "l"(*(const unsigned long long*)&pair_67), "l"(*(const unsigned long long*)&mm_60));
                uint32_t _fp4_e2m1_pack_pair_17;
                uint16_t _e2m1_pair_h_18;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_18) : "f"(g_68.x), "f"(g_68.y));
                _fp4_e2m1_pack_pair_17 = (uint32_t)_e2m1_pair_h_18;
                unsigned int word_69 = _fp4_pair_3 | _fp4_pair_4 << 8 | _fp4_pair_5 << 16 | _fp4_e2m1_pack_pair_17 << 24;
                unsigned int carry_70 = (word_69 & 2004318071) + 2004318071;
                w[1] = word_69 & (carry_70 | 2004318071);
                int cf_71 = f ^ 2;
                unsigned int v4_72[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_72[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_72[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_72[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_72[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2 + 16 * cf_71)));
                float m_73 = m0;
                if ((cf_71 & 2) != 0) {
                    m_73 = m1;
                }
                float2 mm_74 = make_float2(m_73, m_73);
                float2 pair_75 = make_float2(__uint_as_float(v4_72[0] << 16), __uint_as_float(v4_72[0] & 4294901760u));
                float2 g_76;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_76) : "l"(*(const unsigned long long*)&pair_75), "l"(*(const unsigned long long*)&mm_74));
                uint32_t _fp4_pair_6;
                uint16_t _e2m1_pair_h_19;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_19) : "f"(g_76.x), "f"(g_76.y));
                _fp4_pair_6 = (uint32_t)_e2m1_pair_h_19;
                float2 pair_77 = make_float2(__uint_as_float(v4_72[1] << 16), __uint_as_float(v4_72[1] & 4294901760u));
                float2 g_78;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_78) : "l"(*(const unsigned long long*)&pair_77), "l"(*(const unsigned long long*)&mm_74));
                uint32_t _fp4_pair_7;
                uint16_t _e2m1_pair_h_20;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_20) : "f"(g_78.x), "f"(g_78.y));
                _fp4_pair_7 = (uint32_t)_e2m1_pair_h_20;
                float2 pair_79 = make_float2(__uint_as_float(v4_72[2] << 16), __uint_as_float(v4_72[2] & 4294901760u));
                float2 g_80;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_80) : "l"(*(const unsigned long long*)&pair_79), "l"(*(const unsigned long long*)&mm_74));
                uint32_t _fp4_pair_8;
                uint16_t _e2m1_pair_h_21;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_21) : "f"(g_80.x), "f"(g_80.y));
                _fp4_pair_8 = (uint32_t)_e2m1_pair_h_21;
                float2 pair_81 = make_float2(__uint_as_float(v4_72[3] << 16), __uint_as_float(v4_72[3] & 4294901760u));
                float2 g_82;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_82) : "l"(*(const unsigned long long*)&pair_81), "l"(*(const unsigned long long*)&mm_74));
                uint32_t _fp4_e2m1_pack_pair_22;
                uint16_t _e2m1_pair_h_23;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_23) : "f"(g_82.x), "f"(g_82.y));
                _fp4_e2m1_pack_pair_22 = (uint32_t)_e2m1_pair_h_23;
                unsigned int word_83 = _fp4_pair_6 | _fp4_pair_7 << 8 | _fp4_pair_8 << 16 | _fp4_e2m1_pack_pair_22 << 24;
                unsigned int carry_84 = (word_83 & 2004318071) + 2004318071;
                w[2] = word_83 & (carry_84 | 2004318071);
                int cf_85 = f ^ 3;
                unsigned int v4_86[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_86[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_86[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_86[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_86[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2 + 16 * cf_85)));
                float m_87 = m0;
                if ((cf_85 & 2) != 0) {
                    m_87 = m1;
                }
                float2 mm_88 = make_float2(m_87, m_87);
                float2 pair_89 = make_float2(__uint_as_float(v4_86[0] << 16), __uint_as_float(v4_86[0] & 4294901760u));
                float2 g_90;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_90) : "l"(*(const unsigned long long*)&pair_89), "l"(*(const unsigned long long*)&mm_88));
                uint32_t _fp4_pair_9;
                uint16_t _e2m1_pair_h_24;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_24) : "f"(g_90.x), "f"(g_90.y));
                _fp4_pair_9 = (uint32_t)_e2m1_pair_h_24;
                float2 pair_91 = make_float2(__uint_as_float(v4_86[1] << 16), __uint_as_float(v4_86[1] & 4294901760u));
                float2 g_92;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_92) : "l"(*(const unsigned long long*)&pair_91), "l"(*(const unsigned long long*)&mm_88));
                uint32_t _fp4_pair_10;
                uint16_t _e2m1_pair_h_25;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_25) : "f"(g_92.x), "f"(g_92.y));
                _fp4_pair_10 = (uint32_t)_e2m1_pair_h_25;
                float2 pair_93 = make_float2(__uint_as_float(v4_86[2] << 16), __uint_as_float(v4_86[2] & 4294901760u));
                float2 g_94;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_94) : "l"(*(const unsigned long long*)&pair_93), "l"(*(const unsigned long long*)&mm_88));
                uint32_t _fp4_pair_11;
                uint16_t _e2m1_pair_h_26;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_26) : "f"(g_94.x), "f"(g_94.y));
                _fp4_pair_11 = (uint32_t)_e2m1_pair_h_26;
                float2 pair_95 = make_float2(__uint_as_float(v4_86[3] << 16), __uint_as_float(v4_86[3] & 4294901760u));
                float2 g_96;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_96) : "l"(*(const unsigned long long*)&pair_95), "l"(*(const unsigned long long*)&mm_88));
                uint32_t _fp4_e2m1_pack_pair_27;
                uint16_t _e2m1_pair_h_28;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_28) : "f"(g_96.x), "f"(g_96.y));
                _fp4_e2m1_pack_pair_27 = (uint32_t)_e2m1_pair_h_28;
                unsigned int word_97 = _fp4_pair_9 | _fp4_pair_10 << 8 | _fp4_pair_11 << 16 | _fp4_e2m1_pack_pair_27 << 24;
                unsigned int carry_98 = (word_97 & 2004318071) + 2004318071;
                w[3] = word_97 & (carry_98 | 2004318071);
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
                int pp2_99 = lane_1 + 32;
                int base2_100 = xb_off + 64 * pp2_99;
                float bm0_101 = __uint_as_float(bmp[1] << 16);
                float bm1_102 = __uint_as_float(bmp[1] & 4294901760u);
                float _fdiv_29 = __fdiv_rn(mul_f32_rn_noftz(bm1_102, inv), 6.0f);
                float _fdiv_30 = __fdiv_rn(mul_f32_rn_noftz(bm0_101, inv), 6.0f);
                uint16_t _e4m3x2_f32_1;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_fdiv_29), "f"(_fdiv_30));
                uint32_t _e4m3x2_to_f16x2_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_1) : "h"((uint16_t)(_e4m3x2_f32_1)));
                float _cvt_f32_f16_2;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_2) : "h"((uint16_t)(_e4m3x2_to_f16x2_1)));
                float _cvt_f32_f16_3;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_3) : "h"((uint16_t)(_e4m3x2_to_f16x2_1 >> 16)));
                float _fdiv_rn_3 = __fdiv_rn(inv, _cvt_f32_f16_2);
                float _fdiv_rn_4 = __fdiv_rn(inv, _cvt_f32_f16_3);
                float m0_103 = 0.0f;
                float m1_104 = 0.0f;
                if (_cvt_f32_f16_2 > 0.0f) {
                    m0_103 = _fdiv_rn_3;
                }
                if (_cvt_f32_f16_3 > 0.0f) {
                    m1_104 = _fdiv_rn_4;
                }
                unsigned int w_105[4];
                int cf_106 = f ^ 0;
                unsigned int v4_107[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_107[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_107[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_107[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_107[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_100 + 16 * cf_106)));
                float m_108 = m0_103;
                if ((cf_106 & 2) != 0) {
                    m_108 = m1_104;
                }
                float2 mm_109 = make_float2(m_108, m_108);
                float2 pair_110 = make_float2(__uint_as_float(v4_107[0] << 16), __uint_as_float(v4_107[0] & 4294901760u));
                float2 g_111;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_111) : "l"(*(const unsigned long long*)&pair_110), "l"(*(const unsigned long long*)&mm_109));
                uint32_t _fp4_pair_12;
                uint16_t _e2m1_pair_h_31;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_31) : "f"(g_111.x), "f"(g_111.y));
                _fp4_pair_12 = (uint32_t)_e2m1_pair_h_31;
                float2 pair_112 = make_float2(__uint_as_float(v4_107[1] << 16), __uint_as_float(v4_107[1] & 4294901760u));
                float2 g_113;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_113) : "l"(*(const unsigned long long*)&pair_112), "l"(*(const unsigned long long*)&mm_109));
                uint32_t _fp4_pair_13;
                uint16_t _e2m1_pair_h_32;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_32) : "f"(g_113.x), "f"(g_113.y));
                _fp4_pair_13 = (uint32_t)_e2m1_pair_h_32;
                float2 pair_114 = make_float2(__uint_as_float(v4_107[2] << 16), __uint_as_float(v4_107[2] & 4294901760u));
                float2 g_115;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_115) : "l"(*(const unsigned long long*)&pair_114), "l"(*(const unsigned long long*)&mm_109));
                uint32_t _fp4_pair_14;
                uint16_t _e2m1_pair_h_33;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_33) : "f"(g_115.x), "f"(g_115.y));
                _fp4_pair_14 = (uint32_t)_e2m1_pair_h_33;
                float2 pair_116 = make_float2(__uint_as_float(v4_107[3] << 16), __uint_as_float(v4_107[3] & 4294901760u));
                float2 g_117;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_117) : "l"(*(const unsigned long long*)&pair_116), "l"(*(const unsigned long long*)&mm_109));
                uint32_t _fp4_e2m1_pack_pair_34;
                uint16_t _e2m1_pair_h_35;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_35) : "f"(g_117.x), "f"(g_117.y));
                _fp4_e2m1_pack_pair_34 = (uint32_t)_e2m1_pair_h_35;
                unsigned int word_118 = _fp4_pair_12 | _fp4_pair_13 << 8 | _fp4_pair_14 << 16 | _fp4_e2m1_pack_pair_34 << 24;
                unsigned int carry_119 = (word_118 & 2004318071) + 2004318071;
                w_105[0] = word_118 & (carry_119 | 2004318071);
                int cf_120 = f ^ 1;
                unsigned int v4_121[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_121[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_121[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_121[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_121[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_100 + 16 * cf_120)));
                float m_122 = m0_103;
                if ((cf_120 & 2) != 0) {
                    m_122 = m1_104;
                }
                float2 mm_123 = make_float2(m_122, m_122);
                float2 pair_124 = make_float2(__uint_as_float(v4_121[0] << 16), __uint_as_float(v4_121[0] & 4294901760u));
                float2 g_125;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_125) : "l"(*(const unsigned long long*)&pair_124), "l"(*(const unsigned long long*)&mm_123));
                uint32_t _fp4_pair_15;
                uint16_t _e2m1_pair_h_36;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_36) : "f"(g_125.x), "f"(g_125.y));
                _fp4_pair_15 = (uint32_t)_e2m1_pair_h_36;
                float2 pair_126 = make_float2(__uint_as_float(v4_121[1] << 16), __uint_as_float(v4_121[1] & 4294901760u));
                float2 g_127;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_127) : "l"(*(const unsigned long long*)&pair_126), "l"(*(const unsigned long long*)&mm_123));
                uint32_t _fp4_pair_16;
                uint16_t _e2m1_pair_h_37;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_37) : "f"(g_127.x), "f"(g_127.y));
                _fp4_pair_16 = (uint32_t)_e2m1_pair_h_37;
                float2 pair_128 = make_float2(__uint_as_float(v4_121[2] << 16), __uint_as_float(v4_121[2] & 4294901760u));
                float2 g_129;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_129) : "l"(*(const unsigned long long*)&pair_128), "l"(*(const unsigned long long*)&mm_123));
                uint32_t _fp4_pair_17;
                uint16_t _e2m1_pair_h_38;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_38) : "f"(g_129.x), "f"(g_129.y));
                _fp4_pair_17 = (uint32_t)_e2m1_pair_h_38;
                float2 pair_130 = make_float2(__uint_as_float(v4_121[3] << 16), __uint_as_float(v4_121[3] & 4294901760u));
                float2 g_131;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_131) : "l"(*(const unsigned long long*)&pair_130), "l"(*(const unsigned long long*)&mm_123));
                uint32_t _fp4_e2m1_pack_pair_39;
                uint16_t _e2m1_pair_h_40;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_40) : "f"(g_131.x), "f"(g_131.y));
                _fp4_e2m1_pack_pair_39 = (uint32_t)_e2m1_pair_h_40;
                unsigned int word_132 = _fp4_pair_15 | _fp4_pair_16 << 8 | _fp4_pair_17 << 16 | _fp4_e2m1_pack_pair_39 << 24;
                unsigned int carry_133 = (word_132 & 2004318071) + 2004318071;
                w_105[1] = word_132 & (carry_133 | 2004318071);
                int cf_134 = f ^ 2;
                unsigned int v4_135[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_135[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_135[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_135[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_135[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_100 + 16 * cf_134)));
                float m_136 = m0_103;
                if ((cf_134 & 2) != 0) {
                    m_136 = m1_104;
                }
                float2 mm_137 = make_float2(m_136, m_136);
                float2 pair_138 = make_float2(__uint_as_float(v4_135[0] << 16), __uint_as_float(v4_135[0] & 4294901760u));
                float2 g_139;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_139) : "l"(*(const unsigned long long*)&pair_138), "l"(*(const unsigned long long*)&mm_137));
                uint32_t _fp4_pair_18;
                uint16_t _e2m1_pair_h_41;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_41) : "f"(g_139.x), "f"(g_139.y));
                _fp4_pair_18 = (uint32_t)_e2m1_pair_h_41;
                float2 pair_140 = make_float2(__uint_as_float(v4_135[1] << 16), __uint_as_float(v4_135[1] & 4294901760u));
                float2 g_141;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_141) : "l"(*(const unsigned long long*)&pair_140), "l"(*(const unsigned long long*)&mm_137));
                uint32_t _fp4_pair_19;
                uint16_t _e2m1_pair_h_42;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_42) : "f"(g_141.x), "f"(g_141.y));
                _fp4_pair_19 = (uint32_t)_e2m1_pair_h_42;
                float2 pair_142 = make_float2(__uint_as_float(v4_135[2] << 16), __uint_as_float(v4_135[2] & 4294901760u));
                float2 g_143;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_143) : "l"(*(const unsigned long long*)&pair_142), "l"(*(const unsigned long long*)&mm_137));
                uint32_t _fp4_pair_20;
                uint16_t _e2m1_pair_h_43;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_43) : "f"(g_143.x), "f"(g_143.y));
                _fp4_pair_20 = (uint32_t)_e2m1_pair_h_43;
                float2 pair_144 = make_float2(__uint_as_float(v4_135[3] << 16), __uint_as_float(v4_135[3] & 4294901760u));
                float2 g_145;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_145) : "l"(*(const unsigned long long*)&pair_144), "l"(*(const unsigned long long*)&mm_137));
                uint32_t _fp4_e2m1_pack_pair_44;
                uint16_t _e2m1_pair_h_45;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_45) : "f"(g_145.x), "f"(g_145.y));
                _fp4_e2m1_pack_pair_44 = (uint32_t)_e2m1_pair_h_45;
                unsigned int word_146 = _fp4_pair_18 | _fp4_pair_19 << 8 | _fp4_pair_20 << 16 | _fp4_e2m1_pack_pair_44 << 24;
                unsigned int carry_147 = (word_146 & 2004318071) + 2004318071;
                w_105[2] = word_146 & (carry_147 | 2004318071);
                int cf_148 = f ^ 3;
                unsigned int v4_149[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_149[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_149[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_149[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_149[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_100 + 16 * cf_148)));
                float m_150 = m0_103;
                if ((cf_148 & 2) != 0) {
                    m_150 = m1_104;
                }
                float2 mm_151 = make_float2(m_150, m_150);
                float2 pair_152 = make_float2(__uint_as_float(v4_149[0] << 16), __uint_as_float(v4_149[0] & 4294901760u));
                float2 g_153;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_153) : "l"(*(const unsigned long long*)&pair_152), "l"(*(const unsigned long long*)&mm_151));
                uint32_t _fp4_pair_21;
                uint16_t _e2m1_pair_h_46;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_46) : "f"(g_153.x), "f"(g_153.y));
                _fp4_pair_21 = (uint32_t)_e2m1_pair_h_46;
                float2 pair_154 = make_float2(__uint_as_float(v4_149[1] << 16), __uint_as_float(v4_149[1] & 4294901760u));
                float2 g_155;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_155) : "l"(*(const unsigned long long*)&pair_154), "l"(*(const unsigned long long*)&mm_151));
                uint32_t _fp4_pair_22;
                uint16_t _e2m1_pair_h_47;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_47) : "f"(g_155.x), "f"(g_155.y));
                _fp4_pair_22 = (uint32_t)_e2m1_pair_h_47;
                float2 pair_156 = make_float2(__uint_as_float(v4_149[2] << 16), __uint_as_float(v4_149[2] & 4294901760u));
                float2 g_157;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_157) : "l"(*(const unsigned long long*)&pair_156), "l"(*(const unsigned long long*)&mm_151));
                uint32_t _fp4_pair_23;
                uint16_t _e2m1_pair_h_48;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_48) : "f"(g_157.x), "f"(g_157.y));
                _fp4_pair_23 = (uint32_t)_e2m1_pair_h_48;
                float2 pair_158 = make_float2(__uint_as_float(v4_149[3] << 16), __uint_as_float(v4_149[3] & 4294901760u));
                float2 g_159;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_159) : "l"(*(const unsigned long long*)&pair_158), "l"(*(const unsigned long long*)&mm_151));
                uint32_t _fp4_e2m1_pack_pair_49;
                uint16_t _e2m1_pair_h_50;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_50) : "f"(g_159.x), "f"(g_159.y));
                _fp4_e2m1_pack_pair_49 = (uint32_t)_e2m1_pair_h_50;
                unsigned int word_160 = _fp4_pair_21 | _fp4_pair_22 << 8 | _fp4_pair_23 << 16 | _fp4_e2m1_pack_pair_49 << 24;
                unsigned int carry_161 = (word_160 & 2004318071) + 2004318071;
                w_105[3] = word_160 & (carry_161 | 2004318071);
                bool fodd_162 = (f & 1) != 0;
                bool fhi_163 = (f & 2) != 0;
                unsigned int t0_164 = w_105[0];
                unsigned int t1_165 = w_105[1];
                unsigned int t2_166 = w_105[2];
                unsigned int t3_167 = w_105[3];
                if (fodd_162) {
                    t0_164 = w_105[1];
                    t1_165 = w_105[0];
                    t2_166 = w_105[3];
                    t3_167 = w_105[2];
                }
                unsigned int q0_168 = t0_164;
                unsigned int q1_169 = t1_165;
                unsigned int q2_170 = t2_166;
                unsigned int q3_171 = t3_167;
                if (fhi_163) {
                    q0_168 = t2_166;
                    q1_169 = t3_167;
                    q2_170 = t0_164;
                    q3_171 = t1_165;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_99)), "r"(q0_168), "r"(q1_169), "r"(q2_170), "r"(q3_171) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_99)), "h"((uint16_t)(_e4m3x2_f32_1)) : "memory");
                int pp2_172 = lane_1 + 64;
                int base2_173 = xb_off + 64 * pp2_172;
                float bm0_174 = __uint_as_float(bmp[2] << 16);
                float bm1_175 = __uint_as_float(bmp[2] & 4294901760u);
                float _fdiv_51 = __fdiv_rn(mul_f32_rn_noftz(bm1_175, inv), 6.0f);
                float _fdiv_52 = __fdiv_rn(mul_f32_rn_noftz(bm0_174, inv), 6.0f);
                uint16_t _e4m3x2_f32_2;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(_fdiv_51), "f"(_fdiv_52));
                uint32_t _e4m3x2_to_f16x2_2;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_2) : "h"((uint16_t)(_e4m3x2_f32_2)));
                float _cvt_f32_f16_4;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_4) : "h"((uint16_t)(_e4m3x2_to_f16x2_2)));
                float _cvt_f32_f16_5;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_5) : "h"((uint16_t)(_e4m3x2_to_f16x2_2 >> 16)));
                float _fdiv_rn_5 = __fdiv_rn(inv, _cvt_f32_f16_4);
                float _fdiv_rn_6 = __fdiv_rn(inv, _cvt_f32_f16_5);
                float m0_176 = 0.0f;
                float m1_177 = 0.0f;
                if (_cvt_f32_f16_4 > 0.0f) {
                    m0_176 = _fdiv_rn_5;
                }
                if (_cvt_f32_f16_5 > 0.0f) {
                    m1_177 = _fdiv_rn_6;
                }
                unsigned int w_178[4];
                int cf_179 = f ^ 0;
                unsigned int v4_180[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_180[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_180[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_180[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_180[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_173 + 16 * cf_179)));
                float m_181 = m0_176;
                if ((cf_179 & 2) != 0) {
                    m_181 = m1_177;
                }
                float2 mm_182 = make_float2(m_181, m_181);
                float2 pair_183 = make_float2(__uint_as_float(v4_180[0] << 16), __uint_as_float(v4_180[0] & 4294901760u));
                float2 g_184;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_184) : "l"(*(const unsigned long long*)&pair_183), "l"(*(const unsigned long long*)&mm_182));
                uint32_t _fp4_pair_24;
                uint16_t _e2m1_pair_h_53;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_53) : "f"(g_184.x), "f"(g_184.y));
                _fp4_pair_24 = (uint32_t)_e2m1_pair_h_53;
                float2 pair_185 = make_float2(__uint_as_float(v4_180[1] << 16), __uint_as_float(v4_180[1] & 4294901760u));
                float2 g_186;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_186) : "l"(*(const unsigned long long*)&pair_185), "l"(*(const unsigned long long*)&mm_182));
                uint32_t _fp4_pair_25;
                uint16_t _e2m1_pair_h_54;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_54) : "f"(g_186.x), "f"(g_186.y));
                _fp4_pair_25 = (uint32_t)_e2m1_pair_h_54;
                float2 pair_187 = make_float2(__uint_as_float(v4_180[2] << 16), __uint_as_float(v4_180[2] & 4294901760u));
                float2 g_188;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_188) : "l"(*(const unsigned long long*)&pair_187), "l"(*(const unsigned long long*)&mm_182));
                uint32_t _fp4_pair_26;
                uint16_t _e2m1_pair_h_55;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_55) : "f"(g_188.x), "f"(g_188.y));
                _fp4_pair_26 = (uint32_t)_e2m1_pair_h_55;
                float2 pair_189 = make_float2(__uint_as_float(v4_180[3] << 16), __uint_as_float(v4_180[3] & 4294901760u));
                float2 g_190;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_190) : "l"(*(const unsigned long long*)&pair_189), "l"(*(const unsigned long long*)&mm_182));
                uint32_t _fp4_e2m1_pack_pair_56;
                uint16_t _e2m1_pair_h_57;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_57) : "f"(g_190.x), "f"(g_190.y));
                _fp4_e2m1_pack_pair_56 = (uint32_t)_e2m1_pair_h_57;
                unsigned int word_191 = _fp4_pair_24 | _fp4_pair_25 << 8 | _fp4_pair_26 << 16 | _fp4_e2m1_pack_pair_56 << 24;
                unsigned int carry_192 = (word_191 & 2004318071) + 2004318071;
                w_178[0] = word_191 & (carry_192 | 2004318071);
                int cf_193 = f ^ 1;
                unsigned int v4_194[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_194[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_194[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_194[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_194[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_173 + 16 * cf_193)));
                float m_195 = m0_176;
                if ((cf_193 & 2) != 0) {
                    m_195 = m1_177;
                }
                float2 mm_196 = make_float2(m_195, m_195);
                float2 pair_197 = make_float2(__uint_as_float(v4_194[0] << 16), __uint_as_float(v4_194[0] & 4294901760u));
                float2 g_198;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_198) : "l"(*(const unsigned long long*)&pair_197), "l"(*(const unsigned long long*)&mm_196));
                uint32_t _fp4_pair_27;
                uint16_t _e2m1_pair_h_58;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_58) : "f"(g_198.x), "f"(g_198.y));
                _fp4_pair_27 = (uint32_t)_e2m1_pair_h_58;
                float2 pair_199 = make_float2(__uint_as_float(v4_194[1] << 16), __uint_as_float(v4_194[1] & 4294901760u));
                float2 g_200;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_200) : "l"(*(const unsigned long long*)&pair_199), "l"(*(const unsigned long long*)&mm_196));
                uint32_t _fp4_pair_28;
                uint16_t _e2m1_pair_h_59;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_59) : "f"(g_200.x), "f"(g_200.y));
                _fp4_pair_28 = (uint32_t)_e2m1_pair_h_59;
                float2 pair_201 = make_float2(__uint_as_float(v4_194[2] << 16), __uint_as_float(v4_194[2] & 4294901760u));
                float2 g_202;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_202) : "l"(*(const unsigned long long*)&pair_201), "l"(*(const unsigned long long*)&mm_196));
                uint32_t _fp4_pair_29;
                uint16_t _e2m1_pair_h_60;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_60) : "f"(g_202.x), "f"(g_202.y));
                _fp4_pair_29 = (uint32_t)_e2m1_pair_h_60;
                float2 pair_203 = make_float2(__uint_as_float(v4_194[3] << 16), __uint_as_float(v4_194[3] & 4294901760u));
                float2 g_204;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_204) : "l"(*(const unsigned long long*)&pair_203), "l"(*(const unsigned long long*)&mm_196));
                uint32_t _fp4_e2m1_pack_pair_61;
                uint16_t _e2m1_pair_h_62;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_62) : "f"(g_204.x), "f"(g_204.y));
                _fp4_e2m1_pack_pair_61 = (uint32_t)_e2m1_pair_h_62;
                unsigned int word_205 = _fp4_pair_27 | _fp4_pair_28 << 8 | _fp4_pair_29 << 16 | _fp4_e2m1_pack_pair_61 << 24;
                unsigned int carry_206 = (word_205 & 2004318071) + 2004318071;
                w_178[1] = word_205 & (carry_206 | 2004318071);
                int cf_207 = f ^ 2;
                unsigned int v4_208[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_208[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_208[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_208[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_208[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_173 + 16 * cf_207)));
                float m_209 = m0_176;
                if ((cf_207 & 2) != 0) {
                    m_209 = m1_177;
                }
                float2 mm_210 = make_float2(m_209, m_209);
                float2 pair_211 = make_float2(__uint_as_float(v4_208[0] << 16), __uint_as_float(v4_208[0] & 4294901760u));
                float2 g_212;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_212) : "l"(*(const unsigned long long*)&pair_211), "l"(*(const unsigned long long*)&mm_210));
                uint32_t _fp4_pair_30;
                uint16_t _e2m1_pair_h_63;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_63) : "f"(g_212.x), "f"(g_212.y));
                _fp4_pair_30 = (uint32_t)_e2m1_pair_h_63;
                float2 pair_213 = make_float2(__uint_as_float(v4_208[1] << 16), __uint_as_float(v4_208[1] & 4294901760u));
                float2 g_214;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_214) : "l"(*(const unsigned long long*)&pair_213), "l"(*(const unsigned long long*)&mm_210));
                uint32_t _fp4_pair_31;
                uint16_t _e2m1_pair_h_64;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_64) : "f"(g_214.x), "f"(g_214.y));
                _fp4_pair_31 = (uint32_t)_e2m1_pair_h_64;
                float2 pair_215 = make_float2(__uint_as_float(v4_208[2] << 16), __uint_as_float(v4_208[2] & 4294901760u));
                float2 g_216;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_216) : "l"(*(const unsigned long long*)&pair_215), "l"(*(const unsigned long long*)&mm_210));
                uint32_t _fp4_pair_32;
                uint16_t _e2m1_pair_h_65;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_65) : "f"(g_216.x), "f"(g_216.y));
                _fp4_pair_32 = (uint32_t)_e2m1_pair_h_65;
                float2 pair_217 = make_float2(__uint_as_float(v4_208[3] << 16), __uint_as_float(v4_208[3] & 4294901760u));
                float2 g_218;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_218) : "l"(*(const unsigned long long*)&pair_217), "l"(*(const unsigned long long*)&mm_210));
                uint32_t _fp4_e2m1_pack_pair_66;
                uint16_t _e2m1_pair_h_67;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_67) : "f"(g_218.x), "f"(g_218.y));
                _fp4_e2m1_pack_pair_66 = (uint32_t)_e2m1_pair_h_67;
                unsigned int word_219 = _fp4_pair_30 | _fp4_pair_31 << 8 | _fp4_pair_32 << 16 | _fp4_e2m1_pack_pair_66 << 24;
                unsigned int carry_220 = (word_219 & 2004318071) + 2004318071;
                w_178[2] = word_219 & (carry_220 | 2004318071);
                int cf_221 = f ^ 3;
                unsigned int v4_222[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_222[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_222[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_222[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_222[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_173 + 16 * cf_221)));
                float m_223 = m0_176;
                if ((cf_221 & 2) != 0) {
                    m_223 = m1_177;
                }
                float2 mm_224 = make_float2(m_223, m_223);
                float2 pair_225 = make_float2(__uint_as_float(v4_222[0] << 16), __uint_as_float(v4_222[0] & 4294901760u));
                float2 g_226;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_226) : "l"(*(const unsigned long long*)&pair_225), "l"(*(const unsigned long long*)&mm_224));
                uint32_t _fp4_pair_33;
                uint16_t _e2m1_pair_h_68;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_68) : "f"(g_226.x), "f"(g_226.y));
                _fp4_pair_33 = (uint32_t)_e2m1_pair_h_68;
                float2 pair_227 = make_float2(__uint_as_float(v4_222[1] << 16), __uint_as_float(v4_222[1] & 4294901760u));
                float2 g_228;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_228) : "l"(*(const unsigned long long*)&pair_227), "l"(*(const unsigned long long*)&mm_224));
                uint32_t _fp4_pair_34;
                uint16_t _e2m1_pair_h_69;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_69) : "f"(g_228.x), "f"(g_228.y));
                _fp4_pair_34 = (uint32_t)_e2m1_pair_h_69;
                float2 pair_229 = make_float2(__uint_as_float(v4_222[2] << 16), __uint_as_float(v4_222[2] & 4294901760u));
                float2 g_230;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_230) : "l"(*(const unsigned long long*)&pair_229), "l"(*(const unsigned long long*)&mm_224));
                uint32_t _fp4_pair_35;
                uint16_t _e2m1_pair_h_70;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_70) : "f"(g_230.x), "f"(g_230.y));
                _fp4_pair_35 = (uint32_t)_e2m1_pair_h_70;
                float2 pair_231 = make_float2(__uint_as_float(v4_222[3] << 16), __uint_as_float(v4_222[3] & 4294901760u));
                float2 g_232;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_232) : "l"(*(const unsigned long long*)&pair_231), "l"(*(const unsigned long long*)&mm_224));
                uint32_t _fp4_e2m1_pack_pair_71;
                uint16_t _e2m1_pair_h_72;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_72) : "f"(g_232.x), "f"(g_232.y));
                _fp4_e2m1_pack_pair_71 = (uint32_t)_e2m1_pair_h_72;
                unsigned int word_233 = _fp4_pair_33 | _fp4_pair_34 << 8 | _fp4_pair_35 << 16 | _fp4_e2m1_pack_pair_71 << 24;
                unsigned int carry_234 = (word_233 & 2004318071) + 2004318071;
                w_178[3] = word_233 & (carry_234 | 2004318071);
                bool fodd_235 = (f & 1) != 0;
                bool fhi_236 = (f & 2) != 0;
                unsigned int t0_237 = w_178[0];
                unsigned int t1_238 = w_178[1];
                unsigned int t2_239 = w_178[2];
                unsigned int t3_240 = w_178[3];
                if (fodd_235) {
                    t0_237 = w_178[1];
                    t1_238 = w_178[0];
                    t2_239 = w_178[3];
                    t3_240 = w_178[2];
                }
                unsigned int q0_241 = t0_237;
                unsigned int q1_242 = t1_238;
                unsigned int q2_243 = t2_239;
                unsigned int q3_244 = t3_240;
                if (fhi_236) {
                    q0_241 = t2_239;
                    q1_242 = t3_240;
                    q2_243 = t0_237;
                    q3_244 = t1_238;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_172)), "r"(q0_241), "r"(q1_242), "r"(q2_243), "r"(q3_244) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_172)), "h"((uint16_t)(_e4m3x2_f32_2)) : "memory");
                int pp2_245 = lane_1 + 96;
                int base2_246 = xb_off + 64 * pp2_245;
                float bm0_247 = __uint_as_float(bmp[3] << 16);
                float bm1_248 = __uint_as_float(bmp[3] & 4294901760u);
                float _fdiv_73 = __fdiv_rn(mul_f32_rn_noftz(bm1_248, inv), 6.0f);
                float _fdiv_74 = __fdiv_rn(mul_f32_rn_noftz(bm0_247, inv), 6.0f);
                uint16_t _e4m3x2_f32_3;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(_fdiv_73), "f"(_fdiv_74));
                uint32_t _e4m3x2_to_f16x2_3;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_3) : "h"((uint16_t)(_e4m3x2_f32_3)));
                float _cvt_f32_f16_6;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_6) : "h"((uint16_t)(_e4m3x2_to_f16x2_3)));
                float _cvt_f32_f16_7;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(_e4m3x2_to_f16x2_3 >> 16)));
                float _fdiv_rn_7 = __fdiv_rn(inv, _cvt_f32_f16_6);
                float _fdiv_rn_8 = __fdiv_rn(inv, _cvt_f32_f16_7);
                float m0_249 = 0.0f;
                float m1_250 = 0.0f;
                if (_cvt_f32_f16_6 > 0.0f) {
                    m0_249 = _fdiv_rn_7;
                }
                if (_cvt_f32_f16_7 > 0.0f) {
                    m1_250 = _fdiv_rn_8;
                }
                unsigned int w_251[4];
                int cf_252 = f ^ 0;
                unsigned int v4_253[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_253[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_253[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_253[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_253[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_246 + 16 * cf_252)));
                float m_254 = m0_249;
                if ((cf_252 & 2) != 0) {
                    m_254 = m1_250;
                }
                float2 mm_255 = make_float2(m_254, m_254);
                float2 pair_256 = make_float2(__uint_as_float(v4_253[0] << 16), __uint_as_float(v4_253[0] & 4294901760u));
                float2 g_257;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_257) : "l"(*(const unsigned long long*)&pair_256), "l"(*(const unsigned long long*)&mm_255));
                uint32_t _fp4_pair_36;
                uint16_t _e2m1_pair_h_75;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_75) : "f"(g_257.x), "f"(g_257.y));
                _fp4_pair_36 = (uint32_t)_e2m1_pair_h_75;
                float2 pair_258 = make_float2(__uint_as_float(v4_253[1] << 16), __uint_as_float(v4_253[1] & 4294901760u));
                float2 g_259;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_259) : "l"(*(const unsigned long long*)&pair_258), "l"(*(const unsigned long long*)&mm_255));
                uint32_t _fp4_pair_37;
                uint16_t _e2m1_pair_h_76;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_76) : "f"(g_259.x), "f"(g_259.y));
                _fp4_pair_37 = (uint32_t)_e2m1_pair_h_76;
                float2 pair_260 = make_float2(__uint_as_float(v4_253[2] << 16), __uint_as_float(v4_253[2] & 4294901760u));
                float2 g_261;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_261) : "l"(*(const unsigned long long*)&pair_260), "l"(*(const unsigned long long*)&mm_255));
                uint32_t _fp4_pair_38;
                uint16_t _e2m1_pair_h_77;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_77) : "f"(g_261.x), "f"(g_261.y));
                _fp4_pair_38 = (uint32_t)_e2m1_pair_h_77;
                float2 pair_262 = make_float2(__uint_as_float(v4_253[3] << 16), __uint_as_float(v4_253[3] & 4294901760u));
                float2 g_263;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_263) : "l"(*(const unsigned long long*)&pair_262), "l"(*(const unsigned long long*)&mm_255));
                uint32_t _fp4_e2m1_pack_pair_78;
                uint16_t _e2m1_pair_h_79;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_79) : "f"(g_263.x), "f"(g_263.y));
                _fp4_e2m1_pack_pair_78 = (uint32_t)_e2m1_pair_h_79;
                unsigned int word_264 = _fp4_pair_36 | _fp4_pair_37 << 8 | _fp4_pair_38 << 16 | _fp4_e2m1_pack_pair_78 << 24;
                unsigned int carry_265 = (word_264 & 2004318071) + 2004318071;
                w_251[0] = word_264 & (carry_265 | 2004318071);
                int cf_266 = f ^ 1;
                unsigned int v4_267[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_267[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_267[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_267[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_267[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_246 + 16 * cf_266)));
                float m_268 = m0_249;
                if ((cf_266 & 2) != 0) {
                    m_268 = m1_250;
                }
                float2 mm_269 = make_float2(m_268, m_268);
                float2 pair_270 = make_float2(__uint_as_float(v4_267[0] << 16), __uint_as_float(v4_267[0] & 4294901760u));
                float2 g_271;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_271) : "l"(*(const unsigned long long*)&pair_270), "l"(*(const unsigned long long*)&mm_269));
                uint32_t _fp4_pair_39;
                uint16_t _e2m1_pair_h_80;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_80) : "f"(g_271.x), "f"(g_271.y));
                _fp4_pair_39 = (uint32_t)_e2m1_pair_h_80;
                float2 pair_272 = make_float2(__uint_as_float(v4_267[1] << 16), __uint_as_float(v4_267[1] & 4294901760u));
                float2 g_273;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_273) : "l"(*(const unsigned long long*)&pair_272), "l"(*(const unsigned long long*)&mm_269));
                uint32_t _fp4_pair_40;
                uint16_t _e2m1_pair_h_81;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_81) : "f"(g_273.x), "f"(g_273.y));
                _fp4_pair_40 = (uint32_t)_e2m1_pair_h_81;
                float2 pair_274 = make_float2(__uint_as_float(v4_267[2] << 16), __uint_as_float(v4_267[2] & 4294901760u));
                float2 g_275;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_275) : "l"(*(const unsigned long long*)&pair_274), "l"(*(const unsigned long long*)&mm_269));
                uint32_t _fp4_pair_41;
                uint16_t _e2m1_pair_h_82;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_82) : "f"(g_275.x), "f"(g_275.y));
                _fp4_pair_41 = (uint32_t)_e2m1_pair_h_82;
                float2 pair_276 = make_float2(__uint_as_float(v4_267[3] << 16), __uint_as_float(v4_267[3] & 4294901760u));
                float2 g_277;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_277) : "l"(*(const unsigned long long*)&pair_276), "l"(*(const unsigned long long*)&mm_269));
                uint32_t _fp4_e2m1_pack_pair_83;
                uint16_t _e2m1_pair_h_84;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_84) : "f"(g_277.x), "f"(g_277.y));
                _fp4_e2m1_pack_pair_83 = (uint32_t)_e2m1_pair_h_84;
                unsigned int word_278 = _fp4_pair_39 | _fp4_pair_40 << 8 | _fp4_pair_41 << 16 | _fp4_e2m1_pack_pair_83 << 24;
                unsigned int carry_279 = (word_278 & 2004318071) + 2004318071;
                w_251[1] = word_278 & (carry_279 | 2004318071);
                int cf_280 = f ^ 2;
                unsigned int v4_281[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_281[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_281[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_281[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_281[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_246 + 16 * cf_280)));
                float m_282 = m0_249;
                if ((cf_280 & 2) != 0) {
                    m_282 = m1_250;
                }
                float2 mm_283 = make_float2(m_282, m_282);
                float2 pair_284 = make_float2(__uint_as_float(v4_281[0] << 16), __uint_as_float(v4_281[0] & 4294901760u));
                float2 g_285;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_285) : "l"(*(const unsigned long long*)&pair_284), "l"(*(const unsigned long long*)&mm_283));
                uint32_t _fp4_pair_42;
                uint16_t _e2m1_pair_h_85;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_85) : "f"(g_285.x), "f"(g_285.y));
                _fp4_pair_42 = (uint32_t)_e2m1_pair_h_85;
                float2 pair_286 = make_float2(__uint_as_float(v4_281[1] << 16), __uint_as_float(v4_281[1] & 4294901760u));
                float2 g_287;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_287) : "l"(*(const unsigned long long*)&pair_286), "l"(*(const unsigned long long*)&mm_283));
                uint32_t _fp4_pair_43;
                uint16_t _e2m1_pair_h_86;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_86) : "f"(g_287.x), "f"(g_287.y));
                _fp4_pair_43 = (uint32_t)_e2m1_pair_h_86;
                float2 pair_288 = make_float2(__uint_as_float(v4_281[2] << 16), __uint_as_float(v4_281[2] & 4294901760u));
                float2 g_289;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_289) : "l"(*(const unsigned long long*)&pair_288), "l"(*(const unsigned long long*)&mm_283));
                uint32_t _fp4_pair_44;
                uint16_t _e2m1_pair_h_87;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_87) : "f"(g_289.x), "f"(g_289.y));
                _fp4_pair_44 = (uint32_t)_e2m1_pair_h_87;
                float2 pair_290 = make_float2(__uint_as_float(v4_281[3] << 16), __uint_as_float(v4_281[3] & 4294901760u));
                float2 g_291;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_291) : "l"(*(const unsigned long long*)&pair_290), "l"(*(const unsigned long long*)&mm_283));
                uint32_t _fp4_e2m1_pack_pair_88;
                uint16_t _e2m1_pair_h_89;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_89) : "f"(g_291.x), "f"(g_291.y));
                _fp4_e2m1_pack_pair_88 = (uint32_t)_e2m1_pair_h_89;
                unsigned int word_292 = _fp4_pair_42 | _fp4_pair_43 << 8 | _fp4_pair_44 << 16 | _fp4_e2m1_pack_pair_88 << 24;
                unsigned int carry_293 = (word_292 & 2004318071) + 2004318071;
                w_251[2] = word_292 & (carry_293 | 2004318071);
                int cf_294 = f ^ 3;
                unsigned int v4_295[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_295[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_295[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_295[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_295[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_246 + 16 * cf_294)));
                float m_296 = m0_249;
                if ((cf_294 & 2) != 0) {
                    m_296 = m1_250;
                }
                float2 mm_297 = make_float2(m_296, m_296);
                float2 pair_298 = make_float2(__uint_as_float(v4_295[0] << 16), __uint_as_float(v4_295[0] & 4294901760u));
                float2 g_299;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_299) : "l"(*(const unsigned long long*)&pair_298), "l"(*(const unsigned long long*)&mm_297));
                uint32_t _fp4_pair_45;
                uint16_t _e2m1_pair_h_90;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_90) : "f"(g_299.x), "f"(g_299.y));
                _fp4_pair_45 = (uint32_t)_e2m1_pair_h_90;
                float2 pair_300 = make_float2(__uint_as_float(v4_295[1] << 16), __uint_as_float(v4_295[1] & 4294901760u));
                float2 g_301;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_301) : "l"(*(const unsigned long long*)&pair_300), "l"(*(const unsigned long long*)&mm_297));
                uint32_t _fp4_pair_46;
                uint16_t _e2m1_pair_h_91;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_91) : "f"(g_301.x), "f"(g_301.y));
                _fp4_pair_46 = (uint32_t)_e2m1_pair_h_91;
                float2 pair_302 = make_float2(__uint_as_float(v4_295[2] << 16), __uint_as_float(v4_295[2] & 4294901760u));
                float2 g_303;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_303) : "l"(*(const unsigned long long*)&pair_302), "l"(*(const unsigned long long*)&mm_297));
                uint32_t _fp4_pair_47;
                uint16_t _e2m1_pair_h_92;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_92) : "f"(g_303.x), "f"(g_303.y));
                _fp4_pair_47 = (uint32_t)_e2m1_pair_h_92;
                float2 pair_304 = make_float2(__uint_as_float(v4_295[3] << 16), __uint_as_float(v4_295[3] & 4294901760u));
                float2 g_305;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_305) : "l"(*(const unsigned long long*)&pair_304), "l"(*(const unsigned long long*)&mm_297));
                uint32_t _fp4_e2m1_pack_pair_93;
                uint16_t _e2m1_pair_h_94;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_94) : "f"(g_305.x), "f"(g_305.y));
                _fp4_e2m1_pack_pair_93 = (uint32_t)_e2m1_pair_h_94;
                unsigned int word_306 = _fp4_pair_45 | _fp4_pair_46 << 8 | _fp4_pair_47 << 16 | _fp4_e2m1_pack_pair_93 << 24;
                unsigned int carry_307 = (word_306 & 2004318071) + 2004318071;
                w_251[3] = word_306 & (carry_307 | 2004318071);
                bool fodd_308 = (f & 1) != 0;
                bool fhi_309 = (f & 2) != 0;
                unsigned int t0_310 = w_251[0];
                unsigned int t1_311 = w_251[1];
                unsigned int t2_312 = w_251[2];
                unsigned int t3_313 = w_251[3];
                if (fodd_308) {
                    t0_310 = w_251[1];
                    t1_311 = w_251[0];
                    t2_312 = w_251[3];
                    t3_313 = w_251[2];
                }
                unsigned int q0_314 = t0_310;
                unsigned int q1_315 = t1_311;
                unsigned int q2_316 = t2_312;
                unsigned int q3_317 = t3_313;
                if (fhi_309) {
                    q0_314 = t2_312;
                    q1_315 = t3_313;
                    q2_316 = t0_310;
                    q3_317 = t1_311;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_245)), "r"(q0_314), "r"(q1_315), "r"(q2_316), "r"(q3_317) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_245)), "h"((uint16_t)(_e4m3x2_f32_3)) : "memory");
                int pp2_318 = lane_1 + 128;
                int base2_319 = xb_off + 64 * pp2_318;
                float bm0_320 = __uint_as_float(bmp[4] << 16);
                float bm1_321 = __uint_as_float(bmp[4] & 4294901760u);
                float _fdiv_95 = __fdiv_rn(mul_f32_rn_noftz(bm1_321, inv), 6.0f);
                float _fdiv_96 = __fdiv_rn(mul_f32_rn_noftz(bm0_320, inv), 6.0f);
                uint16_t _e4m3x2_f32_4;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_4) : "f"(_fdiv_95), "f"(_fdiv_96));
                uint32_t _e4m3x2_to_f16x2_4;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_4) : "h"((uint16_t)(_e4m3x2_f32_4)));
                float _cvt_f32_f16_8;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_8) : "h"((uint16_t)(_e4m3x2_to_f16x2_4)));
                float _cvt_f32_f16_9;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_9) : "h"((uint16_t)(_e4m3x2_to_f16x2_4 >> 16)));
                float _fdiv_rn_9 = __fdiv_rn(inv, _cvt_f32_f16_8);
                float _fdiv_rn_10 = __fdiv_rn(inv, _cvt_f32_f16_9);
                float m0_322 = 0.0f;
                float m1_323 = 0.0f;
                if (_cvt_f32_f16_8 > 0.0f) {
                    m0_322 = _fdiv_rn_9;
                }
                if (_cvt_f32_f16_9 > 0.0f) {
                    m1_323 = _fdiv_rn_10;
                }
                unsigned int w_324[4];
                int cf_325 = f ^ 0;
                unsigned int v4_326[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_326[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_326[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_326[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_326[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_319 + 16 * cf_325)));
                float m_327 = m0_322;
                if ((cf_325 & 2) != 0) {
                    m_327 = m1_323;
                }
                float2 mm_328 = make_float2(m_327, m_327);
                float2 pair_329 = make_float2(__uint_as_float(v4_326[0] << 16), __uint_as_float(v4_326[0] & 4294901760u));
                float2 g_330;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_330) : "l"(*(const unsigned long long*)&pair_329), "l"(*(const unsigned long long*)&mm_328));
                uint32_t _fp4_pair_48;
                uint16_t _e2m1_pair_h_97;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_97) : "f"(g_330.x), "f"(g_330.y));
                _fp4_pair_48 = (uint32_t)_e2m1_pair_h_97;
                float2 pair_331 = make_float2(__uint_as_float(v4_326[1] << 16), __uint_as_float(v4_326[1] & 4294901760u));
                float2 g_332;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_332) : "l"(*(const unsigned long long*)&pair_331), "l"(*(const unsigned long long*)&mm_328));
                uint32_t _fp4_pair_49;
                uint16_t _e2m1_pair_h_98;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_98) : "f"(g_332.x), "f"(g_332.y));
                _fp4_pair_49 = (uint32_t)_e2m1_pair_h_98;
                float2 pair_333 = make_float2(__uint_as_float(v4_326[2] << 16), __uint_as_float(v4_326[2] & 4294901760u));
                float2 g_334;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_334) : "l"(*(const unsigned long long*)&pair_333), "l"(*(const unsigned long long*)&mm_328));
                uint32_t _fp4_pair_50;
                uint16_t _e2m1_pair_h_99;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_99) : "f"(g_334.x), "f"(g_334.y));
                _fp4_pair_50 = (uint32_t)_e2m1_pair_h_99;
                float2 pair_335 = make_float2(__uint_as_float(v4_326[3] << 16), __uint_as_float(v4_326[3] & 4294901760u));
                float2 g_336;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_336) : "l"(*(const unsigned long long*)&pair_335), "l"(*(const unsigned long long*)&mm_328));
                uint32_t _fp4_e2m1_pack_pair_100;
                uint16_t _e2m1_pair_h_101;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_101) : "f"(g_336.x), "f"(g_336.y));
                _fp4_e2m1_pack_pair_100 = (uint32_t)_e2m1_pair_h_101;
                unsigned int word_337 = _fp4_pair_48 | _fp4_pair_49 << 8 | _fp4_pair_50 << 16 | _fp4_e2m1_pack_pair_100 << 24;
                unsigned int carry_338 = (word_337 & 2004318071) + 2004318071;
                w_324[0] = word_337 & (carry_338 | 2004318071);
                int cf_339 = f ^ 1;
                unsigned int v4_340[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_340[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_340[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_340[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_340[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_319 + 16 * cf_339)));
                float m_341 = m0_322;
                if ((cf_339 & 2) != 0) {
                    m_341 = m1_323;
                }
                float2 mm_342 = make_float2(m_341, m_341);
                float2 pair_343 = make_float2(__uint_as_float(v4_340[0] << 16), __uint_as_float(v4_340[0] & 4294901760u));
                float2 g_344;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_344) : "l"(*(const unsigned long long*)&pair_343), "l"(*(const unsigned long long*)&mm_342));
                uint32_t _fp4_pair_51;
                uint16_t _e2m1_pair_h_102;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_102) : "f"(g_344.x), "f"(g_344.y));
                _fp4_pair_51 = (uint32_t)_e2m1_pair_h_102;
                float2 pair_345 = make_float2(__uint_as_float(v4_340[1] << 16), __uint_as_float(v4_340[1] & 4294901760u));
                float2 g_346;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_346) : "l"(*(const unsigned long long*)&pair_345), "l"(*(const unsigned long long*)&mm_342));
                uint32_t _fp4_pair_52;
                uint16_t _e2m1_pair_h_103;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_103) : "f"(g_346.x), "f"(g_346.y));
                _fp4_pair_52 = (uint32_t)_e2m1_pair_h_103;
                float2 pair_347 = make_float2(__uint_as_float(v4_340[2] << 16), __uint_as_float(v4_340[2] & 4294901760u));
                float2 g_348;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_348) : "l"(*(const unsigned long long*)&pair_347), "l"(*(const unsigned long long*)&mm_342));
                uint32_t _fp4_pair_53;
                uint16_t _e2m1_pair_h_104;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_104) : "f"(g_348.x), "f"(g_348.y));
                _fp4_pair_53 = (uint32_t)_e2m1_pair_h_104;
                float2 pair_349 = make_float2(__uint_as_float(v4_340[3] << 16), __uint_as_float(v4_340[3] & 4294901760u));
                float2 g_350;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_350) : "l"(*(const unsigned long long*)&pair_349), "l"(*(const unsigned long long*)&mm_342));
                uint32_t _fp4_e2m1_pack_pair_105;
                uint16_t _e2m1_pair_h_106;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_106) : "f"(g_350.x), "f"(g_350.y));
                _fp4_e2m1_pack_pair_105 = (uint32_t)_e2m1_pair_h_106;
                unsigned int word_351 = _fp4_pair_51 | _fp4_pair_52 << 8 | _fp4_pair_53 << 16 | _fp4_e2m1_pack_pair_105 << 24;
                unsigned int carry_352 = (word_351 & 2004318071) + 2004318071;
                w_324[1] = word_351 & (carry_352 | 2004318071);
                int cf_353 = f ^ 2;
                unsigned int v4_354[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_354[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_354[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_354[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_354[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_319 + 16 * cf_353)));
                float m_355 = m0_322;
                if ((cf_353 & 2) != 0) {
                    m_355 = m1_323;
                }
                float2 mm_356 = make_float2(m_355, m_355);
                float2 pair_357 = make_float2(__uint_as_float(v4_354[0] << 16), __uint_as_float(v4_354[0] & 4294901760u));
                float2 g_358;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_358) : "l"(*(const unsigned long long*)&pair_357), "l"(*(const unsigned long long*)&mm_356));
                uint32_t _fp4_pair_54;
                uint16_t _e2m1_pair_h_107;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_107) : "f"(g_358.x), "f"(g_358.y));
                _fp4_pair_54 = (uint32_t)_e2m1_pair_h_107;
                float2 pair_359 = make_float2(__uint_as_float(v4_354[1] << 16), __uint_as_float(v4_354[1] & 4294901760u));
                float2 g_360;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_360) : "l"(*(const unsigned long long*)&pair_359), "l"(*(const unsigned long long*)&mm_356));
                uint32_t _fp4_pair_55;
                uint16_t _e2m1_pair_h_108;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_108) : "f"(g_360.x), "f"(g_360.y));
                _fp4_pair_55 = (uint32_t)_e2m1_pair_h_108;
                float2 pair_361 = make_float2(__uint_as_float(v4_354[2] << 16), __uint_as_float(v4_354[2] & 4294901760u));
                float2 g_362;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_362) : "l"(*(const unsigned long long*)&pair_361), "l"(*(const unsigned long long*)&mm_356));
                uint32_t _fp4_pair_56;
                uint16_t _e2m1_pair_h_109;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_109) : "f"(g_362.x), "f"(g_362.y));
                _fp4_pair_56 = (uint32_t)_e2m1_pair_h_109;
                float2 pair_363 = make_float2(__uint_as_float(v4_354[3] << 16), __uint_as_float(v4_354[3] & 4294901760u));
                float2 g_364;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_364) : "l"(*(const unsigned long long*)&pair_363), "l"(*(const unsigned long long*)&mm_356));
                uint32_t _fp4_e2m1_pack_pair_110;
                uint16_t _e2m1_pair_h_111;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_111) : "f"(g_364.x), "f"(g_364.y));
                _fp4_e2m1_pack_pair_110 = (uint32_t)_e2m1_pair_h_111;
                unsigned int word_365 = _fp4_pair_54 | _fp4_pair_55 << 8 | _fp4_pair_56 << 16 | _fp4_e2m1_pack_pair_110 << 24;
                unsigned int carry_366 = (word_365 & 2004318071) + 2004318071;
                w_324[2] = word_365 & (carry_366 | 2004318071);
                int cf_367 = f ^ 3;
                unsigned int v4_368[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_368[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_368[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_368[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_368[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_319 + 16 * cf_367)));
                float m_369 = m0_322;
                if ((cf_367 & 2) != 0) {
                    m_369 = m1_323;
                }
                float2 mm_370 = make_float2(m_369, m_369);
                float2 pair_371 = make_float2(__uint_as_float(v4_368[0] << 16), __uint_as_float(v4_368[0] & 4294901760u));
                float2 g_372;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_372) : "l"(*(const unsigned long long*)&pair_371), "l"(*(const unsigned long long*)&mm_370));
                uint32_t _fp4_pair_57;
                uint16_t _e2m1_pair_h_112;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_112) : "f"(g_372.x), "f"(g_372.y));
                _fp4_pair_57 = (uint32_t)_e2m1_pair_h_112;
                float2 pair_373 = make_float2(__uint_as_float(v4_368[1] << 16), __uint_as_float(v4_368[1] & 4294901760u));
                float2 g_374;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_374) : "l"(*(const unsigned long long*)&pair_373), "l"(*(const unsigned long long*)&mm_370));
                uint32_t _fp4_pair_58;
                uint16_t _e2m1_pair_h_113;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_113) : "f"(g_374.x), "f"(g_374.y));
                _fp4_pair_58 = (uint32_t)_e2m1_pair_h_113;
                float2 pair_375 = make_float2(__uint_as_float(v4_368[2] << 16), __uint_as_float(v4_368[2] & 4294901760u));
                float2 g_376;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_376) : "l"(*(const unsigned long long*)&pair_375), "l"(*(const unsigned long long*)&mm_370));
                uint32_t _fp4_pair_59;
                uint16_t _e2m1_pair_h_114;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_114) : "f"(g_376.x), "f"(g_376.y));
                _fp4_pair_59 = (uint32_t)_e2m1_pair_h_114;
                float2 pair_377 = make_float2(__uint_as_float(v4_368[3] << 16), __uint_as_float(v4_368[3] & 4294901760u));
                float2 g_378;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_378) : "l"(*(const unsigned long long*)&pair_377), "l"(*(const unsigned long long*)&mm_370));
                uint32_t _fp4_e2m1_pack_pair_115;
                uint16_t _e2m1_pair_h_116;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_116) : "f"(g_378.x), "f"(g_378.y));
                _fp4_e2m1_pack_pair_115 = (uint32_t)_e2m1_pair_h_116;
                unsigned int word_379 = _fp4_pair_57 | _fp4_pair_58 << 8 | _fp4_pair_59 << 16 | _fp4_e2m1_pack_pair_115 << 24;
                unsigned int carry_380 = (word_379 & 2004318071) + 2004318071;
                w_324[3] = word_379 & (carry_380 | 2004318071);
                bool fodd_381 = (f & 1) != 0;
                bool fhi_382 = (f & 2) != 0;
                unsigned int t0_383 = w_324[0];
                unsigned int t1_384 = w_324[1];
                unsigned int t2_385 = w_324[2];
                unsigned int t3_386 = w_324[3];
                if (fodd_381) {
                    t0_383 = w_324[1];
                    t1_384 = w_324[0];
                    t2_385 = w_324[3];
                    t3_386 = w_324[2];
                }
                unsigned int q0_387 = t0_383;
                unsigned int q1_388 = t1_384;
                unsigned int q2_389 = t2_385;
                unsigned int q3_390 = t3_386;
                if (fhi_382) {
                    q0_387 = t2_385;
                    q1_388 = t3_386;
                    q2_389 = t0_383;
                    q3_390 = t1_384;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_318)), "r"(q0_387), "r"(q1_388), "r"(q2_389), "r"(q3_390) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_318)), "h"((uint16_t)(_e4m3x2_f32_4)) : "memory");
                int pp2_391 = lane_1 + 160;
                int base2_392 = xb_off + 64 * pp2_391;
                float bm0_393 = __uint_as_float(bmp[5] << 16);
                float bm1_394 = __uint_as_float(bmp[5] & 4294901760u);
                float _fdiv_117 = __fdiv_rn(mul_f32_rn_noftz(bm1_394, inv), 6.0f);
                float _fdiv_118 = __fdiv_rn(mul_f32_rn_noftz(bm0_393, inv), 6.0f);
                uint16_t _e4m3x2_f32_5;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_5) : "f"(_fdiv_117), "f"(_fdiv_118));
                uint32_t _e4m3x2_to_f16x2_5;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_5) : "h"((uint16_t)(_e4m3x2_f32_5)));
                float _cvt_f32_f16_10;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_10) : "h"((uint16_t)(_e4m3x2_to_f16x2_5)));
                float _cvt_f32_f16_11;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_11) : "h"((uint16_t)(_e4m3x2_to_f16x2_5 >> 16)));
                float _fdiv_rn_11 = __fdiv_rn(inv, _cvt_f32_f16_10);
                float _fdiv_rn_12 = __fdiv_rn(inv, _cvt_f32_f16_11);
                float m0_395 = 0.0f;
                float m1_396 = 0.0f;
                if (_cvt_f32_f16_10 > 0.0f) {
                    m0_395 = _fdiv_rn_11;
                }
                if (_cvt_f32_f16_11 > 0.0f) {
                    m1_396 = _fdiv_rn_12;
                }
                unsigned int w_397[4];
                int cf_398 = f ^ 0;
                unsigned int v4_399[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_399[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_399[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_399[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_399[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_392 + 16 * cf_398)));
                float m_400 = m0_395;
                if ((cf_398 & 2) != 0) {
                    m_400 = m1_396;
                }
                float2 mm_401 = make_float2(m_400, m_400);
                float2 pair_402 = make_float2(__uint_as_float(v4_399[0] << 16), __uint_as_float(v4_399[0] & 4294901760u));
                float2 g_403;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_403) : "l"(*(const unsigned long long*)&pair_402), "l"(*(const unsigned long long*)&mm_401));
                uint32_t _fp4_pair_60;
                uint16_t _e2m1_pair_h_119;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_119) : "f"(g_403.x), "f"(g_403.y));
                _fp4_pair_60 = (uint32_t)_e2m1_pair_h_119;
                float2 pair_404 = make_float2(__uint_as_float(v4_399[1] << 16), __uint_as_float(v4_399[1] & 4294901760u));
                float2 g_405;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_405) : "l"(*(const unsigned long long*)&pair_404), "l"(*(const unsigned long long*)&mm_401));
                uint32_t _fp4_pair_61;
                uint16_t _e2m1_pair_h_120;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_120) : "f"(g_405.x), "f"(g_405.y));
                _fp4_pair_61 = (uint32_t)_e2m1_pair_h_120;
                float2 pair_406 = make_float2(__uint_as_float(v4_399[2] << 16), __uint_as_float(v4_399[2] & 4294901760u));
                float2 g_407;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_407) : "l"(*(const unsigned long long*)&pair_406), "l"(*(const unsigned long long*)&mm_401));
                uint32_t _fp4_pair_62;
                uint16_t _e2m1_pair_h_121;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_121) : "f"(g_407.x), "f"(g_407.y));
                _fp4_pair_62 = (uint32_t)_e2m1_pair_h_121;
                float2 pair_408 = make_float2(__uint_as_float(v4_399[3] << 16), __uint_as_float(v4_399[3] & 4294901760u));
                float2 g_409;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_409) : "l"(*(const unsigned long long*)&pair_408), "l"(*(const unsigned long long*)&mm_401));
                uint32_t _fp4_e2m1_pack_pair_122;
                uint16_t _e2m1_pair_h_123;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_123) : "f"(g_409.x), "f"(g_409.y));
                _fp4_e2m1_pack_pair_122 = (uint32_t)_e2m1_pair_h_123;
                unsigned int word_410 = _fp4_pair_60 | _fp4_pair_61 << 8 | _fp4_pair_62 << 16 | _fp4_e2m1_pack_pair_122 << 24;
                unsigned int carry_411 = (word_410 & 2004318071) + 2004318071;
                w_397[0] = word_410 & (carry_411 | 2004318071);
                int cf_412 = f ^ 1;
                unsigned int v4_413[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_413[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_413[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_413[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_413[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_392 + 16 * cf_412)));
                float m_414 = m0_395;
                if ((cf_412 & 2) != 0) {
                    m_414 = m1_396;
                }
                float2 mm_415 = make_float2(m_414, m_414);
                float2 pair_416 = make_float2(__uint_as_float(v4_413[0] << 16), __uint_as_float(v4_413[0] & 4294901760u));
                float2 g_417;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_417) : "l"(*(const unsigned long long*)&pair_416), "l"(*(const unsigned long long*)&mm_415));
                uint32_t _fp4_pair_63;
                uint16_t _e2m1_pair_h_124;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_124) : "f"(g_417.x), "f"(g_417.y));
                _fp4_pair_63 = (uint32_t)_e2m1_pair_h_124;
                float2 pair_418 = make_float2(__uint_as_float(v4_413[1] << 16), __uint_as_float(v4_413[1] & 4294901760u));
                float2 g_419;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_419) : "l"(*(const unsigned long long*)&pair_418), "l"(*(const unsigned long long*)&mm_415));
                uint32_t _fp4_pair_64;
                uint16_t _e2m1_pair_h_125;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_125) : "f"(g_419.x), "f"(g_419.y));
                _fp4_pair_64 = (uint32_t)_e2m1_pair_h_125;
                float2 pair_420 = make_float2(__uint_as_float(v4_413[2] << 16), __uint_as_float(v4_413[2] & 4294901760u));
                float2 g_421;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_421) : "l"(*(const unsigned long long*)&pair_420), "l"(*(const unsigned long long*)&mm_415));
                uint32_t _fp4_pair_65;
                uint16_t _e2m1_pair_h_126;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_126) : "f"(g_421.x), "f"(g_421.y));
                _fp4_pair_65 = (uint32_t)_e2m1_pair_h_126;
                float2 pair_422 = make_float2(__uint_as_float(v4_413[3] << 16), __uint_as_float(v4_413[3] & 4294901760u));
                float2 g_423;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_423) : "l"(*(const unsigned long long*)&pair_422), "l"(*(const unsigned long long*)&mm_415));
                uint32_t _fp4_e2m1_pack_pair_127;
                uint16_t _e2m1_pair_h_128;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_128) : "f"(g_423.x), "f"(g_423.y));
                _fp4_e2m1_pack_pair_127 = (uint32_t)_e2m1_pair_h_128;
                unsigned int word_424 = _fp4_pair_63 | _fp4_pair_64 << 8 | _fp4_pair_65 << 16 | _fp4_e2m1_pack_pair_127 << 24;
                unsigned int carry_425 = (word_424 & 2004318071) + 2004318071;
                w_397[1] = word_424 & (carry_425 | 2004318071);
                int cf_426 = f ^ 2;
                unsigned int v4_427[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_427[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_427[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_427[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_427[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_392 + 16 * cf_426)));
                float m_428 = m0_395;
                if ((cf_426 & 2) != 0) {
                    m_428 = m1_396;
                }
                float2 mm_429 = make_float2(m_428, m_428);
                float2 pair_430 = make_float2(__uint_as_float(v4_427[0] << 16), __uint_as_float(v4_427[0] & 4294901760u));
                float2 g_431;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_431) : "l"(*(const unsigned long long*)&pair_430), "l"(*(const unsigned long long*)&mm_429));
                uint32_t _fp4_pair_66;
                uint16_t _e2m1_pair_h_129;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_129) : "f"(g_431.x), "f"(g_431.y));
                _fp4_pair_66 = (uint32_t)_e2m1_pair_h_129;
                float2 pair_432 = make_float2(__uint_as_float(v4_427[1] << 16), __uint_as_float(v4_427[1] & 4294901760u));
                float2 g_433;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_433) : "l"(*(const unsigned long long*)&pair_432), "l"(*(const unsigned long long*)&mm_429));
                uint32_t _fp4_pair_67;
                uint16_t _e2m1_pair_h_130;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_130) : "f"(g_433.x), "f"(g_433.y));
                _fp4_pair_67 = (uint32_t)_e2m1_pair_h_130;
                float2 pair_434 = make_float2(__uint_as_float(v4_427[2] << 16), __uint_as_float(v4_427[2] & 4294901760u));
                float2 g_435;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_435) : "l"(*(const unsigned long long*)&pair_434), "l"(*(const unsigned long long*)&mm_429));
                uint32_t _fp4_pair_68;
                uint16_t _e2m1_pair_h_131;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_131) : "f"(g_435.x), "f"(g_435.y));
                _fp4_pair_68 = (uint32_t)_e2m1_pair_h_131;
                float2 pair_436 = make_float2(__uint_as_float(v4_427[3] << 16), __uint_as_float(v4_427[3] & 4294901760u));
                float2 g_437;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_437) : "l"(*(const unsigned long long*)&pair_436), "l"(*(const unsigned long long*)&mm_429));
                uint32_t _fp4_e2m1_pack_pair_132;
                uint16_t _e2m1_pair_h_133;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_133) : "f"(g_437.x), "f"(g_437.y));
                _fp4_e2m1_pack_pair_132 = (uint32_t)_e2m1_pair_h_133;
                unsigned int word_438 = _fp4_pair_66 | _fp4_pair_67 << 8 | _fp4_pair_68 << 16 | _fp4_e2m1_pack_pair_132 << 24;
                unsigned int carry_439 = (word_438 & 2004318071) + 2004318071;
                w_397[2] = word_438 & (carry_439 | 2004318071);
                int cf_440 = f ^ 3;
                unsigned int v4_441[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_441[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_441[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_441[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_441[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_392 + 16 * cf_440)));
                float m_442 = m0_395;
                if ((cf_440 & 2) != 0) {
                    m_442 = m1_396;
                }
                float2 mm_443 = make_float2(m_442, m_442);
                float2 pair_444 = make_float2(__uint_as_float(v4_441[0] << 16), __uint_as_float(v4_441[0] & 4294901760u));
                float2 g_445;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_445) : "l"(*(const unsigned long long*)&pair_444), "l"(*(const unsigned long long*)&mm_443));
                uint32_t _fp4_pair_69;
                uint16_t _e2m1_pair_h_134;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_134) : "f"(g_445.x), "f"(g_445.y));
                _fp4_pair_69 = (uint32_t)_e2m1_pair_h_134;
                float2 pair_446 = make_float2(__uint_as_float(v4_441[1] << 16), __uint_as_float(v4_441[1] & 4294901760u));
                float2 g_447;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_447) : "l"(*(const unsigned long long*)&pair_446), "l"(*(const unsigned long long*)&mm_443));
                uint32_t _fp4_pair_70;
                uint16_t _e2m1_pair_h_135;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_135) : "f"(g_447.x), "f"(g_447.y));
                _fp4_pair_70 = (uint32_t)_e2m1_pair_h_135;
                float2 pair_448 = make_float2(__uint_as_float(v4_441[2] << 16), __uint_as_float(v4_441[2] & 4294901760u));
                float2 g_449;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_449) : "l"(*(const unsigned long long*)&pair_448), "l"(*(const unsigned long long*)&mm_443));
                uint32_t _fp4_pair_71;
                uint16_t _e2m1_pair_h_136;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_136) : "f"(g_449.x), "f"(g_449.y));
                _fp4_pair_71 = (uint32_t)_e2m1_pair_h_136;
                float2 pair_450 = make_float2(__uint_as_float(v4_441[3] << 16), __uint_as_float(v4_441[3] & 4294901760u));
                float2 g_451;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_451) : "l"(*(const unsigned long long*)&pair_450), "l"(*(const unsigned long long*)&mm_443));
                uint32_t _fp4_e2m1_pack_pair_137;
                uint16_t _e2m1_pair_h_138;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_138) : "f"(g_451.x), "f"(g_451.y));
                _fp4_e2m1_pack_pair_137 = (uint32_t)_e2m1_pair_h_138;
                unsigned int word_452 = _fp4_pair_69 | _fp4_pair_70 << 8 | _fp4_pair_71 << 16 | _fp4_e2m1_pack_pair_137 << 24;
                unsigned int carry_453 = (word_452 & 2004318071) + 2004318071;
                w_397[3] = word_452 & (carry_453 | 2004318071);
                bool fodd_454 = (f & 1) != 0;
                bool fhi_455 = (f & 2) != 0;
                unsigned int t0_456 = w_397[0];
                unsigned int t1_457 = w_397[1];
                unsigned int t2_458 = w_397[2];
                unsigned int t3_459 = w_397[3];
                if (fodd_454) {
                    t0_456 = w_397[1];
                    t1_457 = w_397[0];
                    t2_458 = w_397[3];
                    t3_459 = w_397[2];
                }
                unsigned int q0_460 = t0_456;
                unsigned int q1_461 = t1_457;
                unsigned int q2_462 = t2_458;
                unsigned int q3_463 = t3_459;
                if (fhi_455) {
                    q0_460 = t2_458;
                    q1_461 = t3_459;
                    q2_462 = t0_456;
                    q3_463 = t1_457;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_391)), "r"(q0_460), "r"(q1_461), "r"(q2_462), "r"(q3_463) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_391)), "h"((uint16_t)(_e4m3x2_f32_5)) : "memory");
                rs = rs_50;
                __syncwarp();
                if (n > k + 1 && lane_1 == 0) {
                    long long next_row2 = (long long)(r0 + warp_0 + 10 * (k + 1));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int stage_1 = 2 * warp_0 + (k + 1 & 1);
                    mbarrier_arrive_expect_tx(xb_full_addr + (stage_1) * 8, 12288);
                    cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)xb_off, x + (next_row2 * 6144), 12288, xb_full_addr + (stage_1) * 8);
                }
                if (lane_1 < 4) {
                    {
                        uint32_t _addr_139 = static_cast<uint32_t>(wsm_addr + (unsigned int)(ob_off + 3072 + 384 + 4 * lane_1));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_139), "f"(rs) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane_1 == 0) {
                    long long row64 = (long long)(r0 + warp_0 + 10 * k);
                    int owner = (int)(row64 % 4);
                    int ix = (int)(row64 / 4);
                    int rs_wo = (int)rs_w;
                    int rs_ro = (int)rs_r;
                    int capo = (int)capo64;
                    int rk = (int)rank64;
                    long long base_w = tab[owner];
                    long long base_r = tab[4 + owner];
                    uint8_t* dst_w = reinterpret_cast<uint8_t*>(base_w);
                    int v_off = rs_wo + (rk * capo + ix) * 3072;
                    int sc_off = rs_wo + 4 * capo * 3072 + (rk * capo + ix) * 384;
                    {
                        void* _cpbulk_dst_140 = reinterpret_cast<void*>(dst_w + v_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_140), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_141 = reinterpret_cast<void*>(dst_w + sc_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_141), "r"(wsm_addr + (unsigned int)(ob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    uint8_t* dst_r = reinterpret_cast<uint8_t*>(base_r);
                    int m_off = rs_ro + (rk * capo + ix) * 16;
                    {
                        void* _cpbulk_dst_142 = reinterpret_cast<void*>(dst_r + m_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_142), "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
            }
            int rb0 = r0 + (rank - r0 % 4 + 4) % 4;
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
                int rs_wo_1 = (int)rs_w;
                int rs_ro_1 = (int)rs_r;
                int capo_1 = (int)capo64;
                int ix_1 = (int)bix64;
                int rk_1 = (int)rank64;
                uint32_t _sysv_poll32_group_0[4];
                do {
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[0]) : "l"(reinterpret_cast<unsigned int*>(ts_rs + (rs_ro_1 + ix_1 * 16))) : "memory");
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[1]) : "l"(reinterpret_cast<unsigned int*>(ts_rs + (rs_ro_1 + (capo_1 + ix_1) * 16))) : "memory");
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[2]) : "l"(reinterpret_cast<unsigned int*>(ts_rs + (rs_ro_1 + (2 * capo_1 + ix_1) * 16))) : "memory");
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[3]) : "l"(reinterpret_cast<unsigned int*>(ts_rs + (rs_ro_1 + (3 * capo_1 + ix_1) * 16))) : "memory");
                } while (
                    (_sysv_poll32_group_0[0] == 0x80000000u) || (_sysv_poll32_group_0[1] == 0x80000000u) || (_sysv_poll32_group_0[2] == 0x80000000u) || (_sysv_poll32_group_0[3] == 0x80000000u)
                );
                float rs_s = __uint_as_float(_sysv_poll32_group_0[0]);
                float rs_s_0 = __uint_as_float(_sysv_poll32_group_0[1]);
                float rs_s_1 = __uint_as_float(_sysv_poll32_group_0[2]);
                float rs_s_2 = __uint_as_float(_sysv_poll32_group_0[3]);
                int v = rs_wo_1 + ix_1 * 3072;
                int sc = rs_wo_1 + 4 * capo_1 * 3072 + ix_1 * 384;
                int v_3 = rs_wo_1 + (capo_1 + ix_1) * 3072;
                int sc_4 = rs_wo_1 + 4 * capo_1 * 3072 + (capo_1 + ix_1) * 384;
                int v_5 = rs_wo_1 + (2 * capo_1 + ix_1) * 3072;
                int sc_6 = rs_wo_1 + 4 * capo_1 * 3072 + (2 * capo_1 + ix_1) * 384;
                int v_7 = rs_wo_1 + (3 * capo_1 + ix_1) * 3072;
                int sc_8 = rs_wo_1 + 4 * capo_1 * 3072 + (3 * capo_1 + ix_1) * 384;
                float am_1 = 0.0f;
                #pragma unroll 1
                for (int jj = 0; jj < 3; jj++) {
                    int pp0 = lane_1 + 64 * jj;
                    int pp1 = pp0 + 32;
                    uint32_t _fp4_wire_values_0[32];
                    uint32_t _fp4_wire_scales_0[8];
                    uint16_t _wire_h16_143_0;
                    uint16_t _wire_h16_143_1;
                    uint16_t _wire_h16_143_2;
                    uint16_t _wire_h16_143_3;
                    uint16_t _wire_h16_143_4;
                    uint16_t _wire_h16_143_5;
                    uint16_t _wire_h16_143_6;
                    uint16_t _wire_h16_143_7;
                    uint32_t _wire_vpend_143 = 255u;
                    uint32_t _wire_spend_143 = 255u;
                    do {
                        if (_wire_vpend_143 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_143 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_3 + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_143 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[8]), "=r"(_fp4_wire_values_0[9]), "=r"(_fp4_wire_values_0[10]), "=r"(_fp4_wire_values_0[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_5 + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_143 & 8u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[12]), "=r"(_fp4_wire_values_0[13]), "=r"(_fp4_wire_values_0[14]), "=r"(_fp4_wire_values_0[15]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_7 + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_143 & 16u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[16]), "=r"(_fp4_wire_values_0[17]), "=r"(_fp4_wire_values_0[18]), "=r"(_fp4_wire_values_0[19]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_143 & 32u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[20]), "=r"(_fp4_wire_values_0[21]), "=r"(_fp4_wire_values_0[22]), "=r"(_fp4_wire_values_0[23]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_3 + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_143 & 64u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[24]), "=r"(_fp4_wire_values_0[25]), "=r"(_fp4_wire_values_0[26]), "=r"(_fp4_wire_values_0[27]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_5 + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_143 & 128u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[28]), "=r"(_fp4_wire_values_0[29]), "=r"(_fp4_wire_values_0[30]), "=r"(_fp4_wire_values_0[31]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_7 + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_spend_143 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_143_0;
                        }
                        if (_wire_spend_143 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_4 + (int)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_143_1;
                        }
                        if (_wire_spend_143 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_6 + (int)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_0[2] = (uint32_t)_wire_h16_143_2;
                        }
                        if (_wire_spend_143 & 8u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_3) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_8 + (int)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_0[3] = (uint32_t)_wire_h16_143_3;
                        }
                        if (_wire_spend_143 & 16u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_4) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_0[4] = (uint32_t)_wire_h16_143_4;
                        }
                        if (_wire_spend_143 & 32u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_5) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_4 + (int)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_0[5] = (uint32_t)_wire_h16_143_5;
                        }
                        if (_wire_spend_143 & 64u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_6) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_6 + (int)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_0[6] = (uint32_t)_wire_h16_143_6;
                        }
                        if (_wire_spend_143 & 128u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_7) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_8 + (int)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_0[7] = (uint32_t)_wire_h16_143_7;
                        }
                        _wire_vpend_143 = ((uint32_t)((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((_fp4_wire_values_0[8] & ~((_fp4_wire_values_0[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[9] & ~((_fp4_wire_values_0[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[10] & ~((_fp4_wire_values_0[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[11] & ~((_fp4_wire_values_0[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2) | ((uint32_t)((((_fp4_wire_values_0[12] & ~((_fp4_wire_values_0[12] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[13] & ~((_fp4_wire_values_0[13] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[14] & ~((_fp4_wire_values_0[14] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[15] & ~((_fp4_wire_values_0[15] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 3) | ((uint32_t)((((_fp4_wire_values_0[16] & ~((_fp4_wire_values_0[16] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[17] & ~((_fp4_wire_values_0[17] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[18] & ~((_fp4_wire_values_0[18] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[19] & ~((_fp4_wire_values_0[19] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 4) | ((uint32_t)((((_fp4_wire_values_0[20] & ~((_fp4_wire_values_0[20] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[21] & ~((_fp4_wire_values_0[21] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[22] & ~((_fp4_wire_values_0[22] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[23] & ~((_fp4_wire_values_0[23] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 5) | ((uint32_t)((((_fp4_wire_values_0[24] & ~((_fp4_wire_values_0[24] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[25] & ~((_fp4_wire_values_0[25] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[26] & ~((_fp4_wire_values_0[26] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[27] & ~((_fp4_wire_values_0[27] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 6) | ((uint32_t)((((_fp4_wire_values_0[28] & ~((_fp4_wire_values_0[28] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[29] & ~((_fp4_wire_values_0[29] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[30] & ~((_fp4_wire_values_0[30] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[31] & ~((_fp4_wire_values_0[31] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 7);
                        _wire_spend_143 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[2] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[3] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[3] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 3) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[4] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[4] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 4) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[5] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[5] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 5) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[6] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[6] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 6) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[7] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[7] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 7);
                    } while ((_wire_vpend_143 | _wire_spend_143) != 0u);
                    float acc[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_6;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_6) : "h"((uint16_t)(_fp4_wire_scales_0[0])));
                    float _cvt_f32_f16_12;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_12) : "h"((uint16_t)(_e4m3x2_to_f16x2_6)));
                    float _cvt_f32_f16_13;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_13) : "h"((uint16_t)(_e4m3x2_to_f16x2_6 >> 16)));
                    float _mul_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(_cvt_f32_f16_12), "f"(rs_s));
                    float _mul_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(_cvt_f32_f16_13), "f"(rs_s));
                    float2 scale0 = make_float2(_mul_0, _mul_0);
                    float2 scale1 = make_float2(_mul_1, _mul_1);
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
                    float2 a_1 = make_float2(acc[0], acc[1]);
                    float2 q = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                    float2 r = fma_f32x2_rn_noftz(q, scale0, a_1);
                    acc[0] = r.x;
                    acc[1] = r.y;
                    float2 a_0 = make_float2(acc[2], acc[3]);
                    float2 q_1 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                    float2 r_2 = fma_f32x2_rn_noftz(q_1, scale0, a_0);
                    acc[2] = r_2.x;
                    acc[3] = r_2.y;
                    float2 a_3 = make_float2(acc[4], acc[5]);
                    float2 q_4 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                    float2 r_5 = fma_f32x2_rn_noftz(q_4, scale0, a_3);
                    acc[4] = r_5.x;
                    acc[5] = r_5.y;
                    float2 a_6 = make_float2(acc[6], acc[7]);
                    float2 q_7 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                    float2 r_8 = fma_f32x2_rn_noftz(q_7, scale0, a_6);
                    acc[6] = r_8.x;
                    acc[7] = r_8.y;
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
                    float2 a_9 = make_float2(acc[8], acc[9]);
                    float2 q_10 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                    float2 r_11 = fma_f32x2_rn_noftz(q_10, scale0, a_9);
                    acc[8] = r_11.x;
                    acc[9] = r_11.y;
                    float2 a_12_1 = make_float2(acc[10], acc[11]);
                    float2 q_13 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                    float2 r_14 = fma_f32x2_rn_noftz(q_13, scale0, a_12_1);
                    acc[10] = r_14.x;
                    acc[11] = r_14.y;
                    float2 a_15 = make_float2(acc[12], acc[13]);
                    float2 q_16 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                    float2 r_17 = fma_f32x2_rn_noftz(q_16, scale0, a_15);
                    acc[12] = r_17.x;
                    acc[13] = r_17.y;
                    float2 a_18 = make_float2(acc[14], acc[15]);
                    float2 q_19 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                    float2 r_20 = fma_f32x2_rn_noftz(q_19, scale0, a_18);
                    acc[14] = r_20.x;
                    acc[15] = r_20.y;
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
                    float2 a_21 = make_float2(acc[16], acc[17]);
                    float2 q_22 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                    float2 r_23 = fma_f32x2_rn_noftz(q_22, scale1, a_21);
                    acc[16] = r_23.x;
                    acc[17] = r_23.y;
                    float2 a_24 = make_float2(acc[18], acc[19]);
                    float2 q_25 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                    float2 r_26 = fma_f32x2_rn_noftz(q_25, scale1, a_24);
                    acc[18] = r_26.x;
                    acc[19] = r_26.y;
                    float2 a_27 = make_float2(acc[20], acc[21]);
                    float2 q_28 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                    float2 r_29 = fma_f32x2_rn_noftz(q_28, scale1, a_27);
                    acc[20] = r_29.x;
                    acc[21] = r_29.y;
                    float2 a_30 = make_float2(acc[22], acc[23]);
                    float2 q_31 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                    float2 r_32 = fma_f32x2_rn_noftz(q_31, scale1, a_30);
                    acc[22] = r_32.x;
                    acc[23] = r_32.y;
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
                    float2 a_33 = make_float2(acc[24], acc[25]);
                    float2 q_34 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                    float2 r_35 = fma_f32x2_rn_noftz(q_34, scale1, a_33);
                    acc[24] = r_35.x;
                    acc[25] = r_35.y;
                    float2 a_36 = make_float2(acc[26], acc[27]);
                    float2 q_37 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                    float2 r_38 = fma_f32x2_rn_noftz(q_37, scale1, a_36);
                    acc[26] = r_38.x;
                    acc[27] = r_38.y;
                    float2 a_39 = make_float2(acc[28], acc[29]);
                    float2 q_40 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                    float2 r_41 = fma_f32x2_rn_noftz(q_40, scale1, a_39);
                    acc[28] = r_41.x;
                    acc[29] = r_41.y;
                    float2 a_42_1 = make_float2(acc[30], acc[31]);
                    float2 q_43 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                    float2 r_44 = fma_f32x2_rn_noftz(q_43, scale1, a_42_1);
                    acc[30] = r_44.x;
                    acc[31] = r_44.y;
                    uint32_t _e4m3x2_to_f16x2_7;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_7) : "h"((uint16_t)(_fp4_wire_scales_0[1])));
                    float _cvt_f32_f16_14;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_14) : "h"((uint16_t)(_e4m3x2_to_f16x2_7)));
                    float _cvt_f32_f16_15;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_15) : "h"((uint16_t)(_e4m3x2_to_f16x2_7 >> 16)));
                    float _mul_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_cvt_f32_f16_14), "f"(rs_s_0));
                    float _mul_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_cvt_f32_f16_15), "f"(rs_s_0));
                    float2 scale0_45 = make_float2(_mul_2, _mul_2);
                    float2 scale1_46 = make_float2(_mul_3, _mul_3);
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
                    float2 a_47 = make_float2(acc[0], acc[1]);
                    float2 q_48 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                    float2 r_49 = fma_f32x2_rn_noftz(q_48, scale0_45, a_47);
                    acc[0] = r_49.x;
                    acc[1] = r_49.y;
                    float2 a_50 = make_float2(acc[2], acc[3]);
                    float2 q_51 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                    float2 r_52 = fma_f32x2_rn_noftz(q_51, scale0_45, a_50);
                    acc[2] = r_52.x;
                    acc[3] = r_52.y;
                    float2 a_53 = make_float2(acc[4], acc[5]);
                    float2 q_54 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                    float2 r_55 = fma_f32x2_rn_noftz(q_54, scale0_45, a_53);
                    acc[4] = r_55.x;
                    acc[5] = r_55.y;
                    float2 a_56 = make_float2(acc[6], acc[7]);
                    float2 q_57 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                    float2 r_58 = fma_f32x2_rn_noftz(q_57, scale0_45, a_56);
                    acc[6] = r_58.x;
                    acc[7] = r_58.y;
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
                    float2 a_59 = make_float2(acc[8], acc[9]);
                    float2 q_60 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                    float2 r_61 = fma_f32x2_rn_noftz(q_60, scale0_45, a_59);
                    acc[8] = r_61.x;
                    acc[9] = r_61.y;
                    float2 a_62 = make_float2(acc[10], acc[11]);
                    float2 q_63 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                    float2 r_64 = fma_f32x2_rn_noftz(q_63, scale0_45, a_62);
                    acc[10] = r_64.x;
                    acc[11] = r_64.y;
                    float2 a_65 = make_float2(acc[12], acc[13]);
                    float2 q_66 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                    float2 r_67 = fma_f32x2_rn_noftz(q_66, scale0_45, a_65);
                    acc[12] = r_67.x;
                    acc[13] = r_67.y;
                    float2 a_68 = make_float2(acc[14], acc[15]);
                    float2 q_69 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                    float2 r_70 = fma_f32x2_rn_noftz(q_69, scale0_45, a_68);
                    acc[14] = r_70.x;
                    acc[15] = r_70.y;
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
                    float2 a_71 = make_float2(acc[16], acc[17]);
                    float2 q_72 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                    float2 r_73 = fma_f32x2_rn_noftz(q_72, scale1_46, a_71);
                    acc[16] = r_73.x;
                    acc[17] = r_73.y;
                    float2 a_74 = make_float2(acc[18], acc[19]);
                    float2 q_75 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                    float2 r_76 = fma_f32x2_rn_noftz(q_75, scale1_46, a_74);
                    acc[18] = r_76.x;
                    acc[19] = r_76.y;
                    float2 a_77 = make_float2(acc[20], acc[21]);
                    float2 q_78 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                    float2 r_79 = fma_f32x2_rn_noftz(q_78, scale1_46, a_77);
                    acc[20] = r_79.x;
                    acc[21] = r_79.y;
                    float2 a_80 = make_float2(acc[22], acc[23]);
                    float2 q_81 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                    float2 r_82 = fma_f32x2_rn_noftz(q_81, scale1_46, a_80);
                    acc[22] = r_82.x;
                    acc[23] = r_82.y;
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
                    float2 a_83 = make_float2(acc[24], acc[25]);
                    float2 q_84 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                    float2 r_85 = fma_f32x2_rn_noftz(q_84, scale1_46, a_83);
                    acc[24] = r_85.x;
                    acc[25] = r_85.y;
                    float2 a_86 = make_float2(acc[26], acc[27]);
                    float2 q_87 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                    float2 r_88 = fma_f32x2_rn_noftz(q_87, scale1_46, a_86);
                    acc[26] = r_88.x;
                    acc[27] = r_88.y;
                    float2 a_89 = make_float2(acc[28], acc[29]);
                    float2 q_90 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                    float2 r_91 = fma_f32x2_rn_noftz(q_90, scale1_46, a_89);
                    acc[28] = r_91.x;
                    acc[29] = r_91.y;
                    float2 a_92 = make_float2(acc[30], acc[31]);
                    float2 q_93 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                    float2 r_94 = fma_f32x2_rn_noftz(q_93, scale1_46, a_92);
                    acc[30] = r_94.x;
                    acc[31] = r_94.y;
                    uint32_t _e4m3x2_to_f16x2_8;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_8) : "h"((uint16_t)(_fp4_wire_scales_0[2])));
                    float _cvt_f32_f16_16;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_16) : "h"((uint16_t)(_e4m3x2_to_f16x2_8)));
                    float _cvt_f32_f16_17;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_17) : "h"((uint16_t)(_e4m3x2_to_f16x2_8 >> 16)));
                    float _mul_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_cvt_f32_f16_16), "f"(rs_s_1));
                    float _mul_5;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_cvt_f32_f16_17), "f"(rs_s_1));
                    float2 scale0_95 = make_float2(_mul_4, _mul_4);
                    float2 scale1_96 = make_float2(_mul_5, _mul_5);
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
                    float2 a_97 = make_float2(acc[0], acc[1]);
                    float2 q_98 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                    float2 r_99 = fma_f32x2_rn_noftz(q_98, scale0_95, a_97);
                    acc[0] = r_99.x;
                    acc[1] = r_99.y;
                    float2 a_100 = make_float2(acc[2], acc[3]);
                    float2 q_101 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                    float2 r_102 = fma_f32x2_rn_noftz(q_101, scale0_95, a_100);
                    acc[2] = r_102.x;
                    acc[3] = r_102.y;
                    float2 a_103 = make_float2(acc[4], acc[5]);
                    float2 q_104 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                    float2 r_105 = fma_f32x2_rn_noftz(q_104, scale0_95, a_103);
                    acc[4] = r_105.x;
                    acc[5] = r_105.y;
                    float2 a_106 = make_float2(acc[6], acc[7]);
                    float2 q_107 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                    float2 r_108 = fma_f32x2_rn_noftz(q_107, scale0_95, a_106);
                    acc[6] = r_108.x;
                    acc[7] = r_108.y;
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
                    float2 a_109 = make_float2(acc[8], acc[9]);
                    float2 q_110 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                    float2 r_111 = fma_f32x2_rn_noftz(q_110, scale0_95, a_109);
                    acc[8] = r_111.x;
                    acc[9] = r_111.y;
                    float2 a_112 = make_float2(acc[10], acc[11]);
                    float2 q_113 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                    float2 r_114 = fma_f32x2_rn_noftz(q_113, scale0_95, a_112);
                    acc[10] = r_114.x;
                    acc[11] = r_114.y;
                    float2 a_115 = make_float2(acc[12], acc[13]);
                    float2 q_116 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                    float2 r_117 = fma_f32x2_rn_noftz(q_116, scale0_95, a_115);
                    acc[12] = r_117.x;
                    acc[13] = r_117.y;
                    float2 a_118 = make_float2(acc[14], acc[15]);
                    float2 q_119 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                    float2 r_120 = fma_f32x2_rn_noftz(q_119, scale0_95, a_118);
                    acc[14] = r_120.x;
                    acc[15] = r_120.y;
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
                    float2 a_121 = make_float2(acc[16], acc[17]);
                    float2 q_122 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                    float2 r_123 = fma_f32x2_rn_noftz(q_122, scale1_96, a_121);
                    acc[16] = r_123.x;
                    acc[17] = r_123.y;
                    float2 a_124 = make_float2(acc[18], acc[19]);
                    float2 q_125 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                    float2 r_126 = fma_f32x2_rn_noftz(q_125, scale1_96, a_124);
                    acc[18] = r_126.x;
                    acc[19] = r_126.y;
                    float2 a_127 = make_float2(acc[20], acc[21]);
                    float2 q_128 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                    float2 r_129 = fma_f32x2_rn_noftz(q_128, scale1_96, a_127);
                    acc[20] = r_129.x;
                    acc[21] = r_129.y;
                    float2 a_130 = make_float2(acc[22], acc[23]);
                    float2 q_131 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                    float2 r_132 = fma_f32x2_rn_noftz(q_131, scale1_96, a_130);
                    acc[22] = r_132.x;
                    acc[23] = r_132.y;
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
                    float2 a_133 = make_float2(acc[24], acc[25]);
                    float2 q_134 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                    float2 r_135 = fma_f32x2_rn_noftz(q_134, scale1_96, a_133);
                    acc[24] = r_135.x;
                    acc[25] = r_135.y;
                    float2 a_136 = make_float2(acc[26], acc[27]);
                    float2 q_137 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                    float2 r_138 = fma_f32x2_rn_noftz(q_137, scale1_96, a_136);
                    acc[26] = r_138.x;
                    acc[27] = r_138.y;
                    float2 a_139 = make_float2(acc[28], acc[29]);
                    float2 q_140 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                    float2 r_141 = fma_f32x2_rn_noftz(q_140, scale1_96, a_139);
                    acc[28] = r_141.x;
                    acc[29] = r_141.y;
                    float2 a_142 = make_float2(acc[30], acc[31]);
                    float2 q_143 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                    float2 r_144 = fma_f32x2_rn_noftz(q_143, scale1_96, a_142);
                    acc[30] = r_144.x;
                    acc[31] = r_144.y;
                    uint32_t _e4m3x2_to_f16x2_9;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_9) : "h"((uint16_t)(_fp4_wire_scales_0[3])));
                    float _cvt_f32_f16_18;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_18) : "h"((uint16_t)(_e4m3x2_to_f16x2_9)));
                    float _cvt_f32_f16_19;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_19) : "h"((uint16_t)(_e4m3x2_to_f16x2_9 >> 16)));
                    float _mul_6;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_cvt_f32_f16_18), "f"(rs_s_2));
                    float _mul_7;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_cvt_f32_f16_19), "f"(rs_s_2));
                    float2 scale0_145 = make_float2(_mul_6, _mul_6);
                    float2 scale1_146 = make_float2(_mul_7, _mul_7);
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
                    float2 a_147 = make_float2(acc[0], acc[1]);
                    float2 q_148 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                    float2 r_149 = fma_f32x2_rn_noftz(q_148, scale0_145, a_147);
                    acc[0] = r_149.x;
                    acc[1] = r_149.y;
                    float2 a_150 = make_float2(acc[2], acc[3]);
                    float2 q_151 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                    float2 r_152 = fma_f32x2_rn_noftz(q_151, scale0_145, a_150);
                    acc[2] = r_152.x;
                    acc[3] = r_152.y;
                    float2 a_153 = make_float2(acc[4], acc[5]);
                    float2 q_154 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                    float2 r_155 = fma_f32x2_rn_noftz(q_154, scale0_145, a_153);
                    acc[4] = r_155.x;
                    acc[5] = r_155.y;
                    float2 a_156 = make_float2(acc[6], acc[7]);
                    float2 q_157 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                    float2 r_158 = fma_f32x2_rn_noftz(q_157, scale0_145, a_156);
                    acc[6] = r_158.x;
                    acc[7] = r_158.y;
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
                    float2 a_159 = make_float2(acc[8], acc[9]);
                    float2 q_160 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                    float2 r_161 = fma_f32x2_rn_noftz(q_160, scale0_145, a_159);
                    acc[8] = r_161.x;
                    acc[9] = r_161.y;
                    float2 a_162 = make_float2(acc[10], acc[11]);
                    float2 q_163 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                    float2 r_164 = fma_f32x2_rn_noftz(q_163, scale0_145, a_162);
                    acc[10] = r_164.x;
                    acc[11] = r_164.y;
                    float2 a_165 = make_float2(acc[12], acc[13]);
                    float2 q_166 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                    float2 r_167 = fma_f32x2_rn_noftz(q_166, scale0_145, a_165);
                    acc[12] = r_167.x;
                    acc[13] = r_167.y;
                    float2 a_168 = make_float2(acc[14], acc[15]);
                    float2 q_169 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                    float2 r_170 = fma_f32x2_rn_noftz(q_169, scale0_145, a_168);
                    acc[14] = r_170.x;
                    acc[15] = r_170.y;
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
                    float2 a_171 = make_float2(acc[16], acc[17]);
                    float2 q_172 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                    float2 r_173 = fma_f32x2_rn_noftz(q_172, scale1_146, a_171);
                    acc[16] = r_173.x;
                    acc[17] = r_173.y;
                    float2 a_174 = make_float2(acc[18], acc[19]);
                    float2 q_175 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                    float2 r_176 = fma_f32x2_rn_noftz(q_175, scale1_146, a_174);
                    acc[18] = r_176.x;
                    acc[19] = r_176.y;
                    float2 a_177 = make_float2(acc[20], acc[21]);
                    float2 q_178 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                    float2 r_179 = fma_f32x2_rn_noftz(q_178, scale1_146, a_177);
                    acc[20] = r_179.x;
                    acc[21] = r_179.y;
                    float2 a_180 = make_float2(acc[22], acc[23]);
                    float2 q_181 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                    float2 r_182 = fma_f32x2_rn_noftz(q_181, scale1_146, a_180);
                    acc[22] = r_182.x;
                    acc[23] = r_182.y;
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
                    float2 a_183 = make_float2(acc[24], acc[25]);
                    float2 q_184 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                    float2 r_185 = fma_f32x2_rn_noftz(q_184, scale1_146, a_183);
                    acc[24] = r_185.x;
                    acc[25] = r_185.y;
                    float2 a_186 = make_float2(acc[26], acc[27]);
                    float2 q_187 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                    float2 r_188 = fma_f32x2_rn_noftz(q_187, scale1_146, a_186);
                    acc[26] = r_188.x;
                    acc[27] = r_188.y;
                    float2 a_189 = make_float2(acc[28], acc[29]);
                    float2 q_190 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                    float2 r_191 = fma_f32x2_rn_noftz(q_190, scale1_146, a_189);
                    acc[28] = r_191.x;
                    acc[29] = r_191.y;
                    float2 a_192 = make_float2(acc[30], acc[31]);
                    float2 q_193 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                    float2 r_194 = fma_f32x2_rn_noftz(q_193, scale1_146, a_192);
                    acc[30] = r_194.x;
                    acc[31] = r_194.y;
                    float cur = am_1;
                    cur = fmaxf(cur, fabsf(acc[0]));
                    cur = fmaxf(cur, fabsf(acc[1]));
                    cur = fmaxf(cur, fabsf(acc[2]));
                    cur = fmaxf(cur, fabsf(acc[3]));
                    cur = fmaxf(cur, fabsf(acc[4]));
                    cur = fmaxf(cur, fabsf(acc[5]));
                    cur = fmaxf(cur, fabsf(acc[6]));
                    cur = fmaxf(cur, fabsf(acc[7]));
                    cur = fmaxf(cur, fabsf(acc[8]));
                    cur = fmaxf(cur, fabsf(acc[9]));
                    cur = fmaxf(cur, fabsf(acc[10]));
                    cur = fmaxf(cur, fabsf(acc[11]));
                    cur = fmaxf(cur, fabsf(acc[12]));
                    cur = fmaxf(cur, fabsf(acc[13]));
                    cur = fmaxf(cur, fabsf(acc[14]));
                    cur = fmaxf(cur, fabsf(acc[15]));
                    cur = fmaxf(cur, fabsf(acc[16]));
                    cur = fmaxf(cur, fabsf(acc[17]));
                    cur = fmaxf(cur, fabsf(acc[18]));
                    cur = fmaxf(cur, fabsf(acc[19]));
                    cur = fmaxf(cur, fabsf(acc[20]));
                    cur = fmaxf(cur, fabsf(acc[21]));
                    cur = fmaxf(cur, fabsf(acc[22]));
                    cur = fmaxf(cur, fabsf(acc[23]));
                    cur = fmaxf(cur, fabsf(acc[24]));
                    cur = fmaxf(cur, fabsf(acc[25]));
                    cur = fmaxf(cur, fabsf(acc[26]));
                    cur = fmaxf(cur, fabsf(acc[27]));
                    cur = fmaxf(cur, fabsf(acc[28]));
                    cur = fmaxf(cur, fabsf(acc[29]));
                    cur = fmaxf(cur, fabsf(acc[30]));
                    cur = fmaxf(cur, fabsf(acc[31]));
                    am_1 = cur;
                    float acc_195[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_10;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_10) : "h"((uint16_t)(_fp4_wire_scales_0[4])));
                    float _cvt_f32_f16_20;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_20) : "h"((uint16_t)(_e4m3x2_to_f16x2_10)));
                    float _cvt_f32_f16_21;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_21) : "h"((uint16_t)(_e4m3x2_to_f16x2_10 >> 16)));
                    float _mul_8;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_cvt_f32_f16_20), "f"(rs_s));
                    float _mul_9;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_cvt_f32_f16_21), "f"(rs_s));
                    float2 scale0_196 = make_float2(_mul_8, _mul_8);
                    float2 scale1_197 = make_float2(_mul_9, _mul_9);
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
                    float2 a_198 = make_float2(acc_195[0], acc_195[1]);
                    float2 q_199 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                    float2 r_200 = fma_f32x2_rn_noftz(q_199, scale0_196, a_198);
                    acc_195[0] = r_200.x;
                    acc_195[1] = r_200.y;
                    float2 a_201 = make_float2(acc_195[2], acc_195[3]);
                    float2 q_202 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                    float2 r_203 = fma_f32x2_rn_noftz(q_202, scale0_196, a_201);
                    acc_195[2] = r_203.x;
                    acc_195[3] = r_203.y;
                    float2 a_204 = make_float2(acc_195[4], acc_195[5]);
                    float2 q_205 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                    float2 r_206 = fma_f32x2_rn_noftz(q_205, scale0_196, a_204);
                    acc_195[4] = r_206.x;
                    acc_195[5] = r_206.y;
                    float2 a_207 = make_float2(acc_195[6], acc_195[7]);
                    float2 q_208 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                    float2 r_209 = fma_f32x2_rn_noftz(q_208, scale0_196, a_207);
                    acc_195[6] = r_209.x;
                    acc_195[7] = r_209.y;
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
                    float2 a_210 = make_float2(acc_195[8], acc_195[9]);
                    float2 q_211 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                    float2 r_212 = fma_f32x2_rn_noftz(q_211, scale0_196, a_210);
                    acc_195[8] = r_212.x;
                    acc_195[9] = r_212.y;
                    float2 a_213 = make_float2(acc_195[10], acc_195[11]);
                    float2 q_214 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                    float2 r_215 = fma_f32x2_rn_noftz(q_214, scale0_196, a_213);
                    acc_195[10] = r_215.x;
                    acc_195[11] = r_215.y;
                    float2 a_216 = make_float2(acc_195[12], acc_195[13]);
                    float2 q_217 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                    float2 r_218 = fma_f32x2_rn_noftz(q_217, scale0_196, a_216);
                    acc_195[12] = r_218.x;
                    acc_195[13] = r_218.y;
                    float2 a_219 = make_float2(acc_195[14], acc_195[15]);
                    float2 q_220 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                    float2 r_221 = fma_f32x2_rn_noftz(q_220, scale0_196, a_219);
                    acc_195[14] = r_221.x;
                    acc_195[15] = r_221.y;
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
                    float2 a_222 = make_float2(acc_195[16], acc_195[17]);
                    float2 q_223 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                    float2 r_224 = fma_f32x2_rn_noftz(q_223, scale1_197, a_222);
                    acc_195[16] = r_224.x;
                    acc_195[17] = r_224.y;
                    float2 a_225 = make_float2(acc_195[18], acc_195[19]);
                    float2 q_226 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                    float2 r_227 = fma_f32x2_rn_noftz(q_226, scale1_197, a_225);
                    acc_195[18] = r_227.x;
                    acc_195[19] = r_227.y;
                    float2 a_228 = make_float2(acc_195[20], acc_195[21]);
                    float2 q_229 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                    float2 r_230 = fma_f32x2_rn_noftz(q_229, scale1_197, a_228);
                    acc_195[20] = r_230.x;
                    acc_195[21] = r_230.y;
                    float2 a_231 = make_float2(acc_195[22], acc_195[23]);
                    float2 q_232 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                    float2 r_233 = fma_f32x2_rn_noftz(q_232, scale1_197, a_231);
                    acc_195[22] = r_233.x;
                    acc_195[23] = r_233.y;
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
                    float2 a_234 = make_float2(acc_195[24], acc_195[25]);
                    float2 q_235 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                    float2 r_236 = fma_f32x2_rn_noftz(q_235, scale1_197, a_234);
                    acc_195[24] = r_236.x;
                    acc_195[25] = r_236.y;
                    float2 a_237 = make_float2(acc_195[26], acc_195[27]);
                    float2 q_238 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                    float2 r_239 = fma_f32x2_rn_noftz(q_238, scale1_197, a_237);
                    acc_195[26] = r_239.x;
                    acc_195[27] = r_239.y;
                    float2 a_240 = make_float2(acc_195[28], acc_195[29]);
                    float2 q_241 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                    float2 r_242 = fma_f32x2_rn_noftz(q_241, scale1_197, a_240);
                    acc_195[28] = r_242.x;
                    acc_195[29] = r_242.y;
                    float2 a_243 = make_float2(acc_195[30], acc_195[31]);
                    float2 q_244 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                    float2 r_245 = fma_f32x2_rn_noftz(q_244, scale1_197, a_243);
                    acc_195[30] = r_245.x;
                    acc_195[31] = r_245.y;
                    uint32_t _e4m3x2_to_f16x2_11;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_11) : "h"((uint16_t)(_fp4_wire_scales_0[5])));
                    float _cvt_f32_f16_22;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_22) : "h"((uint16_t)(_e4m3x2_to_f16x2_11)));
                    float _cvt_f32_f16_23;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_23) : "h"((uint16_t)(_e4m3x2_to_f16x2_11 >> 16)));
                    float _mul_10;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(_cvt_f32_f16_22), "f"(rs_s_0));
                    float _mul_11;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_cvt_f32_f16_23), "f"(rs_s_0));
                    float2 scale0_246 = make_float2(_mul_10, _mul_10);
                    float2 scale1_247 = make_float2(_mul_11, _mul_11);
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
                    float2 a_248 = make_float2(acc_195[0], acc_195[1]);
                    float2 q_249 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                    float2 r_250 = fma_f32x2_rn_noftz(q_249, scale0_246, a_248);
                    acc_195[0] = r_250.x;
                    acc_195[1] = r_250.y;
                    float2 a_251 = make_float2(acc_195[2], acc_195[3]);
                    float2 q_252 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                    float2 r_253 = fma_f32x2_rn_noftz(q_252, scale0_246, a_251);
                    acc_195[2] = r_253.x;
                    acc_195[3] = r_253.y;
                    float2 a_254 = make_float2(acc_195[4], acc_195[5]);
                    float2 q_255 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                    float2 r_256 = fma_f32x2_rn_noftz(q_255, scale0_246, a_254);
                    acc_195[4] = r_256.x;
                    acc_195[5] = r_256.y;
                    float2 a_257 = make_float2(acc_195[6], acc_195[7]);
                    float2 q_258 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                    float2 r_259 = fma_f32x2_rn_noftz(q_258, scale0_246, a_257);
                    acc_195[6] = r_259.x;
                    acc_195[7] = r_259.y;
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
                    float2 a_260 = make_float2(acc_195[8], acc_195[9]);
                    float2 q_261 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                    float2 r_262 = fma_f32x2_rn_noftz(q_261, scale0_246, a_260);
                    acc_195[8] = r_262.x;
                    acc_195[9] = r_262.y;
                    float2 a_263 = make_float2(acc_195[10], acc_195[11]);
                    float2 q_264 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                    float2 r_265 = fma_f32x2_rn_noftz(q_264, scale0_246, a_263);
                    acc_195[10] = r_265.x;
                    acc_195[11] = r_265.y;
                    float2 a_266 = make_float2(acc_195[12], acc_195[13]);
                    float2 q_267 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                    float2 r_268 = fma_f32x2_rn_noftz(q_267, scale0_246, a_266);
                    acc_195[12] = r_268.x;
                    acc_195[13] = r_268.y;
                    float2 a_269 = make_float2(acc_195[14], acc_195[15]);
                    float2 q_270 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                    float2 r_271 = fma_f32x2_rn_noftz(q_270, scale0_246, a_269);
                    acc_195[14] = r_271.x;
                    acc_195[15] = r_271.y;
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
                    float2 a_272 = make_float2(acc_195[16], acc_195[17]);
                    float2 q_273 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                    float2 r_274 = fma_f32x2_rn_noftz(q_273, scale1_247, a_272);
                    acc_195[16] = r_274.x;
                    acc_195[17] = r_274.y;
                    float2 a_275 = make_float2(acc_195[18], acc_195[19]);
                    float2 q_276 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                    float2 r_277 = fma_f32x2_rn_noftz(q_276, scale1_247, a_275);
                    acc_195[18] = r_277.x;
                    acc_195[19] = r_277.y;
                    float2 a_278 = make_float2(acc_195[20], acc_195[21]);
                    float2 q_279 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                    float2 r_280 = fma_f32x2_rn_noftz(q_279, scale1_247, a_278);
                    acc_195[20] = r_280.x;
                    acc_195[21] = r_280.y;
                    float2 a_281 = make_float2(acc_195[22], acc_195[23]);
                    float2 q_282 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                    float2 r_283 = fma_f32x2_rn_noftz(q_282, scale1_247, a_281);
                    acc_195[22] = r_283.x;
                    acc_195[23] = r_283.y;
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
                    float2 a_284 = make_float2(acc_195[24], acc_195[25]);
                    float2 q_285 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                    float2 r_286 = fma_f32x2_rn_noftz(q_285, scale1_247, a_284);
                    acc_195[24] = r_286.x;
                    acc_195[25] = r_286.y;
                    float2 a_287 = make_float2(acc_195[26], acc_195[27]);
                    float2 q_288 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                    float2 r_289 = fma_f32x2_rn_noftz(q_288, scale1_247, a_287);
                    acc_195[26] = r_289.x;
                    acc_195[27] = r_289.y;
                    float2 a_290 = make_float2(acc_195[28], acc_195[29]);
                    float2 q_291 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                    float2 r_292 = fma_f32x2_rn_noftz(q_291, scale1_247, a_290);
                    acc_195[28] = r_292.x;
                    acc_195[29] = r_292.y;
                    float2 a_293 = make_float2(acc_195[30], acc_195[31]);
                    float2 q_294 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                    float2 r_295 = fma_f32x2_rn_noftz(q_294, scale1_247, a_293);
                    acc_195[30] = r_295.x;
                    acc_195[31] = r_295.y;
                    uint32_t _e4m3x2_to_f16x2_12;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_12) : "h"((uint16_t)(_fp4_wire_scales_0[6])));
                    float _cvt_f32_f16_24;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_24) : "h"((uint16_t)(_e4m3x2_to_f16x2_12)));
                    float _cvt_f32_f16_25;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_25) : "h"((uint16_t)(_e4m3x2_to_f16x2_12 >> 16)));
                    float _mul_12;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(_cvt_f32_f16_24), "f"(rs_s_1));
                    float _mul_13;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(_cvt_f32_f16_25), "f"(rs_s_1));
                    float2 scale0_296 = make_float2(_mul_12, _mul_12);
                    float2 scale1_297 = make_float2(_mul_13, _mul_13);
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
                    float2 a_298 = make_float2(acc_195[0], acc_195[1]);
                    float2 q_299 = make_float2(_unpack_fp4_24[0], _unpack_fp4_24[1]);
                    float2 r_300 = fma_f32x2_rn_noftz(q_299, scale0_296, a_298);
                    acc_195[0] = r_300.x;
                    acc_195[1] = r_300.y;
                    float2 a_301 = make_float2(acc_195[2], acc_195[3]);
                    float2 q_302 = make_float2(_unpack_fp4_24[2], _unpack_fp4_24[3]);
                    float2 r_303 = fma_f32x2_rn_noftz(q_302, scale0_296, a_301);
                    acc_195[2] = r_303.x;
                    acc_195[3] = r_303.y;
                    float2 a_304 = make_float2(acc_195[4], acc_195[5]);
                    float2 q_305 = make_float2(_unpack_fp4_24[4], _unpack_fp4_24[5]);
                    float2 r_306 = fma_f32x2_rn_noftz(q_305, scale0_296, a_304);
                    acc_195[4] = r_306.x;
                    acc_195[5] = r_306.y;
                    float2 a_307 = make_float2(acc_195[6], acc_195[7]);
                    float2 q_308 = make_float2(_unpack_fp4_24[6], _unpack_fp4_24[7]);
                    float2 r_309 = fma_f32x2_rn_noftz(q_308, scale0_296, a_307);
                    acc_195[6] = r_309.x;
                    acc_195[7] = r_309.y;
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
                    float2 a_310 = make_float2(acc_195[8], acc_195[9]);
                    float2 q_311 = make_float2(_unpack_fp4_25[0], _unpack_fp4_25[1]);
                    float2 r_312 = fma_f32x2_rn_noftz(q_311, scale0_296, a_310);
                    acc_195[8] = r_312.x;
                    acc_195[9] = r_312.y;
                    float2 a_313 = make_float2(acc_195[10], acc_195[11]);
                    float2 q_314 = make_float2(_unpack_fp4_25[2], _unpack_fp4_25[3]);
                    float2 r_315 = fma_f32x2_rn_noftz(q_314, scale0_296, a_313);
                    acc_195[10] = r_315.x;
                    acc_195[11] = r_315.y;
                    float2 a_316 = make_float2(acc_195[12], acc_195[13]);
                    float2 q_317 = make_float2(_unpack_fp4_25[4], _unpack_fp4_25[5]);
                    float2 r_318 = fma_f32x2_rn_noftz(q_317, scale0_296, a_316);
                    acc_195[12] = r_318.x;
                    acc_195[13] = r_318.y;
                    float2 a_319 = make_float2(acc_195[14], acc_195[15]);
                    float2 q_320 = make_float2(_unpack_fp4_25[6], _unpack_fp4_25[7]);
                    float2 r_321 = fma_f32x2_rn_noftz(q_320, scale0_296, a_319);
                    acc_195[14] = r_321.x;
                    acc_195[15] = r_321.y;
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
                    float2 a_322 = make_float2(acc_195[16], acc_195[17]);
                    float2 q_323 = make_float2(_unpack_fp4_26[0], _unpack_fp4_26[1]);
                    float2 r_324 = fma_f32x2_rn_noftz(q_323, scale1_297, a_322);
                    acc_195[16] = r_324.x;
                    acc_195[17] = r_324.y;
                    float2 a_325 = make_float2(acc_195[18], acc_195[19]);
                    float2 q_326 = make_float2(_unpack_fp4_26[2], _unpack_fp4_26[3]);
                    float2 r_327 = fma_f32x2_rn_noftz(q_326, scale1_297, a_325);
                    acc_195[18] = r_327.x;
                    acc_195[19] = r_327.y;
                    float2 a_328 = make_float2(acc_195[20], acc_195[21]);
                    float2 q_329 = make_float2(_unpack_fp4_26[4], _unpack_fp4_26[5]);
                    float2 r_330 = fma_f32x2_rn_noftz(q_329, scale1_297, a_328);
                    acc_195[20] = r_330.x;
                    acc_195[21] = r_330.y;
                    float2 a_331 = make_float2(acc_195[22], acc_195[23]);
                    float2 q_332 = make_float2(_unpack_fp4_26[6], _unpack_fp4_26[7]);
                    float2 r_333 = fma_f32x2_rn_noftz(q_332, scale1_297, a_331);
                    acc_195[22] = r_333.x;
                    acc_195[23] = r_333.y;
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
                    float2 a_334 = make_float2(acc_195[24], acc_195[25]);
                    float2 q_335 = make_float2(_unpack_fp4_27[0], _unpack_fp4_27[1]);
                    float2 r_336 = fma_f32x2_rn_noftz(q_335, scale1_297, a_334);
                    acc_195[24] = r_336.x;
                    acc_195[25] = r_336.y;
                    float2 a_337 = make_float2(acc_195[26], acc_195[27]);
                    float2 q_338 = make_float2(_unpack_fp4_27[2], _unpack_fp4_27[3]);
                    float2 r_339 = fma_f32x2_rn_noftz(q_338, scale1_297, a_337);
                    acc_195[26] = r_339.x;
                    acc_195[27] = r_339.y;
                    float2 a_340 = make_float2(acc_195[28], acc_195[29]);
                    float2 q_341 = make_float2(_unpack_fp4_27[4], _unpack_fp4_27[5]);
                    float2 r_342 = fma_f32x2_rn_noftz(q_341, scale1_297, a_340);
                    acc_195[28] = r_342.x;
                    acc_195[29] = r_342.y;
                    float2 a_343 = make_float2(acc_195[30], acc_195[31]);
                    float2 q_344 = make_float2(_unpack_fp4_27[6], _unpack_fp4_27[7]);
                    float2 r_345 = fma_f32x2_rn_noftz(q_344, scale1_297, a_343);
                    acc_195[30] = r_345.x;
                    acc_195[31] = r_345.y;
                    uint32_t _e4m3x2_to_f16x2_13;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_13) : "h"((uint16_t)(_fp4_wire_scales_0[7])));
                    float _cvt_f32_f16_26;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_26) : "h"((uint16_t)(_e4m3x2_to_f16x2_13)));
                    float _cvt_f32_f16_27;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_27) : "h"((uint16_t)(_e4m3x2_to_f16x2_13 >> 16)));
                    float _mul_14;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(_cvt_f32_f16_26), "f"(rs_s_2));
                    float _mul_15;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(_cvt_f32_f16_27), "f"(rs_s_2));
                    float2 scale0_346 = make_float2(_mul_14, _mul_14);
                    float2 scale1_347 = make_float2(_mul_15, _mul_15);
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
                    float2 a_348 = make_float2(acc_195[0], acc_195[1]);
                    float2 q_349 = make_float2(_unpack_fp4_28[0], _unpack_fp4_28[1]);
                    float2 r_350 = fma_f32x2_rn_noftz(q_349, scale0_346, a_348);
                    acc_195[0] = r_350.x;
                    acc_195[1] = r_350.y;
                    float2 a_351 = make_float2(acc_195[2], acc_195[3]);
                    float2 q_352 = make_float2(_unpack_fp4_28[2], _unpack_fp4_28[3]);
                    float2 r_353 = fma_f32x2_rn_noftz(q_352, scale0_346, a_351);
                    acc_195[2] = r_353.x;
                    acc_195[3] = r_353.y;
                    float2 a_354 = make_float2(acc_195[4], acc_195[5]);
                    float2 q_355 = make_float2(_unpack_fp4_28[4], _unpack_fp4_28[5]);
                    float2 r_356 = fma_f32x2_rn_noftz(q_355, scale0_346, a_354);
                    acc_195[4] = r_356.x;
                    acc_195[5] = r_356.y;
                    float2 a_357 = make_float2(acc_195[6], acc_195[7]);
                    float2 q_358 = make_float2(_unpack_fp4_28[6], _unpack_fp4_28[7]);
                    float2 r_359 = fma_f32x2_rn_noftz(q_358, scale0_346, a_357);
                    acc_195[6] = r_359.x;
                    acc_195[7] = r_359.y;
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
                    float2 a_360 = make_float2(acc_195[8], acc_195[9]);
                    float2 q_361 = make_float2(_unpack_fp4_29[0], _unpack_fp4_29[1]);
                    float2 r_362 = fma_f32x2_rn_noftz(q_361, scale0_346, a_360);
                    acc_195[8] = r_362.x;
                    acc_195[9] = r_362.y;
                    float2 a_363 = make_float2(acc_195[10], acc_195[11]);
                    float2 q_364 = make_float2(_unpack_fp4_29[2], _unpack_fp4_29[3]);
                    float2 r_365 = fma_f32x2_rn_noftz(q_364, scale0_346, a_363);
                    acc_195[10] = r_365.x;
                    acc_195[11] = r_365.y;
                    float2 a_366 = make_float2(acc_195[12], acc_195[13]);
                    float2 q_367 = make_float2(_unpack_fp4_29[4], _unpack_fp4_29[5]);
                    float2 r_368 = fma_f32x2_rn_noftz(q_367, scale0_346, a_366);
                    acc_195[12] = r_368.x;
                    acc_195[13] = r_368.y;
                    float2 a_369 = make_float2(acc_195[14], acc_195[15]);
                    float2 q_370 = make_float2(_unpack_fp4_29[6], _unpack_fp4_29[7]);
                    float2 r_371 = fma_f32x2_rn_noftz(q_370, scale0_346, a_369);
                    acc_195[14] = r_371.x;
                    acc_195[15] = r_371.y;
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
                    float2 a_372 = make_float2(acc_195[16], acc_195[17]);
                    float2 q_373 = make_float2(_unpack_fp4_30[0], _unpack_fp4_30[1]);
                    float2 r_374 = fma_f32x2_rn_noftz(q_373, scale1_347, a_372);
                    acc_195[16] = r_374.x;
                    acc_195[17] = r_374.y;
                    float2 a_375 = make_float2(acc_195[18], acc_195[19]);
                    float2 q_376 = make_float2(_unpack_fp4_30[2], _unpack_fp4_30[3]);
                    float2 r_377 = fma_f32x2_rn_noftz(q_376, scale1_347, a_375);
                    acc_195[18] = r_377.x;
                    acc_195[19] = r_377.y;
                    float2 a_378 = make_float2(acc_195[20], acc_195[21]);
                    float2 q_379 = make_float2(_unpack_fp4_30[4], _unpack_fp4_30[5]);
                    float2 r_380 = fma_f32x2_rn_noftz(q_379, scale1_347, a_378);
                    acc_195[20] = r_380.x;
                    acc_195[21] = r_380.y;
                    float2 a_381 = make_float2(acc_195[22], acc_195[23]);
                    float2 q_382 = make_float2(_unpack_fp4_30[6], _unpack_fp4_30[7]);
                    float2 r_383 = fma_f32x2_rn_noftz(q_382, scale1_347, a_381);
                    acc_195[22] = r_383.x;
                    acc_195[23] = r_383.y;
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
                    float2 a_384 = make_float2(acc_195[24], acc_195[25]);
                    float2 q_385 = make_float2(_unpack_fp4_31[0], _unpack_fp4_31[1]);
                    float2 r_386 = fma_f32x2_rn_noftz(q_385, scale1_347, a_384);
                    acc_195[24] = r_386.x;
                    acc_195[25] = r_386.y;
                    float2 a_387 = make_float2(acc_195[26], acc_195[27]);
                    float2 q_388 = make_float2(_unpack_fp4_31[2], _unpack_fp4_31[3]);
                    float2 r_389 = fma_f32x2_rn_noftz(q_388, scale1_347, a_387);
                    acc_195[26] = r_389.x;
                    acc_195[27] = r_389.y;
                    float2 a_390 = make_float2(acc_195[28], acc_195[29]);
                    float2 q_391 = make_float2(_unpack_fp4_31[4], _unpack_fp4_31[5]);
                    float2 r_392 = fma_f32x2_rn_noftz(q_391, scale1_347, a_390);
                    acc_195[28] = r_392.x;
                    acc_195[29] = r_392.y;
                    float2 a_393 = make_float2(acc_195[30], acc_195[31]);
                    float2 q_394 = make_float2(_unpack_fp4_31[6], _unpack_fp4_31[7]);
                    float2 r_395 = fma_f32x2_rn_noftz(q_394, scale1_347, a_393);
                    acc_195[30] = r_395.x;
                    acc_195[31] = r_395.y;
                    float cur_396 = am_1;
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[0]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[1]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[2]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[3]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[4]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[5]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[6]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[7]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[8]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[9]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[10]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[11]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[12]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[13]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[14]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[15]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[16]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[17]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[18]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[19]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[20]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[21]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[22]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[23]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[24]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[25]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[26]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[27]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[28]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[29]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[30]));
                    cur_396 = fmaxf(cur_396, fabsf(acc_195[31]));
                    am_1 = cur_396;
                }
                float rs2 = 1.0f;
                float _warp_reduce_144 = am_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_144 = max_noftz(_warp_reduce_144, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_144, offset));
                am_1 = _warp_reduce_144;
                float _fdiv_rn_13 = __fdiv_rn(2688.0f, am_1);
                float inv_1 = 0.0f;
                if (am_1 > 0.0f) {
                    inv_1 = _fdiv_rn_13;
                }
                float rs_1 = __fdiv_rn(am_1, 2688.0f);
                rs2 = rs_1;
                #pragma unroll 1
                for (int j = 0; j < 6; j++) {
                    int pp_1 = lane_1 + 32 * j;
                    unsigned int _vec_load_0[4];
                    {
                        uint32_t _uv4_145_0;
                        uint32_t _uv4_145_1;
                        uint32_t _uv4_145_2;
                        uint32_t _uv4_145_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_145_0), "=r"(_uv4_145_1), "=r"(_uv4_145_2), "=r"(_uv4_145_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp_1))) + (0))) : "memory");
                        _vec_load_0[0 + 0] = _uv4_145_0;
                        _vec_load_0[0 + 1] = _uv4_145_1;
                        _vec_load_0[0 + 2] = _uv4_145_2;
                        _vec_load_0[0 + 3] = _uv4_145_3;
                    }
                    unsigned int _vec_load_1[1];
                    {
                        uint32_t _scalar_bits_146;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_146) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc + (int)(4 * (pp_1 >> 1)))) + (0))) : "memory");
                        _vec_load_1[0] = (unsigned int)_scalar_bits_146;
                    }
                    unsigned int c = _vec_load_1[0] >> (unsigned int)(16 * (pp_1 & 1)) & 65535;
                    unsigned int _vec_load_2[4];
                    {
                        uint32_t _uv4_147_0;
                        uint32_t _uv4_147_1;
                        uint32_t _uv4_147_2;
                        uint32_t _uv4_147_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_147_0), "=r"(_uv4_147_1), "=r"(_uv4_147_2), "=r"(_uv4_147_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (v_3 + (int)(16 * pp_1))) + (0))) : "memory");
                        _vec_load_2[0 + 0] = _uv4_147_0;
                        _vec_load_2[0 + 1] = _uv4_147_1;
                        _vec_load_2[0 + 2] = _uv4_147_2;
                        _vec_load_2[0 + 3] = _uv4_147_3;
                    }
                    unsigned int _vec_load_3[1];
                    {
                        uint32_t _scalar_bits_148;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_148) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_4 + (int)(4 * (pp_1 >> 1)))) + (0))) : "memory");
                        _vec_load_3[0] = (unsigned int)_scalar_bits_148;
                    }
                    unsigned int c_0 = _vec_load_3[0] >> (unsigned int)(16 * (pp_1 & 1)) & 65535;
                    unsigned int _vec_load_4[4];
                    {
                        uint32_t _uv4_149_0;
                        uint32_t _uv4_149_1;
                        uint32_t _uv4_149_2;
                        uint32_t _uv4_149_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_149_0), "=r"(_uv4_149_1), "=r"(_uv4_149_2), "=r"(_uv4_149_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (v_5 + (int)(16 * pp_1))) + (0))) : "memory");
                        _vec_load_4[0 + 0] = _uv4_149_0;
                        _vec_load_4[0 + 1] = _uv4_149_1;
                        _vec_load_4[0 + 2] = _uv4_149_2;
                        _vec_load_4[0 + 3] = _uv4_149_3;
                    }
                    unsigned int _vec_load_5[1];
                    {
                        uint32_t _scalar_bits_150;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_150) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_6 + (int)(4 * (pp_1 >> 1)))) + (0))) : "memory");
                        _vec_load_5[0] = (unsigned int)_scalar_bits_150;
                    }
                    unsigned int c_1 = _vec_load_5[0] >> (unsigned int)(16 * (pp_1 & 1)) & 65535;
                    unsigned int _vec_load_6[4];
                    {
                        uint32_t _uv4_151_0;
                        uint32_t _uv4_151_1;
                        uint32_t _uv4_151_2;
                        uint32_t _uv4_151_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_151_0), "=r"(_uv4_151_1), "=r"(_uv4_151_2), "=r"(_uv4_151_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (v_7 + (int)(16 * pp_1))) + (0))) : "memory");
                        _vec_load_6[0 + 0] = _uv4_151_0;
                        _vec_load_6[0 + 1] = _uv4_151_1;
                        _vec_load_6[0 + 2] = _uv4_151_2;
                        _vec_load_6[0 + 3] = _uv4_151_3;
                    }
                    unsigned int _vec_load_7[1];
                    {
                        uint32_t _scalar_bits_152;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_152) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_8 + (int)(4 * (pp_1 >> 1)))) + (0))) : "memory");
                        _vec_load_7[0] = (unsigned int)_scalar_bits_152;
                    }
                    unsigned int c_2 = _vec_load_7[0] >> (unsigned int)(16 * (pp_1 & 1)) & 65535;
                    float acc2[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_14;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_14) : "h"((uint16_t)(c)));
                    float _cvt_f32_f16_28;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_28) : "h"((uint16_t)(_e4m3x2_to_f16x2_14)));
                    float _cvt_f32_f16_29;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_29) : "h"((uint16_t)(_e4m3x2_to_f16x2_14 >> 16)));
                    float _mul_16;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_cvt_f32_f16_28), "f"(rs_s));
                    float _mul_17;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_cvt_f32_f16_29), "f"(rs_s));
                    float2 scale0_1 = make_float2(_mul_16, _mul_16);
                    float2 scale1_1 = make_float2(_mul_17, _mul_17);
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
                    float2 a_4 = make_float2(acc2[0], acc2[1]);
                    float2 q_2 = make_float2(_unpack_fp4_32[0], _unpack_fp4_32[1]);
                    float2 r_1 = fma_f32x2_rn_noftz(q_2, scale0_1, a_4);
                    acc2[0] = r_1.x;
                    acc2[1] = r_1.y;
                    float2 a_3_1 = make_float2(acc2[2], acc2[3]);
                    float2 q_4_1 = make_float2(_unpack_fp4_32[2], _unpack_fp4_32[3]);
                    float2 r_5_1 = fma_f32x2_rn_noftz(q_4_1, scale0_1, a_3_1);
                    acc2[2] = r_5_1.x;
                    acc2[3] = r_5_1.y;
                    float2 a_6_1 = make_float2(acc2[4], acc2[5]);
                    float2 q_7_1 = make_float2(_unpack_fp4_32[4], _unpack_fp4_32[5]);
                    float2 r_8_1 = fma_f32x2_rn_noftz(q_7_1, scale0_1, a_6_1);
                    acc2[4] = r_8_1.x;
                    acc2[5] = r_8_1.y;
                    float2 a_9_1 = make_float2(acc2[6], acc2[7]);
                    float2 q_10_1 = make_float2(_unpack_fp4_32[6], _unpack_fp4_32[7]);
                    float2 r_11_1 = fma_f32x2_rn_noftz(q_10_1, scale0_1, a_9_1);
                    acc2[6] = r_11_1.x;
                    acc2[7] = r_11_1.y;
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
                    float2 a_12_2 = make_float2(acc2[8], acc2[9]);
                    float2 q_13_1 = make_float2(_unpack_fp4_33[0], _unpack_fp4_33[1]);
                    float2 r_14_1 = fma_f32x2_rn_noftz(q_13_1, scale0_1, a_12_2);
                    acc2[8] = r_14_1.x;
                    acc2[9] = r_14_1.y;
                    float2 a_15_1 = make_float2(acc2[10], acc2[11]);
                    float2 q_16_1 = make_float2(_unpack_fp4_33[2], _unpack_fp4_33[3]);
                    float2 r_17_1 = fma_f32x2_rn_noftz(q_16_1, scale0_1, a_15_1);
                    acc2[10] = r_17_1.x;
                    acc2[11] = r_17_1.y;
                    float2 a_18_1 = make_float2(acc2[12], acc2[13]);
                    float2 q_19_1 = make_float2(_unpack_fp4_33[4], _unpack_fp4_33[5]);
                    float2 r_20_1 = fma_f32x2_rn_noftz(q_19_1, scale0_1, a_18_1);
                    acc2[12] = r_20_1.x;
                    acc2[13] = r_20_1.y;
                    float2 a_21_1 = make_float2(acc2[14], acc2[15]);
                    float2 q_22_1 = make_float2(_unpack_fp4_33[6], _unpack_fp4_33[7]);
                    float2 r_23_1 = fma_f32x2_rn_noftz(q_22_1, scale0_1, a_21_1);
                    acc2[14] = r_23_1.x;
                    acc2[15] = r_23_1.y;
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
                    float2 a_24_1 = make_float2(acc2[16], acc2[17]);
                    float2 q_25_1 = make_float2(_unpack_fp4_34[0], _unpack_fp4_34[1]);
                    float2 r_26_1 = fma_f32x2_rn_noftz(q_25_1, scale1_1, a_24_1);
                    acc2[16] = r_26_1.x;
                    acc2[17] = r_26_1.y;
                    float2 a_27_1 = make_float2(acc2[18], acc2[19]);
                    float2 q_28_1 = make_float2(_unpack_fp4_34[2], _unpack_fp4_34[3]);
                    float2 r_29_1 = fma_f32x2_rn_noftz(q_28_1, scale1_1, a_27_1);
                    acc2[18] = r_29_1.x;
                    acc2[19] = r_29_1.y;
                    float2 a_30_1 = make_float2(acc2[20], acc2[21]);
                    float2 q_31_1 = make_float2(_unpack_fp4_34[4], _unpack_fp4_34[5]);
                    float2 r_32_1 = fma_f32x2_rn_noftz(q_31_1, scale1_1, a_30_1);
                    acc2[20] = r_32_1.x;
                    acc2[21] = r_32_1.y;
                    float2 a_33_1 = make_float2(acc2[22], acc2[23]);
                    float2 q_34_1 = make_float2(_unpack_fp4_34[6], _unpack_fp4_34[7]);
                    float2 r_35_1 = fma_f32x2_rn_noftz(q_34_1, scale1_1, a_33_1);
                    acc2[22] = r_35_1.x;
                    acc2[23] = r_35_1.y;
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
                    float2 a_36_1 = make_float2(acc2[24], acc2[25]);
                    float2 q_37_1 = make_float2(_unpack_fp4_35[0], _unpack_fp4_35[1]);
                    float2 r_38_1 = fma_f32x2_rn_noftz(q_37_1, scale1_1, a_36_1);
                    acc2[24] = r_38_1.x;
                    acc2[25] = r_38_1.y;
                    float2 a_39_1 = make_float2(acc2[26], acc2[27]);
                    float2 q_40_1 = make_float2(_unpack_fp4_35[2], _unpack_fp4_35[3]);
                    float2 r_41_1 = fma_f32x2_rn_noftz(q_40_1, scale1_1, a_39_1);
                    acc2[26] = r_41_1.x;
                    acc2[27] = r_41_1.y;
                    float2 a_42_2 = make_float2(acc2[28], acc2[29]);
                    float2 q_43_1 = make_float2(_unpack_fp4_35[4], _unpack_fp4_35[5]);
                    float2 r_44_1 = fma_f32x2_rn_noftz(q_43_1, scale1_1, a_42_2);
                    acc2[28] = r_44_1.x;
                    acc2[29] = r_44_1.y;
                    float2 a_45 = make_float2(acc2[30], acc2[31]);
                    float2 q_46 = make_float2(_unpack_fp4_35[6], _unpack_fp4_35[7]);
                    float2 r_47 = fma_f32x2_rn_noftz(q_46, scale1_1, a_45);
                    acc2[30] = r_47.x;
                    acc2[31] = r_47.y;
                    uint32_t _e4m3x2_to_f16x2_15;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_15) : "h"((uint16_t)(c_0)));
                    float _cvt_f32_f16_30;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_30) : "h"((uint16_t)(_e4m3x2_to_f16x2_15)));
                    float _cvt_f32_f16_31;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_31) : "h"((uint16_t)(_e4m3x2_to_f16x2_15 >> 16)));
                    float _mul_18;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(_cvt_f32_f16_30), "f"(rs_s_0));
                    float _mul_19;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_cvt_f32_f16_31), "f"(rs_s_0));
                    float2 scale0_48 = make_float2(_mul_18, _mul_18);
                    float2 scale1_49 = make_float2(_mul_19, _mul_19);
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
                    float2 a_50_1 = make_float2(acc2[0], acc2[1]);
                    float2 q_51_1 = make_float2(_unpack_fp4_36[0], _unpack_fp4_36[1]);
                    float2 r_52_1 = fma_f32x2_rn_noftz(q_51_1, scale0_48, a_50_1);
                    acc2[0] = r_52_1.x;
                    acc2[1] = r_52_1.y;
                    float2 a_53_1 = make_float2(acc2[2], acc2[3]);
                    float2 q_54_1 = make_float2(_unpack_fp4_36[2], _unpack_fp4_36[3]);
                    float2 r_55_1 = fma_f32x2_rn_noftz(q_54_1, scale0_48, a_53_1);
                    acc2[2] = r_55_1.x;
                    acc2[3] = r_55_1.y;
                    float2 a_56_1 = make_float2(acc2[4], acc2[5]);
                    float2 q_57_1 = make_float2(_unpack_fp4_36[4], _unpack_fp4_36[5]);
                    float2 r_58_1 = fma_f32x2_rn_noftz(q_57_1, scale0_48, a_56_1);
                    acc2[4] = r_58_1.x;
                    acc2[5] = r_58_1.y;
                    float2 a_59_1 = make_float2(acc2[6], acc2[7]);
                    float2 q_60_1 = make_float2(_unpack_fp4_36[6], _unpack_fp4_36[7]);
                    float2 r_61_1 = fma_f32x2_rn_noftz(q_60_1, scale0_48, a_59_1);
                    acc2[6] = r_61_1.x;
                    acc2[7] = r_61_1.y;
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
                    float2 a_62_1 = make_float2(acc2[8], acc2[9]);
                    float2 q_63_1 = make_float2(_unpack_fp4_37[0], _unpack_fp4_37[1]);
                    float2 r_64_1 = fma_f32x2_rn_noftz(q_63_1, scale0_48, a_62_1);
                    acc2[8] = r_64_1.x;
                    acc2[9] = r_64_1.y;
                    float2 a_65_1 = make_float2(acc2[10], acc2[11]);
                    float2 q_66_1 = make_float2(_unpack_fp4_37[2], _unpack_fp4_37[3]);
                    float2 r_67_1 = fma_f32x2_rn_noftz(q_66_1, scale0_48, a_65_1);
                    acc2[10] = r_67_1.x;
                    acc2[11] = r_67_1.y;
                    float2 a_68_1 = make_float2(acc2[12], acc2[13]);
                    float2 q_69_1 = make_float2(_unpack_fp4_37[4], _unpack_fp4_37[5]);
                    float2 r_70_1 = fma_f32x2_rn_noftz(q_69_1, scale0_48, a_68_1);
                    acc2[12] = r_70_1.x;
                    acc2[13] = r_70_1.y;
                    float2 a_71_1 = make_float2(acc2[14], acc2[15]);
                    float2 q_72_1 = make_float2(_unpack_fp4_37[6], _unpack_fp4_37[7]);
                    float2 r_73_1 = fma_f32x2_rn_noftz(q_72_1, scale0_48, a_71_1);
                    acc2[14] = r_73_1.x;
                    acc2[15] = r_73_1.y;
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
                    float2 a_74_1 = make_float2(acc2[16], acc2[17]);
                    float2 q_75_1 = make_float2(_unpack_fp4_38[0], _unpack_fp4_38[1]);
                    float2 r_76_1 = fma_f32x2_rn_noftz(q_75_1, scale1_49, a_74_1);
                    acc2[16] = r_76_1.x;
                    acc2[17] = r_76_1.y;
                    float2 a_77_1 = make_float2(acc2[18], acc2[19]);
                    float2 q_78_1 = make_float2(_unpack_fp4_38[2], _unpack_fp4_38[3]);
                    float2 r_79_1 = fma_f32x2_rn_noftz(q_78_1, scale1_49, a_77_1);
                    acc2[18] = r_79_1.x;
                    acc2[19] = r_79_1.y;
                    float2 a_80_1 = make_float2(acc2[20], acc2[21]);
                    float2 q_81_1 = make_float2(_unpack_fp4_38[4], _unpack_fp4_38[5]);
                    float2 r_82_1 = fma_f32x2_rn_noftz(q_81_1, scale1_49, a_80_1);
                    acc2[20] = r_82_1.x;
                    acc2[21] = r_82_1.y;
                    float2 a_83_1 = make_float2(acc2[22], acc2[23]);
                    float2 q_84_1 = make_float2(_unpack_fp4_38[6], _unpack_fp4_38[7]);
                    float2 r_85_1 = fma_f32x2_rn_noftz(q_84_1, scale1_49, a_83_1);
                    acc2[22] = r_85_1.x;
                    acc2[23] = r_85_1.y;
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
                    float2 a_86_1 = make_float2(acc2[24], acc2[25]);
                    float2 q_87_1 = make_float2(_unpack_fp4_39[0], _unpack_fp4_39[1]);
                    float2 r_88_1 = fma_f32x2_rn_noftz(q_87_1, scale1_49, a_86_1);
                    acc2[24] = r_88_1.x;
                    acc2[25] = r_88_1.y;
                    float2 a_89_1 = make_float2(acc2[26], acc2[27]);
                    float2 q_90_1 = make_float2(_unpack_fp4_39[2], _unpack_fp4_39[3]);
                    float2 r_91_1 = fma_f32x2_rn_noftz(q_90_1, scale1_49, a_89_1);
                    acc2[26] = r_91_1.x;
                    acc2[27] = r_91_1.y;
                    float2 a_92_1 = make_float2(acc2[28], acc2[29]);
                    float2 q_93_1 = make_float2(_unpack_fp4_39[4], _unpack_fp4_39[5]);
                    float2 r_94_1 = fma_f32x2_rn_noftz(q_93_1, scale1_49, a_92_1);
                    acc2[28] = r_94_1.x;
                    acc2[29] = r_94_1.y;
                    float2 a_95 = make_float2(acc2[30], acc2[31]);
                    float2 q_96 = make_float2(_unpack_fp4_39[6], _unpack_fp4_39[7]);
                    float2 r_97 = fma_f32x2_rn_noftz(q_96, scale1_49, a_95);
                    acc2[30] = r_97.x;
                    acc2[31] = r_97.y;
                    uint32_t _e4m3x2_to_f16x2_16;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_16) : "h"((uint16_t)(c_1)));
                    float _cvt_f32_f16_32;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_32) : "h"((uint16_t)(_e4m3x2_to_f16x2_16)));
                    float _cvt_f32_f16_33;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_33) : "h"((uint16_t)(_e4m3x2_to_f16x2_16 >> 16)));
                    float _mul_20;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(_cvt_f32_f16_32), "f"(rs_s_1));
                    float _mul_21;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(_cvt_f32_f16_33), "f"(rs_s_1));
                    float2 scale0_98 = make_float2(_mul_20, _mul_20);
                    float2 scale1_99 = make_float2(_mul_21, _mul_21);
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
                    float2 a_100_1 = make_float2(acc2[0], acc2[1]);
                    float2 q_101_1 = make_float2(_unpack_fp4_40[0], _unpack_fp4_40[1]);
                    float2 r_102_1 = fma_f32x2_rn_noftz(q_101_1, scale0_98, a_100_1);
                    acc2[0] = r_102_1.x;
                    acc2[1] = r_102_1.y;
                    float2 a_103_1 = make_float2(acc2[2], acc2[3]);
                    float2 q_104_1 = make_float2(_unpack_fp4_40[2], _unpack_fp4_40[3]);
                    float2 r_105_1 = fma_f32x2_rn_noftz(q_104_1, scale0_98, a_103_1);
                    acc2[2] = r_105_1.x;
                    acc2[3] = r_105_1.y;
                    float2 a_106_1 = make_float2(acc2[4], acc2[5]);
                    float2 q_107_1 = make_float2(_unpack_fp4_40[4], _unpack_fp4_40[5]);
                    float2 r_108_1 = fma_f32x2_rn_noftz(q_107_1, scale0_98, a_106_1);
                    acc2[4] = r_108_1.x;
                    acc2[5] = r_108_1.y;
                    float2 a_109_1 = make_float2(acc2[6], acc2[7]);
                    float2 q_110_1 = make_float2(_unpack_fp4_40[6], _unpack_fp4_40[7]);
                    float2 r_111_1 = fma_f32x2_rn_noftz(q_110_1, scale0_98, a_109_1);
                    acc2[6] = r_111_1.x;
                    acc2[7] = r_111_1.y;
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
                    float2 a_112_1 = make_float2(acc2[8], acc2[9]);
                    float2 q_113_1 = make_float2(_unpack_fp4_41[0], _unpack_fp4_41[1]);
                    float2 r_114_1 = fma_f32x2_rn_noftz(q_113_1, scale0_98, a_112_1);
                    acc2[8] = r_114_1.x;
                    acc2[9] = r_114_1.y;
                    float2 a_115_1 = make_float2(acc2[10], acc2[11]);
                    float2 q_116_1 = make_float2(_unpack_fp4_41[2], _unpack_fp4_41[3]);
                    float2 r_117_1 = fma_f32x2_rn_noftz(q_116_1, scale0_98, a_115_1);
                    acc2[10] = r_117_1.x;
                    acc2[11] = r_117_1.y;
                    float2 a_118_1 = make_float2(acc2[12], acc2[13]);
                    float2 q_119_1 = make_float2(_unpack_fp4_41[4], _unpack_fp4_41[5]);
                    float2 r_120_1 = fma_f32x2_rn_noftz(q_119_1, scale0_98, a_118_1);
                    acc2[12] = r_120_1.x;
                    acc2[13] = r_120_1.y;
                    float2 a_121_1 = make_float2(acc2[14], acc2[15]);
                    float2 q_122_1 = make_float2(_unpack_fp4_41[6], _unpack_fp4_41[7]);
                    float2 r_123_1 = fma_f32x2_rn_noftz(q_122_1, scale0_98, a_121_1);
                    acc2[14] = r_123_1.x;
                    acc2[15] = r_123_1.y;
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
                    float2 a_124_1 = make_float2(acc2[16], acc2[17]);
                    float2 q_125_1 = make_float2(_unpack_fp4_42[0], _unpack_fp4_42[1]);
                    float2 r_126_1 = fma_f32x2_rn_noftz(q_125_1, scale1_99, a_124_1);
                    acc2[16] = r_126_1.x;
                    acc2[17] = r_126_1.y;
                    float2 a_127_1 = make_float2(acc2[18], acc2[19]);
                    float2 q_128_1 = make_float2(_unpack_fp4_42[2], _unpack_fp4_42[3]);
                    float2 r_129_1 = fma_f32x2_rn_noftz(q_128_1, scale1_99, a_127_1);
                    acc2[18] = r_129_1.x;
                    acc2[19] = r_129_1.y;
                    float2 a_130_1 = make_float2(acc2[20], acc2[21]);
                    float2 q_131_1 = make_float2(_unpack_fp4_42[4], _unpack_fp4_42[5]);
                    float2 r_132_1 = fma_f32x2_rn_noftz(q_131_1, scale1_99, a_130_1);
                    acc2[20] = r_132_1.x;
                    acc2[21] = r_132_1.y;
                    float2 a_133_1 = make_float2(acc2[22], acc2[23]);
                    float2 q_134_1 = make_float2(_unpack_fp4_42[6], _unpack_fp4_42[7]);
                    float2 r_135_1 = fma_f32x2_rn_noftz(q_134_1, scale1_99, a_133_1);
                    acc2[22] = r_135_1.x;
                    acc2[23] = r_135_1.y;
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
                    float2 a_136_1 = make_float2(acc2[24], acc2[25]);
                    float2 q_137_1 = make_float2(_unpack_fp4_43[0], _unpack_fp4_43[1]);
                    float2 r_138_1 = fma_f32x2_rn_noftz(q_137_1, scale1_99, a_136_1);
                    acc2[24] = r_138_1.x;
                    acc2[25] = r_138_1.y;
                    float2 a_139_1 = make_float2(acc2[26], acc2[27]);
                    float2 q_140_1 = make_float2(_unpack_fp4_43[2], _unpack_fp4_43[3]);
                    float2 r_141_1 = fma_f32x2_rn_noftz(q_140_1, scale1_99, a_139_1);
                    acc2[26] = r_141_1.x;
                    acc2[27] = r_141_1.y;
                    float2 a_142_1 = make_float2(acc2[28], acc2[29]);
                    float2 q_143_1 = make_float2(_unpack_fp4_43[4], _unpack_fp4_43[5]);
                    float2 r_144_1 = fma_f32x2_rn_noftz(q_143_1, scale1_99, a_142_1);
                    acc2[28] = r_144_1.x;
                    acc2[29] = r_144_1.y;
                    float2 a_145 = make_float2(acc2[30], acc2[31]);
                    float2 q_146 = make_float2(_unpack_fp4_43[6], _unpack_fp4_43[7]);
                    float2 r_147 = fma_f32x2_rn_noftz(q_146, scale1_99, a_145);
                    acc2[30] = r_147.x;
                    acc2[31] = r_147.y;
                    uint32_t _e4m3x2_to_f16x2_17;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_17) : "h"((uint16_t)(c_2)));
                    float _cvt_f32_f16_34;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_34) : "h"((uint16_t)(_e4m3x2_to_f16x2_17)));
                    float _cvt_f32_f16_35;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_35) : "h"((uint16_t)(_e4m3x2_to_f16x2_17 >> 16)));
                    float _mul_22;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(_cvt_f32_f16_34), "f"(rs_s_2));
                    float _mul_23;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_cvt_f32_f16_35), "f"(rs_s_2));
                    float2 scale0_148 = make_float2(_mul_22, _mul_22);
                    float2 scale1_149 = make_float2(_mul_23, _mul_23);
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
                    float2 a_150_1 = make_float2(acc2[0], acc2[1]);
                    float2 q_151_1 = make_float2(_unpack_fp4_44[0], _unpack_fp4_44[1]);
                    float2 r_152_1 = fma_f32x2_rn_noftz(q_151_1, scale0_148, a_150_1);
                    acc2[0] = r_152_1.x;
                    acc2[1] = r_152_1.y;
                    float2 a_153_1 = make_float2(acc2[2], acc2[3]);
                    float2 q_154_1 = make_float2(_unpack_fp4_44[2], _unpack_fp4_44[3]);
                    float2 r_155_1 = fma_f32x2_rn_noftz(q_154_1, scale0_148, a_153_1);
                    acc2[2] = r_155_1.x;
                    acc2[3] = r_155_1.y;
                    float2 a_156_1 = make_float2(acc2[4], acc2[5]);
                    float2 q_157_1 = make_float2(_unpack_fp4_44[4], _unpack_fp4_44[5]);
                    float2 r_158_1 = fma_f32x2_rn_noftz(q_157_1, scale0_148, a_156_1);
                    acc2[4] = r_158_1.x;
                    acc2[5] = r_158_1.y;
                    float2 a_159_1 = make_float2(acc2[6], acc2[7]);
                    float2 q_160_1 = make_float2(_unpack_fp4_44[6], _unpack_fp4_44[7]);
                    float2 r_161_1 = fma_f32x2_rn_noftz(q_160_1, scale0_148, a_159_1);
                    acc2[6] = r_161_1.x;
                    acc2[7] = r_161_1.y;
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
                    float2 a_162_1 = make_float2(acc2[8], acc2[9]);
                    float2 q_163_1 = make_float2(_unpack_fp4_45[0], _unpack_fp4_45[1]);
                    float2 r_164_1 = fma_f32x2_rn_noftz(q_163_1, scale0_148, a_162_1);
                    acc2[8] = r_164_1.x;
                    acc2[9] = r_164_1.y;
                    float2 a_165_1 = make_float2(acc2[10], acc2[11]);
                    float2 q_166_1 = make_float2(_unpack_fp4_45[2], _unpack_fp4_45[3]);
                    float2 r_167_1 = fma_f32x2_rn_noftz(q_166_1, scale0_148, a_165_1);
                    acc2[10] = r_167_1.x;
                    acc2[11] = r_167_1.y;
                    float2 a_168_1 = make_float2(acc2[12], acc2[13]);
                    float2 q_169_1 = make_float2(_unpack_fp4_45[4], _unpack_fp4_45[5]);
                    float2 r_170_1 = fma_f32x2_rn_noftz(q_169_1, scale0_148, a_168_1);
                    acc2[12] = r_170_1.x;
                    acc2[13] = r_170_1.y;
                    float2 a_171_1 = make_float2(acc2[14], acc2[15]);
                    float2 q_172_1 = make_float2(_unpack_fp4_45[6], _unpack_fp4_45[7]);
                    float2 r_173_1 = fma_f32x2_rn_noftz(q_172_1, scale0_148, a_171_1);
                    acc2[14] = r_173_1.x;
                    acc2[15] = r_173_1.y;
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
                    float2 a_174_1 = make_float2(acc2[16], acc2[17]);
                    float2 q_175_1 = make_float2(_unpack_fp4_46[0], _unpack_fp4_46[1]);
                    float2 r_176_1 = fma_f32x2_rn_noftz(q_175_1, scale1_149, a_174_1);
                    acc2[16] = r_176_1.x;
                    acc2[17] = r_176_1.y;
                    float2 a_177_1 = make_float2(acc2[18], acc2[19]);
                    float2 q_178_1 = make_float2(_unpack_fp4_46[2], _unpack_fp4_46[3]);
                    float2 r_179_1 = fma_f32x2_rn_noftz(q_178_1, scale1_149, a_177_1);
                    acc2[18] = r_179_1.x;
                    acc2[19] = r_179_1.y;
                    float2 a_180_1 = make_float2(acc2[20], acc2[21]);
                    float2 q_181_1 = make_float2(_unpack_fp4_46[4], _unpack_fp4_46[5]);
                    float2 r_182_1 = fma_f32x2_rn_noftz(q_181_1, scale1_149, a_180_1);
                    acc2[20] = r_182_1.x;
                    acc2[21] = r_182_1.y;
                    float2 a_183_1 = make_float2(acc2[22], acc2[23]);
                    float2 q_184_1 = make_float2(_unpack_fp4_46[6], _unpack_fp4_46[7]);
                    float2 r_185_1 = fma_f32x2_rn_noftz(q_184_1, scale1_149, a_183_1);
                    acc2[22] = r_185_1.x;
                    acc2[23] = r_185_1.y;
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
                    float2 a_186_1 = make_float2(acc2[24], acc2[25]);
                    float2 q_187_1 = make_float2(_unpack_fp4_47[0], _unpack_fp4_47[1]);
                    float2 r_188_1 = fma_f32x2_rn_noftz(q_187_1, scale1_149, a_186_1);
                    acc2[24] = r_188_1.x;
                    acc2[25] = r_188_1.y;
                    float2 a_189_1 = make_float2(acc2[26], acc2[27]);
                    float2 q_190_1 = make_float2(_unpack_fp4_47[2], _unpack_fp4_47[3]);
                    float2 r_191_1 = fma_f32x2_rn_noftz(q_190_1, scale1_149, a_189_1);
                    acc2[26] = r_191_1.x;
                    acc2[27] = r_191_1.y;
                    float2 a_192_1 = make_float2(acc2[28], acc2[29]);
                    float2 q_193_1 = make_float2(_unpack_fp4_47[4], _unpack_fp4_47[5]);
                    float2 r_194_1 = fma_f32x2_rn_noftz(q_193_1, scale1_149, a_192_1);
                    acc2[28] = r_194_1.x;
                    acc2[29] = r_194_1.y;
                    float2 a_195 = make_float2(acc2[30], acc2[31]);
                    float2 q_196 = make_float2(_unpack_fp4_47[6], _unpack_fp4_47[7]);
                    float2 r_197 = fma_f32x2_rn_noftz(q_196, scale1_149, a_195);
                    acc2[30] = r_197.x;
                    acc2[31] = r_197.y;
                    unsigned int q_198[4];
                    float bm0_1 = 0.0f;
                    float bm1_1 = 0.0f;
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[0]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[16]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[1]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[17]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[2]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[18]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[3]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[19]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[4]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[20]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[5]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[21]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[6]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[22]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[7]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[23]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[8]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[24]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[9]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[25]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[10]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[26]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[11]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[27]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[12]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[28]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[13]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[29]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[14]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[30]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2[15]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2[31]));
                    float _fdiv_153 = __fdiv_rn(mul_f32_rn_noftz(bm1_1, inv_1), 6.0f);
                    float _fdiv_154 = __fdiv_rn(mul_f32_rn_noftz(bm0_1, inv_1), 6.0f);
                    uint16_t _e4m3x2_f32_6;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_6) : "f"(_fdiv_153), "f"(_fdiv_154));
                    uint32_t _e4m3x2_to_f16x2_18;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_18) : "h"((uint16_t)(_e4m3x2_f32_6)));
                    float _cvt_f32_f16_36;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_36) : "h"((uint16_t)(_e4m3x2_to_f16x2_18)));
                    float _cvt_f32_f16_37;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_37) : "h"((uint16_t)(_e4m3x2_to_f16x2_18 >> 16)));
                    float _fdiv_rn_14 = __fdiv_rn(inv_1, _cvt_f32_f16_36);
                    float _fdiv_rn_15 = __fdiv_rn(inv_1, _cvt_f32_f16_37);
                    float m0_1 = 0.0f;
                    float m1_1 = 0.0f;
                    if (_cvt_f32_f16_36 > 0.0f) {
                        m0_1 = _fdiv_rn_14;
                    }
                    if (_cvt_f32_f16_37 > 0.0f) {
                        m1_1 = _fdiv_rn_15;
                    }
                    float2 mm_1 = make_float2(m0_1, m0_1);
                    float2 pair_1 = make_float2(acc2[0], acc2[1]);
                    float2 g_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1) : "l"(*(const unsigned long long*)&pair_1), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t _fp4_pair_72;
                    uint16_t _e2m1_pair_h_155;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_155) : "f"(g_1.x), "f"(g_1.y));
                    _fp4_pair_72 = (uint32_t)_e2m1_pair_h_155;
                    float2 pair_199_1 = make_float2(acc2[2], acc2[3]);
                    float2 g_200_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_200_1) : "l"(*(const unsigned long long*)&pair_199_1), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t _fp4_pair_73;
                    uint16_t _e2m1_pair_h_156;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_156) : "f"(g_200_1.x), "f"(g_200_1.y));
                    _fp4_pair_73 = (uint32_t)_e2m1_pair_h_156;
                    float2 pair_201_1 = make_float2(acc2[4], acc2[5]);
                    float2 g_202_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_202_1) : "l"(*(const unsigned long long*)&pair_201_1), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t _fp4_pair_74;
                    uint16_t _e2m1_pair_h_157;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_157) : "f"(g_202_1.x), "f"(g_202_1.y));
                    _fp4_pair_74 = (uint32_t)_e2m1_pair_h_157;
                    float2 pair_203_1 = make_float2(acc2[6], acc2[7]);
                    float2 g_204_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_204_1) : "l"(*(const unsigned long long*)&pair_203_1), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t _fp4_e2m1_pack_pair_158;
                    uint16_t _e2m1_pair_h_159;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_159) : "f"(g_204_1.x), "f"(g_204_1.y));
                    _fp4_e2m1_pack_pair_158 = (uint32_t)_e2m1_pair_h_159;
                    unsigned int word_1 = _fp4_pair_72 | _fp4_pair_73 << 8 | _fp4_pair_74 << 16 | _fp4_e2m1_pack_pair_158 << 24;
                    unsigned int carry_1 = (word_1 & 2004318071) + 2004318071;
                    q_198[0] = word_1 & (carry_1 | 2004318071);
                    float2 mm_205 = make_float2(m0_1, m0_1);
                    float2 pair_206 = make_float2(acc2[8], acc2[9]);
                    float2 g_207;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_207) : "l"(*(const unsigned long long*)&pair_206), "l"(*(const unsigned long long*)&mm_205));
                    uint32_t _fp4_pair_75;
                    uint16_t _e2m1_pair_h_160;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_160) : "f"(g_207.x), "f"(g_207.y));
                    _fp4_pair_75 = (uint32_t)_e2m1_pair_h_160;
                    float2 pair_208 = make_float2(acc2[10], acc2[11]);
                    float2 g_209;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_209) : "l"(*(const unsigned long long*)&pair_208), "l"(*(const unsigned long long*)&mm_205));
                    uint32_t _fp4_pair_76;
                    uint16_t _e2m1_pair_h_161;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_161) : "f"(g_209.x), "f"(g_209.y));
                    _fp4_pair_76 = (uint32_t)_e2m1_pair_h_161;
                    float2 pair_210 = make_float2(acc2[12], acc2[13]);
                    float2 g_211;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_211) : "l"(*(const unsigned long long*)&pair_210), "l"(*(const unsigned long long*)&mm_205));
                    uint32_t _fp4_pair_77;
                    uint16_t _e2m1_pair_h_162;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_162) : "f"(g_211.x), "f"(g_211.y));
                    _fp4_pair_77 = (uint32_t)_e2m1_pair_h_162;
                    float2 pair_212 = make_float2(acc2[14], acc2[15]);
                    float2 g_213;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_213) : "l"(*(const unsigned long long*)&pair_212), "l"(*(const unsigned long long*)&mm_205));
                    uint32_t _fp4_e2m1_pack_pair_163;
                    uint16_t _e2m1_pair_h_164;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_164) : "f"(g_213.x), "f"(g_213.y));
                    _fp4_e2m1_pack_pair_163 = (uint32_t)_e2m1_pair_h_164;
                    unsigned int word_214 = _fp4_pair_75 | _fp4_pair_76 << 8 | _fp4_pair_77 << 16 | _fp4_e2m1_pack_pair_163 << 24;
                    unsigned int carry_215 = (word_214 & 2004318071) + 2004318071;
                    q_198[1] = word_214 & (carry_215 | 2004318071);
                    float2 mm_216 = make_float2(m1_1, m1_1);
                    float2 pair_217_1 = make_float2(acc2[16], acc2[17]);
                    float2 g_218_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_218_1) : "l"(*(const unsigned long long*)&pair_217_1), "l"(*(const unsigned long long*)&mm_216));
                    uint32_t _fp4_pair_78;
                    uint16_t _e2m1_pair_h_165;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_165) : "f"(g_218_1.x), "f"(g_218_1.y));
                    _fp4_pair_78 = (uint32_t)_e2m1_pair_h_165;
                    float2 pair_219 = make_float2(acc2[18], acc2[19]);
                    float2 g_220;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_220) : "l"(*(const unsigned long long*)&pair_219), "l"(*(const unsigned long long*)&mm_216));
                    uint32_t _fp4_pair_79;
                    uint16_t _e2m1_pair_h_166;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_166) : "f"(g_220.x), "f"(g_220.y));
                    _fp4_pair_79 = (uint32_t)_e2m1_pair_h_166;
                    float2 pair_221 = make_float2(acc2[20], acc2[21]);
                    float2 g_222;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_222) : "l"(*(const unsigned long long*)&pair_221), "l"(*(const unsigned long long*)&mm_216));
                    uint32_t _fp4_pair_80;
                    uint16_t _e2m1_pair_h_167;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_167) : "f"(g_222.x), "f"(g_222.y));
                    _fp4_pair_80 = (uint32_t)_e2m1_pair_h_167;
                    float2 pair_223 = make_float2(acc2[22], acc2[23]);
                    float2 g_224;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_224) : "l"(*(const unsigned long long*)&pair_223), "l"(*(const unsigned long long*)&mm_216));
                    uint32_t _fp4_e2m1_pack_pair_168;
                    uint16_t _e2m1_pair_h_169;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_169) : "f"(g_224.x), "f"(g_224.y));
                    _fp4_e2m1_pack_pair_168 = (uint32_t)_e2m1_pair_h_169;
                    unsigned int word_225 = _fp4_pair_78 | _fp4_pair_79 << 8 | _fp4_pair_80 << 16 | _fp4_e2m1_pack_pair_168 << 24;
                    unsigned int carry_226 = (word_225 & 2004318071) + 2004318071;
                    q_198[2] = word_225 & (carry_226 | 2004318071);
                    float2 mm_227 = make_float2(m1_1, m1_1);
                    float2 pair_228 = make_float2(acc2[24], acc2[25]);
                    float2 g_229;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_229) : "l"(*(const unsigned long long*)&pair_228), "l"(*(const unsigned long long*)&mm_227));
                    uint32_t _fp4_pair_81;
                    uint16_t _e2m1_pair_h_170;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_170) : "f"(g_229.x), "f"(g_229.y));
                    _fp4_pair_81 = (uint32_t)_e2m1_pair_h_170;
                    float2 pair_230 = make_float2(acc2[26], acc2[27]);
                    float2 g_231;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_231) : "l"(*(const unsigned long long*)&pair_230), "l"(*(const unsigned long long*)&mm_227));
                    uint32_t _fp4_pair_82;
                    uint16_t _e2m1_pair_h_171;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_171) : "f"(g_231.x), "f"(g_231.y));
                    _fp4_pair_82 = (uint32_t)_e2m1_pair_h_171;
                    float2 pair_232 = make_float2(acc2[28], acc2[29]);
                    float2 g_233;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_233) : "l"(*(const unsigned long long*)&pair_232), "l"(*(const unsigned long long*)&mm_227));
                    uint32_t _fp4_pair_83;
                    uint16_t _e2m1_pair_h_172;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_172) : "f"(g_233.x), "f"(g_233.y));
                    _fp4_pair_83 = (uint32_t)_e2m1_pair_h_172;
                    float2 pair_234 = make_float2(acc2[30], acc2[31]);
                    float2 g_235;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_235) : "l"(*(const unsigned long long*)&pair_234), "l"(*(const unsigned long long*)&mm_227));
                    uint32_t _fp4_e2m1_pack_pair_173;
                    uint16_t _e2m1_pair_h_174;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_174) : "f"(g_235.x), "f"(g_235.y));
                    _fp4_e2m1_pack_pair_173 = (uint32_t)_e2m1_pair_h_174;
                    unsigned int word_236 = _fp4_pair_81 | _fp4_pair_82 << 8 | _fp4_pair_83 << 16 | _fp4_e2m1_pack_pair_173 << 24;
                    unsigned int carry_237 = (word_236 & 2004318071) + 2004318071;
                    q_198[3] = word_236 & (carry_237 | 2004318071);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(bob_off + 16 * pp_1)), "r"(q_198[0]), "r"(q_198[1]), "r"(q_198[2]), "r"(q_198[3]) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(bob_off + 3072 + 2 * pp_1)), "h"((uint16_t)(_e4m3x2_f32_6)) : "memory");
                    float d[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_19;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_19) : "h"((uint16_t)(_e4m3x2_f32_6)));
                    float _cvt_f32_f16_38;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_38) : "h"((uint16_t)(_e4m3x2_to_f16x2_19)));
                    float _cvt_f32_f16_39;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_39) : "h"((uint16_t)(_e4m3x2_to_f16x2_19 >> 16)));
                    float _mul_24;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(_cvt_f32_f16_38), "f"(rs2));
                    float _mul_25;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(_cvt_f32_f16_39), "f"(rs2));
                    float2 scale0_238 = make_float2(_mul_24, _mul_24);
                    float2 scale1_239 = make_float2(_mul_25, _mul_25);
                    float _unpack_fp4_48[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_198[0]));
                        (&_unpack_fp4_48[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_48[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_48[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_48[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_48[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_48[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_48[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_48[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_48[0]), "=f"(_unpack_fp4_48[1]), "=f"(_unpack_fp4_48[2]), "=f"(_unpack_fp4_48[3]), "=f"(_unpack_fp4_48[4]), "=f"(_unpack_fp4_48[5]), "=f"(_unpack_fp4_48[6]), "=f"(_unpack_fp4_48[7]) : "r"(q_198[0]));
                        #endif
                    }
                    float2 a_240_1 = make_float2(d[0], d[1]);
                    float2 q_241_1 = make_float2(_unpack_fp4_48[0], _unpack_fp4_48[1]);
                    float2 r_242_1 = fma_f32x2_rn_noftz(q_241_1, scale0_238, a_240_1);
                    d[0] = r_242_1.x;
                    d[1] = r_242_1.y;
                    float2 a_243_1 = make_float2(d[2], d[3]);
                    float2 q_244_1 = make_float2(_unpack_fp4_48[2], _unpack_fp4_48[3]);
                    float2 r_245_1 = fma_f32x2_rn_noftz(q_244_1, scale0_238, a_243_1);
                    d[2] = r_245_1.x;
                    d[3] = r_245_1.y;
                    float2 a_246 = make_float2(d[4], d[5]);
                    float2 q_247 = make_float2(_unpack_fp4_48[4], _unpack_fp4_48[5]);
                    float2 r_248 = fma_f32x2_rn_noftz(q_247, scale0_238, a_246);
                    d[4] = r_248.x;
                    d[5] = r_248.y;
                    float2 a_249 = make_float2(d[6], d[7]);
                    float2 q_250 = make_float2(_unpack_fp4_48[6], _unpack_fp4_48[7]);
                    float2 r_251 = fma_f32x2_rn_noftz(q_250, scale0_238, a_249);
                    d[6] = r_251.x;
                    d[7] = r_251.y;
                    float _unpack_fp4_49[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_198[1 + 0]));
                        (&_unpack_fp4_49[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_49[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_49[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_49[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_49[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_49[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_49[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_49[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_49[0]), "=f"(_unpack_fp4_49[1]), "=f"(_unpack_fp4_49[2]), "=f"(_unpack_fp4_49[3]), "=f"(_unpack_fp4_49[4]), "=f"(_unpack_fp4_49[5]), "=f"(_unpack_fp4_49[6]), "=f"(_unpack_fp4_49[7]) : "r"(q_198[1 + 0]));
                        #endif
                    }
                    float2 a_252 = make_float2(d[8], d[9]);
                    float2 q_253 = make_float2(_unpack_fp4_49[0], _unpack_fp4_49[1]);
                    float2 r_254 = fma_f32x2_rn_noftz(q_253, scale0_238, a_252);
                    d[8] = r_254.x;
                    d[9] = r_254.y;
                    float2 a_255 = make_float2(d[10], d[11]);
                    float2 q_256 = make_float2(_unpack_fp4_49[2], _unpack_fp4_49[3]);
                    float2 r_257 = fma_f32x2_rn_noftz(q_256, scale0_238, a_255);
                    d[10] = r_257.x;
                    d[11] = r_257.y;
                    float2 a_258 = make_float2(d[12], d[13]);
                    float2 q_259 = make_float2(_unpack_fp4_49[4], _unpack_fp4_49[5]);
                    float2 r_260 = fma_f32x2_rn_noftz(q_259, scale0_238, a_258);
                    d[12] = r_260.x;
                    d[13] = r_260.y;
                    float2 a_261 = make_float2(d[14], d[15]);
                    float2 q_262 = make_float2(_unpack_fp4_49[6], _unpack_fp4_49[7]);
                    float2 r_263 = fma_f32x2_rn_noftz(q_262, scale0_238, a_261);
                    d[14] = r_263.x;
                    d[15] = r_263.y;
                    float _unpack_fp4_50[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_198[2 + 0]));
                        (&_unpack_fp4_50[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_50[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_50[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_50[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_50[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_50[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_50[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_50[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_50[0]), "=f"(_unpack_fp4_50[1]), "=f"(_unpack_fp4_50[2]), "=f"(_unpack_fp4_50[3]), "=f"(_unpack_fp4_50[4]), "=f"(_unpack_fp4_50[5]), "=f"(_unpack_fp4_50[6]), "=f"(_unpack_fp4_50[7]) : "r"(q_198[2 + 0]));
                        #endif
                    }
                    float2 a_264 = make_float2(d[16], d[17]);
                    float2 q_265 = make_float2(_unpack_fp4_50[0], _unpack_fp4_50[1]);
                    float2 r_266 = fma_f32x2_rn_noftz(q_265, scale1_239, a_264);
                    d[16] = r_266.x;
                    d[17] = r_266.y;
                    float2 a_267 = make_float2(d[18], d[19]);
                    float2 q_268 = make_float2(_unpack_fp4_50[2], _unpack_fp4_50[3]);
                    float2 r_269 = fma_f32x2_rn_noftz(q_268, scale1_239, a_267);
                    d[18] = r_269.x;
                    d[19] = r_269.y;
                    float2 a_270 = make_float2(d[20], d[21]);
                    float2 q_271 = make_float2(_unpack_fp4_50[4], _unpack_fp4_50[5]);
                    float2 r_272 = fma_f32x2_rn_noftz(q_271, scale1_239, a_270);
                    d[20] = r_272.x;
                    d[21] = r_272.y;
                    float2 a_273 = make_float2(d[22], d[23]);
                    float2 q_274 = make_float2(_unpack_fp4_50[6], _unpack_fp4_50[7]);
                    float2 r_275 = fma_f32x2_rn_noftz(q_274, scale1_239, a_273);
                    d[22] = r_275.x;
                    d[23] = r_275.y;
                    float _unpack_fp4_51[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_198[3 + 0]));
                        (&_unpack_fp4_51[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_51[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_51[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_51[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_51[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_51[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_51[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_51[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_51[0]), "=f"(_unpack_fp4_51[1]), "=f"(_unpack_fp4_51[2]), "=f"(_unpack_fp4_51[3]), "=f"(_unpack_fp4_51[4]), "=f"(_unpack_fp4_51[5]), "=f"(_unpack_fp4_51[6]), "=f"(_unpack_fp4_51[7]) : "r"(q_198[3 + 0]));
                        #endif
                    }
                    float2 a_276 = make_float2(d[24], d[25]);
                    float2 q_277 = make_float2(_unpack_fp4_51[0], _unpack_fp4_51[1]);
                    float2 r_278 = fma_f32x2_rn_noftz(q_277, scale1_239, a_276);
                    d[24] = r_278.x;
                    d[25] = r_278.y;
                    float2 a_279 = make_float2(d[26], d[27]);
                    float2 q_280 = make_float2(_unpack_fp4_51[2], _unpack_fp4_51[3]);
                    float2 r_281 = fma_f32x2_rn_noftz(q_280, scale1_239, a_279);
                    d[26] = r_281.x;
                    d[27] = r_281.y;
                    float2 a_282 = make_float2(d[28], d[29]);
                    float2 q_283 = make_float2(_unpack_fp4_51[4], _unpack_fp4_51[5]);
                    float2 r_284 = fma_f32x2_rn_noftz(q_283, scale1_239, a_282);
                    d[28] = r_284.x;
                    d[29] = r_284.y;
                    float2 a_285 = make_float2(d[30], d[31]);
                    float2 q_286 = make_float2(_unpack_fp4_51[6], _unpack_fp4_51[7]);
                    float2 r_287 = fma_f32x2_rn_noftz(q_286, scale1_239, a_285);
                    d[30] = r_287.x;
                    d[31] = r_287.y;
                    unsigned int d_bf16[16];
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
                    int c_288 = lane_1;
                    int l = c_288 >> 2;
                    unsigned int w4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l + (c_288 & 3 ^ l >> 1 & 3)))));
                    long long idx = brow64 * 3072 + (long long)(4 * (128 * j + c_288));
                    {
                        int4 _iv4 = make_int4(w4[0 + 0], w4[0 + 1], w4[0 + 2], w4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx) = _iv4;
                    }
                    int c_289 = lane_1 + 32;
                    int l_290 = c_289 >> 2;
                    unsigned int w4_291[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_291[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_291[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_291[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_291[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_290 + (c_289 & 3 ^ l_290 >> 1 & 3)))));
                    long long idx_292 = brow64 * 3072 + (long long)(4 * (128 * j + c_289));
                    {
                        int4 _iv4 = make_int4(w4_291[0 + 0], w4_291[0 + 1], w4_291[0 + 2], w4_291[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_292) = _iv4;
                    }
                    int c_293 = lane_1 + 64;
                    int l_294 = c_293 >> 2;
                    unsigned int w4_295[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_295[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_295[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_295[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_295[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_294 + (c_293 & 3 ^ l_294 >> 1 & 3)))));
                    long long idx_296 = brow64 * 3072 + (long long)(4 * (128 * j + c_293));
                    {
                        int4 _iv4 = make_int4(w4_295[0 + 0], w4_295[0 + 1], w4_295[0 + 2], w4_295[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_296) = _iv4;
                    }
                    int c_297 = lane_1 + 96;
                    int l_298 = c_297 >> 2;
                    unsigned int w4_299[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_299[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_299[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_299[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_299[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_298 + (c_297 & 3 ^ l_298 >> 1 & 3)))));
                    long long idx_300 = brow64 * 3072 + (long long)(4 * (128 * j + c_297));
                    {
                        int4 _iv4 = make_int4(w4_299[0 + 0], w4_299[0 + 1], w4_299[0 + 2], w4_299[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_300) = _iv4;
                    }
                    __syncwarp();
                }
                if (lane_1 < 4) {
                    {
                        uint32_t _addr_175 = static_cast<uint32_t>(wsm_addr + (unsigned int)(bob_off + 3072 + 384 + 4 * lane_1));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_175), "f"(rs2) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane_1 == 0) {
                    int ag_wo = (int)ag_w;
                    int ag_ro = (int)ag_r;
                    int capo_0 = (int)capo64;
                    int ix_1_1 = (int)bix64;
                    int rk_2 = (int)rank64;
                    int v_off_1 = ag_wo + (rk_2 * capo_0 + ix_1_1) * 3072;
                    int sc_off_1 = ag_wo + 4 * capo_0 * 3072 + (rk_2 * capo_0 + ix_1_1) * 384;
                    int m_off_1 = ag_ro + (rk_2 * capo_0 + ix_1_1) * 16;
                    int peer = (rank + 1) % 4;
                    uint8_t* pw = reinterpret_cast<uint8_t*>(tab[peer]);
                    uint8_t* pr = reinterpret_cast<uint8_t*>(tab[4 + peer]);
                    {
                        void* _cpbulk_dst_176 = reinterpret_cast<void*>(pw + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_176), "r"(wsm_addr + (unsigned int)bob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_177 = reinterpret_cast<void*>(pw + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_177), "r"(wsm_addr + (unsigned int)(bob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_178 = reinterpret_cast<void*>(pr + m_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_178), "r"(wsm_addr + (unsigned int)(bob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_3 = (rank + 2) % 4;
                    uint8_t* pw_4 = reinterpret_cast<uint8_t*>(tab[peer_3]);
                    uint8_t* pr_5 = reinterpret_cast<uint8_t*>(tab[4 + peer_3]);
                    {
                        void* _cpbulk_dst_179 = reinterpret_cast<void*>(pw_4 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_179), "r"(wsm_addr + (unsigned int)bob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_180 = reinterpret_cast<void*>(pw_4 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_180), "r"(wsm_addr + (unsigned int)(bob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_181 = reinterpret_cast<void*>(pr_5 + m_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_181), "r"(wsm_addr + (unsigned int)(bob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_6 = (rank + 3) % 4;
                    uint8_t* pw_7 = reinterpret_cast<uint8_t*>(tab[peer_6]);
                    uint8_t* pr_8 = reinterpret_cast<uint8_t*>(tab[4 + peer_6]);
                    {
                        void* _cpbulk_dst_182 = reinterpret_cast<void*>(pw_7 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_182), "r"(wsm_addr + (unsigned int)bob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_183 = reinterpret_cast<void*>(pw_7 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_183), "r"(wsm_addr + (unsigned int)(bob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_184 = reinterpret_cast<void*>(pr_8 + m_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_184), "r"(wsm_addr + (unsigned int)(bob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                kb = kb + 1;
            }
            int fme = rb0 - r0;
            #pragma unroll 1
            for (int c_3 = wC; c_3 < nC; c_3 += 10) {
                int blk = c_3 / 3;
                int w_1 = c_3 % 3;
                int bump = 0;
                if (w_1 >= fme) {
                    bump = 1;
                }
                int crow = r0 + blk * 4 + w_1 + bump;
                long long cix64 = (long long)(crow / 4);
                long long crow64 = (long long)crow;
                long long csrc64 = (long long)(crow % 4);
                int ag_wo_1 = (int)ag_w;
                int ag_ro_1 = (int)ag_r;
                int capo_2 = (int)capo64;
                int src = (int)csrc64;
                int ix_2 = (int)cix64;
                int m_off_2 = ag_ro_1 + (src * capo_2 + ix_2) * 16;
                float rs_2 = 0.0f;
                uint32_t _sysv_poll32_group_1[1];
                do {
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_1[0]) : "l"(reinterpret_cast<unsigned int*>(ts_rs + m_off_2)) : "memory");
                } while (
                    (_sysv_poll32_group_1[0] == 0x80000000u)
                );
                rs_2 = __uint_as_float(_sysv_poll32_group_1[0]);
                int v_1 = ag_wo_1 + (src * capo_2 + ix_2) * 3072;
                int sc_1 = ag_wo_1 + 4 * capo_2 * 3072 + (src * capo_2 + ix_2) * 384;
                int ppj = lane_1;
                int ppj_0 = lane_1 + 32;
                int ppj_1 = lane_1 + 64;
                int ppj_2 = lane_1 + 96;
                int ppj_3 = lane_1 + 128;
                int ppj_4 = lane_1 + 160;
                uint32_t _fp4_wire_values_1[24];
                uint32_t _fp4_wire_scales_1[6];
                uint16_t _wire_h16_185_0;
                uint16_t _wire_h16_185_1;
                uint16_t _wire_h16_185_2;
                uint16_t _wire_h16_185_3;
                uint16_t _wire_h16_185_4;
                uint16_t _wire_h16_185_5;
                uint32_t _wire_vpend_185 = 63u;
                uint32_t _wire_spend_185 = 63u;
                do {
                    if (_wire_vpend_185 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[0]), "=r"(_fp4_wire_values_1[1]), "=r"(_fp4_wire_values_1[2]), "=r"(_fp4_wire_values_1[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * ppj)))) : "memory");
                    }
                    if (_wire_vpend_185 & 2u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[4]), "=r"(_fp4_wire_values_1[5]), "=r"(_fp4_wire_values_1[6]), "=r"(_fp4_wire_values_1[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * ppj_0)))) : "memory");
                    }
                    if (_wire_vpend_185 & 4u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[8]), "=r"(_fp4_wire_values_1[9]), "=r"(_fp4_wire_values_1[10]), "=r"(_fp4_wire_values_1[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * ppj_1)))) : "memory");
                    }
                    if (_wire_vpend_185 & 8u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[12]), "=r"(_fp4_wire_values_1[13]), "=r"(_fp4_wire_values_1[14]), "=r"(_fp4_wire_values_1[15]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * ppj_2)))) : "memory");
                    }
                    if (_wire_vpend_185 & 16u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[16]), "=r"(_fp4_wire_values_1[17]), "=r"(_fp4_wire_values_1[18]), "=r"(_fp4_wire_values_1[19]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * ppj_3)))) : "memory");
                    }
                    if (_wire_vpend_185 & 32u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[20]), "=r"(_fp4_wire_values_1[21]), "=r"(_fp4_wire_values_1[22]), "=r"(_fp4_wire_values_1[23]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * ppj_4)))) : "memory");
                    }
                    if (_wire_spend_185 & 1u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_185_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * ppj)))) : "memory");
                        _fp4_wire_scales_1[0] = (uint32_t)_wire_h16_185_0;
                    }
                    if (_wire_spend_185 & 2u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_185_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * ppj_0)))) : "memory");
                        _fp4_wire_scales_1[1] = (uint32_t)_wire_h16_185_1;
                    }
                    if (_wire_spend_185 & 4u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_185_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * ppj_1)))) : "memory");
                        _fp4_wire_scales_1[2] = (uint32_t)_wire_h16_185_2;
                    }
                    if (_wire_spend_185 & 8u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_185_3) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * ppj_2)))) : "memory");
                        _fp4_wire_scales_1[3] = (uint32_t)_wire_h16_185_3;
                    }
                    if (_wire_spend_185 & 16u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_185_4) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * ppj_3)))) : "memory");
                        _fp4_wire_scales_1[4] = (uint32_t)_wire_h16_185_4;
                    }
                    if (_wire_spend_185 & 32u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_185_5) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * ppj_4)))) : "memory");
                        _fp4_wire_scales_1[5] = (uint32_t)_wire_h16_185_5;
                    }
                    _wire_vpend_185 = ((uint32_t)((((_fp4_wire_values_1[0] & ~((_fp4_wire_values_1[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[1] & ~((_fp4_wire_values_1[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[2] & ~((_fp4_wire_values_1[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[3] & ~((_fp4_wire_values_1[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_1[4] & ~((_fp4_wire_values_1[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[5] & ~((_fp4_wire_values_1[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[6] & ~((_fp4_wire_values_1[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[7] & ~((_fp4_wire_values_1[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((_fp4_wire_values_1[8] & ~((_fp4_wire_values_1[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[9] & ~((_fp4_wire_values_1[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[10] & ~((_fp4_wire_values_1[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[11] & ~((_fp4_wire_values_1[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2) | ((uint32_t)((((_fp4_wire_values_1[12] & ~((_fp4_wire_values_1[12] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[13] & ~((_fp4_wire_values_1[13] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[14] & ~((_fp4_wire_values_1[14] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[15] & ~((_fp4_wire_values_1[15] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 3) | ((uint32_t)((((_fp4_wire_values_1[16] & ~((_fp4_wire_values_1[16] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[17] & ~((_fp4_wire_values_1[17] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[18] & ~((_fp4_wire_values_1[18] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[19] & ~((_fp4_wire_values_1[19] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 4) | ((uint32_t)((((_fp4_wire_values_1[20] & ~((_fp4_wire_values_1[20] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[21] & ~((_fp4_wire_values_1[21] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[22] & ~((_fp4_wire_values_1[22] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[23] & ~((_fp4_wire_values_1[23] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 5);
                    _wire_spend_185 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[2] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[3] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[3] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 3) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[4] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[4] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 4) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[5] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[5] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 5);
                } while ((_wire_vpend_185 | _wire_spend_185) != 0u);
                float acc_1[32] = {0};
                uint32_t _e4m3x2_to_f16x2_20;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_20) : "h"((uint16_t)(_fp4_wire_scales_1[0])));
                float _cvt_f32_f16_40;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_40) : "h"((uint16_t)(_e4m3x2_to_f16x2_20)));
                float _cvt_f32_f16_41;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_41) : "h"((uint16_t)(_e4m3x2_to_f16x2_20 >> 16)));
                float _mul_26;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(_cvt_f32_f16_40), "f"(rs_2));
                float _mul_27;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(_cvt_f32_f16_41), "f"(rs_2));
                float2 scale0_2 = make_float2(_mul_26, _mul_26);
                float2 scale1_2 = make_float2(_mul_27, _mul_27);
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
                float2 a_5 = make_float2(acc_1[0], acc_1[1]);
                float2 q_3 = make_float2(_unpack_fp4_52[0], _unpack_fp4_52[1]);
                float2 r_3 = fma_f32x2_rn_noftz(q_3, scale0_2, a_5);
                acc_1[0] = r_3.x;
                acc_1[1] = r_3.y;
                float2 a_5_1 = make_float2(acc_1[2], acc_1[3]);
                float2 q_6 = make_float2(_unpack_fp4_52[2], _unpack_fp4_52[3]);
                float2 r_7 = fma_f32x2_rn_noftz(q_6, scale0_2, a_5_1);
                acc_1[2] = r_7.x;
                acc_1[3] = r_7.y;
                float2 a_8 = make_float2(acc_1[4], acc_1[5]);
                float2 q_9 = make_float2(_unpack_fp4_52[4], _unpack_fp4_52[5]);
                float2 r_10 = fma_f32x2_rn_noftz(q_9, scale0_2, a_8);
                acc_1[4] = r_10.x;
                acc_1[5] = r_10.y;
                float2 a_11 = make_float2(acc_1[6], acc_1[7]);
                float2 q_12 = make_float2(_unpack_fp4_52[6], _unpack_fp4_52[7]);
                float2 r_13 = fma_f32x2_rn_noftz(q_12, scale0_2, a_11);
                acc_1[6] = r_13.x;
                acc_1[7] = r_13.y;
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
                float2 a_14 = make_float2(acc_1[8], acc_1[9]);
                float2 q_15 = make_float2(_unpack_fp4_53[0], _unpack_fp4_53[1]);
                float2 r_16 = fma_f32x2_rn_noftz(q_15, scale0_2, a_14);
                acc_1[8] = r_16.x;
                acc_1[9] = r_16.y;
                float2 a_17 = make_float2(acc_1[10], acc_1[11]);
                float2 q_18 = make_float2(_unpack_fp4_53[2], _unpack_fp4_53[3]);
                float2 r_19 = fma_f32x2_rn_noftz(q_18, scale0_2, a_17);
                acc_1[10] = r_19.x;
                acc_1[11] = r_19.y;
                float2 a_20 = make_float2(acc_1[12], acc_1[13]);
                float2 q_21 = make_float2(_unpack_fp4_53[4], _unpack_fp4_53[5]);
                float2 r_22 = fma_f32x2_rn_noftz(q_21, scale0_2, a_20);
                acc_1[12] = r_22.x;
                acc_1[13] = r_22.y;
                float2 a_23 = make_float2(acc_1[14], acc_1[15]);
                float2 q_24 = make_float2(_unpack_fp4_53[6], _unpack_fp4_53[7]);
                float2 r_25 = fma_f32x2_rn_noftz(q_24, scale0_2, a_23);
                acc_1[14] = r_25.x;
                acc_1[15] = r_25.y;
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
                float2 a_26 = make_float2(acc_1[16], acc_1[17]);
                float2 q_27 = make_float2(_unpack_fp4_54[0], _unpack_fp4_54[1]);
                float2 r_28 = fma_f32x2_rn_noftz(q_27, scale1_2, a_26);
                acc_1[16] = r_28.x;
                acc_1[17] = r_28.y;
                float2 a_29 = make_float2(acc_1[18], acc_1[19]);
                float2 q_30 = make_float2(_unpack_fp4_54[2], _unpack_fp4_54[3]);
                float2 r_31 = fma_f32x2_rn_noftz(q_30, scale1_2, a_29);
                acc_1[18] = r_31.x;
                acc_1[19] = r_31.y;
                float2 a_32_1 = make_float2(acc_1[20], acc_1[21]);
                float2 q_33 = make_float2(_unpack_fp4_54[4], _unpack_fp4_54[5]);
                float2 r_34 = fma_f32x2_rn_noftz(q_33, scale1_2, a_32_1);
                acc_1[20] = r_34.x;
                acc_1[21] = r_34.y;
                float2 a_35 = make_float2(acc_1[22], acc_1[23]);
                float2 q_36 = make_float2(_unpack_fp4_54[6], _unpack_fp4_54[7]);
                float2 r_37 = fma_f32x2_rn_noftz(q_36, scale1_2, a_35);
                acc_1[22] = r_37.x;
                acc_1[23] = r_37.y;
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
                float2 a_38 = make_float2(acc_1[24], acc_1[25]);
                float2 q_39 = make_float2(_unpack_fp4_55[0], _unpack_fp4_55[1]);
                float2 r_40 = fma_f32x2_rn_noftz(q_39, scale1_2, a_38);
                acc_1[24] = r_40.x;
                acc_1[25] = r_40.y;
                float2 a_41 = make_float2(acc_1[26], acc_1[27]);
                float2 q_42 = make_float2(_unpack_fp4_55[2], _unpack_fp4_55[3]);
                float2 r_43 = fma_f32x2_rn_noftz(q_42, scale1_2, a_41);
                acc_1[26] = r_43.x;
                acc_1[27] = r_43.y;
                float2 a_44 = make_float2(acc_1[28], acc_1[29]);
                float2 q_45 = make_float2(_unpack_fp4_55[4], _unpack_fp4_55[5]);
                float2 r_46 = fma_f32x2_rn_noftz(q_45, scale1_2, a_44);
                acc_1[28] = r_46.x;
                acc_1[29] = r_46.y;
                float2 a_47_1 = make_float2(acc_1[30], acc_1[31]);
                float2 q_48_1 = make_float2(_unpack_fp4_55[6], _unpack_fp4_55[7]);
                float2 r_49_1 = fma_f32x2_rn_noftz(q_48_1, scale1_2, a_47_1);
                acc_1[30] = r_49_1.x;
                acc_1[31] = r_49_1.y;
                unsigned int acc_bf16[16];
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
                int c_50 = lane_1;
                int l_1 = c_50 >> 2;
                unsigned int w4_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1 + (c_50 & 3 ^ l_1 >> 1 & 3)))));
                long long idx_1 = crow64 * 3072 + (long long)(4 * c_50);
                {
                    int4 _iv4 = make_int4(w4_1[0 + 0], w4_1[0 + 1], w4_1[0 + 2], w4_1[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1) = _iv4;
                }
                int c_51 = lane_1 + 32;
                int l_52 = c_51 >> 2;
                unsigned int w4_53[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_53[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_53[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_53[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_53[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_52 + (c_51 & 3 ^ l_52 >> 1 & 3)))));
                long long idx_54 = crow64 * 3072 + (long long)(4 * c_51);
                {
                    int4 _iv4 = make_int4(w4_53[0 + 0], w4_53[0 + 1], w4_53[0 + 2], w4_53[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_54) = _iv4;
                }
                int c_55 = lane_1 + 64;
                int l_56 = c_55 >> 2;
                unsigned int w4_57[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_57[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_57[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_57[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_57[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_56 + (c_55 & 3 ^ l_56 >> 1 & 3)))));
                long long idx_58 = crow64 * 3072 + (long long)(4 * c_55);
                {
                    int4 _iv4 = make_int4(w4_57[0 + 0], w4_57[0 + 1], w4_57[0 + 2], w4_57[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_58) = _iv4;
                }
                int c_59 = lane_1 + 96;
                int l_60 = c_59 >> 2;
                unsigned int w4_61[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_61[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_61[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_61[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_61[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_60 + (c_59 & 3 ^ l_60 >> 1 & 3)))));
                long long idx_62 = crow64 * 3072 + (long long)(4 * c_59);
                {
                    int4 _iv4 = make_int4(w4_61[0 + 0], w4_61[0 + 1], w4_61[0 + 2], w4_61[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_62) = _iv4;
                }
                __syncwarp();
                float acc_63[32] = {0};
                uint32_t _e4m3x2_to_f16x2_21;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_21) : "h"((uint16_t)(_fp4_wire_scales_1[1])));
                float _cvt_f32_f16_42;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_42) : "h"((uint16_t)(_e4m3x2_to_f16x2_21)));
                float _cvt_f32_f16_43;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_43) : "h"((uint16_t)(_e4m3x2_to_f16x2_21 >> 16)));
                float _mul_28;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(_cvt_f32_f16_42), "f"(rs_2));
                float _mul_29;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(_cvt_f32_f16_43), "f"(rs_2));
                float2 scale0_64 = make_float2(_mul_28, _mul_28);
                float2 scale1_65 = make_float2(_mul_29, _mul_29);
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
                float2 a_66 = make_float2(acc_63[0], acc_63[1]);
                float2 q_67 = make_float2(_unpack_fp4_56[0], _unpack_fp4_56[1]);
                float2 r_68 = fma_f32x2_rn_noftz(q_67, scale0_64, a_66);
                acc_63[0] = r_68.x;
                acc_63[1] = r_68.y;
                float2 a_69 = make_float2(acc_63[2], acc_63[3]);
                float2 q_70 = make_float2(_unpack_fp4_56[2], _unpack_fp4_56[3]);
                float2 r_71 = fma_f32x2_rn_noftz(q_70, scale0_64, a_69);
                acc_63[2] = r_71.x;
                acc_63[3] = r_71.y;
                float2 a_72 = make_float2(acc_63[4], acc_63[5]);
                float2 q_73 = make_float2(_unpack_fp4_56[4], _unpack_fp4_56[5]);
                float2 r_74 = fma_f32x2_rn_noftz(q_73, scale0_64, a_72);
                acc_63[4] = r_74.x;
                acc_63[5] = r_74.y;
                float2 a_75 = make_float2(acc_63[6], acc_63[7]);
                float2 q_76 = make_float2(_unpack_fp4_56[6], _unpack_fp4_56[7]);
                float2 r_77 = fma_f32x2_rn_noftz(q_76, scale0_64, a_75);
                acc_63[6] = r_77.x;
                acc_63[7] = r_77.y;
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
                float2 a_78 = make_float2(acc_63[8], acc_63[9]);
                float2 q_79 = make_float2(_unpack_fp4_57[0], _unpack_fp4_57[1]);
                float2 r_80 = fma_f32x2_rn_noftz(q_79, scale0_64, a_78);
                acc_63[8] = r_80.x;
                acc_63[9] = r_80.y;
                float2 a_81 = make_float2(acc_63[10], acc_63[11]);
                float2 q_82 = make_float2(_unpack_fp4_57[2], _unpack_fp4_57[3]);
                float2 r_83 = fma_f32x2_rn_noftz(q_82, scale0_64, a_81);
                acc_63[10] = r_83.x;
                acc_63[11] = r_83.y;
                float2 a_84 = make_float2(acc_63[12], acc_63[13]);
                float2 q_85 = make_float2(_unpack_fp4_57[4], _unpack_fp4_57[5]);
                float2 r_86 = fma_f32x2_rn_noftz(q_85, scale0_64, a_84);
                acc_63[12] = r_86.x;
                acc_63[13] = r_86.y;
                float2 a_87 = make_float2(acc_63[14], acc_63[15]);
                float2 q_88 = make_float2(_unpack_fp4_57[6], _unpack_fp4_57[7]);
                float2 r_89 = fma_f32x2_rn_noftz(q_88, scale0_64, a_87);
                acc_63[14] = r_89.x;
                acc_63[15] = r_89.y;
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
                float2 a_90 = make_float2(acc_63[16], acc_63[17]);
                float2 q_91 = make_float2(_unpack_fp4_58[0], _unpack_fp4_58[1]);
                float2 r_92 = fma_f32x2_rn_noftz(q_91, scale1_65, a_90);
                acc_63[16] = r_92.x;
                acc_63[17] = r_92.y;
                float2 a_93 = make_float2(acc_63[18], acc_63[19]);
                float2 q_94 = make_float2(_unpack_fp4_58[2], _unpack_fp4_58[3]);
                float2 r_95 = fma_f32x2_rn_noftz(q_94, scale1_65, a_93);
                acc_63[18] = r_95.x;
                acc_63[19] = r_95.y;
                float2 a_96 = make_float2(acc_63[20], acc_63[21]);
                float2 q_97 = make_float2(_unpack_fp4_58[4], _unpack_fp4_58[5]);
                float2 r_98 = fma_f32x2_rn_noftz(q_97, scale1_65, a_96);
                acc_63[20] = r_98.x;
                acc_63[21] = r_98.y;
                float2 a_99 = make_float2(acc_63[22], acc_63[23]);
                float2 q_100 = make_float2(_unpack_fp4_58[6], _unpack_fp4_58[7]);
                float2 r_101 = fma_f32x2_rn_noftz(q_100, scale1_65, a_99);
                acc_63[22] = r_101.x;
                acc_63[23] = r_101.y;
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
                float2 a_102 = make_float2(acc_63[24], acc_63[25]);
                float2 q_103 = make_float2(_unpack_fp4_59[0], _unpack_fp4_59[1]);
                float2 r_104 = fma_f32x2_rn_noftz(q_103, scale1_65, a_102);
                acc_63[24] = r_104.x;
                acc_63[25] = r_104.y;
                float2 a_105 = make_float2(acc_63[26], acc_63[27]);
                float2 q_106 = make_float2(_unpack_fp4_59[2], _unpack_fp4_59[3]);
                float2 r_107 = fma_f32x2_rn_noftz(q_106, scale1_65, a_105);
                acc_63[26] = r_107.x;
                acc_63[27] = r_107.y;
                float2 a_108 = make_float2(acc_63[28], acc_63[29]);
                float2 q_109 = make_float2(_unpack_fp4_59[4], _unpack_fp4_59[5]);
                float2 r_110 = fma_f32x2_rn_noftz(q_109, scale1_65, a_108);
                acc_63[28] = r_110.x;
                acc_63[29] = r_110.y;
                float2 a_111 = make_float2(acc_63[30], acc_63[31]);
                float2 q_112 = make_float2(_unpack_fp4_59[6], _unpack_fp4_59[7]);
                float2 r_113 = fma_f32x2_rn_noftz(q_112, scale1_65, a_111);
                acc_63[30] = r_113.x;
                acc_63[31] = r_113.y;
                unsigned int acc_63_bf16[16];
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
                int c_114 = lane_1;
                int l_115 = c_114 >> 2;
                unsigned int w4_116[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_116[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_116[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_116[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_116[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_115 + (c_114 & 3 ^ l_115 >> 1 & 3)))));
                long long idx_117 = crow64 * 3072 + (long long)(4 * (128 + c_114));
                {
                    int4 _iv4 = make_int4(w4_116[0 + 0], w4_116[0 + 1], w4_116[0 + 2], w4_116[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_117) = _iv4;
                }
                int c_118 = lane_1 + 32;
                int l_119 = c_118 >> 2;
                unsigned int w4_120[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_120[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_120[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_120[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_120[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_119 + (c_118 & 3 ^ l_119 >> 1 & 3)))));
                long long idx_121 = crow64 * 3072 + (long long)(4 * (128 + c_118));
                {
                    int4 _iv4 = make_int4(w4_120[0 + 0], w4_120[0 + 1], w4_120[0 + 2], w4_120[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_121) = _iv4;
                }
                int c_122 = lane_1 + 64;
                int l_123 = c_122 >> 2;
                unsigned int w4_124[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_124[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_124[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_124[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_124[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_123 + (c_122 & 3 ^ l_123 >> 1 & 3)))));
                long long idx_125 = crow64 * 3072 + (long long)(4 * (128 + c_122));
                {
                    int4 _iv4 = make_int4(w4_124[0 + 0], w4_124[0 + 1], w4_124[0 + 2], w4_124[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_125) = _iv4;
                }
                int c_126 = lane_1 + 96;
                int l_127 = c_126 >> 2;
                unsigned int w4_128[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_128[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_128[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_128[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_128[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_127 + (c_126 & 3 ^ l_127 >> 1 & 3)))));
                long long idx_129 = crow64 * 3072 + (long long)(4 * (128 + c_126));
                {
                    int4 _iv4 = make_int4(w4_128[0 + 0], w4_128[0 + 1], w4_128[0 + 2], w4_128[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_129) = _iv4;
                }
                __syncwarp();
                float acc_130[32] = {0};
                uint32_t _e4m3x2_to_f16x2_22;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_22) : "h"((uint16_t)(_fp4_wire_scales_1[2])));
                float _cvt_f32_f16_44;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_44) : "h"((uint16_t)(_e4m3x2_to_f16x2_22)));
                float _cvt_f32_f16_45;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_45) : "h"((uint16_t)(_e4m3x2_to_f16x2_22 >> 16)));
                float _mul_30;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(_cvt_f32_f16_44), "f"(rs_2));
                float _mul_31;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(_cvt_f32_f16_45), "f"(rs_2));
                float2 scale0_131 = make_float2(_mul_30, _mul_30);
                float2 scale1_132 = make_float2(_mul_31, _mul_31);
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
                float2 a_133_2 = make_float2(acc_130[0], acc_130[1]);
                float2 q_134_2 = make_float2(_unpack_fp4_60[0], _unpack_fp4_60[1]);
                float2 r_135_2 = fma_f32x2_rn_noftz(q_134_2, scale0_131, a_133_2);
                acc_130[0] = r_135_2.x;
                acc_130[1] = r_135_2.y;
                float2 a_136_2 = make_float2(acc_130[2], acc_130[3]);
                float2 q_137_2 = make_float2(_unpack_fp4_60[2], _unpack_fp4_60[3]);
                float2 r_138_2 = fma_f32x2_rn_noftz(q_137_2, scale0_131, a_136_2);
                acc_130[2] = r_138_2.x;
                acc_130[3] = r_138_2.y;
                float2 a_139_2 = make_float2(acc_130[4], acc_130[5]);
                float2 q_140_2 = make_float2(_unpack_fp4_60[4], _unpack_fp4_60[5]);
                float2 r_141_2 = fma_f32x2_rn_noftz(q_140_2, scale0_131, a_139_2);
                acc_130[4] = r_141_2.x;
                acc_130[5] = r_141_2.y;
                float2 a_142_2 = make_float2(acc_130[6], acc_130[7]);
                float2 q_143_2 = make_float2(_unpack_fp4_60[6], _unpack_fp4_60[7]);
                float2 r_144_2 = fma_f32x2_rn_noftz(q_143_2, scale0_131, a_142_2);
                acc_130[6] = r_144_2.x;
                acc_130[7] = r_144_2.y;
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
                float2 a_145_1 = make_float2(acc_130[8], acc_130[9]);
                float2 q_146_1 = make_float2(_unpack_fp4_61[0], _unpack_fp4_61[1]);
                float2 r_147_1 = fma_f32x2_rn_noftz(q_146_1, scale0_131, a_145_1);
                acc_130[8] = r_147_1.x;
                acc_130[9] = r_147_1.y;
                float2 a_148 = make_float2(acc_130[10], acc_130[11]);
                float2 q_149 = make_float2(_unpack_fp4_61[2], _unpack_fp4_61[3]);
                float2 r_150 = fma_f32x2_rn_noftz(q_149, scale0_131, a_148);
                acc_130[10] = r_150.x;
                acc_130[11] = r_150.y;
                float2 a_151 = make_float2(acc_130[12], acc_130[13]);
                float2 q_152 = make_float2(_unpack_fp4_61[4], _unpack_fp4_61[5]);
                float2 r_153 = fma_f32x2_rn_noftz(q_152, scale0_131, a_151);
                acc_130[12] = r_153.x;
                acc_130[13] = r_153.y;
                float2 a_154 = make_float2(acc_130[14], acc_130[15]);
                float2 q_155 = make_float2(_unpack_fp4_61[6], _unpack_fp4_61[7]);
                float2 r_156 = fma_f32x2_rn_noftz(q_155, scale0_131, a_154);
                acc_130[14] = r_156.x;
                acc_130[15] = r_156.y;
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
                float2 a_157 = make_float2(acc_130[16], acc_130[17]);
                float2 q_158 = make_float2(_unpack_fp4_62[0], _unpack_fp4_62[1]);
                float2 r_159 = fma_f32x2_rn_noftz(q_158, scale1_132, a_157);
                acc_130[16] = r_159.x;
                acc_130[17] = r_159.y;
                float2 a_160 = make_float2(acc_130[18], acc_130[19]);
                float2 q_161 = make_float2(_unpack_fp4_62[2], _unpack_fp4_62[3]);
                float2 r_162 = fma_f32x2_rn_noftz(q_161, scale1_132, a_160);
                acc_130[18] = r_162.x;
                acc_130[19] = r_162.y;
                float2 a_163 = make_float2(acc_130[20], acc_130[21]);
                float2 q_164 = make_float2(_unpack_fp4_62[4], _unpack_fp4_62[5]);
                float2 r_165 = fma_f32x2_rn_noftz(q_164, scale1_132, a_163);
                acc_130[20] = r_165.x;
                acc_130[21] = r_165.y;
                float2 a_166 = make_float2(acc_130[22], acc_130[23]);
                float2 q_167 = make_float2(_unpack_fp4_62[6], _unpack_fp4_62[7]);
                float2 r_168 = fma_f32x2_rn_noftz(q_167, scale1_132, a_166);
                acc_130[22] = r_168.x;
                acc_130[23] = r_168.y;
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
                float2 a_169 = make_float2(acc_130[24], acc_130[25]);
                float2 q_170 = make_float2(_unpack_fp4_63[0], _unpack_fp4_63[1]);
                float2 r_171 = fma_f32x2_rn_noftz(q_170, scale1_132, a_169);
                acc_130[24] = r_171.x;
                acc_130[25] = r_171.y;
                float2 a_172 = make_float2(acc_130[26], acc_130[27]);
                float2 q_173 = make_float2(_unpack_fp4_63[2], _unpack_fp4_63[3]);
                float2 r_174 = fma_f32x2_rn_noftz(q_173, scale1_132, a_172);
                acc_130[26] = r_174.x;
                acc_130[27] = r_174.y;
                float2 a_175 = make_float2(acc_130[28], acc_130[29]);
                float2 q_176 = make_float2(_unpack_fp4_63[4], _unpack_fp4_63[5]);
                float2 r_177 = fma_f32x2_rn_noftz(q_176, scale1_132, a_175);
                acc_130[28] = r_177.x;
                acc_130[29] = r_177.y;
                float2 a_178 = make_float2(acc_130[30], acc_130[31]);
                float2 q_179 = make_float2(_unpack_fp4_63[6], _unpack_fp4_63[7]);
                float2 r_180 = fma_f32x2_rn_noftz(q_179, scale1_132, a_178);
                acc_130[30] = r_180.x;
                acc_130[31] = r_180.y;
                unsigned int acc_130_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_130[_lp*2 + 0], acc_130[_lp*2+1 + 0]));
                    acc_130_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_130_bf16[0]), "r"(acc_130_bf16[1]), "r"(acc_130_bf16[2]), "r"(acc_130_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_130_bf16[4]), "r"(acc_130_bf16[5]), "r"(acc_130_bf16[6]), "r"(acc_130_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_130_bf16[8]), "r"(acc_130_bf16[9]), "r"(acc_130_bf16[10]), "r"(acc_130_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_130_bf16[12]), "r"(acc_130_bf16[13]), "r"(acc_130_bf16[14]), "r"(acc_130_bf16[15]) : "memory");
                __syncwarp();
                int c_181 = lane_1;
                int l_182 = c_181 >> 2;
                unsigned int w4_183[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_183[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_183[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_183[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_183[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_182 + (c_181 & 3 ^ l_182 >> 1 & 3)))));
                long long idx_184 = crow64 * 3072 + (long long)(4 * (256 + c_181));
                {
                    int4 _iv4 = make_int4(w4_183[0 + 0], w4_183[0 + 1], w4_183[0 + 2], w4_183[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_184) = _iv4;
                }
                int c_185 = lane_1 + 32;
                int l_186 = c_185 >> 2;
                unsigned int w4_187[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_187[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_187[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_187[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_187[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_186 + (c_185 & 3 ^ l_186 >> 1 & 3)))));
                long long idx_188 = crow64 * 3072 + (long long)(4 * (256 + c_185));
                {
                    int4 _iv4 = make_int4(w4_187[0 + 0], w4_187[0 + 1], w4_187[0 + 2], w4_187[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_188) = _iv4;
                }
                int c_189 = lane_1 + 64;
                int l_190 = c_189 >> 2;
                unsigned int w4_191[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_191[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_191[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_191[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_191[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_190 + (c_189 & 3 ^ l_190 >> 1 & 3)))));
                long long idx_192 = crow64 * 3072 + (long long)(4 * (256 + c_189));
                {
                    int4 _iv4 = make_int4(w4_191[0 + 0], w4_191[0 + 1], w4_191[0 + 2], w4_191[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_192) = _iv4;
                }
                int c_193 = lane_1 + 96;
                int l_194 = c_193 >> 2;
                unsigned int w4_195[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_195[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_195[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_195[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_195[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_194 + (c_193 & 3 ^ l_194 >> 1 & 3)))));
                long long idx_196 = crow64 * 3072 + (long long)(4 * (256 + c_193));
                {
                    int4 _iv4 = make_int4(w4_195[0 + 0], w4_195[0 + 1], w4_195[0 + 2], w4_195[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_196) = _iv4;
                }
                __syncwarp();
                float acc_197[32] = {0};
                uint32_t _e4m3x2_to_f16x2_23;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_23) : "h"((uint16_t)(_fp4_wire_scales_1[3])));
                float _cvt_f32_f16_46;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_46) : "h"((uint16_t)(_e4m3x2_to_f16x2_23)));
                float _cvt_f32_f16_47;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_47) : "h"((uint16_t)(_e4m3x2_to_f16x2_23 >> 16)));
                float _mul_32;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(_cvt_f32_f16_46), "f"(rs_2));
                float _mul_33;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(_cvt_f32_f16_47), "f"(rs_2));
                float2 scale0_198 = make_float2(_mul_32, _mul_32);
                float2 scale1_199 = make_float2(_mul_33, _mul_33);
                float _unpack_fp4_64[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[12 + 0]));
                    (&_unpack_fp4_64[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_64[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_64[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_64[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_64[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_64[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_64[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_64[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_64[0]), "=f"(_unpack_fp4_64[1]), "=f"(_unpack_fp4_64[2]), "=f"(_unpack_fp4_64[3]), "=f"(_unpack_fp4_64[4]), "=f"(_unpack_fp4_64[5]), "=f"(_unpack_fp4_64[6]), "=f"(_unpack_fp4_64[7]) : "r"(_fp4_wire_values_1[12 + 0]));
                    #endif
                }
                float2 a_200 = make_float2(acc_197[0], acc_197[1]);
                float2 q_201 = make_float2(_unpack_fp4_64[0], _unpack_fp4_64[1]);
                float2 r_202 = fma_f32x2_rn_noftz(q_201, scale0_198, a_200);
                acc_197[0] = r_202.x;
                acc_197[1] = r_202.y;
                float2 a_203 = make_float2(acc_197[2], acc_197[3]);
                float2 q_204 = make_float2(_unpack_fp4_64[2], _unpack_fp4_64[3]);
                float2 r_205 = fma_f32x2_rn_noftz(q_204, scale0_198, a_203);
                acc_197[2] = r_205.x;
                acc_197[3] = r_205.y;
                float2 a_206 = make_float2(acc_197[4], acc_197[5]);
                float2 q_207 = make_float2(_unpack_fp4_64[4], _unpack_fp4_64[5]);
                float2 r_208 = fma_f32x2_rn_noftz(q_207, scale0_198, a_206);
                acc_197[4] = r_208.x;
                acc_197[5] = r_208.y;
                float2 a_209 = make_float2(acc_197[6], acc_197[7]);
                float2 q_210 = make_float2(_unpack_fp4_64[6], _unpack_fp4_64[7]);
                float2 r_211 = fma_f32x2_rn_noftz(q_210, scale0_198, a_209);
                acc_197[6] = r_211.x;
                acc_197[7] = r_211.y;
                float _unpack_fp4_65[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[13 + 0]));
                    (&_unpack_fp4_65[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_65[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_65[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_65[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_65[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_65[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_65[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_65[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_65[0]), "=f"(_unpack_fp4_65[1]), "=f"(_unpack_fp4_65[2]), "=f"(_unpack_fp4_65[3]), "=f"(_unpack_fp4_65[4]), "=f"(_unpack_fp4_65[5]), "=f"(_unpack_fp4_65[6]), "=f"(_unpack_fp4_65[7]) : "r"(_fp4_wire_values_1[13 + 0]));
                    #endif
                }
                float2 a_212 = make_float2(acc_197[8], acc_197[9]);
                float2 q_213 = make_float2(_unpack_fp4_65[0], _unpack_fp4_65[1]);
                float2 r_214 = fma_f32x2_rn_noftz(q_213, scale0_198, a_212);
                acc_197[8] = r_214.x;
                acc_197[9] = r_214.y;
                float2 a_215 = make_float2(acc_197[10], acc_197[11]);
                float2 q_216 = make_float2(_unpack_fp4_65[2], _unpack_fp4_65[3]);
                float2 r_217 = fma_f32x2_rn_noftz(q_216, scale0_198, a_215);
                acc_197[10] = r_217.x;
                acc_197[11] = r_217.y;
                float2 a_218 = make_float2(acc_197[12], acc_197[13]);
                float2 q_219 = make_float2(_unpack_fp4_65[4], _unpack_fp4_65[5]);
                float2 r_220 = fma_f32x2_rn_noftz(q_219, scale0_198, a_218);
                acc_197[12] = r_220.x;
                acc_197[13] = r_220.y;
                float2 a_221 = make_float2(acc_197[14], acc_197[15]);
                float2 q_222 = make_float2(_unpack_fp4_65[6], _unpack_fp4_65[7]);
                float2 r_223 = fma_f32x2_rn_noftz(q_222, scale0_198, a_221);
                acc_197[14] = r_223.x;
                acc_197[15] = r_223.y;
                float _unpack_fp4_66[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[14 + 0]));
                    (&_unpack_fp4_66[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_66[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_66[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_66[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_66[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_66[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_66[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_66[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_66[0]), "=f"(_unpack_fp4_66[1]), "=f"(_unpack_fp4_66[2]), "=f"(_unpack_fp4_66[3]), "=f"(_unpack_fp4_66[4]), "=f"(_unpack_fp4_66[5]), "=f"(_unpack_fp4_66[6]), "=f"(_unpack_fp4_66[7]) : "r"(_fp4_wire_values_1[14 + 0]));
                    #endif
                }
                float2 a_224 = make_float2(acc_197[16], acc_197[17]);
                float2 q_225 = make_float2(_unpack_fp4_66[0], _unpack_fp4_66[1]);
                float2 r_226 = fma_f32x2_rn_noftz(q_225, scale1_199, a_224);
                acc_197[16] = r_226.x;
                acc_197[17] = r_226.y;
                float2 a_227 = make_float2(acc_197[18], acc_197[19]);
                float2 q_228 = make_float2(_unpack_fp4_66[2], _unpack_fp4_66[3]);
                float2 r_229 = fma_f32x2_rn_noftz(q_228, scale1_199, a_227);
                acc_197[18] = r_229.x;
                acc_197[19] = r_229.y;
                float2 a_230 = make_float2(acc_197[20], acc_197[21]);
                float2 q_231 = make_float2(_unpack_fp4_66[4], _unpack_fp4_66[5]);
                float2 r_232 = fma_f32x2_rn_noftz(q_231, scale1_199, a_230);
                acc_197[20] = r_232.x;
                acc_197[21] = r_232.y;
                float2 a_233 = make_float2(acc_197[22], acc_197[23]);
                float2 q_234 = make_float2(_unpack_fp4_66[6], _unpack_fp4_66[7]);
                float2 r_235 = fma_f32x2_rn_noftz(q_234, scale1_199, a_233);
                acc_197[22] = r_235.x;
                acc_197[23] = r_235.y;
                float _unpack_fp4_67[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[15 + 0]));
                    (&_unpack_fp4_67[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_67[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_67[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_67[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_67[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_67[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_67[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_67[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_67[0]), "=f"(_unpack_fp4_67[1]), "=f"(_unpack_fp4_67[2]), "=f"(_unpack_fp4_67[3]), "=f"(_unpack_fp4_67[4]), "=f"(_unpack_fp4_67[5]), "=f"(_unpack_fp4_67[6]), "=f"(_unpack_fp4_67[7]) : "r"(_fp4_wire_values_1[15 + 0]));
                    #endif
                }
                float2 a_236 = make_float2(acc_197[24], acc_197[25]);
                float2 q_237 = make_float2(_unpack_fp4_67[0], _unpack_fp4_67[1]);
                float2 r_238 = fma_f32x2_rn_noftz(q_237, scale1_199, a_236);
                acc_197[24] = r_238.x;
                acc_197[25] = r_238.y;
                float2 a_239 = make_float2(acc_197[26], acc_197[27]);
                float2 q_240 = make_float2(_unpack_fp4_67[2], _unpack_fp4_67[3]);
                float2 r_241 = fma_f32x2_rn_noftz(q_240, scale1_199, a_239);
                acc_197[26] = r_241.x;
                acc_197[27] = r_241.y;
                float2 a_242 = make_float2(acc_197[28], acc_197[29]);
                float2 q_243 = make_float2(_unpack_fp4_67[4], _unpack_fp4_67[5]);
                float2 r_244 = fma_f32x2_rn_noftz(q_243, scale1_199, a_242);
                acc_197[28] = r_244.x;
                acc_197[29] = r_244.y;
                float2 a_245 = make_float2(acc_197[30], acc_197[31]);
                float2 q_246 = make_float2(_unpack_fp4_67[6], _unpack_fp4_67[7]);
                float2 r_247 = fma_f32x2_rn_noftz(q_246, scale1_199, a_245);
                acc_197[30] = r_247.x;
                acc_197[31] = r_247.y;
                unsigned int acc_197_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_197[_lp*2 + 0], acc_197[_lp*2+1 + 0]));
                    acc_197_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_197_bf16[0]), "r"(acc_197_bf16[1]), "r"(acc_197_bf16[2]), "r"(acc_197_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_197_bf16[4]), "r"(acc_197_bf16[5]), "r"(acc_197_bf16[6]), "r"(acc_197_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_197_bf16[8]), "r"(acc_197_bf16[9]), "r"(acc_197_bf16[10]), "r"(acc_197_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_197_bf16[12]), "r"(acc_197_bf16[13]), "r"(acc_197_bf16[14]), "r"(acc_197_bf16[15]) : "memory");
                __syncwarp();
                int c_248 = lane_1;
                int l_249 = c_248 >> 2;
                unsigned int w4_250[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_250[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_250[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_250[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_250[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_249 + (c_248 & 3 ^ l_249 >> 1 & 3)))));
                long long idx_251 = crow64 * 3072 + (long long)(4 * (384 + c_248));
                {
                    int4 _iv4 = make_int4(w4_250[0 + 0], w4_250[0 + 1], w4_250[0 + 2], w4_250[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_251) = _iv4;
                }
                int c_252 = lane_1 + 32;
                int l_253 = c_252 >> 2;
                unsigned int w4_254[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_254[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_254[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_254[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_254[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_253 + (c_252 & 3 ^ l_253 >> 1 & 3)))));
                long long idx_255 = crow64 * 3072 + (long long)(4 * (384 + c_252));
                {
                    int4 _iv4 = make_int4(w4_254[0 + 0], w4_254[0 + 1], w4_254[0 + 2], w4_254[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_255) = _iv4;
                }
                int c_256 = lane_1 + 64;
                int l_257 = c_256 >> 2;
                unsigned int w4_258[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_258[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_258[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_258[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_258[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_257 + (c_256 & 3 ^ l_257 >> 1 & 3)))));
                long long idx_259 = crow64 * 3072 + (long long)(4 * (384 + c_256));
                {
                    int4 _iv4 = make_int4(w4_258[0 + 0], w4_258[0 + 1], w4_258[0 + 2], w4_258[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_259) = _iv4;
                }
                int c_260 = lane_1 + 96;
                int l_261 = c_260 >> 2;
                unsigned int w4_262[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_262[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_262[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_262[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_262[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_261 + (c_260 & 3 ^ l_261 >> 1 & 3)))));
                long long idx_263 = crow64 * 3072 + (long long)(4 * (384 + c_260));
                {
                    int4 _iv4 = make_int4(w4_262[0 + 0], w4_262[0 + 1], w4_262[0 + 2], w4_262[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_263) = _iv4;
                }
                __syncwarp();
                float acc_264[32] = {0};
                uint32_t _e4m3x2_to_f16x2_24;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_24) : "h"((uint16_t)(_fp4_wire_scales_1[4])));
                float _cvt_f32_f16_48;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_48) : "h"((uint16_t)(_e4m3x2_to_f16x2_24)));
                float _cvt_f32_f16_49;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_49) : "h"((uint16_t)(_e4m3x2_to_f16x2_24 >> 16)));
                float _mul_34;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(_cvt_f32_f16_48), "f"(rs_2));
                float _mul_35;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(_cvt_f32_f16_49), "f"(rs_2));
                float2 scale0_265 = make_float2(_mul_34, _mul_34);
                float2 scale1_266 = make_float2(_mul_35, _mul_35);
                float _unpack_fp4_68[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[16 + 0]));
                    (&_unpack_fp4_68[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_68[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_68[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_68[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_68[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_68[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_68[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_68[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_68[0]), "=f"(_unpack_fp4_68[1]), "=f"(_unpack_fp4_68[2]), "=f"(_unpack_fp4_68[3]), "=f"(_unpack_fp4_68[4]), "=f"(_unpack_fp4_68[5]), "=f"(_unpack_fp4_68[6]), "=f"(_unpack_fp4_68[7]) : "r"(_fp4_wire_values_1[16 + 0]));
                    #endif
                }
                float2 a_267_1 = make_float2(acc_264[0], acc_264[1]);
                float2 q_268_1 = make_float2(_unpack_fp4_68[0], _unpack_fp4_68[1]);
                float2 r_269_1 = fma_f32x2_rn_noftz(q_268_1, scale0_265, a_267_1);
                acc_264[0] = r_269_1.x;
                acc_264[1] = r_269_1.y;
                float2 a_270_1 = make_float2(acc_264[2], acc_264[3]);
                float2 q_271_1 = make_float2(_unpack_fp4_68[2], _unpack_fp4_68[3]);
                float2 r_272_1 = fma_f32x2_rn_noftz(q_271_1, scale0_265, a_270_1);
                acc_264[2] = r_272_1.x;
                acc_264[3] = r_272_1.y;
                float2 a_273_1 = make_float2(acc_264[4], acc_264[5]);
                float2 q_274_1 = make_float2(_unpack_fp4_68[4], _unpack_fp4_68[5]);
                float2 r_275_1 = fma_f32x2_rn_noftz(q_274_1, scale0_265, a_273_1);
                acc_264[4] = r_275_1.x;
                acc_264[5] = r_275_1.y;
                float2 a_276_1 = make_float2(acc_264[6], acc_264[7]);
                float2 q_277_1 = make_float2(_unpack_fp4_68[6], _unpack_fp4_68[7]);
                float2 r_278_1 = fma_f32x2_rn_noftz(q_277_1, scale0_265, a_276_1);
                acc_264[6] = r_278_1.x;
                acc_264[7] = r_278_1.y;
                float _unpack_fp4_69[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[17 + 0]));
                    (&_unpack_fp4_69[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_69[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_69[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_69[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_69[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_69[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_69[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_69[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_69[0]), "=f"(_unpack_fp4_69[1]), "=f"(_unpack_fp4_69[2]), "=f"(_unpack_fp4_69[3]), "=f"(_unpack_fp4_69[4]), "=f"(_unpack_fp4_69[5]), "=f"(_unpack_fp4_69[6]), "=f"(_unpack_fp4_69[7]) : "r"(_fp4_wire_values_1[17 + 0]));
                    #endif
                }
                float2 a_279_1 = make_float2(acc_264[8], acc_264[9]);
                float2 q_280_1 = make_float2(_unpack_fp4_69[0], _unpack_fp4_69[1]);
                float2 r_281_1 = fma_f32x2_rn_noftz(q_280_1, scale0_265, a_279_1);
                acc_264[8] = r_281_1.x;
                acc_264[9] = r_281_1.y;
                float2 a_282_1 = make_float2(acc_264[10], acc_264[11]);
                float2 q_283_1 = make_float2(_unpack_fp4_69[2], _unpack_fp4_69[3]);
                float2 r_284_1 = fma_f32x2_rn_noftz(q_283_1, scale0_265, a_282_1);
                acc_264[10] = r_284_1.x;
                acc_264[11] = r_284_1.y;
                float2 a_285_1 = make_float2(acc_264[12], acc_264[13]);
                float2 q_286_1 = make_float2(_unpack_fp4_69[4], _unpack_fp4_69[5]);
                float2 r_287_1 = fma_f32x2_rn_noftz(q_286_1, scale0_265, a_285_1);
                acc_264[12] = r_287_1.x;
                acc_264[13] = r_287_1.y;
                float2 a_288 = make_float2(acc_264[14], acc_264[15]);
                float2 q_289 = make_float2(_unpack_fp4_69[6], _unpack_fp4_69[7]);
                float2 r_290 = fma_f32x2_rn_noftz(q_289, scale0_265, a_288);
                acc_264[14] = r_290.x;
                acc_264[15] = r_290.y;
                float _unpack_fp4_70[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[18 + 0]));
                    (&_unpack_fp4_70[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_70[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_70[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_70[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_70[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_70[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_70[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_70[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_70[0]), "=f"(_unpack_fp4_70[1]), "=f"(_unpack_fp4_70[2]), "=f"(_unpack_fp4_70[3]), "=f"(_unpack_fp4_70[4]), "=f"(_unpack_fp4_70[5]), "=f"(_unpack_fp4_70[6]), "=f"(_unpack_fp4_70[7]) : "r"(_fp4_wire_values_1[18 + 0]));
                    #endif
                }
                float2 a_291 = make_float2(acc_264[16], acc_264[17]);
                float2 q_292 = make_float2(_unpack_fp4_70[0], _unpack_fp4_70[1]);
                float2 r_293 = fma_f32x2_rn_noftz(q_292, scale1_266, a_291);
                acc_264[16] = r_293.x;
                acc_264[17] = r_293.y;
                float2 a_294 = make_float2(acc_264[18], acc_264[19]);
                float2 q_295 = make_float2(_unpack_fp4_70[2], _unpack_fp4_70[3]);
                float2 r_296 = fma_f32x2_rn_noftz(q_295, scale1_266, a_294);
                acc_264[18] = r_296.x;
                acc_264[19] = r_296.y;
                float2 a_297 = make_float2(acc_264[20], acc_264[21]);
                float2 q_298 = make_float2(_unpack_fp4_70[4], _unpack_fp4_70[5]);
                float2 r_299 = fma_f32x2_rn_noftz(q_298, scale1_266, a_297);
                acc_264[20] = r_299.x;
                acc_264[21] = r_299.y;
                float2 a_300 = make_float2(acc_264[22], acc_264[23]);
                float2 q_301 = make_float2(_unpack_fp4_70[6], _unpack_fp4_70[7]);
                float2 r_302 = fma_f32x2_rn_noftz(q_301, scale1_266, a_300);
                acc_264[22] = r_302.x;
                acc_264[23] = r_302.y;
                float _unpack_fp4_71[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[19 + 0]));
                    (&_unpack_fp4_71[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_71[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_71[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_71[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_71[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_71[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_71[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_71[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_71[0]), "=f"(_unpack_fp4_71[1]), "=f"(_unpack_fp4_71[2]), "=f"(_unpack_fp4_71[3]), "=f"(_unpack_fp4_71[4]), "=f"(_unpack_fp4_71[5]), "=f"(_unpack_fp4_71[6]), "=f"(_unpack_fp4_71[7]) : "r"(_fp4_wire_values_1[19 + 0]));
                    #endif
                }
                float2 a_303 = make_float2(acc_264[24], acc_264[25]);
                float2 q_304 = make_float2(_unpack_fp4_71[0], _unpack_fp4_71[1]);
                float2 r_305 = fma_f32x2_rn_noftz(q_304, scale1_266, a_303);
                acc_264[24] = r_305.x;
                acc_264[25] = r_305.y;
                float2 a_306 = make_float2(acc_264[26], acc_264[27]);
                float2 q_307 = make_float2(_unpack_fp4_71[2], _unpack_fp4_71[3]);
                float2 r_308 = fma_f32x2_rn_noftz(q_307, scale1_266, a_306);
                acc_264[26] = r_308.x;
                acc_264[27] = r_308.y;
                float2 a_309 = make_float2(acc_264[28], acc_264[29]);
                float2 q_310 = make_float2(_unpack_fp4_71[4], _unpack_fp4_71[5]);
                float2 r_311 = fma_f32x2_rn_noftz(q_310, scale1_266, a_309);
                acc_264[28] = r_311.x;
                acc_264[29] = r_311.y;
                float2 a_312 = make_float2(acc_264[30], acc_264[31]);
                float2 q_313 = make_float2(_unpack_fp4_71[6], _unpack_fp4_71[7]);
                float2 r_314 = fma_f32x2_rn_noftz(q_313, scale1_266, a_312);
                acc_264[30] = r_314.x;
                acc_264[31] = r_314.y;
                unsigned int acc_264_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_264[_lp*2 + 0], acc_264[_lp*2+1 + 0]));
                    acc_264_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_264_bf16[0]), "r"(acc_264_bf16[1]), "r"(acc_264_bf16[2]), "r"(acc_264_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_264_bf16[4]), "r"(acc_264_bf16[5]), "r"(acc_264_bf16[6]), "r"(acc_264_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_264_bf16[8]), "r"(acc_264_bf16[9]), "r"(acc_264_bf16[10]), "r"(acc_264_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_264_bf16[12]), "r"(acc_264_bf16[13]), "r"(acc_264_bf16[14]), "r"(acc_264_bf16[15]) : "memory");
                __syncwarp();
                int c_315 = lane_1;
                int l_316 = c_315 >> 2;
                unsigned int w4_317[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_317[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_317[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_317[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_317[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_316 + (c_315 & 3 ^ l_316 >> 1 & 3)))));
                long long idx_318 = crow64 * 3072 + (long long)(4 * (512 + c_315));
                {
                    int4 _iv4 = make_int4(w4_317[0 + 0], w4_317[0 + 1], w4_317[0 + 2], w4_317[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_318) = _iv4;
                }
                int c_319 = lane_1 + 32;
                int l_320 = c_319 >> 2;
                unsigned int w4_321[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_321[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_321[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_321[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_321[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_320 + (c_319 & 3 ^ l_320 >> 1 & 3)))));
                long long idx_322 = crow64 * 3072 + (long long)(4 * (512 + c_319));
                {
                    int4 _iv4 = make_int4(w4_321[0 + 0], w4_321[0 + 1], w4_321[0 + 2], w4_321[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_322) = _iv4;
                }
                int c_323 = lane_1 + 64;
                int l_324 = c_323 >> 2;
                unsigned int w4_325[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_325[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_325[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_325[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_325[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_324 + (c_323 & 3 ^ l_324 >> 1 & 3)))));
                long long idx_326 = crow64 * 3072 + (long long)(4 * (512 + c_323));
                {
                    int4 _iv4 = make_int4(w4_325[0 + 0], w4_325[0 + 1], w4_325[0 + 2], w4_325[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_326) = _iv4;
                }
                int c_327 = lane_1 + 96;
                int l_328 = c_327 >> 2;
                unsigned int w4_329[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_329[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_329[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_329[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_329[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_328 + (c_327 & 3 ^ l_328 >> 1 & 3)))));
                long long idx_330 = crow64 * 3072 + (long long)(4 * (512 + c_327));
                {
                    int4 _iv4 = make_int4(w4_329[0 + 0], w4_329[0 + 1], w4_329[0 + 2], w4_329[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_330) = _iv4;
                }
                __syncwarp();
                float acc_331[32] = {0};
                uint32_t _e4m3x2_to_f16x2_25;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_25) : "h"((uint16_t)(_fp4_wire_scales_1[5])));
                float _cvt_f32_f16_50;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_50) : "h"((uint16_t)(_e4m3x2_to_f16x2_25)));
                float _cvt_f32_f16_51;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_51) : "h"((uint16_t)(_e4m3x2_to_f16x2_25 >> 16)));
                float _mul_36;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(_cvt_f32_f16_50), "f"(rs_2));
                float _mul_37;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(_cvt_f32_f16_51), "f"(rs_2));
                float2 scale0_332 = make_float2(_mul_36, _mul_36);
                float2 scale1_333 = make_float2(_mul_37, _mul_37);
                float _unpack_fp4_72[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[20 + 0]));
                    (&_unpack_fp4_72[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_72[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_72[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_72[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_72[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_72[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_72[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_72[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_72[0]), "=f"(_unpack_fp4_72[1]), "=f"(_unpack_fp4_72[2]), "=f"(_unpack_fp4_72[3]), "=f"(_unpack_fp4_72[4]), "=f"(_unpack_fp4_72[5]), "=f"(_unpack_fp4_72[6]), "=f"(_unpack_fp4_72[7]) : "r"(_fp4_wire_values_1[20 + 0]));
                    #endif
                }
                float2 a_334_1 = make_float2(acc_331[0], acc_331[1]);
                float2 q_335_1 = make_float2(_unpack_fp4_72[0], _unpack_fp4_72[1]);
                float2 r_336_1 = fma_f32x2_rn_noftz(q_335_1, scale0_332, a_334_1);
                acc_331[0] = r_336_1.x;
                acc_331[1] = r_336_1.y;
                float2 a_337_1 = make_float2(acc_331[2], acc_331[3]);
                float2 q_338_1 = make_float2(_unpack_fp4_72[2], _unpack_fp4_72[3]);
                float2 r_339_1 = fma_f32x2_rn_noftz(q_338_1, scale0_332, a_337_1);
                acc_331[2] = r_339_1.x;
                acc_331[3] = r_339_1.y;
                float2 a_340_1 = make_float2(acc_331[4], acc_331[5]);
                float2 q_341_1 = make_float2(_unpack_fp4_72[4], _unpack_fp4_72[5]);
                float2 r_342_1 = fma_f32x2_rn_noftz(q_341_1, scale0_332, a_340_1);
                acc_331[4] = r_342_1.x;
                acc_331[5] = r_342_1.y;
                float2 a_343_1 = make_float2(acc_331[6], acc_331[7]);
                float2 q_344_1 = make_float2(_unpack_fp4_72[6], _unpack_fp4_72[7]);
                float2 r_345_1 = fma_f32x2_rn_noftz(q_344_1, scale0_332, a_343_1);
                acc_331[6] = r_345_1.x;
                acc_331[7] = r_345_1.y;
                float _unpack_fp4_73[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[21 + 0]));
                    (&_unpack_fp4_73[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_73[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_73[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_73[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_73[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_73[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_73[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_73[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_73[0]), "=f"(_unpack_fp4_73[1]), "=f"(_unpack_fp4_73[2]), "=f"(_unpack_fp4_73[3]), "=f"(_unpack_fp4_73[4]), "=f"(_unpack_fp4_73[5]), "=f"(_unpack_fp4_73[6]), "=f"(_unpack_fp4_73[7]) : "r"(_fp4_wire_values_1[21 + 0]));
                    #endif
                }
                float2 a_346 = make_float2(acc_331[8], acc_331[9]);
                float2 q_347 = make_float2(_unpack_fp4_73[0], _unpack_fp4_73[1]);
                float2 r_348 = fma_f32x2_rn_noftz(q_347, scale0_332, a_346);
                acc_331[8] = r_348.x;
                acc_331[9] = r_348.y;
                float2 a_349 = make_float2(acc_331[10], acc_331[11]);
                float2 q_350 = make_float2(_unpack_fp4_73[2], _unpack_fp4_73[3]);
                float2 r_351 = fma_f32x2_rn_noftz(q_350, scale0_332, a_349);
                acc_331[10] = r_351.x;
                acc_331[11] = r_351.y;
                float2 a_352 = make_float2(acc_331[12], acc_331[13]);
                float2 q_353 = make_float2(_unpack_fp4_73[4], _unpack_fp4_73[5]);
                float2 r_354 = fma_f32x2_rn_noftz(q_353, scale0_332, a_352);
                acc_331[12] = r_354.x;
                acc_331[13] = r_354.y;
                float2 a_355 = make_float2(acc_331[14], acc_331[15]);
                float2 q_356 = make_float2(_unpack_fp4_73[6], _unpack_fp4_73[7]);
                float2 r_357 = fma_f32x2_rn_noftz(q_356, scale0_332, a_355);
                acc_331[14] = r_357.x;
                acc_331[15] = r_357.y;
                float _unpack_fp4_74[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[22 + 0]));
                    (&_unpack_fp4_74[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_74[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_74[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_74[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_74[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_74[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_74[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_74[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_74[0]), "=f"(_unpack_fp4_74[1]), "=f"(_unpack_fp4_74[2]), "=f"(_unpack_fp4_74[3]), "=f"(_unpack_fp4_74[4]), "=f"(_unpack_fp4_74[5]), "=f"(_unpack_fp4_74[6]), "=f"(_unpack_fp4_74[7]) : "r"(_fp4_wire_values_1[22 + 0]));
                    #endif
                }
                float2 a_358 = make_float2(acc_331[16], acc_331[17]);
                float2 q_359 = make_float2(_unpack_fp4_74[0], _unpack_fp4_74[1]);
                float2 r_360 = fma_f32x2_rn_noftz(q_359, scale1_333, a_358);
                acc_331[16] = r_360.x;
                acc_331[17] = r_360.y;
                float2 a_361 = make_float2(acc_331[18], acc_331[19]);
                float2 q_362 = make_float2(_unpack_fp4_74[2], _unpack_fp4_74[3]);
                float2 r_363 = fma_f32x2_rn_noftz(q_362, scale1_333, a_361);
                acc_331[18] = r_363.x;
                acc_331[19] = r_363.y;
                float2 a_364 = make_float2(acc_331[20], acc_331[21]);
                float2 q_365 = make_float2(_unpack_fp4_74[4], _unpack_fp4_74[5]);
                float2 r_366 = fma_f32x2_rn_noftz(q_365, scale1_333, a_364);
                acc_331[20] = r_366.x;
                acc_331[21] = r_366.y;
                float2 a_367 = make_float2(acc_331[22], acc_331[23]);
                float2 q_368 = make_float2(_unpack_fp4_74[6], _unpack_fp4_74[7]);
                float2 r_369 = fma_f32x2_rn_noftz(q_368, scale1_333, a_367);
                acc_331[22] = r_369.x;
                acc_331[23] = r_369.y;
                float _unpack_fp4_75[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[23 + 0]));
                    (&_unpack_fp4_75[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_75[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_75[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_75[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_75[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_75[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_75[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_75[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_75[0]), "=f"(_unpack_fp4_75[1]), "=f"(_unpack_fp4_75[2]), "=f"(_unpack_fp4_75[3]), "=f"(_unpack_fp4_75[4]), "=f"(_unpack_fp4_75[5]), "=f"(_unpack_fp4_75[6]), "=f"(_unpack_fp4_75[7]) : "r"(_fp4_wire_values_1[23 + 0]));
                    #endif
                }
                float2 a_370 = make_float2(acc_331[24], acc_331[25]);
                float2 q_371 = make_float2(_unpack_fp4_75[0], _unpack_fp4_75[1]);
                float2 r_372 = fma_f32x2_rn_noftz(q_371, scale1_333, a_370);
                acc_331[24] = r_372.x;
                acc_331[25] = r_372.y;
                float2 a_373 = make_float2(acc_331[26], acc_331[27]);
                float2 q_374 = make_float2(_unpack_fp4_75[2], _unpack_fp4_75[3]);
                float2 r_375 = fma_f32x2_rn_noftz(q_374, scale1_333, a_373);
                acc_331[26] = r_375.x;
                acc_331[27] = r_375.y;
                float2 a_376 = make_float2(acc_331[28], acc_331[29]);
                float2 q_377 = make_float2(_unpack_fp4_75[4], _unpack_fp4_75[5]);
                float2 r_378 = fma_f32x2_rn_noftz(q_377, scale1_333, a_376);
                acc_331[28] = r_378.x;
                acc_331[29] = r_378.y;
                float2 a_379 = make_float2(acc_331[30], acc_331[31]);
                float2 q_380 = make_float2(_unpack_fp4_75[6], _unpack_fp4_75[7]);
                float2 r_381 = fma_f32x2_rn_noftz(q_380, scale1_333, a_379);
                acc_331[30] = r_381.x;
                acc_331[31] = r_381.y;
                unsigned int acc_331_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_331[_lp*2 + 0], acc_331[_lp*2+1 + 0]));
                    acc_331_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_331_bf16[0]), "r"(acc_331_bf16[1]), "r"(acc_331_bf16[2]), "r"(acc_331_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_331_bf16[4]), "r"(acc_331_bf16[5]), "r"(acc_331_bf16[6]), "r"(acc_331_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_331_bf16[8]), "r"(acc_331_bf16[9]), "r"(acc_331_bf16[10]), "r"(acc_331_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_331_bf16[12]), "r"(acc_331_bf16[13]), "r"(acc_331_bf16[14]), "r"(acc_331_bf16[15]) : "memory");
                __syncwarp();
                int c_382 = lane_1;
                int l_383 = c_382 >> 2;
                unsigned int w4_384[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_384[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_384[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_384[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_384[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_383 + (c_382 & 3 ^ l_383 >> 1 & 3)))));
                long long idx_385 = crow64 * 3072 + (long long)(4 * (640 + c_382));
                {
                    int4 _iv4 = make_int4(w4_384[0 + 0], w4_384[0 + 1], w4_384[0 + 2], w4_384[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_385) = _iv4;
                }
                int c_386 = lane_1 + 32;
                int l_387 = c_386 >> 2;
                unsigned int w4_388[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_388[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_388[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_388[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_388[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_387 + (c_386 & 3 ^ l_387 >> 1 & 3)))));
                long long idx_389 = crow64 * 3072 + (long long)(4 * (640 + c_386));
                {
                    int4 _iv4 = make_int4(w4_388[0 + 0], w4_388[0 + 1], w4_388[0 + 2], w4_388[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_389) = _iv4;
                }
                int c_390 = lane_1 + 64;
                int l_391 = c_390 >> 2;
                unsigned int w4_392[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_392[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_392[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_392[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_392[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_391 + (c_390 & 3 ^ l_391 >> 1 & 3)))));
                long long idx_393 = crow64 * 3072 + (long long)(4 * (640 + c_390));
                {
                    int4 _iv4 = make_int4(w4_392[0 + 0], w4_392[0 + 1], w4_392[0 + 2], w4_392[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_393) = _iv4;
                }
                int c_394 = lane_1 + 96;
                int l_395 = c_394 >> 2;
                unsigned int w4_396[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_396[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_396[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_396[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_396[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_395 + (c_394 & 3 ^ l_395 >> 1 & 3)))));
                long long idx_397 = crow64 * 3072 + (long long)(4 * (640 + c_394));
                {
                    int4 _iv4 = make_int4(w4_396[0 + 0], w4_396[0 + 1], w4_396[0 + 2], w4_396[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_397) = _iv4;
                }
                __syncwarp();
            }
            if (lane_1 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            if (tid == 0) {
                int base_2 = bid * 4;
                ctrl[base_2 + 2] = (unsigned int)rows;
                ctrl[base_2 + 3] = 2;
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
            if (prev_algo != 0) {
                long long bid64_1 = (long long)bid;
                long long g64_1 = (long long)gridDim.x;
                long long lane64 = (long long)lane_0;
                long long pr64 = (long long)prev_rows;
                long long pro64 = (pr64 + 3) / 4;
                long long pb64 = (long long)pbuf;
                long long capo64_1 = (long long)((cap + 4 - 1) / 4);
                long long sa_w_1 = 4 * capo64_1 * 3456;
                long long sa_r_1 = 4 * capo64_1 * 16;
                long long rs_w_1 = pb64 * (2 * sa_w_1);
                long long rs_r_1 = pb64 * (2 * sa_r_1);
                long long v_base = rs_w_1;
                long long s_base = rs_w_1 + 4 * capo64_1 * 3072;
                long long m_base = rs_r_1;
                unsigned int sv4[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                unsigned int ss4[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                unsigned int sr4[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                long long u = pro64 * 3072 / 16;
                long long lo = u * bid64_1 / g64_1;
                long long hi = u * (bid64_1 + 1) / g64_1;
                long long g0 = lo + lane64;
                #pragma unroll 1
                for (long long g_2 = g0; g_2 < hi; g_2 += 32) {
                    long long widx = v_base + 16 * g_2 >> 2;
                    {
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx) = _iv4;
                    }
                }
                long long u_0 = pro64 * 384 / 16;
                long long lo_1 = u_0 * bid64_1 / g64_1;
                long long hi_2 = u_0 * (bid64_1 + 1) / g64_1;
                long long g0_3 = lo_1 + lane64;
                #pragma unroll 1
                for (long long g_3 = g0_3; g_3 < hi_2; g_3 += 32) {
                    long long widx_1 = s_base + 16 * g_3 >> 2;
                    {
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_1) = _iv4;
                    }
                }
                long long u_4 = pro64 * 16 / 16;
                long long lo_5 = u_4 * bid64_1 / g64_1;
                long long hi_6 = u_4 * (bid64_1 + 1) / g64_1;
                long long g0_7 = lo_5 + lane64;
                #pragma unroll 1
                for (long long g_4 = g0_7; g_4 < hi_6; g_4 += 32) {
                    long long widx_2 = m_base + 16 * g_4 >> 2;
                    {
                        int4 _iv4 = make_int4(sr4[0 + 0], sr4[0 + 1], sr4[0 + 2], sr4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_2) = _iv4;
                    }
                }
                long long v_base_8 = rs_w_1 + capo64_1 * 3072;
                long long s_base_9 = rs_w_1 + 4 * capo64_1 * 3072 + capo64_1 * 384;
                long long m_base_10 = rs_r_1 + capo64_1 * 16;
                unsigned int sv4_11[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                unsigned int ss4_12[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                unsigned int sr4_13[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                long long u_14 = pro64 * 3072 / 16;
                long long lo_15 = u_14 * bid64_1 / g64_1;
                long long hi_16 = u_14 * (bid64_1 + 1) / g64_1;
                long long g0_17 = lo_15 + lane64;
                #pragma unroll 1
                for (long long g_5 = g0_17; g_5 < hi_16; g_5 += 32) {
                    long long widx_3 = v_base_8 + 16 * g_5 >> 2;
                    {
                        int4 _iv4 = make_int4(sv4_11[0 + 0], sv4_11[0 + 1], sv4_11[0 + 2], sv4_11[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_3) = _iv4;
                    }
                }
                long long u_18 = pro64 * 384 / 16;
                long long lo_19 = u_18 * bid64_1 / g64_1;
                long long hi_20 = u_18 * (bid64_1 + 1) / g64_1;
                long long g0_21 = lo_19 + lane64;
                #pragma unroll 1
                for (long long g_6 = g0_21; g_6 < hi_20; g_6 += 32) {
                    long long widx_4 = s_base_9 + 16 * g_6 >> 2;
                    {
                        int4 _iv4 = make_int4(ss4_12[0 + 0], ss4_12[0 + 1], ss4_12[0 + 2], ss4_12[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_4) = _iv4;
                    }
                }
                long long u_22 = pro64 * 16 / 16;
                long long lo_23 = u_22 * bid64_1 / g64_1;
                long long hi_24 = u_22 * (bid64_1 + 1) / g64_1;
                long long g0_25 = lo_23 + lane64;
                #pragma unroll 1
                for (long long g_7 = g0_25; g_7 < hi_24; g_7 += 32) {
                    long long widx_5 = m_base_10 + 16 * g_7 >> 2;
                    {
                        int4 _iv4 = make_int4(sr4_13[0 + 0], sr4_13[0 + 1], sr4_13[0 + 2], sr4_13[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_5) = _iv4;
                    }
                }
                long long v_base_26 = rs_w_1 + 2 * capo64_1 * 3072;
                long long s_base_27 = rs_w_1 + 4 * capo64_1 * 3072 + 2 * capo64_1 * 384;
                long long m_base_28 = rs_r_1 + 2 * capo64_1 * 16;
                unsigned int sv4_29[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                unsigned int ss4_30[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                unsigned int sr4_31[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                long long u_32 = pro64 * 3072 / 16;
                long long lo_33 = u_32 * bid64_1 / g64_1;
                long long hi_34 = u_32 * (bid64_1 + 1) / g64_1;
                long long g0_35 = lo_33 + lane64;
                #pragma unroll 1
                for (long long g_8 = g0_35; g_8 < hi_34; g_8 += 32) {
                    long long widx_6 = v_base_26 + 16 * g_8 >> 2;
                    {
                        int4 _iv4 = make_int4(sv4_29[0 + 0], sv4_29[0 + 1], sv4_29[0 + 2], sv4_29[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_6) = _iv4;
                    }
                }
                long long u_36 = pro64 * 384 / 16;
                long long lo_37 = u_36 * bid64_1 / g64_1;
                long long hi_38 = u_36 * (bid64_1 + 1) / g64_1;
                long long g0_39 = lo_37 + lane64;
                #pragma unroll 1
                for (long long g_9 = g0_39; g_9 < hi_38; g_9 += 32) {
                    long long widx_7 = s_base_27 + 16 * g_9 >> 2;
                    {
                        int4 _iv4 = make_int4(ss4_30[0 + 0], ss4_30[0 + 1], ss4_30[0 + 2], ss4_30[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_7) = _iv4;
                    }
                }
                long long u_40 = pro64 * 16 / 16;
                long long lo_41 = u_40 * bid64_1 / g64_1;
                long long hi_42 = u_40 * (bid64_1 + 1) / g64_1;
                long long g0_43 = lo_41 + lane64;
                #pragma unroll 1
                for (long long g_10 = g0_43; g_10 < hi_42; g_10 += 32) {
                    long long widx_8 = m_base_28 + 16 * g_10 >> 2;
                    {
                        int4 _iv4 = make_int4(sr4_31[0 + 0], sr4_31[0 + 1], sr4_31[0 + 2], sr4_31[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_8) = _iv4;
                    }
                }
                long long v_base_44 = rs_w_1 + 3 * capo64_1 * 3072;
                long long s_base_45 = rs_w_1 + 4 * capo64_1 * 3072 + 3 * capo64_1 * 384;
                long long m_base_46 = rs_r_1 + 3 * capo64_1 * 16;
                unsigned int sv4_47[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                unsigned int ss4_48[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                unsigned int sr4_49[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                long long u_50 = pro64 * 3072 / 16;
                long long lo_51 = u_50 * bid64_1 / g64_1;
                long long hi_52 = u_50 * (bid64_1 + 1) / g64_1;
                long long g0_53 = lo_51 + lane64;
                #pragma unroll 1
                for (long long g_11 = g0_53; g_11 < hi_52; g_11 += 32) {
                    long long widx_9 = v_base_44 + 16 * g_11 >> 2;
                    {
                        int4 _iv4 = make_int4(sv4_47[0 + 0], sv4_47[0 + 1], sv4_47[0 + 2], sv4_47[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_9) = _iv4;
                    }
                }
                long long u_54 = pro64 * 384 / 16;
                long long lo_55 = u_54 * bid64_1 / g64_1;
                long long hi_56 = u_54 * (bid64_1 + 1) / g64_1;
                long long g0_57 = lo_55 + lane64;
                #pragma unroll 1
                for (long long g_12 = g0_57; g_12 < hi_56; g_12 += 32) {
                    long long widx_10 = s_base_45 + 16 * g_12 >> 2;
                    {
                        int4 _iv4 = make_int4(ss4_48[0 + 0], ss4_48[0 + 1], ss4_48[0 + 2], ss4_48[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_10) = _iv4;
                    }
                }
                long long u_58 = pro64 * 16 / 16;
                long long lo_59 = u_58 * bid64_1 / g64_1;
                long long hi_60 = u_58 * (bid64_1 + 1) / g64_1;
                long long g0_61 = lo_59 + lane64;
                #pragma unroll 1
                for (long long g_13 = g0_61; g_13 < hi_60; g_13 += 32) {
                    long long widx_11 = m_base_46 + 16 * g_13 >> 2;
                    {
                        int4 _iv4 = make_int4(sr4_49[0 + 0], sr4_49[0 + 1], sr4_49[0 + 2], sr4_49[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_11) = _iv4;
                    }
                }
                if (0 != rank) {
                    long long v_base2 = rs_w_1 + sa_w_1;
                    long long s_base2 = rs_w_1 + sa_w_1 + 4 * capo64_1 * 3072;
                    long long m_base2 = rs_r_1 + sa_r_1;
                    unsigned int sv4_0[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                    unsigned int ss4_1[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                    unsigned int sr4_2[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                    long long u_3 = pro64 * 3072 / 16;
                    long long lo_4 = u_3 * bid64_1 / g64_1;
                    long long hi_5 = u_3 * (bid64_1 + 1) / g64_1;
                    long long g0_6 = lo_4 + lane64;
                    #pragma unroll 1
                    for (long long g_14 = g0_6; g_14 < hi_5; g_14 += 32) {
                        long long widx_12 = v_base2 + 16 * g_14 >> 2;
                        {
                            int4 _iv4 = make_int4(sv4_0[0 + 0], sv4_0[0 + 1], sv4_0[0 + 2], sv4_0[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_12) = _iv4;
                        }
                    }
                    long long u_7 = pro64 * 384 / 16;
                    long long lo_8 = u_7 * bid64_1 / g64_1;
                    long long hi_9 = u_7 * (bid64_1 + 1) / g64_1;
                    long long g0_10 = lo_8 + lane64;
                    #pragma unroll 1
                    for (long long g_15 = g0_10; g_15 < hi_9; g_15 += 32) {
                        long long widx_13 = s_base2 + 16 * g_15 >> 2;
                        {
                            int4 _iv4 = make_int4(ss4_1[0 + 0], ss4_1[0 + 1], ss4_1[0 + 2], ss4_1[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_13) = _iv4;
                        }
                    }
                    long long u_11 = pro64 * 16 / 16;
                    long long lo_12 = u_11 * bid64_1 / g64_1;
                    long long hi_13 = u_11 * (bid64_1 + 1) / g64_1;
                    long long g0_14 = lo_12 + lane64;
                    #pragma unroll 1
                    for (long long g_16 = g0_14; g_16 < hi_13; g_16 += 32) {
                        long long widx_14 = m_base2 + 16 * g_16 >> 2;
                        {
                            int4 _iv4 = make_int4(sr4_2[0 + 0], sr4_2[0 + 1], sr4_2[0 + 2], sr4_2[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_14) = _iv4;
                        }
                    }
                }
                if (1 != rank) {
                    long long v_base2_1 = rs_w_1 + sa_w_1 + capo64_1 * 3072;
                    long long s_base2_1 = rs_w_1 + sa_w_1 + 4 * capo64_1 * 3072 + capo64_1 * 384;
                    long long m_base2_1 = rs_r_1 + sa_r_1 + capo64_1 * 16;
                    unsigned int sv4_0_1[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                    unsigned int ss4_1_1[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                    unsigned int sr4_2_1[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                    long long u_3_1 = pro64 * 3072 / 16;
                    long long lo_4_1 = u_3_1 * bid64_1 / g64_1;
                    long long hi_5_1 = u_3_1 * (bid64_1 + 1) / g64_1;
                    long long g0_6_1 = lo_4_1 + lane64;
                    #pragma unroll 1
                    for (long long g_17 = g0_6_1; g_17 < hi_5_1; g_17 += 32) {
                        long long widx_15 = v_base2_1 + 16 * g_17 >> 2;
                        {
                            int4 _iv4 = make_int4(sv4_0_1[0 + 0], sv4_0_1[0 + 1], sv4_0_1[0 + 2], sv4_0_1[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_15) = _iv4;
                        }
                    }
                    long long u_7_1 = pro64 * 384 / 16;
                    long long lo_8_1 = u_7_1 * bid64_1 / g64_1;
                    long long hi_9_1 = u_7_1 * (bid64_1 + 1) / g64_1;
                    long long g0_10_1 = lo_8_1 + lane64;
                    #pragma unroll 1
                    for (long long g_18 = g0_10_1; g_18 < hi_9_1; g_18 += 32) {
                        long long widx_16 = s_base2_1 + 16 * g_18 >> 2;
                        {
                            int4 _iv4 = make_int4(ss4_1_1[0 + 0], ss4_1_1[0 + 1], ss4_1_1[0 + 2], ss4_1_1[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_16) = _iv4;
                        }
                    }
                    long long u_11_1 = pro64 * 16 / 16;
                    long long lo_12_1 = u_11_1 * bid64_1 / g64_1;
                    long long hi_13_1 = u_11_1 * (bid64_1 + 1) / g64_1;
                    long long g0_14_1 = lo_12_1 + lane64;
                    #pragma unroll 1
                    for (long long g_19 = g0_14_1; g_19 < hi_13_1; g_19 += 32) {
                        long long widx_17 = m_base2_1 + 16 * g_19 >> 2;
                        {
                            int4 _iv4 = make_int4(sr4_2_1[0 + 0], sr4_2_1[0 + 1], sr4_2_1[0 + 2], sr4_2_1[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_17) = _iv4;
                        }
                    }
                }
                if (2 != rank) {
                    long long v_base2_2 = rs_w_1 + sa_w_1 + 2 * capo64_1 * 3072;
                    long long s_base2_2 = rs_w_1 + sa_w_1 + 4 * capo64_1 * 3072 + 2 * capo64_1 * 384;
                    long long m_base2_2 = rs_r_1 + sa_r_1 + 2 * capo64_1 * 16;
                    unsigned int sv4_0_2[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                    unsigned int ss4_1_2[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                    unsigned int sr4_2_2[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                    long long u_3_2 = pro64 * 3072 / 16;
                    long long lo_4_2 = u_3_2 * bid64_1 / g64_1;
                    long long hi_5_2 = u_3_2 * (bid64_1 + 1) / g64_1;
                    long long g0_6_2 = lo_4_2 + lane64;
                    #pragma unroll 1
                    for (long long g_20 = g0_6_2; g_20 < hi_5_2; g_20 += 32) {
                        long long widx_18 = v_base2_2 + 16 * g_20 >> 2;
                        {
                            int4 _iv4 = make_int4(sv4_0_2[0 + 0], sv4_0_2[0 + 1], sv4_0_2[0 + 2], sv4_0_2[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_18) = _iv4;
                        }
                    }
                    long long u_7_2 = pro64 * 384 / 16;
                    long long lo_8_2 = u_7_2 * bid64_1 / g64_1;
                    long long hi_9_2 = u_7_2 * (bid64_1 + 1) / g64_1;
                    long long g0_10_2 = lo_8_2 + lane64;
                    #pragma unroll 1
                    for (long long g_21 = g0_10_2; g_21 < hi_9_2; g_21 += 32) {
                        long long widx_19 = s_base2_2 + 16 * g_21 >> 2;
                        {
                            int4 _iv4 = make_int4(ss4_1_2[0 + 0], ss4_1_2[0 + 1], ss4_1_2[0 + 2], ss4_1_2[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_19) = _iv4;
                        }
                    }
                    long long u_11_2 = pro64 * 16 / 16;
                    long long lo_12_2 = u_11_2 * bid64_1 / g64_1;
                    long long hi_13_2 = u_11_2 * (bid64_1 + 1) / g64_1;
                    long long g0_14_2 = lo_12_2 + lane64;
                    #pragma unroll 1
                    for (long long g_22 = g0_14_2; g_22 < hi_13_2; g_22 += 32) {
                        long long widx_20 = m_base2_2 + 16 * g_22 >> 2;
                        {
                            int4 _iv4 = make_int4(sr4_2_2[0 + 0], sr4_2_2[0 + 1], sr4_2_2[0 + 2], sr4_2_2[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_20) = _iv4;
                        }
                    }
                }
                if (3 != rank) {
                    long long v_base2_3 = rs_w_1 + sa_w_1 + 3 * capo64_1 * 3072;
                    long long s_base2_3 = rs_w_1 + sa_w_1 + 4 * capo64_1 * 3072 + 3 * capo64_1 * 384;
                    long long m_base2_3 = rs_r_1 + sa_r_1 + 3 * capo64_1 * 16;
                    unsigned int sv4_0_3[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                    unsigned int ss4_1_3[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                    unsigned int sr4_2_3[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                    long long u_3_3 = pro64 * 3072 / 16;
                    long long lo_4_3 = u_3_3 * bid64_1 / g64_1;
                    long long hi_5_3 = u_3_3 * (bid64_1 + 1) / g64_1;
                    long long g0_6_3 = lo_4_3 + lane64;
                    #pragma unroll 1
                    for (long long g_23 = g0_6_3; g_23 < hi_5_3; g_23 += 32) {
                        long long widx_21 = v_base2_3 + 16 * g_23 >> 2;
                        {
                            int4 _iv4 = make_int4(sv4_0_3[0 + 0], sv4_0_3[0 + 1], sv4_0_3[0 + 2], sv4_0_3[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_21) = _iv4;
                        }
                    }
                    long long u_7_3 = pro64 * 384 / 16;
                    long long lo_8_3 = u_7_3 * bid64_1 / g64_1;
                    long long hi_9_3 = u_7_3 * (bid64_1 + 1) / g64_1;
                    long long g0_10_3 = lo_8_3 + lane64;
                    #pragma unroll 1
                    for (long long g_24 = g0_10_3; g_24 < hi_9_3; g_24 += 32) {
                        long long widx_22 = s_base2_3 + 16 * g_24 >> 2;
                        {
                            int4 _iv4 = make_int4(ss4_1_3[0 + 0], ss4_1_3[0 + 1], ss4_1_3[0 + 2], ss4_1_3[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_22) = _iv4;
                        }
                    }
                    long long u_11_3 = pro64 * 16 / 16;
                    long long lo_12_3 = u_11_3 * bid64_1 / g64_1;
                    long long hi_13_3 = u_11_3 * (bid64_1 + 1) / g64_1;
                    long long g0_14_3 = lo_12_3 + lane64;
                    #pragma unroll 1
                    for (long long g_25 = g0_14_3; g_25 < hi_13_3; g_25 += 32) {
                        long long widx_23 = m_base2_3 + 16 * g_25 >> 2;
                        {
                            int4 _iv4 = make_int4(sr4_2_3[0 + 0], sr4_2_3[0 + 1], sr4_2_3[0 + 2], sr4_2_3[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_23) = _iv4;
                        }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
